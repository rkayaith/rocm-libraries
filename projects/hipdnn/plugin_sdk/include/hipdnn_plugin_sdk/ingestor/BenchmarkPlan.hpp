// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

#include <hipdnn_plugin_sdk/EnginePluginTypeTraits.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/WinnerCache.hpp>
#include <hipdnn_plugin_sdk/interfaces/IPlan.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// Timing constants, not knobs: Part 1 fixes the mechanism. These are starting values,
/// not measured ones -- the gfx942 run confirmed sampling executes and selects, but did
/// not measure per-candidate variance, so whether this many iterations makes a pointwise
/// kernel's timing stable is still open (plan §9 uncertainty 1).
constexpr int BENCHMARK_WARMUP_RUNS = 1;
constexpr int BENCHMARK_ITERATIONS = 5;

namespace detail
{

/// Owns at most one hipEvent_t and always destroys it, including when a candidate's
/// execute() throws between construction and destruction.
class ScopedHipEvent
{
public:
    ScopedHipEvent()
    {
        if(hipEventCreate(&_event) != hipSuccess)
        {
            _event = nullptr;
        }
    }

    ~ScopedHipEvent()
    {
        if(_event != nullptr)
        {
            // Nothing actionable in a destructor; the event is being discarded anyway.
            static_cast<void>(hipEventDestroy(_event));
        }
    }

    ScopedHipEvent(const ScopedHipEvent&) = delete;
    ScopedHipEvent& operator=(const ScopedHipEvent&) = delete;
    ScopedHipEvent(ScopedHipEvent&&) = delete;
    ScopedHipEvent& operator=(ScopedHipEvent&&) = delete;

    bool valid() const
    {
        return _event != nullptr;
    }

    hipEvent_t get() const
    {
        return _event;
    }

private:
    hipEvent_t _event = nullptr;
};

} // namespace detail

/// A composite IPlan owning every knob-filtered catalog entry as its own GenericPlan,
/// timing each on the first execute() and delegating every call after to the fastest.
/// Wraps GenericPlan rather than widening it: a single-kernel plan's construction,
/// workspace query, and null-prepared check stay exactly what they are today.
///
/// Timing goes through IPlan::execute() only, on the handle's own stream -- this class
/// never touches a dispatcher, a PreparedDispatch, or any HIP launch API beyond the
/// bracketing events.
template <typename THandle>
class BenchmarkPlan : public IPlan<THandle>
{
public:
    /// One sub-plan plus the kernel it was built for. The ids ride alongside because
    /// IPlan has no kernel accessor and must not grow one: kernelId is what the selection
    /// log names, and packId/dispatchId are the staleness cross-check a cached ranking is
    /// validated against on a later run.
    ///
    /// The two staleness ids default and sit after `plan` so a caller that does not cache
    /// -- every existing SDK fixture -- constructs a Candidate exactly as before. A
    /// default-constructed packId never matches a real one, so an uncached candidate
    /// simply never resolves against a record, which is the correct outcome.
    struct Candidate
    {
        DescriptorId kernelId;
        std::unique_ptr<IPlan<THandle>> plan;
        DescriptorId packId{};
        DescriptorId dispatchId{};
    };

    /// Invoked once, with every usable candidate in benchmarked order, after sampling
    /// resolves the winner. An **absent callback means no caching** -- the flag-off path,
    /// the SDK fixtures, and any provider that does not want a cache simply pass nothing,
    /// so this stays a plain std::function rather than a cache reference. BenchmarkPlan
    /// deliberately knows nothing about the cache's type or lifetime.
    using RecordRankingFn = std::function<void(std::vector<RankedEntry>)>;

    /// @param handle Used once, here, to size every sub-plan's workspace requirement;
    ///        execute() always uses the handle its own caller passes.
    /// @throws HipdnnPluginException(INTERNAL_ERROR) if @p candidates is empty.
    BenchmarkPlan(std::vector<Candidate> candidates,
                  const THandle& handle,
                  RecordRankingFn recordRanking = {})
        : _candidates(std::move(candidates))
        , _recordRanking(std::move(recordRanking))
    {
        static_assert(HasGetStream<THandle>::value,
                      "BenchmarkPlan requires THandle to have a 'hipStream_t getStream() const' "
                      "method");

        if(_candidates.empty())
        {
            throw HipdnnPluginException(HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                                        "BenchmarkPlan constructed with no candidates");
        }

        for(const auto& candidate : _candidates)
        {
            _workspaceBytes = std::max(_workspaceBytes, candidate.plan->getWorkspaceSize(handle));
        }
    }

    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    size_t getWorkspaceSize(const THandle& /*handle*/) const override
    {
        return _workspaceBytes;
    }

    /// Sampling runs candidates against the caller's real buffers, so a candidate that
    /// fails mid-loop can leave a partial result behind. What makes that safe is that
    /// this function always ends with the delegated execute below, which overwrites the
    /// output with the winner's. Never add an early return between resolveChosen() and
    /// that delegation.
    // NOLINTNEXTLINE(portability-template-virtual-member-function)
    void execute(const THandle& handle,
                 const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                 uint32_t numDeviceBuffers,
                 void* workspace = nullptr) const override
    {
        const size_t chosen = resolveChosen(handle, deviceBuffers, numDeviceBuffers, workspace);
        _candidates[chosen].plan->execute(handle, deviceBuffers, numDeviceBuffers, workspace);
    }

private:
    /// Resolves _chosen on the first call under the mutex; every later call returns the
    /// cached winner without re-sampling. Two threads racing the first execute() see one
    /// sampling pass, not two.
    ///
    /// The lock is held across the whole sweep, so a second thread executing with
    /// different buffers blocks until sampling finishes rather than proceeding in
    /// parallel. That is the deliberate trade for sampling exactly once: a
    /// double-checked lock here would let two threads sample concurrently, against each
    /// other's buffers. Only the first execute() pays it.
    size_t resolveChosen(const THandle& handle,
                         const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                         uint32_t numDeviceBuffers,
                         void* workspace) const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        if(_chosen.has_value())
        {
            return *_chosen;
        }

        // Every usable candidate's time is retained, not just the running minimum: the
        // winner cache stores the whole ranking so a later run whose knob filter excludes
        // the winner can still serve the best surviving candidate.
        std::vector<std::pair<double, size_t>> ranked;
        ranked.reserve(_candidates.size());

        for(size_t index = 0; index < _candidates.size(); ++index)
        {
            const auto timeMs
                = sampleCandidate(index, handle, deviceBuffers, numDeviceBuffers, workspace);
            if(!timeMs.has_value())
            {
                // Omitted from the ranking, never appended with a sentinel time: a
                // candidate that threw or failed to time is known-broken, and recording
                // it as a low-ranked fallback would let it be served ahead of the normal
                // ranked path on a later run.
                continue;
            }
            ranked.emplace_back(*timeMs, index);
        }

        // stable_sort, not sort: ties must resolve to the lowest candidate index. A plain
        // std::sort would reorder equal times arbitrarily and silently change which
        // kernel wins -- and micro-kernels at a few microseconds do tie.
        std::stable_sort(ranked.begin(), ranked.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.first < rhs.first;
        });

        size_t best = 0;
        if(ranked.empty())
        {
            HIPDNN_PLUGIN_LOG_ERROR("ingestor: benchmarking found no usable candidate among "
                                    << _candidates.size() << " kernel(s); defaulting to "
                                    << toString(_candidates.front().kernelId));
            // Nothing is recorded here: an all-unusable sweep has no ranking, and caching
            // index 0 would cache a guess.
        }
        else
        {
            best = ranked.front().second;
            HIPDNN_PLUGIN_LOG_INFO("ingestor: benchmarking selected kernel "
                                   << toString(_candidates[best].kernelId) << " in "
                                   << ranked.front().first << " ms among " << _candidates.size()
                                   << " candidate(s)");

            if(_recordRanking)
            {
                std::vector<RankedEntry> entries;
                entries.reserve(ranked.size());
                for(const auto& [timeMs, index] : ranked)
                {
                    const auto& candidate = _candidates[index];
                    entries.push_back(RankedEntry{
                        candidate.kernelId, candidate.packId, candidate.dispatchId, timeMs});
                }
                _recordRanking(std::move(entries));
            }
        }

        _chosen = best;
        return best;
    }

protected:
    /// The minimum timed execute() over BENCHMARK_ITERATIONS, after BENCHMARK_WARMUP_RUNS
    /// untimed ones, or nullopt if the candidate threw or a HIP event call failed -- both
    /// are a loss for this candidate, never a throw out of resolveChosen().
    ///
    /// Virtual purely as a **test seam**, and it changes no production behaviour: the
    /// timed path below needs real hipEvents, so on a machine with no device every
    /// candidate scores unusable and the ranking is always empty. That would leave the
    /// write-back path -- and the story's central claim, that a sampled ranking survives
    /// its plan -- green while proving nothing. A test subclass returns deterministic
    /// times instead, so the ranking, the omission of failed candidates, and the
    /// all-unusable case are all assertable without hardware. The device path itself is
    /// proven on gfx942.
    virtual std::optional<double> sampleCandidate(size_t index,
                                                  const THandle& handle,
                                                  const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                                  uint32_t numDeviceBuffers,
                                                  void* workspace) const
    {
        const auto& candidate = _candidates[index];
        try
        {
            for(int warmup = 0; warmup < BENCHMARK_WARMUP_RUNS; ++warmup)
            {
                candidate.plan->execute(handle, deviceBuffers, numDeviceBuffers, workspace);
            }

            double bestMs = std::numeric_limits<double>::max();
            for(int iteration = 0; iteration < BENCHMARK_ITERATIONS; ++iteration)
            {
                const auto sampleMs
                    = timeOneExecute(candidate, handle, deviceBuffers, numDeviceBuffers, workspace);
                if(!sampleMs.has_value())
                {
                    HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                                           << toString(candidate.kernelId)
                                           << "' failed to time a launch; scored unusable");
                    return std::nullopt;
                }
                bestMs = std::min(bestMs, *sampleMs);
            }
            return bestMs;
        }
        catch(const std::exception& error)
        {
            HIPDNN_PLUGIN_LOG_WARN("ingestor: benchmarking candidate '"
                                   << toString(candidate.kernelId)
                                   << "' threw and is scored unusable: " << error.what());
            return std::nullopt;
        }
    }

private:
    /// One warmup-free, timed execute() bracketed by hipEvents on handle.getStream() --
    /// never the null stream, or a plan on a non-default stream measures nothing.
    std::optional<double> timeOneExecute(const Candidate& candidate,
                                         const THandle& handle,
                                         const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                                         uint32_t numDeviceBuffers,
                                         void* workspace) const
    {
        const detail::ScopedHipEvent start;
        const detail::ScopedHipEvent stop;
        if(!start.valid() || !stop.valid())
        {
            return std::nullopt;
        }

        const auto stream = handle.getStream();
        if(hipEventRecord(start.get(), stream) != hipSuccess)
        {
            return std::nullopt;
        }

        candidate.plan->execute(handle, deviceBuffers, numDeviceBuffers, workspace);

        if(hipEventRecord(stop.get(), stream) != hipSuccess
           || hipEventSynchronize(stop.get()) != hipSuccess)
        {
            return std::nullopt;
        }

        float elapsedMs = 0.0F;
        if(hipEventElapsedTime(&elapsedMs, start.get(), stop.get()) != hipSuccess)
        {
            return std::nullopt;
        }
        return static_cast<double>(elapsedMs);
    }

    std::vector<Candidate> _candidates;
    RecordRankingFn _recordRanking;
    size_t _workspaceBytes = 0;
    mutable std::optional<size_t> _chosen;
    mutable std::mutex _mutex;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
