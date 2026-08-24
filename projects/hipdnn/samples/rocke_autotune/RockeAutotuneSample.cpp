// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file RockeAutotuneSample.cpp
 * @brief Autotuning a real SDPA graph across competing rocKE kernel variants, and
 *        showing the measured winner reach the cache and be reused.
 *
 * Follows the pattern of AutotuneSample.cpp: a main() with scenarios in individual
 * functions. Run with --help for usage.
 *
 * HOW THIS DIFFERS FROM AutotuneSample.cpp
 * ----------------------------------------
 * That sample autotunes a convolution over MIOpen/CK's built-in engines and writes a
 * heuristic config file it deletes in the same process. It never shows a cache being
 * READ back, and never involves the kernel ingestor.
 *
 * This one runs several rocKE-authored SDPA kernels that genuinely compete for one
 * graph, then shows the exact-match ranking cache written and then hit.
 *
 * WHY THE VARIANTS COMPETE
 * ------------------------
 * The attention_dense pack's `kernel_match` compares the seven geometry fields a rocKE
 * kernel bakes in -- dtype, head_size, head counts, sequence lengths, causal -- and
 * deliberately NOT `block_n`, the KV tile length. Several kernels built for one geometry
 * at different `block_n` therefore all match the same graph, and the pack's `score()`
 * ranks them. `block_n` is the knob the engine's UED exposes, which is what gives an
 * autotuner something to turn.
 *
 * The descriptor set this expects (examples/descriptors/rocKE/
 * gfx942_attention_dense_autotune) ships three such kernels at block_n 32/64/128, bf16,
 * head_size 64, seqlen 256. head_size 64 rather than 128 because at D=128 only block_n
 * 32 and 64 fit gfx942's 64 KB LDS budget -- D=64 leaves three real competitors.
 *
 * OBSERVING THE CACHE
 * -------------------
 * There is no `bool wasCacheHit` in the public API. Two channels exist; this uses both:
 *
 *   * WRITE side: autotuneExhaustiveSweep() reports AutotuneCacheWriteOutcome. A first
 *     sweep reports WRITTEN; an immediate re-sweep measuring the same order has nothing
 *     to add and reports UNCHANGED.
 *   * READ side: log fragments only. The backend's config heuristic logs "exact-match
 *     cache hit" / "exact-match cache miss" at info level, so this installs a log
 *     callback and inspects what was emitted.
 *
 * The cache keys on (graph content, device). Nothing here asserts WHICH block_n wins:
 * that is a hardware measurement, and pinning it would make the sample fail on a part
 * where a different tile is genuinely faster.
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Logging.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include "../utils/Helpers.hpp"

namespace
{

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
namespace utilities = hipdnn_data_sdk::utilities;

/// The geometry the competing descriptors are built for. A rocKE kernel bakes its shape
/// in, so a graph differing in any of these is declined by every variant and there is
/// nothing to tune.
constexpr int64_t BATCH = 1;
constexpr int64_t NUM_QUERY_HEADS = 8;
constexpr int64_t NUM_KV_HEADS = 8;
constexpr int64_t SEQLEN_Q = 256;
constexpr int64_t SEQLEN_KV = 256;
constexpr int64_t HEAD_SIZE = 64;

constexpr int64_t UID_Q = 1;
constexpr int64_t UID_K = 2;
constexpr int64_t UID_V = 3;
constexpr int64_t UID_O = 4;

/// Log fragments the INGESTOR emits while racing its candidates and while reusing a
/// benchmarked record. The count of raced kernels is reported on the SELECTED line
/// ("among N candidate(s)") rather than one line per candidate, so that is what the
/// sample parses.
constexpr const char* BENCHMARK_SELECTED_FRAGMENT = "ingestor: benchmarking selected kernel";
constexpr const char* RECORD_REUSED_FRAGMENT = "from a benchmarked record of";

/// Captures backend log lines so the sample can observe the read-side cache decision.
class LogCapture
{
public:
    static LogCapture& instance()
    {
        static LogCapture s_instance;
        return s_instance;
    }

    void clear()
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        _lines.clear();
    }

    void record(const char* message)
    {
        if(message == nullptr)
        {
            return;
        }
        const std::lock_guard<std::mutex> lock(_mutex);
        _lines.emplace_back(message);
    }

    /// The most recent captured line containing @p fragment, or empty when none did.
    std::string lastContaining(const std::string& fragment)
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        for(auto it = _lines.rbegin(); it != _lines.rend(); ++it)
        {
            if(it->find(fragment) != std::string::npos)
            {
                return *it;
            }
        }
        return {};
    }

    /// The candidate count the ingestor reports on its selection line, which reads
    /// "... in 0.0245 ms among 3 candidate(s)". Zero when the line is absent or does not
    /// carry a count, which is what a run that never raced anything looks like.
    static size_t candidatesFrom(const std::string& selectedLine)
    {
        const auto at = selectedLine.rfind("among ");
        if(at == std::string::npos)
        {
            return 0;
        }
        return static_cast<size_t>(std::strtoul(selectedLine.c_str() + at + 6, nullptr, 10));
    }

    /// How many captured lines contain @p fragment. The candidate count is how the
    /// sample shows a real race happened rather than a single kernel being rubber-stamped.
    size_t countContaining(const std::string& fragment)
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        size_t hits = 0;
        for(const auto& line : _lines)
        {
            if(line.find(fragment) != std::string::npos)
            {
                ++hits;
            }
        }
        return hits;
    }

private:
    std::mutex _mutex;
    std::vector<std::string> _lines;
};

void logCallback(hipdnnUserLogCallbackHandle_t /*userHandle*/,
                 hipdnnSeverity_t /*severity*/,
                 const char* message)
{
    LogCapture::instance().record(message);
}

/// A BSHD-strided rank-4 operand. Dims are [B, H, S, D] -- hipDNN's canonical SDPA
/// order -- while memory is token-major with heads interleaved, the layout the rocKE
/// dense kernel bakes into its addressing.
std::shared_ptr<TensorAttributes>
    makeOperand(int64_t uid, const std::string& name, int64_t heads, int64_t sequence)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_uid(uid)
        .set_name(name)
        .set_dim({BATCH, heads, sequence, HEAD_SIZE})
        .set_stride({sequence * heads * HEAD_SIZE, HEAD_SIZE, heads * HEAD_SIZE, 1})
        .set_data_type(DataType::BFLOAT16);
    return tensor;
}

/// The one graph shape every competing variant claims: a single causal SDPA node.
std::shared_ptr<Graph> buildSdpaGraph()
{
    auto graph = std::make_shared<Graph>();
    graph->set_name("rocke_autotune_sdpa")
        .set_io_data_type(DataType::BFLOAT16)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto q = makeOperand(UID_Q, "Q", NUM_QUERY_HEADS, SEQLEN_Q);
    auto k = makeOperand(UID_K, "K", NUM_KV_HEADS, SEQLEN_KV);
    auto v = makeOperand(UID_V, "V", NUM_KV_HEADS, SEQLEN_KV);

    SdpaAttributes attributes;
    attributes.set_name("rocke_autotune_sdpa")
        .set_causal_mask(true)
        .set_attn_scale(1.0F / std::sqrt(static_cast<float>(HEAD_SIZE)));

    auto [o, stats] = graph->sdpa(q, k, v, attributes);
    static_cast<void>(stats);
    // O declares the same BSHD layout as its inputs: the kernel writes its output with
    // the addressing it reads them with, and inferred strides would not match.
    o->set_uid(UID_O)
        .set_name("O")
        .set_output(true)
        .set_dim({BATCH, NUM_QUERY_HEADS, SEQLEN_Q, HEAD_SIZE})
        .set_stride(
            {SEQLEN_Q * NUM_QUERY_HEADS * HEAD_SIZE, HEAD_SIZE, NUM_QUERY_HEADS * HEAD_SIZE, 1})
        .set_data_type(DataType::BFLOAT16);

    return graph;
}

/// Device-side operands, kept alive for the whole run so a variant pack stays valid.
struct Operands
{
    std::vector<std::unique_ptr<utilities::Tensor<hipdnn_data_sdk::types::bfloat16>>> tensors;
    std::unordered_map<int64_t, void*> variantPack;
};

Operands allocateOperands()
{
    using bf16 = hipdnn_data_sdk::types::bfloat16;
    Operands operands;

    const auto add = [&operands](int64_t uid, int64_t heads, int64_t sequence, unsigned seed) {
        auto tensor = std::make_unique<utilities::Tensor<bf16>>(
            std::vector<int64_t>{BATCH, heads, sequence, HEAD_SIZE},
            std::vector<int64_t>{sequence * heads * HEAD_SIZE, HEAD_SIZE, heads * HEAD_SIZE, 1});
        tensor->fillTensorWithRandomValues(-1.0F, 1.0F, seed);
        operands.variantPack[uid] = tensor->memory().deviceData();
        operands.tensors.push_back(std::move(tensor));
    };

    add(UID_Q, NUM_QUERY_HEADS, SEQLEN_Q, 1);
    add(UID_K, NUM_KV_HEADS, SEQLEN_KV, 2);
    add(UID_V, NUM_KV_HEADS, SEQLEN_KV, 3);
    add(UID_O, NUM_QUERY_HEADS, SEQLEN_Q, 4);
    return operands;
}

const char* describe(AutotuneCacheWriteOutcome outcome)
{
    switch(outcome)
    {
    case AutotuneCacheWriteOutcome::WRITTEN:
        return "WRITTEN";
    case AutotuneCacheWriteOutcome::UNCHANGED:
        return "UNCHANGED  (the stored ranking already matched)";
    case AutotuneCacheWriteOutcome::DECLINED_DISABLED:
        return "DECLINED_DISABLED  (HIPDNN_DISABLE_EXACT_ENGINE_CACHE)";
    case AutotuneCacheWriteOutcome::DECLINED_UNKEYABLE:
        return "DECLINED_UNKEYABLE  (graph/device did not reduce to a cache key)";
    case AutotuneCacheWriteOutcome::NOT_ATTEMPTED_NO_SUCCESSFUL_ENGINE:
        return "NOT_ATTEMPTED_NO_SUCCESSFUL_ENGINE";
    }
    return "UNKNOWN";
}

void printRanking(const std::vector<AutotuneResult>& results)
{
    // robust ms first, because that is what autotuneExhaustiveSweep() ranks on: the
    // mean after discarding the slow tail, i.e. what the engine usually costs. min ms
    // is shown beside it since a large gap between the two is the volatility the robust
    // statistic exists to ignore.
    std::cout << "      rank  engine                             robust ms    min ms\n";
    int rank = 0;
    for(const auto& result : results)
    {
        std::cout << "      " << std::setw(4) << rank++ << "  " << std::left << std::setw(36)
                  << result.engineName.substr(0, 36) << std::right << std::fixed
                  << std::setprecision(4) << std::setw(9) << result.robustTimeMs << std::setw(10)
                  << result.minTimeMs << '\n';
    }
}

/// Scenario 1: which engine serves this graph?
///
/// Expect exactly ONE. The competing block_n kernels live inside a single engine, so
/// the frontend sees one engine id no matter how many variants are packed -- the count
/// here says nothing about how many kernels will race. Scenario 2 is where the variants
/// become visible. This exists to confirm the rocKE engine is reachable at all before
/// anything is measured.
bool showCandidates(hipdnnHandle_t handle)
{
    std::cout << "\n[1] Engine serving the SDPA graph\n";
    auto graph = buildSdpaGraph();

    auto result = graph->build_operation_graph(handle);
    if(result.is_bad())
    {
        std::cout << "    graph not supported here: " << result.err_msg << '\n'
                  << "    Needs the rocKE attention_dense descriptors packed for this device.\n";
        return false;
    }

    std::vector<int64_t> engineIds;
    result = graph->get_ranked_engine_ids(engineIds);
    if(result.is_bad())
    {
        std::cout << "    no engine offered itself: " << result.err_msg << '\n';
        return false;
    }
    std::cout << "    engines offering themselves: " << engineIds.size() << '\n';
    for(const auto id : engineIds)
    {
        std::cout << "      engine id " << id << '\n';
    }
    if(engineIds.empty())
    {
        std::cout << "    NOTE: nothing serves this graph, so there is nothing to race.\n";
        return false;
    }
    return true;
}

/// Scenario 2: an exhaustive sweep races the kernels and populates both caches.
///
/// One call drives both layers, which is why it is the right entry point even though
/// this graph has a single engine:
///
///   * EXHAUSTIVE mode first builds temporary PRIMING plans carrying the
///     `global.benchmarking` knob. That knob is what makes the ingestor race its
///     candidates -- BenchmarkPlan times every applicable block_n kernel and hands the
///     fastest to recordWinner(), which writes the ingestor winner cache.
///   * The sweep then benchmarks the engine with real plans and writes the measured
///     ranking to the frontend exact-match cache, reporting the outcome.
///
/// So the kernel-level choice and the engine-level ranking both happen here, and both
/// caches are populated by the one call.
bool sweepAndCache(hipdnnHandle_t handle)
{
    std::cout << "\n[2] Exhaustive sweep: race the kernels, cache the winner\n";

    hipdnnSeverity_t savedLevel = HIPDNN_SEV_OFF;
    HIPDNN_FE_CHECK(getGlobalLogLevel(savedLevel));
    HIPDNN_FE_CHECK(setGlobalLogLevel(HIPDNN_SEV_INFO));
    HIPDNN_FE_CHECK(setUserLogCallback(
        logCallback, HIPDNN_SEV_INFO, LogCallbackMode::SYNC, &LogCapture::instance()));
    LogCapture::instance().clear();

    auto graph = buildSdpaGraph();
    HIPDNN_FE_CHECK(graph->build_operation_graph(handle));
    HIPDNN_FE_CHECK(graph->add_all_engines());

    auto operands = allocateOperands();
    int64_t maxWorkspace = 0;
    HIPDNN_FE_CHECK(graph->get_estimated_max_workspace_size(maxWorkspace));
    const utilities::Workspace workspace(static_cast<size_t>(maxWorkspace));

    AutotuneConfig config;
    config.warmupIterations = 1;

    std::vector<AutotuneResult> results;
    auto outcome = AutotuneCacheWriteOutcome::NOT_ATTEMPTED_NO_SUCCESSFUL_ENGINE;
    HIPDNN_FE_CHECK(graph->autotuneExhaustiveSweep(handle,
                                                   operands.variantPack,
                                                   workspace.get(),
                                                   maxWorkspace,
                                                   config,
                                                   {},
                                                   &results,
                                                   &outcome));

    const auto selected = LogCapture::instance().lastContaining(BENCHMARK_SELECTED_FRAGMENT);
    const auto raced = LogCapture::candidatesFrom(selected);

    setUserLogCallback(logCallback, HIPDNN_SEV_OFF, LogCallbackMode::SYNC, &LogCapture::instance());
    setGlobalLogLevel(savedLevel);

    // Kernel level: did the ingestor actually race the competing block_n variants?
    std::cout << "    kernel candidates raced by the ingestor: " << raced << '\n';
    if(!selected.empty())
    {
        std::cout << "    " << selected << '\n';
    }
    if(raced < 2)
    {
        std::cout << "    NOTE: fewer than two kernels were raced, so nothing was really\n"
                     "    chosen at the kernel level. Check that the competing block_n\n"
                     "    descriptors were packed for this arch.\n";
    }

    // Engine level: the measured ranking and where it went.
    std::cout << "    engines benchmarked: " << results.size() << '\n';
    printRanking(results);
    std::cout << "    exact-match cache write: " << describe(outcome) << '\n';

    return raced >= 2 || outcome == AutotuneCacheWriteOutcome::WRITTEN;
}

/// Scenario 3: a second graph reuses the benchmarked record instead of re-racing.
///
/// Same process, fresh graph. A hit logs that the engine served a kernel "from a
/// benchmarked record"; a miss re-races and logs candidates again. Neither outcome is
/// reported through a return value anywhere in the API, so the log is the evidence.
bool showWinnerReuse(hipdnnHandle_t handle)
{
    std::cout << "\n[3] A fresh graph reuses the benchmarked winner\n";

    hipdnnSeverity_t savedLevel = HIPDNN_SEV_OFF;
    HIPDNN_FE_CHECK(getGlobalLogLevel(savedLevel));
    HIPDNN_FE_CHECK(setGlobalLogLevel(HIPDNN_SEV_INFO));
    HIPDNN_FE_CHECK(setUserLogCallback(
        logCallback, HIPDNN_SEV_INFO, LogCallbackMode::SYNC, &LogCapture::instance()));
    LogCapture::instance().clear();

    auto graph = buildSdpaGraph();
    HIPDNN_FE_CHECK(graph->build_operation_graph(handle));
    HIPDNN_FE_CHECK(graph->build(handle));

    auto operands = allocateOperands();
    int64_t workspaceSize = 0;
    HIPDNN_FE_CHECK(graph->get_workspace_size(workspaceSize));
    const utilities::Workspace workspace(static_cast<size_t>(workspaceSize));
    HIPDNN_FE_CHECK(graph->execute(handle, operands.variantPack, workspace.get()));

    const auto reused = LogCapture::instance().lastContaining(RECORD_REUSED_FRAGMENT);
    const auto reRaced = LogCapture::candidatesFrom(
        LogCapture::instance().lastContaining(BENCHMARK_SELECTED_FRAGMENT));

    setUserLogCallback(logCallback, HIPDNN_SEV_OFF, LogCallbackMode::SYNC, &LogCapture::instance());
    setGlobalLogLevel(savedLevel);

    if(!reused.empty())
    {
        std::cout << "    REUSED: " << reused << '\n'
                  << "    The winner came from the recorded ranking rather than being\n"
                     "    re-measured, which is the reuse this sample exists to show.\n";
        return true;
    }
    std::cout << "    the record was not reused; " << reRaced
              << " candidate(s) were benchmarked again.\n"
                 "    A record is keyed on (graph content, device), so this means the second\n"
                 "    graph did not key to the first, or caching is off (HIPDNN_DISABLE_CACHE).\n";
    return false;
}

} // namespace

int main(int argc, char** argv)
{
    for(int i = 1; i < argc; ++i)
    {
        if(std::strcmp(argv[i], "--help") == 0)
        {
            std::cout
                << "Autotunes an SDPA graph across competing rocKE kernel variants, then\n"
                   "shows the measured winner being cached and reused.\n\n"
                   "Requires HIPDNN_ENABLE_KERNEL_INGESTOR=ON, HIPDNN_ENABLE_SDPA=ON, and the\n"
                   "rocKE attention_dense descriptors packed for this device's arch.\n\n"
                   "  --help         this message\n"
                   "  --verify-cpu   accepted for consistency with the other samples\n";
            return 0;
        }
    }

    RETURN_SUCCESS_IF_NO_DEVICE();
    initializeFrontendLogging();

    // The ingestor only races its candidates when benchmarking is on, and it is off by
    // default. Set the process-wide override before the first handle so the engine sees
    // it, unless the caller already chose a value.
    if(hipdnn_data_sdk::utilities::getEnv("HIPDNN_FORCE_BENCHMARKING").empty())
    {
        static char forceBenchmarking[] = "HIPDNN_FORCE_BENCHMARKING=1";
        putenv(forceBenchmarking);
        std::cout << "(enabled HIPDNN_FORCE_BENCHMARKING=1 for this process)\n";
    }

    hipdnnHandle_t handle = nullptr;
    HIPDNN_CHECK(hipdnnCreate(&handle));

    std::cout << "=== rocKE autotune and winner cache ===\n"
              << "graph: SDPA b" << BATCH << " hq" << NUM_QUERY_HEADS << " kv" << NUM_KV_HEADS
              << " s" << SEQLEN_Q << " d" << HEAD_SIZE << " bf16 causal\n";

    bool ok = showCandidates(handle);
    if(ok)
    {
        ok = sweepAndCache(handle);
    }
    if(ok)
    {
        ok = showWinnerReuse(handle);
    }

    HIPDNN_CHECK(hipdnnDestroy(handle));

    // An environment without the packed rocKE descriptors is not a failure of the
    // sample: it reports what was missing and exits 0, the way the other samples treat
    // an unsupported configuration.
    std::cout << "\n=== " << (ok ? "demonstrated end to end" : "incomplete, see notes above")
              << " ===\n";
    return 0;
}
