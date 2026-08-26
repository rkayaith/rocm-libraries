// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_SDPA

#include <cmath>
#include <string>
#include <type_traits>
#include <vector>

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/IntegrationGraphVerificationHarness.hpp"
#include "harness/TestConfig.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_integration_tests;

/**
 * @file IntegrationGpuSdpaRockeAttentionDense.cpp
 * @brief Cross-provider forward SDPA coverage for the rocKE-authored
 *        attention_dense packs, verified against the shared GPU/CPU reference.
 *
 * The provider's own suite
 * (hip-kernel-provider/src/integration_tests/.../IntegrationGpuRockeAttentionDense.cpp)
 * proves the build-time half: an authored rocKE descriptor is lowered through comgr,
 * packed, discovered, loaded, dispatched, and -- for the autotune pack -- that every
 * block_n variant matches a CPU reference. This file proves the part that suite
 * structurally cannot: that the same kernels satisfy the *shared* engine contract, judged
 * by the same reference executor and the same tolerance machinery every other engine is
 * judged by, from a binary that knows nothing about rocKE.
 *
 * Why this is C++ rather than a bundle, against the bundle-first rule in ../../README.md:
 * only the batch case list is graph verification, and those cases exist here to be
 * CAPTURED into bundles (`--capture-bundles`), which is the documented way to author a
 * sweep that no tool can yet generate. The remaining cases assert engine *decline* --
 * that a graph the rocKE kernels must not serve is refused rather than served wrongly.
 * A bundle cannot express that: a declined graph is exactly what the bundle harness
 * reports as EngineNotApplicable and skips.
 *
 * Registered with add_cpp_graph_test_sources(), so it is gated behind
 * -DBUILD_CPP_GRAPH_TESTS=ON like every other C++ graph test here.
 *
 * ## The geometry is not free
 *
 * A rocKE kernel bakes its problem shape into the code object as compile-time constants.
 * `attentionDenseKernelMatches` compares seven metadata fields for exact equality --
 * dtype, head_size, num_query_heads, num_kv_heads, seqlen_q, seqlen_kv, causal -- so any
 * other value is DECLINED, not served slowly. `batch` is deliberately absent from that
 * list: it is a grid dimension (`attention_dense_grid` launches
 * `(ceil(seqlen_q/block_m), num_query_heads, batch)`), so it is the one axis a test may
 * legitimately sweep. That is why the correctness cases below vary batch and nothing else.
 *
 * Two packs ship, and they differ only in head size:
 *   - attention_dense           d128, single variant
 *   - attention_dense_autotune  d64,  three block_n variants that the ingestor races
 *
 * ## Layout is load-bearing, not incidental
 *
 * The kernel's addressing is BSHD (token-major, heads interleaved): a token step is
 * `Hq*D` and a head step is `D`. Dims stay hipDNN-canonical `[B,H,S,D]`; only the strides
 * differ. `hasBshdStrides` rejects anything else -- which matters because a BHSD operand
 * is not a slow path but a silently wrong one: the kernel would index it as BSHD, read
 * the wrong elements, and produce plausible garbage without faulting. The BHSD decline
 * case below is that guard's regression test.
 */
namespace
{

/// Which shipped pack a case targets. They differ only in head size, but they are
/// distinct descriptor sets, distinct engines, and distinct kpack members, so a case has
/// to say which one it means.
enum class RockePack
{
    DENSE, ///< attention_dense, d128, one variant
    AUTOTUNE, ///< attention_dense_autotune, d64, three block_n variants
};

/// The UED names the two packs declare. Only ever hashed to an id here -- a descriptor
/// -backed engine does not publish its name through hipdnnGetEngineInfo_ext (see
/// runGraphTest), so these must match attention_dense.ued.json and
/// attention_dense_autotune.ued.json exactly or the id is simply a different engine.
constexpr const char* ROCKE_DENSE_ENGINE_NAME = "hipkernel:AttentionDense";
constexpr const char* ROCKE_AUTOTUNE_ENGINE_NAME = "hipkernel:AttentionDenseAutotune";

/// Head size is a property of the pack, not a free parameter: it is one of the seven
/// fields the kernel matcher compares exactly.
constexpr int64_t headSizeFor(RockePack pack)
{
    return pack == RockePack::DENSE ? 128 : 64;
}

/// The spec both packs are compiled for. Everything here except `batch` is pinned by the
/// descriptors' metadata; changing any of it turns a case from "verifies the kernel" into
/// "verifies that the engine declines", which is what the decline cases are for.
constexpr int64_t NUM_QUERY_HEADS = 8;
constexpr int64_t NUM_KV_HEADS = 8;
constexpr int64_t SEQLEN_Q = 256;
constexpr int64_t SEQLEN_KV = 256;

/// bf16 in, fp32 accumulate, bf16 out. The reference computes the same maths in fp32 and
/// rounds once at the end, so the delta is output rounding plus accumulation order.
/// Matches the tolerance the provider's own suite uses for these kernels.
constexpr float BF16_TOLERANCE = 2e-2f;

struct RockeSdpaCase
{
    RockePack pack;
    int64_t batch;
    /// false => BSHD, the layout the kernels declare and the only one they may serve.
    /// true  => BHSD, which must be declined rather than silently misread.
    bool bhsdLayout;
    /// false => causal, which both packs bake in as `causal: 1`.
    bool nonCausal;
    /// Set when the case must NOT be served by a rocKE kernel. The graph is still built
    /// and still has to be handled sanely; it just must not come back with an answer.
    bool expectDecline;
    unsigned int seed;
    std::string note;
    /// Zero means "the value the pack was compiled for". A non-zero override drives a
    /// geometry field away from the baked one, which must produce a DECLINE: each of
    /// these is compared for exact equality by attentionDenseKernelMatches, so a served
    /// graph would be running a kernel built for different extents.
    int64_t numQueryHeadsOverride = 0;
    int64_t numKvHeadsOverride = 0;
    int64_t seqLenQOverride = 0;
    int64_t seqLenKvOverride = 0;
};

/// Correctness across batch, plus the declines that keep "it ran" from being confused
/// with "it was right".
///
/// Batch is swept 1/2/4 because it is the axis that was silently unscoped. `batch` is a
/// compile-time field -- the rocKE kernel bakes it into its buffer-resource extents, and
/// rocKE's own launcher cache keys on a name carrying `_b{batch}` for exactly that reason
/// -- but the KMD schema had no `batch` field, so the matcher could not compare it and a
/// batch-4 graph was served a batch-1 kernel. Only batch 0 was computed; the rest of the
/// output was never written, at 35%/54% mismatch for B=2/B=4.
///
/// These cases found that defect. The fix is a batch VARIANT per shipped batch plus a
/// matcher that compares the field, so all three batches are now served by their own
/// kernel and all three must be numerically correct. A batch outside the packed set is
/// declined rather than mis-served, which is the same contract every other baked
/// dimension already had.
///
/// Background: Results/rocke-attention-dense-batch-not-matched.md.
///
/// Both packs get the same sweep: d64 and d128 are separately compiled code objects, and
/// their near-identical mismatch rates are what identified the shared match contract
/// rather than either kernel as the carrier.
std::vector<RockeSdpaCase> getRockeSdpaCases()
{
    std::vector<RockeSdpaCase> cases;

    for(const auto pack : {RockePack::DENSE, RockePack::AUTOTUNE})
    {
        const std::string packName = pack == RockePack::DENSE ? "dense_d128" : "autotune_d64";

        for(const int64_t batch : {int64_t{1}, int64_t{2}, int64_t{4}})
        {
            cases.push_back({pack,
                             batch,
                             /*bhsdLayout=*/false,
                             /*nonCausal=*/false,
                             /*expectDecline=*/false,
                             0xC0FFEEu + static_cast<unsigned int>(batch),
                             packName + "_causal_bshd_batch" + std::to_string(batch)});
        }

        // BHSD is the dangerous one: the kernel cannot fault on it, it would just index a
        // BSHD layout over BHSD memory. hasBshdStrides() is the only thing standing
        // between that and a wrong answer, so it gets a standing regression test.
        cases.push_back({pack,
                         /*batch=*/2,
                         /*bhsdLayout=*/true,
                         /*nonCausal=*/false,
                         /*expectDecline=*/true,
                         0xBADF00Du,
                         packName + "_declines_bhsd_layout"});

        // `causal` is one of the seven exactly-compared metadata fields, and both packs
        // are compiled with it set. A non-causal graph must therefore find no matching
        // kernel rather than get a causal one applied to it.
        cases.push_back({pack,
                         /*batch=*/2,
                         /*bhsdLayout=*/false,
                         /*nonCausal=*/true,
                         /*expectDecline=*/true,
                         0xFEEDu,
                         packName + "_declines_non_causal"});

        // The regression guard for the batch defect itself. batch=3 is deliberately NOT
        // packed, and the whole bug was that an unpacked batch matched anyway and got a
        // batch-1 kernel's buffer extents. A decline here is the fix holding; a pass
        // means the matcher stopped comparing batch again and the silent-wrong-output
        // path is back.
        cases.push_back({pack,
                         /*batch=*/3,
                         /*bhsdLayout=*/false,
                         /*nonCausal=*/false,
                         /*expectDecline=*/true,
                         0xB47C4u,
                         packName + "_declines_unpacked_batch3"});

        // The remaining four exactly-compared geometry fields. Each is the same class of
        // hazard batch turned out to be: the kernel bakes the extent in, so a graph that
        // differs must be DECLINED, and a match would mean running a kernel built for a
        // different problem.
        //
        // num_kv_heads is the one to worry about in practice: GQA/MQA (kv < q) is the
        // common production shape, these packs are compiled MHA (kv == q == 8), and the
        // KV-head count changes the broadcast addressing. A mis-served GQA graph reads
        // the wrong KV head and returns a plausible wrong answer -- batch's failure mode
        // exactly.
        cases.push_back({pack,
                         /*batch=*/1,
                         false,
                         false,
                         /*expectDecline=*/true,
                         0x6A1u,
                         packName + "_declines_gqa_kv2",
                         /*numQueryHeadsOverride=*/0,
                         /*numKvHeadsOverride=*/2});
        // hq AND kv move together: the frontend requires num_heads % num_heads_k == 0,
        // so hq=4 with kv=8 is an ILLEGAL graph that validate() rejects before the engine
        // is ever consulted -- a decline for the wrong reason, which is no evidence at
        // all. hq=kv=4 is a legal MHA graph differing from the packs (hq=kv=8) in head
        // count alone, so a decline here is attributable to num_query_heads/num_kv_heads.
        cases.push_back({pack,
                         /*batch=*/1,
                         false,
                         false,
                         /*expectDecline=*/true,
                         0x6A2u,
                         packName + "_declines_wrong_head_count",
                         /*numQueryHeadsOverride=*/4,
                         /*numKvHeadsOverride=*/4});
        cases.push_back({pack,
                         /*batch=*/1,
                         false,
                         false,
                         /*expectDecline=*/true,
                         0x6A3u,
                         packName + "_declines_wrong_seqlen_q",
                         /*numQueryHeadsOverride=*/0,
                         /*numKvHeadsOverride=*/0,
                         /*seqLenQOverride=*/128});
        cases.push_back({pack,
                         /*batch=*/1,
                         false,
                         false,
                         /*expectDecline=*/true,
                         0x6A4u,
                         packName + "_declines_wrong_seqlen_kv",
                         /*numQueryHeadsOverride=*/0,
                         /*numKvHeadsOverride=*/0,
                         /*seqLenQOverride=*/0,
                         /*seqLenKvOverride=*/512});
    }

    // Head size crossed between the packs: d64 geometry must not be served by the d128
    // pack and vice versa. Expressed as a decline case on each pack's own engine, so it
    // fails if either pack ever widens its matcher to accept the other's shape.
    cases.push_back({RockePack::DENSE,
                     /*batch=*/2,
                     /*bhsdLayout=*/false,
                     /*nonCausal=*/false,
                     /*expectDecline=*/true,
                     0x5EEDu,
                     "dense_d128_declines_foreign_head_size"});
    cases.push_back({RockePack::AUTOTUNE,
                     /*batch=*/2,
                     /*bhsdLayout=*/false,
                     /*nonCausal=*/false,
                     /*expectDecline=*/true,
                     0x5EEEu,
                     "autotune_d64_declines_foreign_head_size"});

    return cases;
}

/// True when the case deliberately asks for the *other* pack's head size.
bool usesForeignHeadSize(const RockeSdpaCase& tc)
{
    return tc.note.find("foreign_head_size") != std::string::npos;
}

int64_t headSizeForCase(const RockeSdpaCase& tc)
{
    const auto own = headSizeFor(tc.pack);
    if(!usesForeignHeadSize(tc))
    {
        return own;
    }
    return own == 128 ? 64 : 128;
}

template <typename DataType>
class RockeAttentionDense : public IntegrationGraphVerificationHarness<DataType, RockeSdpaCase>
{
public:
    static_assert(std::is_same_v<DataType, bfloat16>,
                  "both rocKE attention_dense packs are compiled for bf16 only");

    struct GraphOutputs
    {
        std::shared_ptr<graph::TensorAttributes> o;
    };

    static std::pair<graph::Graph, GraphOutputs> buildGraph(hipdnnHandle_t handle,
                                                            const RockeSdpaCase& tc)
    {
        graph::Graph graphObj;
        graphObj.set_name("RockeAttentionDenseTest");

        const auto ioType = getDataTypeEnumFromType<DataType>();
        graphObj.set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_io_data_type(ioType);

        const int64_t headSize = headSizeForCase(tc);
        // An override of 0 means "use the baked value", so a case opts into exactly the
        // one axis it is probing and every other axis stays matchable. Without that, a
        // decline would not identify WHICH field caused it.
        const int64_t hq
            = tc.numQueryHeadsOverride != 0 ? tc.numQueryHeadsOverride : NUM_QUERY_HEADS;
        const int64_t hkv = tc.numKvHeadsOverride != 0 ? tc.numKvHeadsOverride : NUM_KV_HEADS;
        const int64_t sq = tc.seqLenQOverride != 0 ? tc.seqLenQOverride : SEQLEN_Q;
        const int64_t skv = tc.seqLenKvOverride != 0 ? tc.seqLenKvOverride : SEQLEN_KV;

        const std::vector<int64_t> qDims{tc.batch, hq, sq, headSize};
        const std::vector<int64_t> kvDims{tc.batch, hkv, skv, headSize};

        // Dims are hipDNN-canonical [B,H,S,D] in both cases; only the strides move. That
        // is the whole point: a BHSD operand is indistinguishable by shape and is caught
        // by strides alone.
        const auto& strideOrder
            = tc.bhsdLayout ? TensorLayout::BHSD.strideOrder : TensorLayout::BSHD.strideOrder;
        auto makeIo = [&](const std::string& name, const std::vector<int64_t>& dims) {
            return std::make_shared<graph::TensorAttributes>(graph::makeTensorAttributes(
                name, ioType, dims, generateStrides(dims, strideOrder)));
        };

        auto q = makeIo("Q", qDims);
        auto k = makeIo("K", kvDims);
        auto v = makeIo("V", kvDims);

        graph::SdpaAttributes sdpaAttrs;
        sdpaAttrs.set_attn_scale(1.0f / std::sqrt(static_cast<float>(headSize)));
        if(!tc.nonCausal)
        {
            sdpaAttrs.set_causal_mask(true);
        }

        auto [o, stats] = graphObj.sdpa(q, k, v, sdpaAttrs);
        if(stats != nullptr)
        {
            throw std::runtime_error("generate_stats was not requested but a stats tensor "
                                     "was produced");
        }
        // O carries the same layout as the inputs. Left to inference it would pick up
        // packed BHSD strides and disagree with where the kernel actually writes.
        o->set_output(true)
            .set_dim(qDims)
            .set_stride(generateStrides(qDims, strideOrder))
            .set_data_type(ioType);

        auto validateResult = graphObj.validate();
        if(validateResult.is_bad())
        {
            // The frontend rejected the GRAPH, so no engine was ever consulted. For a
            // decline case that is a false positive waiting to happen: the case would
            // "prove" the engine refused something the frontend never let through. Say so
            // explicitly, because the bare validator message ("num_heads must be divisible
            // by num_heads_k") reads like a product complaint rather than a malformed
            // test input.
            throw std::runtime_error(
                "Failed to validate graph: " + validateResult.get_message()
                + " -- this graph is malformed, so it tests nothing about the rocKE engine. "
                  "A decline case must be a LEGAL graph that the engine refuses, not one "
                  "the frontend rejects first.");
        }

        auto buildResult = graphObj.build_operation_graph(handle);
        if(buildResult.is_bad())
        {
            throw std::runtime_error("Failed to build operation graph: "
                                     + buildResult.get_message());
        }

        return std::make_pair(std::move(graphObj), GraphOutputs{o});
    }

protected:
    void runGraphTest() override
    {
        const auto& testCase = this->GetParam();

        auto [graphObj, outputs] = buildGraph(getSharedHandle(), testCase);

        this->setTestCaseNote(testCase.note);
        this->inputFillRecipes().setGlobalSeed(testCase.seed);

        // The rocKE engine must be among the applicable ones. Checked explicitly rather
        // than left to the harness, because "some engine served this graph" is exactly
        // the answer that would look green while never touching rocKE: sortEngineIds()
        // ranks ASM_SDPA above a discovered engine, so ASM wins any graph both can serve.
        //
        // Selected by id hashed from the UED name, not by --test-engine: a descriptor
        // -backed engine publishes no name through hipdnnGetEngineInfo_ext, so
        // --test-engine rejects it as "not loaded" even though it is loaded and
        // dispatching. See Results/rocke-ingestor-engine-names-unpublished.md.
        const int64_t rockeEngineId = hipdnn_data_sdk::utilities::engineNameToId(
            testCase.pack == RockePack::DENSE ? ROCKE_DENSE_ENGINE_NAME
                                              : ROCKE_AUTOTUNE_ENGINE_NAME);

        std::vector<int64_t> rankedEngineIds;
        const auto status = graphObj.get_ranked_engine_ids(rankedEngineIds);
        const bool offeredItself
            = status.is_good()
              && std::find(rankedEngineIds.begin(), rankedEngineIds.end(), rockeEngineId)
                     != rankedEngineIds.end();

        if(testCase.expectDecline)
        {
            EXPECT_FALSE(offeredItself)
                << "the rocKE engine offered to serve " << testCase.note
                << ". These kernels bake their geometry and BSHD addressing in as "
                   "compile-time constants, so accepting this graph does not mean a slower "
                   "path -- it means reading the wrong elements and returning a plausible "
                   "wrong answer.";
            return;
        }

        // A skip here would silently drop the only rocKE coverage in this project, so the
        // arch gate is explicit: off gfx942 the pack is dropped by design and there is
        // nothing to test; on gfx942 an absent engine is a real failure.
        if(!offeredItself)
        {
            const auto& arch = TestConfig::get().getCurrentArch();
            if(arch.find("gfx942") == std::string::npos)
            {
                GTEST_SKIP() << "the rocKE attention_dense descriptors declare arch [gfx942]; "
                                "this device is "
                             << arch << ", so the ingestor drops the pack";
            }
            FAIL() << "the rocKE engine did not offer itself for " << testCase.note << " on "
                   << arch
                   << ". Either the production packs were not built into this tree "
                      "(-DHIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE=ON with a source root) "
                      "or the kernel matcher no longer accepts this geometry.";
        }

        this->registerValidator(outputs.o, BF16_TOLERANCE);

        // Stock verifyGraph(). It fills the gpu and reference bundles from ONE
        // generateBundles() and one seeded fill, so both sides see byte-identical inputs
        // -- which is the property any numeric comparison depends on.
        //
        // Pinning rocKE is left to the build: ASM_SDPA is disabled, so the rocKE engine
        // is the only one that serves these graphs and the heuristic's first choice IS
        // the engine under test. That is simpler and less error-prone than re-planning
        // per engine inside the harness.
        graphObj.set_preferred_engine_id_ext(rockeEngineId);
        this->verifyGraph(graphObj);

        int64_t servingEngineId = 0;
        ASSERT_EQ(graphObj.get_execution_plan_engine_id(servingEngineId).code, ErrorCode::OK);
        EXPECT_EQ(servingEngineId, rockeEngineId)
            << "engine id " << servingEngineId << " served the graph, not the rocKE engine "
            << rockeEngineId << ", so the numeric agreement above is not evidence about rocKE";
    }
};

struct RockeCaseName
{
    std::string operator()(const testing::TestParamInfo<RockeSdpaCase>& info) const
    {
        return info.param.note;
    }
};

using IntegrationGpuSdpaRockeAttentionDenseBfp16 = RockeAttentionDense<bfloat16>;

} // namespace

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(IntegrationGpuSdpaRockeAttentionDenseBfp16);
TEST_P(IntegrationGpuSdpaRockeAttentionDenseBfp16, Correctness)
{
    runGraphTest();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         IntegrationGpuSdpaRockeAttentionDenseBfp16,
                         testing::ValuesIn(getRockeSdpaCases()),
                         RockeCaseName());

#endif // HIPDNN_ENABLE_SDPA
