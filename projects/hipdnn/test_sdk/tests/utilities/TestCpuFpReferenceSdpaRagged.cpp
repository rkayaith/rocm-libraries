// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ============================================================================
// Validates the ragged CPU reference (CpuFpReferenceSdpaRagged) against the
// trusted dense CpuFpReferenceSdpa. Inputs are RFC-0014 ragged tensors
// (ShallowRaggedTensor over packed buffers + per-primary ragged_offset aux);
// each batch is extracted into a dense [1,H,seqlen_b,D] tensor, the dense
// reference is run on it, and its output is compared with the corresponding
// entries of the ragged reference output. This is the middle link of the
// validation chain: dense CPU (trusted) -> CPU ragged (here) -> GPU ragged
// (TestGpuFpReferenceSdpaRagged). CPU-only; no device required.
// ============================================================================

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShallowRaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpaRagged.hpp>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_test_sdk::utilities;

namespace
{

// Sequence axis for our [B, H, S, D] ragged SDPA tensors (seq is dim index 2).
constexpr int SEQ_AXIS = 2;

int64_t sum(const std::vector<int64_t>& v)
{
    return std::accumulate(v.begin(), v.end(), int64_t{0});
}

int64_t maxOf(const std::vector<int64_t>& v)
{
    return *std::max_element(v.begin(), v.end());
}

// BSHD-layout element strides for packed rank-4 [B, H, S, D]: seq stride = strides[2] = H*D.
std::vector<int64_t> bshd(const std::vector<int64_t>& dims)
{
    return {dims[1] * dims[2] * dims[3], dims[3], dims[1] * dims[3], 1};
}

std::vector<int64_t> cumTokens(const std::vector<int64_t>& lengths)
{
    std::vector<int64_t> cum(lengths.size() + 1, 0);
    for(size_t i = 0; i < lengths.size(); ++i)
    {
        cum[i + 1] = cum[i] + lengths[i];
    }
    return cum;
}

// Rank-4 [B+1,1,1,1] INT32 ragged_offset aux = cumTokens * seqStride (element units, RFC-0014).
std::shared_ptr<ITensor> makeOffsetAux(const std::vector<int64_t>& cum, int64_t seqStride)
{
    auto aux = std::make_shared<Tensor<int32_t>>(
        std::vector<int64_t>{static_cast<int64_t>(cum.size()), 1, 1, 1});
    for(size_t i = 0; i < cum.size(); ++i)
    {
        aux->setHostValue(
            static_cast<int32_t>(cum[i] * seqStride), static_cast<int64_t>(i), 0, 0, 0);
    }
    return aux;
}

// Deterministic fill of a packed backing buffer in [-1, 1); avoids <random> cross-TU drift.
void fillPacked(std::vector<float>& buf, unsigned int seed)
{
    uint32_t state = seed;
    for(auto& x : buf)
    {
        state = state * 1664525U + 1013904223U;
        const float u = static_cast<float>(state >> 8) / static_cast<float>(1U << 24); // [0,1)
        x = 2.0f * u - 1.0f;
    }
}

// Extract a dense [1, heads, seqLen, dim] slice for batch b from a ragged tensor (batch-relative
// seq index; ragged addressing handles the packing).
Tensor<float> extractDenseSlice(TensorBase<float>& ragged, int64_t b, int64_t seqLen)
{
    const auto heads = ragged.dims()[1];
    const auto dim = ragged.dims()[3];
    Tensor<float> dense({1, heads, seqLen, dim});
    for(int64_t h = 0; h < heads; ++h)
    {
        for(int64_t s = 0; s < seqLen; ++s)
        {
            for(int64_t d = 0; d < dim; ++d)
            {
                dense(0, h, s, d) = ragged.getHostValue(std::vector<int64_t>{b, h, s, d});
            }
        }
    }
    dense.memory().markHostModified();
    return dense;
}

// Build packed ragged inputs (ShallowRaggedTensor), run the ragged CPU reference, then validate each
// batch against the dense CPU reference on [1,H,seqlen_b,D] slices.
void checkRaggedVsDense(const std::vector<int64_t>& seqQ,
                        const std::vector<int64_t>& seqKv,
                        int64_t numHeads,
                        int64_t numHeadsKv,
                        int64_t headDim,
                        int64_t headDimV,
                        int64_t leftBound = -1,
                        int64_t rightBound = -1,
                        bool topLeftAlignment = true)
{
    ASSERT_EQ(seqQ.size(), seqKv.size());
    const auto batch = static_cast<int64_t>(seqQ.size());
    const auto sMaxQ = maxOf(seqQ);
    const auto sMaxKv = maxOf(seqKv);
    const auto totalQ = sum(seqQ);
    const auto totalKv = sum(seqKv);
    const auto cumQ = cumTokens(seqQ);
    const auto cumKv = cumTokens(seqKv);

    const std::vector<int64_t> qDims = {batch, numHeads, sMaxQ, headDim};
    const std::vector<int64_t> kDims = {batch, numHeadsKv, sMaxKv, headDim};
    const std::vector<int64_t> vDims = {batch, numHeadsKv, sMaxKv, headDimV};
    const std::vector<int64_t> oDims = {batch, numHeads, sMaxQ, headDimV};

    // Packed backing buffers (sized to ragged_offset[B]); borrowed by the ShallowRaggedTensors.
    std::vector<float> qBack(static_cast<size_t>(totalQ * numHeads * headDim));
    std::vector<float> kBack(static_cast<size_t>(totalKv * numHeadsKv * headDim));
    std::vector<float> vBack(static_cast<size_t>(totalKv * numHeadsKv * headDimV));
    std::vector<float> oBack(static_cast<size_t>(totalQ * numHeads * headDimV), 0.0f);
    fillPacked(qBack, 11);
    fillPacked(kBack, 22);
    fillPacked(vBack, 33);

    auto offQ = makeOffsetAux(cumQ, numHeads * headDim);
    auto offK = makeOffsetAux(cumKv, numHeadsKv * headDim);
    auto offV = makeOffsetAux(cumKv, numHeadsKv * headDimV);
    auto offO = makeOffsetAux(cumQ, numHeads * headDimV);

    ShallowRaggedTensor<float> q(qBack.data(), qDims, bshd(qDims), SEQ_AXIS, offQ);
    ShallowRaggedTensor<float> k(kBack.data(), kDims, bshd(kDims), SEQ_AXIS, offK);
    ShallowRaggedTensor<float> v(vBack.data(), vDims, bshd(vDims), SEQ_AXIS, offV);
    ShallowRaggedTensor<float> o(oBack.data(), oDims, bshd(oDims), SEQ_AXIS, offO);

    CpuFpReferenceSdpaRagged::forward<float, float, float, float, float>(
        q, k, v, o, std::nullopt, leftBound, rightBound, topLeftAlignment);

    for(int64_t b = 0; b < batch; ++b)
    {
        const auto sQ = seqQ[static_cast<size_t>(b)];
        const auto sKv = seqKv[static_cast<size_t>(b)];
        auto qd = extractDenseSlice(q, b, sQ);
        auto kd = extractDenseSlice(k, b, sKv);
        auto vd = extractDenseSlice(v, b, sKv);
        Tensor<float> oDense({1, numHeads, sQ, headDimV});

        CpuFpReferenceSdpa::forward<float, float, float, float, float>(qd,
                                                                       kd,
                                                                       vd,
                                                                       oDense,
                                                                       std::nullopt,
                                                                       /*attnMask=*/nullptr,
                                                                       leftBound,
                                                                       rightBound,
                                                                       topLeftAlignment);

        for(int64_t s = 0; s < sQ; ++s)
        {
            for(int64_t h = 0; h < numHeads; ++h)
            {
                for(int64_t dv = 0; dv < headDimV; ++dv)
                {
                    const auto got = o.getHostValue(std::vector<int64_t>{b, h, s, dv});
                    EXPECT_NEAR(got, oDense(0, h, s, dv), 1e-4f)
                        << "ragged vs dense mismatch batch " << b << " token " << s << " head " << h
                        << " dv " << dv;
                }
            }
        }
    }
}

} // namespace

TEST(TestCpuFpReferenceSdpaRaggedFp32, RaggedBasicMha)
{
    checkRaggedVsDense({3, 5, 1}, {3, 5, 1}, 4, 4, 16, 16);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, RaggedCrossAttention)
{
    checkRaggedVsDense({2, 4, 3}, {5, 1, 6}, 2, 2, 16, 16);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, RaggedCausalTopLeft)
{
    checkRaggedVsDense({4, 7}, {4, 7}, 2, 2, 16, 16, -1, 0, true);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, RaggedCausalBottomRight)
{
    checkRaggedVsDense({3, 5}, {6, 8}, 2, 2, 16, 16, -1, 0, false);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, RaggedSlidingWindow)
{
    checkRaggedVsDense({8, 6}, {8, 6}, 2, 2, 16, 16, 2, 2, true);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, RaggedGqa)
{
    checkRaggedVsDense({5, 3}, {5, 3}, 8, 2, 16, 16);
}

TEST(TestCpuFpReferenceSdpaRaggedFp32, RaggedAsymmetricHeadDim)
{
    // hdim_q = 192, hdim_v = 128 (asymmetric head dims, as on the ASM v3 path).
    checkRaggedVsDense({3, 5}, {3, 5}, 2, 2, 192, 128);
}
