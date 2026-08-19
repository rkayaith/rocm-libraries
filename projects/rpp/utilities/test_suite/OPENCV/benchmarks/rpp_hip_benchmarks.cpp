/*
MIT License

Copyright (c) 2019 - 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "benchmarks_common.h"
#include <chrono>

using namespace std::chrono;

// Forward declaration of helper function from rpp_test_suite_image.h
// We declare it here to avoid multiple definition errors
inline void set_descriptor_dims_and_strides_local(RpptDescPtr descPtr, int noOfImages, int maxHeight,
                                                   int maxWidth, int numChannels, int offsetInBytes,
                                                   int additionalStride = 0) {
    descPtr->numDims = 4;
    descPtr->offsetInBytes = offsetInBytes;
    descPtr->n = noOfImages;
    descPtr->h = maxHeight;
    descPtr->c = numChannels;

    // Set w to actual image width - kernel should only process actual image pixels
    descPtr->w = maxWidth;

    // NOTE: Strides will be set by update_strides_from_layout() after layout is assigned
    // For strides, we need to use ALIGNED width for memory layout
    int alignedWidth = (maxWidth / 8) * 8 + 8 + additionalStride;
    descPtr->strides.nStride = descPtr->h * alignedWidth * descPtr->c;
    descPtr->strides.hStride = alignedWidth * descPtr->c;
    descPtr->strides.wStride = descPtr->c;
    descPtr->strides.cStride = 1;
}

// Helper function to update strides based on layout
inline void update_strides_from_layout(RpptDescPtr descPtr) {
    int h = descPtr->h;
    int w = descPtr->w;
    int c = descPtr->c;

    if (descPtr->layout == RpptLayout::NCHW) {
        // NCHW layout: Batch-Channels-Height-Width
        descPtr->strides.nStride = c * h * w;
        descPtr->strides.cStride = h * w;
        descPtr->strides.hStride = w;
        descPtr->strides.wStride = 1;
    } else if (descPtr->layout == RpptLayout::NHWC) {
        // NHWC layout: Batch-Height-Width-Channels
        descPtr->strides.nStride = h * w * c;
        descPtr->strides.hStride = w * c;
        descPtr->strides.wStride = c;
        descPtr->strides.cStride = 1;
    }
    // Add other layouts as needed
}

// Helper function to convert RpptDesc to RpptGenericDesc for 3D operations
inline void convert_desc_to_generic_desc_3d(RpptDescPtr descPtr, RpptGenericDescPtr genDescPtr) {
    genDescPtr->numDims = 5;  // NCDHW format
    genDescPtr->offsetInBytes = descPtr->offsetInBytes;
    genDescPtr->dataType = descPtr->dataType;
    // Convert 2D layout to 3D layout equivalent for 3D arithmetic operations
    // NCHW -> NCDHW, NHWC -> NDHWC to match RpptLayout enum for 3D tensors
    if (descPtr->layout == RpptLayout::NCHW) {
        genDescPtr->layout = RpptLayout::NCDHW;
    } else if (descPtr->layout == RpptLayout::NHWC) {
        genDescPtr->layout = RpptLayout::NDHWC;
    } else {
        genDescPtr->layout = descPtr->layout;  // Preserve other layouts
    }

    // Map 2D (NCHW/NHWC) to 3D (NCDHW) by adding depth=1
    genDescPtr->dims[0] = descPtr->n;  // batch
    genDescPtr->dims[1] = descPtr->c;  // channels
    genDescPtr->dims[2] = 1;           // depth (always 1 for 2D images)
    genDescPtr->dims[3] = descPtr->h;  // height
    genDescPtr->dims[4] = descPtr->w;  // width

    // Set strides for 3D layout (after conversion from 2D)
    // For NCDHW (from NCHW): strides are [N, C, D, H, W]
    // For NDHWC (from NHWC): strides are [N, D, H, W, C]
    if (descPtr->layout == RpptLayout::NCHW) {
        // NCDHW layout: Batch-Channels-Depth-Height-Width
        genDescPtr->strides[0] = descPtr->strides.nStride;              // N stride
        genDescPtr->strides[1] = descPtr->strides.cStride;              // C stride
        genDescPtr->strides[2] = descPtr->strides.hStride * descPtr->h; // D stride = h*w (since depth=1)
        genDescPtr->strides[3] = descPtr->strides.hStride;              // H stride
        genDescPtr->strides[4] = descPtr->strides.wStride;              // W stride
    } else {  // NHWC -> NDHWC
        // NDHWC layout: Batch-Depth-Height-Width-Channels
        // With depth=1, the ordering is [N, D, H, W, C]
        genDescPtr->strides[0] = descPtr->strides.nStride;              // N stride = d*h*w*c
        genDescPtr->strides[1] = descPtr->strides.hStride * descPtr->h; // D stride = h*w*c (since depth=1)
        genDescPtr->strides[2] = descPtr->strides.hStride;              // H stride = w*c
        genDescPtr->strides[3] = descPtr->strides.wStride;              // W stride = c
        genDescPtr->strides[4] = descPtr->strides.cStride;              // C stride = 1
    }
}

// Helper function to convert RpptROI to RpptROI3D for 3D operations
inline void convert_roi_to_roi3d(RpptROIPtr roiPtr, RpptROI3DPtr roi3dPtr, int batchSize) {
    for (int i = 0; i < batchSize; ++i) {
        roi3dPtr[i].xyzwhdROI.xyz.x = roiPtr[i].xywhROI.xy.x;
        roi3dPtr[i].xyzwhdROI.xyz.y = roiPtr[i].xywhROI.xy.y;
        roi3dPtr[i].xyzwhdROI.xyz.z = 0;  // depth starts at 0
        roi3dPtr[i].xyzwhdROI.roiWidth = roiPtr[i].xywhROI.roiWidth;
        roi3dPtr[i].xyzwhdROI.roiHeight = roiPtr[i].xywhROI.roiHeight;
        roi3dPtr[i].xyzwhdROI.roiDepth = 1;  // depth is always 1 for 2D images
    }
}

// Macro for checking HIP status
#define CHECK_HIP_STATUS(call)                                                                  \
    do {                                                                                        \
        hipError_t err = (call);                                                                \
        if (err != hipSuccess) {                                                                \
            std::cerr << "HIP error in " << #call << ": " << hipGetErrorString(err) << std::endl; \
        }                                                                                       \
    } while (0)

// Macro for checking RPP status
#define CHECK_RPP_STATUS(call, func_name)                                                       \
    do {                                                                                        \
        RppStatus status = (call);                                                              \
        if (status != RPP_SUCCESS) {                                                            \
            std::cerr << "RPP HIP " << func_name << " failed with status: " << status << std::endl; \
        }                                                                                       \
    } while (0)

// ==================== RPP HIP/GPU BENCHMARK FUNCTIONS ====================
//
// IMPORTANT: Memory allocation pattern for RPP HIP backend:
// 1. Image data (input/output): Device memory (hipMalloc)
// 2. Parameter tensors (alpha, beta, gamma, etc.): Pinned HOST memory (hipHostMalloc)
// 3. ROI tensors: Pinned HOST memory (hipHostMalloc)
//
// The RPP HIP backend expects parameter and ROI pointers to be in HOST memory,
// NOT device memory. Use hipHostMalloc for pinned host memory allocation.
//

void benchmark_RPP_HIP_Brightness(const vector<Mat>& imgs, bool isColor, float alpha, float beta,
                                  rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    Rpp32f *alphaTensor, *betaTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&alphaTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&betaTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        alphaTensor[i] = alpha;
        betaTensor[i] = beta;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_brightness(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                            &alphaTensor[i], &betaTensor[i],
                                            &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Brightness");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(alphaTensor));
    CHECK_HIP_STATUS(hipHostFree(betaTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "alpha=" << alpha << ", beta=" << beta;
    printResult("RPP HIP Brightness", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_GammaCorrection(const vector<Mat>& imgs, bool isColor, float gamma,
                                      rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    Rpp32f *gammaTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&gammaTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        gammaTensor[i] = gamma;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_gamma_correction(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                  &gammaTensor[i], &roiTensor[i], RpptRoiType::XYWH,
                                                  handle, RPP_HIP_BACKEND),
                            "GammaCorrection");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(gammaTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "gamma=" << gamma;
    printResult("RPP HIP GammaCorrection", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

// ==================== TODO: REMAINING OPERATORS ====================
// The following operators are not yet implemented.
// Implementation pattern for all operators:
//   1. Setup descriptors and ROIs
//   2. Allocate GPU memory for ALL images (outside timing loop)
//   3. Copy data to GPU (outside timing loop)
//   4. START TIMING
//   5. Benchmark loop: for(k) { for(i) { RPP function call } }
//   6. Synchronize stream
//   7. END TIMING
//   8. Free GPU memory
// See implemented operators above (Brightness, GammaCorrection, etc.) for reference.

#define HIP_PLACEHOLDER(func_name) \
    cout << "RPP HIP " << func_name << " - Not yet implemented (TODO)" << endl;

void benchmark_RPP_HIP_Blend(const vector<Mat>& imgs, bool isColor, float alpha,
                            rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<Mat> imgs2(num_images);
    for (int i = 0; i < num_images; ++i) {
        imgs[i].convertTo(imgs2[i], -1, 0.8, 30);
    }

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs1(num_images);
    vector<Rpp8u*> d_inputs2(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    Rpp32f *alphaTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&alphaTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        alphaTensor[i] = alpha;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t imageBufferSize = imgs[i].total() * imgs[i].elemSize();

        CHECK_HIP_STATUS(hipMalloc(&d_inputs1[i], imageBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_inputs2[i], imageBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], imageBufferSize));

        CHECK_HIP_STATUS(hipMemcpy(d_inputs1[i], imgs[i].data, imageBufferSize, hipMemcpyHostToDevice));
        CHECK_HIP_STATUS(hipMemcpy(d_inputs2[i], imgs2[i].data, imageBufferSize, hipMemcpyHostToDevice));
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_blend(d_inputs1[i], d_inputs2[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                       &alphaTensor[i], &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Blend");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs1[i]));
        CHECK_HIP_STATUS(hipFree(d_inputs2[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(alphaTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    
    ostringstream params;
    params << "alpha=" << alpha;
    printResult("RPP HIP Blend", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Contrast(const vector<Mat>& imgs, bool isColor, float contrastFactor,
                               float contrastCenter, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    Rpp32f *contrastFactorTensor;
    Rpp32f *contrastCenterTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&contrastFactorTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&contrastCenterTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        contrastFactorTensor[i] = contrastFactor;
        contrastCenterTensor[i] = contrastCenter;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_contrast(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                          &contrastFactorTensor[i], &contrastCenterTensor[i],
                                          &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Contrast");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(contrastFactorTensor));
    CHECK_HIP_STATUS(hipHostFree(contrastCenterTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "factor=" << contrastFactor << ", center=" << contrastCenter;
    printResult("RPP HIP Contrast", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Exposure(const vector<Mat>& imgs, bool isColor, float exposureFactor,
                               rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    Rpp32f *exposureTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&exposureTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        exposureTensor[i] = exposureFactor;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_exposure(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                          &exposureTensor[i], &roiTensor[i], RpptRoiType::XYWH,
                                          handle, RPP_HIP_BACKEND),
                            "Exposure");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(exposureTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "factor=" << exposureFactor;
    printResult("RPP HIP Exposure", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Hue(const vector<Mat>& imgs, float hueDelta, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = 3;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    Rpp32f *hueTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&hueTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = RpptLayout::NHWC;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        hueTensor[i] = hueDelta;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_hue(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                     &hueTensor[i], &roiTensor[i], RpptRoiType::XYWH,
                                     handle, RPP_HIP_BACKEND),
                            "Hue");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(hueTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "hue=" << hueDelta;
    printResult("RPP HIP Hue", imgs.size(), true,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Saturation(const vector<Mat>& imgs, float satFactor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = 3;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    Rpp32f *satTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&satTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = RpptLayout::NHWC;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        satTensor[i] = satFactor;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_saturation(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                            &satTensor[i], &roiTensor[i], RpptRoiType::XYWH,
                                            handle, RPP_HIP_BACKEND),
                            "Saturation");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(satTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "factor=" << satFactor;
    printResult("RPP HIP Saturation", imgs.size(), true,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_ColorToGreyscale(const vector<Mat>& imgs, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int srcChannels = 3;
    int dstChannels = 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    for (int i = 0; i < num_images; ++i) {
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, srcChannels, 0);
        srcDescs[i].layout = RpptLayout::NHWC;
        srcDescs[i].dataType = RpptDataType::U8;

        set_descriptor_dims_and_strides_local(&dstDescs[i], 1, imgs[i].rows, imgs[i].cols, dstChannels, 0);
        dstDescs[i].layout = RpptLayout::NCHW;
        dstDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&dstDescs[i]);

        size_t srcBufferSize = imgs[i].total() * imgs[i].elemSize();
        size_t dstBufferSize = imgs[i].rows * imgs[i].cols;

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], srcBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], dstBufferSize));

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], imgs[i].data, srcBufferSize, hipMemcpyHostToDevice));
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_color_to_greyscale(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                     RpptSubpixelLayout::BGRtype, handle, RPP_HIP_BACKEND),
                            "ColorToGreyscale");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }

    printResult("RPP HIP ColorToGreyscale", imgs.size(), true,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_ColorJitter(const vector<Mat>& imgs, float brightness, float contrast,
                                  float saturation, float hue, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = 3;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    Rpp32f *brightnessTensor;
    Rpp32f *contrastTensor;
    Rpp32f *saturationTensor;
    Rpp32f *hueTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&brightnessTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&contrastTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&saturationTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&hueTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = RpptLayout::NHWC;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        brightnessTensor[i] = brightness;
        contrastTensor[i] = contrast;
        saturationTensor[i] = saturation;
        hueTensor[i] = hue;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_color_jitter(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                              &brightnessTensor[i], &contrastTensor[i],
                                              &saturationTensor[i], &hueTensor[i],
                                              &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "ColorJitter");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(brightnessTensor));
    CHECK_HIP_STATUS(hipHostFree(contrastTensor));
    CHECK_HIP_STATUS(hipHostFree(saturationTensor));
    CHECK_HIP_STATUS(hipHostFree(hueTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "brightness=" << brightness << ", contrast=" << contrast
           << ", saturation=" << saturation << ", hue=" << hue;
    printResult("RPP HIP ColorJitter", imgs.size(), true,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_BoxFilter(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;
    Rpp32u kSize = static_cast<Rpp32u>(kernelSize);

    // Calculate required offset for border handling: offsetInBytes >= 12 * (kernelSize / 2)
    int offsetInBytes = 12 * (kernelSize / 2);

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, offsetInBytes);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_box_filter(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                            kSize, borderType, &roiTensor[i], RpptRoiType::XYWH,
                                            handle, RPP_HIP_BACKEND),
                            "BoxFilter");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    
    ostringstream params;
    params << "kernel=" << kernelSize;
    printResult("RPP HIP BoxFilter", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_MedianFilter(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                   rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;
    Rpp32u kSize = static_cast<Rpp32u>(kernelSize);

    // Calculate required offset for border handling
    int offsetInBytes = 12 * (kernelSize / 2);

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, offsetInBytes);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_median_filter(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                               kSize, borderType, &roiTensor[i], RpptRoiType::XYWH,
                                               handle, RPP_HIP_BACKEND),
                            "MedianFilter");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    
    ostringstream params;
    params << "kernel=" << kernelSize;
    printResult("RPP HIP MedianFilter", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_GaussianFilter(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                     double sigma, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;
    Rpp32u kSize = static_cast<Rpp32u>(kernelSize);

    // Calculate required offset for border handling
    int offsetInBytes = 12 * (kernelSize / 2);

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    float stdDev = static_cast<float>(sigma);
    Rpp32f *stdDevTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&stdDevTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, offsetInBytes);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        stdDevTensor[i] = stdDev;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_gaussian_filter(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                  &stdDevTensor[i], kSize, borderType,
                                                  &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "GaussianFilter");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(stdDevTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));


    ostringstream params;
    params << "kernel=" << kernelSize << ", sigma=" << sigma;
    printResult("RPP HIP GaussianFilter", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_SobelFilter(const vector<Mat>& imgs, bool isColor, int sobelType,
                                  rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;
    int dstChannels = 1;
    Rpp32u kernelSize = 3;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;

        set_descriptor_dims_and_strides_local(&dstDescs[i], 1, imgs[i].rows, imgs[i].cols, dstChannels, 0);
        dstDescs[i].layout = RpptLayout::NCHW;
        dstDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&dstDescs[i]);

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t srcBufferSize = imgs[i].total() * imgs[i].elemSize();
        size_t dstBufferSize = imgs[i].rows * imgs[i].cols;

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], srcBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], dstBufferSize));

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], imgs[i].data, srcBufferSize, hipMemcpyHostToDevice));
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_sobel_filter(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                              sobelType, kernelSize, &roiTensor[i], RpptRoiType::XYWH,
                                              handle, RPP_HIP_BACKEND),
                            "SobelFilter");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    
    ostringstream params;
    params << "type=" << sobelType;
    printResult("RPP HIP SobelFilter", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Erode(const vector<Mat>& imgs, bool isColor, int kernelSize,
                            rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    // Calculate required offset for border handling
    int offsetInBytes = 12 * (kernelSize / 2);

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, offsetInBytes);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_erode(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                       kernelSize, &roiTensor[i], RpptRoiType::XYWH,
                                       handle, RPP_HIP_BACKEND),
                            "Erode");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "kernel=" << kernelSize;
    printResult("RPP HIP Erode", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Dilate(const vector<Mat>& imgs, bool isColor, int kernelSize,
                             rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    // Calculate required offset for border handling
    int offsetInBytes = 12 * (kernelSize / 2);

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, offsetInBytes);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_dilate(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                        kernelSize, &roiTensor[i], RpptRoiType::XYWH,
                                        handle, RPP_HIP_BACKEND),
                            "Dilate");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "kernel=" << kernelSize;
    printResult("RPP HIP Dilate", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Emboss(const vector<Mat>& imgs, bool isColor, int kernelSize, float strength,
                             rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;
    Rpp32u kSize = static_cast<Rpp32u>(kernelSize);

    // Calculate required offset for border handling
    int offsetInBytes = 12 * (kernelSize / 2);

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    Rpp32f *strengthTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&strengthTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, offsetInBytes);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        strengthTensor[i] = strength;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_emboss(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                        &strengthTensor[i], kSize, borderType,
                                        &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Emboss");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(strengthTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "kernel=" << kernelSize << ", strength=" << strength;
    printResult("RPP HIP Emboss", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Crop(const vector<Mat>& imgs, bool isColor, int cropWidth, int cropHeight,
                           rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);

        set_descriptor_dims_and_strides_local(&dstDescs[i], 1, cropHeight, cropWidth, numChannels, 0);
        dstDescs[i].layout = layout;
        dstDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&dstDescs[i]);

        roiTensor[i].xywhROI.xy.x = (imgs[i].cols - cropWidth) / 2;
        roiTensor[i].xywhROI.xy.y = (imgs[i].rows - cropHeight) / 2;
        roiTensor[i].xywhROI.roiWidth = cropWidth;
        roiTensor[i].xywhROI.roiHeight = cropHeight;

        // Calculate buffer sizes based on aligned descriptor dimensions
        size_t srcAlignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);
        size_t dstAlignedBufferSize = dstDescs[i].n * dstDescs[i].h * dstDescs[i].w * dstDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], srcAlignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], dstAlignedBufferSize));

        // Copy source image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[srcAlignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, srcAlignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_crop(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                      &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Crop");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "size=" << cropWidth << "x" << cropHeight;
    printResult("RPP HIP Crop", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Resize(const vector<Mat>& imgs, bool isColor, int dstW, int dstH,
                             RpptInterpolationType interpType, const string& interpName,
                             rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptImagePatch *dstImgSizes;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&dstImgSizes, num_images * sizeof(RpptImagePatch)));
    // Allocate 256 ROI elements per image as workaround for kernel bug (launches 256 threads without bounds check)
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * 256 * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);

        set_descriptor_dims_and_strides_local(&dstDescs[i], 1, dstH, dstW, numChannels, 0);
        dstDescs[i].layout = layout;
        dstDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&dstDescs[i]);

        dstImgSizes[i].width = dstW;
        dstImgSizes[i].height = dstH;

        // Each image gets 256 ROI slots (only first is used, rest are buffer for buggy kernel)
        roiTensor[i * 256].xywhROI.xy.x = 0;
        roiTensor[i * 256].xywhROI.xy.y = 0;
        roiTensor[i * 256].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i * 256].xywhROI.roiHeight = imgs[i].rows;

        size_t srcAlignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);
        size_t dstAlignedBufferSize = dstDescs[i].n * dstDescs[i].h * dstDescs[i].w * dstDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], srcAlignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], dstAlignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[srcAlignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, srcAlignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            // Pass pointer to the i-th 256-element ROI block
            CHECK_RPP_STATUS(rppt_resize(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                        &dstImgSizes[i], interpType, &roiTensor[i * 256], RpptRoiType::XYWH,
                                        handle, RPP_HIP_BACKEND),
                            "Resize");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(dstImgSizes));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "type=" << interpName << ", size=" << dstW << "x" << dstH;
    printResult("RPP HIP Resize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Flip(const vector<Mat>& imgs, bool isColor, int flipCode, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    Rpp32u horizontalFlag = (flipCode == 1 || flipCode == -1) ? 1 : 0;
    Rpp32u verticalFlag = (flipCode == 0 || flipCode == -1) ? 1 : 0;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    Rpp32u *horizontalTensor;
    Rpp32u *verticalTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&horizontalTensor, num_images * sizeof(Rpp32u)));
    CHECK_HIP_STATUS(hipHostMalloc(&verticalTensor, num_images * sizeof(Rpp32u)));
    // Allocate 256 ROI elements per image as workaround for kernel bug (launches 256 threads without bounds check)
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * 256 * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        horizontalTensor[i] = horizontalFlag;
        verticalTensor[i] = verticalFlag;
        // Each image gets 256 ROI slots (only first is used, rest are buffer for buggy kernel)
        roiTensor[i * 256].xywhROI.xy.x = 0;
        roiTensor[i * 256].xywhROI.xy.y = 0;
        roiTensor[i * 256].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i * 256].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned stride (not descriptor width which is actual image size)
        // Buffer size = nStride because we only have n=1 image
        size_t alignedBufferSize = srcDescs[i].strides.nStride * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].strides.hStride / srcDescs[i].c;  // Extract aligned width from stride
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        // CRITICAL: Sync before resetting ROI to prevent CPU-GPU race
        // The GPU may still be reading roiTensor from the previous iteration
        if (k > 0) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
        }

        for (int i = 0; i < num_images; ++i) {       
            roiTensor[i * 256].xywhROI.xy.x = 0;
            roiTensor[i * 256].xywhROI.xy.y = 0;
            roiTensor[i * 256].xywhROI.roiWidth = imgs[i].cols;
            roiTensor[i * 256].xywhROI.roiHeight = imgs[i].rows;
        }

        for (int i = 0; i < num_images; ++i) {
            // Pass pointer to the i-th 256-element ROI block
            CHECK_RPP_STATUS(rppt_flip(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                      &horizontalTensor[i], &verticalTensor[i],
                                      &roiTensor[i * 256], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Flip");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(horizontalTensor));
    CHECK_HIP_STATUS(hipHostFree(verticalTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    string name = (flipCode == 1) ? "Horizontal" : (flipCode == 0) ? "Vertical" : "Both";
    printResult("RPP HIP Flip", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), "type=" + name);
}

void benchmark_RPP_HIP_Rotate(const vector<Mat>& imgs, bool isColor, float angleDeg,
                             rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    Rpp32f *angleTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&angleTensor, num_images * sizeof(Rpp32f)));
    // Allocate 256 ROI elements per image as workaround for kernel bug (launches 256 threads without bounds check)
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * 256 * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        angleTensor[i] = angleDeg;
        // Each image gets 256 ROI slots (only first is used, rest are buffer for buggy kernel)
        roiTensor[i * 256].xywhROI.xy.x = 0;
        roiTensor[i * 256].xywhROI.xy.y = 0;
        roiTensor[i * 256].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i * 256].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions (not original image size)
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        // Source has original width, destination has aligned width
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            // Pass pointer to the i-th 256-element ROI block
            CHECK_RPP_STATUS(rppt_rotate(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                        &angleTensor[i], RpptInterpolationType::BILINEAR,
                                        &roiTensor[i * 256], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Rotate");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(angleTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "angle=" << angleDeg << "deg";
    printResult("RPP HIP Rotate", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_WarpAffine(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    float *affineTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&affineTensor, num_images * 6 * sizeof(float)));
    // Allocate 256 ROI elements per image as workaround for kernel bug (launches 256 threads without bounds check)
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * 256 * sizeof(RpptROI)));

    // Affine transformation matrix: [a11, a12, a13, a21, a22, a23]
    // Same transformation for all images
    float affine[6] = {1.0f, 0.1f, 10.0f, 0.1f, 1.0f, 10.0f};

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Copy affine matrix for this image
        memcpy(&affineTensor[i * 6], affine, 6 * sizeof(float));

        // Each image gets 256 ROI slots (only first is used, rest are buffer for buggy kernel)
        roiTensor[i * 256].xywhROI.xy.x = 0;
        roiTensor[i * 256].xywhROI.xy.y = 0;
        roiTensor[i * 256].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i * 256].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            // Pass pointer to the i-th 256-element ROI block
            CHECK_RPP_STATUS(rppt_warp_affine(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                             &affineTensor[i * 6], RpptInterpolationType::BILINEAR,
                                             &roiTensor[i * 256], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "WarpAffine");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(affineTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP WarpAffine", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_WarpPerspective(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    float *perspectiveTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&perspectiveTensor, num_images * 9 * sizeof(float)));
    // Allocate 256 ROI elements per image as workaround for kernel bug (launches 256 threads without bounds check)
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * 256 * sizeof(RpptROI)));

    // Perspective transformation matrix: 3x3 homography matrix
    // Same transformation for all images
    float perspectiveMatrix[9] = {1.0f, 0.1f, 0.0f, 0.1f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Copy perspective matrix for this image
        memcpy(&perspectiveTensor[i * 9], perspectiveMatrix, 9 * sizeof(float));

        // Each image gets 256 ROI slots (only first is used, rest are buffer for buggy kernel)
        roiTensor[i * 256].xywhROI.xy.x = 0;
        roiTensor[i * 256].xywhROI.xy.y = 0;
        roiTensor[i * 256].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i * 256].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            // Pass pointer to the i-th 256-element ROI block
            CHECK_RPP_STATUS(rppt_warp_perspective(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                   &perspectiveTensor[i * 9], RpptInterpolationType::BILINEAR,
                                                   &roiTensor[i * 256], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "WarpPerspective");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(perspectiveTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP WarpPerspective", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_Fisheye(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptROI *roiTensor;
    // Allocate 256 ROI elements per image as workaround for kernel bug (launches 256 threads without bounds check)
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * 256 * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Each image gets 256 ROI slots (only first is used, rest are buffer for buggy kernel)
        roiTensor[i * 256].xywhROI.xy.x = 0;
        roiTensor[i * 256].xywhROI.xy.y = 0;
        roiTensor[i * 256].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i * 256].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            // Pass pointer to the i-th 256-element ROI block
            CHECK_RPP_STATUS(rppt_fisheye(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                         &roiTensor[i * 256], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Fisheye");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP Fisheye", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_LensCorrection(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    if (num_images == 0) return;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Camera matrix and distortion coefficients in pinned host memory
    Rpp32f *cameraMatrixTensor;
    Rpp32f *distortionCoeffsTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&cameraMatrixTensor, num_images * 9 * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&distortionCoeffsTensor, num_images * 8 * sizeof(Rpp32f)));
    // Allocate 256 ROI elements per image as workaround for kernel bug (launches 256 threads without bounds check)
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * 256 * sizeof(RpptROI)));

    // Sample camera calibration parameters
    Rpp32f sampleCameraMatrix[9] = {534.07088364f, 0.0f, 341.53407554f, 0.0f, 534.11914595f, 232.94565259f, 0.0f, 0.0f, 1.0f};
    Rpp32f sampleDistortion[8] = {-0.29297164f, 0.10770696f, 0.00131038f, -0.0000311f, 0.0434798f, 0.0f, 0.0f, 0.0f};

    // Allocate remap tables in device memory
    // Table descriptor for remap tables
    RpptDesc tableDesc;
    set_descriptor_dims_and_strides_local(&tableDesc, num_images, imgs[0].rows, imgs[0].cols, 1, 0);
    tableDesc.layout = RpptLayout::NCHW;
    tableDesc.dataType = RpptDataType::F32;
    update_strides_from_layout(&tableDesc);

    size_t tableSize = num_images * imgs[0].rows * imgs[0].cols * sizeof(Rpp32f);
    Rpp32f *d_rowRemapTable, *d_colRemapTable;
    CHECK_HIP_STATUS(hipMalloc(&d_rowRemapTable, tableSize));
    CHECK_HIP_STATUS(hipMalloc(&d_colRemapTable, tableSize));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Copy camera matrix and distortion coefficients for this image
        memcpy(&cameraMatrixTensor[i * 9], sampleCameraMatrix, 9 * sizeof(Rpp32f));
        memcpy(&distortionCoeffsTensor[i * 8], sampleDistortion, 8 * sizeof(Rpp32f));

        // Each image gets 256 ROI slots (only first is used, rest are buffer for buggy kernel)
        roiTensor[i * 256].xywhROI.xy.x = 0;
        roiTensor[i * 256].xywhROI.xy.y = 0;
        roiTensor[i * 256].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i * 256].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            // Pass pointer to the i-th 256-element ROI block
            CHECK_RPP_STATUS(rppt_lens_correction(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                  d_rowRemapTable, d_colRemapTable, &tableDesc,
                                                  &cameraMatrixTensor[i * 9], &distortionCoeffsTensor[i * 8],
                                                  &roiTensor[i * 256], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "LensCorrection");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipFree(d_rowRemapTable));
    CHECK_HIP_STATUS(hipFree(d_colRemapTable));
    CHECK_HIP_STATUS(hipHostFree(cameraMatrixTensor));
    CHECK_HIP_STATUS(hipHostFree(distortionCoeffsTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP LensCorrection", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_AddScalar(const vector<Mat>& imgs, bool isColor, float addVal,
                                rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Convert to F32 and create batch buffer
    int imageSize = height * width * channels;
    vector<Rpp32f> h_inputBuffer(num_images * imageSize);

    for (int i = 0; i < num_images; ++i) {
        Mat imgF32;
        imgs[i].convertTo(imgF32, CV_32F);
        memcpy(h_inputBuffer.data() + i * imageSize, imgF32.data, imageSize * sizeof(Rpp32f));
    }

    // Allocate device memory
    Rpp32f *d_inputBuffer, *d_outputBuffer;
    size_t bufferSize = num_images * imageSize * sizeof(Rpp32f);
    CHECK_HIP_STATUS(hipMalloc(&d_inputBuffer, bufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_outputBuffer, bufferSize));
    CHECK_HIP_STATUS(hipMemcpy(d_inputBuffer, h_inputBuffer.data(), bufferSize, hipMemcpyHostToDevice));

    // Allocate pinned host memory for add tensor
    Rpp32f *addTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&addTensor, num_images * sizeof(Rpp32f)));
    for (int i = 0; i < num_images; ++i) {
        addTensor[i] = addVal;
    }

    // Create 5D generic descriptor (NDHWC or NCDHW) with depth=1 for 2D images
    RpptGenericDesc genericDesc;
    genericDesc.numDims = 5;
    genericDesc.offsetInBytes = 0;
    genericDesc.dataType = RpptDataType::F32;

    if (isColor) {
        genericDesc.layout = RpptLayout::NDHWC;
        genericDesc.dims[0] = num_images;
        genericDesc.dims[1] = 1;
        genericDesc.dims[2] = height;
        genericDesc.dims[3] = width;
        genericDesc.dims[4] = channels;
        genericDesc.strides[4] = 1;
        genericDesc.strides[3] = channels;
        genericDesc.strides[2] = width * channels;
        genericDesc.strides[1] = height * width * channels;
        genericDesc.strides[0] = 1 * height * width * channels;
    } else {
        genericDesc.layout = RpptLayout::NCDHW;
        genericDesc.dims[0] = num_images;
        genericDesc.dims[1] = channels;
        genericDesc.dims[2] = 1;
        genericDesc.dims[3] = height;
        genericDesc.dims[4] = width;
        genericDesc.strides[4] = 1;
        genericDesc.strides[3] = width;
        genericDesc.strides[2] = height * width;
        genericDesc.strides[1] = 1 * height * width;
        genericDesc.strides[0] = channels * 1 * height * width;
    }

    RpptGenericDesc srcGenericDesc = genericDesc;
    RpptGenericDesc dstGenericDesc = genericDesc;

    // Allocate ROI3D array in pinned host memory
    RpptROI3D *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI3D)));
    for (int i = 0; i < num_images; ++i) {
        roiTensor[i].xyzwhdROI.xyz.x = 0;
        roiTensor[i].xyzwhdROI.xyz.y = 0;
        roiTensor[i].xyzwhdROI.xyz.z = 0;
        roiTensor[i].xyzwhdROI.roiWidth = width;
        roiTensor[i].xyzwhdROI.roiHeight = height;
        roiTensor[i].xyzwhdROI.roiDepth = 1;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }

        CHECK_RPP_STATUS(rppt_add_scalar(d_inputBuffer, &srcGenericDesc, d_outputBuffer,
                                        &dstGenericDesc, addTensor, roiTensor,
                                        RpptRoi3DType::XYZWHD, handle, RPP_HIP_BACKEND),
                        "AddScalar");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_inputBuffer));
    CHECK_HIP_STATUS(hipFree(d_outputBuffer));
    CHECK_HIP_STATUS(hipHostFree(addTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    
    ostringstream params;
    params << "value=" << addVal;
    printResult("RPP HIP AddScalar", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_SubtractScalar(const vector<Mat>& imgs, bool isColor, float subVal,
                                     rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Convert to F32 and create batch buffer
    int imageSize = height * width * channels;
    vector<Rpp32f> h_inputBuffer(num_images * imageSize);

    for (int i = 0; i < num_images; ++i) {
        Mat imgF32;
        imgs[i].convertTo(imgF32, CV_32F);
        memcpy(h_inputBuffer.data() + i * imageSize, imgF32.data, imageSize * sizeof(Rpp32f));
    }

    // Allocate device memory
    Rpp32f *d_inputBuffer, *d_outputBuffer;
    size_t bufferSize = num_images * imageSize * sizeof(Rpp32f);
    CHECK_HIP_STATUS(hipMalloc(&d_inputBuffer, bufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_outputBuffer, bufferSize));
    CHECK_HIP_STATUS(hipMemcpy(d_inputBuffer, h_inputBuffer.data(), bufferSize, hipMemcpyHostToDevice));

    // Allocate pinned host memory for subtract tensor
    Rpp32f *subtractTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&subtractTensor, num_images * sizeof(Rpp32f)));
    for (int i = 0; i < num_images; ++i) {
        subtractTensor[i] = subVal;
    }

    // Create 5D generic descriptor
    RpptGenericDesc genericDesc;
    genericDesc.numDims = 5;
    genericDesc.offsetInBytes = 0;
    genericDesc.dataType = RpptDataType::F32;

    if (isColor) {
        genericDesc.layout = RpptLayout::NDHWC;
        genericDesc.dims[0] = num_images;
        genericDesc.dims[1] = 1;
        genericDesc.dims[2] = height;
        genericDesc.dims[3] = width;
        genericDesc.dims[4] = channels;
        genericDesc.strides[4] = 1;
        genericDesc.strides[3] = channels;
        genericDesc.strides[2] = width * channels;
        genericDesc.strides[1] = height * width * channels;
        genericDesc.strides[0] = 1 * height * width * channels;
    } else {
        genericDesc.layout = RpptLayout::NCDHW;
        genericDesc.dims[0] = num_images;
        genericDesc.dims[1] = channels;
        genericDesc.dims[2] = 1;
        genericDesc.dims[3] = height;
        genericDesc.dims[4] = width;
        genericDesc.strides[4] = 1;
        genericDesc.strides[3] = width;
        genericDesc.strides[2] = height * width;
        genericDesc.strides[1] = 1 * height * width;
        genericDesc.strides[0] = channels * 1 * height * width;
    }

    RpptGenericDesc srcGenericDesc = genericDesc;
    RpptGenericDesc dstGenericDesc = genericDesc;

    // Allocate ROI3D array in pinned host memory
    RpptROI3D *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI3D)));
    for (int i = 0; i < num_images; ++i) {
        roiTensor[i].xyzwhdROI.xyz.x = 0;
        roiTensor[i].xyzwhdROI.xyz.y = 0;
        roiTensor[i].xyzwhdROI.xyz.z = 0;
        roiTensor[i].xyzwhdROI.roiWidth = width;
        roiTensor[i].xyzwhdROI.roiHeight = height;
        roiTensor[i].xyzwhdROI.roiDepth = 1;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }

        CHECK_RPP_STATUS(rppt_subtract_scalar(d_inputBuffer, &srcGenericDesc, d_outputBuffer,
                                              &dstGenericDesc, subtractTensor, roiTensor,
                                              RpptRoi3DType::XYZWHD, handle, RPP_HIP_BACKEND),
                        "SubtractScalar");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_inputBuffer));
    CHECK_HIP_STATUS(hipFree(d_outputBuffer));
    CHECK_HIP_STATUS(hipHostFree(subtractTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    
    ostringstream params;
    params << "value=" << subVal;
    printResult("RPP HIP SubtractScalar", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_MultiplyScalar(const vector<Mat>& imgs, bool isColor, float mulVal,
                                     rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Convert to F32 and create batch buffer
    int imageSize = height * width * channels;
    vector<Rpp32f> h_inputBuffer(num_images * imageSize);

    for (int i = 0; i < num_images; ++i) {
        Mat imgF32;
        imgs[i].convertTo(imgF32, CV_32F);
        memcpy(h_inputBuffer.data() + i * imageSize, imgF32.data, imageSize * sizeof(Rpp32f));
    }

    // Allocate device memory
    Rpp32f *d_inputBuffer, *d_outputBuffer;
    size_t bufferSize = num_images * imageSize * sizeof(Rpp32f);
    CHECK_HIP_STATUS(hipMalloc(&d_inputBuffer, bufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_outputBuffer, bufferSize));
    CHECK_HIP_STATUS(hipMemcpy(d_inputBuffer, h_inputBuffer.data(), bufferSize, hipMemcpyHostToDevice));

    // Allocate pinned host memory for multiply tensor
    Rpp32f *mulTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&mulTensor, num_images * sizeof(Rpp32f)));
    for (int i = 0; i < num_images; ++i) {
        mulTensor[i] = mulVal;
    }

    // Create 5D generic descriptor
    RpptGenericDesc genericDesc;
    genericDesc.numDims = 5;
    genericDesc.offsetInBytes = 0;
    genericDesc.dataType = RpptDataType::F32;

    if (isColor) {
        genericDesc.layout = RpptLayout::NDHWC;
        genericDesc.dims[0] = num_images;
        genericDesc.dims[1] = 1;
        genericDesc.dims[2] = height;
        genericDesc.dims[3] = width;
        genericDesc.dims[4] = channels;
        genericDesc.strides[4] = 1;
        genericDesc.strides[3] = channels;
        genericDesc.strides[2] = width * channels;
        genericDesc.strides[1] = height * width * channels;
        genericDesc.strides[0] = 1 * height * width * channels;
    } else {
        genericDesc.layout = RpptLayout::NCDHW;
        genericDesc.dims[0] = num_images;
        genericDesc.dims[1] = channels;
        genericDesc.dims[2] = 1;
        genericDesc.dims[3] = height;
        genericDesc.dims[4] = width;
        genericDesc.strides[4] = 1;
        genericDesc.strides[3] = width;
        genericDesc.strides[2] = height * width;
        genericDesc.strides[1] = 1 * height * width;
        genericDesc.strides[0] = channels * 1 * height * width;
    }

    RpptGenericDesc srcGenericDesc = genericDesc;
    RpptGenericDesc dstGenericDesc = genericDesc;

    // Allocate ROI3D array in pinned host memory
    RpptROI3D *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI3D)));
    for (int i = 0; i < num_images; ++i) {
        roiTensor[i].xyzwhdROI.xyz.x = 0;
        roiTensor[i].xyzwhdROI.xyz.y = 0;
        roiTensor[i].xyzwhdROI.xyz.z = 0;
        roiTensor[i].xyzwhdROI.roiWidth = width;
        roiTensor[i].xyzwhdROI.roiHeight = height;
        roiTensor[i].xyzwhdROI.roiDepth = 1;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }

        CHECK_RPP_STATUS(rppt_multiply_scalar(d_inputBuffer, &srcGenericDesc, d_outputBuffer,
                                              &dstGenericDesc, mulTensor, roiTensor,
                                              RpptRoi3DType::XYZWHD, handle, RPP_HIP_BACKEND),
                        "MultiplyScalar");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_inputBuffer));
    CHECK_HIP_STATUS(hipFree(d_outputBuffer));
    CHECK_HIP_STATUS(hipHostFree(mulTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    
    ostringstream params;
    params << "value=" << mulVal;
    printResult("RPP HIP MultiplyScalar", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_BitwiseAnd(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<Mat> imgs2(num_images);
    for (int i = 0; i < num_images; ++i) {
        imgs[i].copyTo(imgs2[i]);
    }

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs1(num_images);
    vector<Rpp8u*> d_inputs2(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs1[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_inputs2[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs1[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs2[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs2[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_bitwise_and(d_inputs1[i], d_inputs2[i], &srcDescs[i],
                                             d_outputs[i], &dstDescs[i], &roiTensor[i],
                                             RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "BitwiseAnd");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs1[i]));
        CHECK_HIP_STATUS(hipFree(d_inputs2[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP BitwiseAnd", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_BitwiseOr(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<Mat> imgs2(num_images);
    for (int i = 0; i < num_images; ++i) {
        imgs[i].copyTo(imgs2[i]);
    }

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs1(num_images);
    vector<Rpp8u*> d_inputs2(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs1[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_inputs2[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs1[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs2[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs2[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_bitwise_or(d_inputs1[i], d_inputs2[i], &srcDescs[i],
                                            d_outputs[i], &dstDescs[i], &roiTensor[i],
                                            RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "BitwiseOr");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs1[i]));
        CHECK_HIP_STATUS(hipFree(d_inputs2[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP BitwiseOr", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_BitwiseNot(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_bitwise_not(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                             &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "BitwiseNot");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP BitwiseNot", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_BitwiseXor(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<Mat> imgs2(num_images);
    for (int i = 0; i < num_images; ++i) {
        imgs[i].copyTo(imgs2[i]);
    }

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs1(num_images);
    vector<Rpp8u*> d_inputs2(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs1[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_inputs2[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs1[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs2[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs2[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_bitwise_xor(d_inputs1[i], d_inputs2[i], &srcDescs[i],
                                             d_outputs[i], &dstDescs[i], &roiTensor[i],
                                             RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "BitwiseXor");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs1[i]));
        CHECK_HIP_STATUS(hipFree(d_inputs2[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP BitwiseXor", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_Threshold(const vector<Mat>& imgs, bool isColor, double thresh,
                                rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors in pinned host memory
    Rpp32f *minTensor;
    Rpp32f *maxTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&minTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&maxTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Binary threshold: pixels below thresh -> 0, pixels >= thresh -> 255
        minTensor[i] = (Rpp32f)thresh;
        maxTensor[i] = 255.0f;

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Allocate device memory
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    rppSetBatchSize(handle, 1);

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_threshold(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                           &minTensor[i], &maxTensor[i],
                                           &roiTensor[i], RpptRoiType::XYWH,
                                           handle, RPP_HIP_BACKEND),
                            "Threshold");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(minTensor));
    CHECK_HIP_STATUS(hipHostFree(maxTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "threshold=" << thresh;
    printResult("RPP HIP Threshold", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_HistogramEqualize(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_histogram_equalize(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                     &roiTensor[i], RpptRoiType::XYWH,
                                                     handle, RPP_HIP_BACKEND),
                            "HistogramEqualize");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP HistogramEqualize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_LUT(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate LUT and ROI tensors in pinned host memory
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    // Allocate device memory for LUT (256 entries for 8-bit input)
    Rpp8u *d_lut;
    CHECK_HIP_STATUS(hipMalloc(&d_lut, 256 * sizeof(Rpp8u)));

    // Create inverse LUT (invert pixel values)
    Rpp8u h_lut[256];
    for (int i = 0; i < 256; ++i) {
        h_lut[i] = 255 - i;  // Invert: 0→255, 255→0
    }
    CHECK_HIP_STATUS(hipMemcpy(d_lut, h_lut, 256 * sizeof(Rpp8u), hipMemcpyHostToDevice));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_lut(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                     d_lut, &roiTensor[i], RpptRoiType::XYWH,
                                     handle, RPP_HIP_BACKEND),
                            "LUT");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipFree(d_lut));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "lut=inverse";
    printResult("RPP HIP LUT", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Magnitude(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs1(num_images);
    vector<Rpp8u*> d_inputs2(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate ROI tensor in pinned host memory
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs1[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_inputs2[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs1[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));

        // Create second input (shifted version for demonstration)
        for (size_t j = 0; j < alignedBufferSize; ++j) {
            h_tempBuffer[j] = (h_tempBuffer[j] >> 1);  // Half the values
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs2[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            // Magnitude: sqrt(src1^2 + src2^2)
            CHECK_RPP_STATUS(rppt_magnitude(d_inputs1[i], d_inputs2[i], &srcDescs[i],
                                           d_outputs[i], &dstDescs[i],
                                           &roiTensor[i], RpptRoiType::XYWH,
                                           handle, RPP_HIP_BACKEND),
                            "Magnitude");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs1[i]));
        CHECK_HIP_STATUS(hipFree(d_inputs2[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "inputs=2";
    printResult("RPP HIP Magnitude", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Phase(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs1(num_images);
    vector<Rpp8u*> d_inputs2(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate ROI tensor in pinned host memory
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs1[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_inputs2[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs1[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));

        // Create second input (shifted version for demonstration)
        for (size_t j = 0; j < alignedBufferSize; ++j) {
            h_tempBuffer[j] = (h_tempBuffer[j] >> 1);  // Half the values
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs2[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            // Phase: atan2(src2, src1) in degrees
            CHECK_RPP_STATUS(rppt_phase(d_inputs1[i], d_inputs2[i], &srcDescs[i],
                                       d_outputs[i], &dstDescs[i],
                                       &roiTensor[i], RpptRoiType::XYWH,
                                       handle, RPP_HIP_BACKEND),
                            "Phase");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs1[i]));
        CHECK_HIP_STATUS(hipFree(d_inputs2[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "inputs=2";
    printResult("RPP HIP Phase", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Remap(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptDesc> tableDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);
    vector<Rpp32f*> d_rowRemapTable(num_images);
    vector<Rpp32f*> d_colRemapTable(num_images);

    // Allocate ROI tensor in pinned host memory
    RpptROI *roiTensor;
    // Allocate 256 ROI elements per image as workaround for kernel bug (launches 256 threads without bounds check)
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * 256 * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Table descriptor (same dimensions as image, 1 channel for row/col maps)
        set_descriptor_dims_and_strides_local(&tableDescs[i], 1, imgs[i].rows, imgs[i].cols, 1, 0);
        tableDescs[i].layout = RpptLayout::NCHW;
        tableDescs[i].dataType = RpptDataType::F32;
        update_strides_from_layout(&tableDescs[i]);

        // Each image gets 256 ROI slots (only first is used, rest are buffer for buggy kernel)
        roiTensor[i * 256].xywhROI.xy.x = 0;
        roiTensor[i * 256].xywhROI.xy.y = 0;
        roiTensor[i * 256].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i * 256].xywhROI.roiHeight = imgs[i].rows;

        int height = imgs[i].rows;
        int width = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;

        // Calculate buffer sizes
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);
        size_t tableSize = tableDescs[i].n * tableDescs[i].h * tableDescs[i].w * sizeof(Rpp32f);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_rowRemapTable[i], tableSize));
        CHECK_HIP_STATUS(hipMalloc(&d_colRemapTable[i], tableSize));

        // Copy image data
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int elementsPerRow = width * numChannels;

        for (int row = 0; row < height; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;

        // Create remap tables (sine wave distortion to match other implementations)
        Rpp32f* h_rowTable = new Rpp32f[tableSize / sizeof(Rpp32f)]();
        Rpp32f* h_colTable = new Rpp32f[tableSize / sizeof(Rpp32f)]();

        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < tableDescs[i].w; ++col) {
                int idx = row * tableDescs[i].w + col;
                if (col < width) {
                    h_rowTable[idx] = (Rpp32f)row + sin(col * 0.01f) * 5.0f;  // Sine wave on rows
                    h_colTable[idx] = (Rpp32f)col + cos(row * 0.01f) * 5.0f;  // Cosine wave on cols
                } else {
                    h_rowTable[idx] = (Rpp32f)row;  // Padding
                    h_colTable[idx] = 0.0f;  // Padding
                }
            }
        }

        CHECK_HIP_STATUS(hipMemcpy(d_rowRemapTable[i], h_rowTable, tableSize, hipMemcpyHostToDevice));
        CHECK_HIP_STATUS(hipMemcpy(d_colRemapTable[i], h_colTable, tableSize, hipMemcpyHostToDevice));
        delete[] h_rowTable;
        delete[] h_colTable;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            // Pass pointer to the i-th 256-element ROI block
            CHECK_RPP_STATUS(rppt_remap(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                       d_rowRemapTable[i], d_colRemapTable[i], &tableDescs[i],
                                       RpptInterpolationType::BILINEAR,
                                       &roiTensor[i * 256], RpptRoiType::XYWH,
                                       handle, RPP_HIP_BACKEND),
                            "Remap");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
        CHECK_HIP_STATUS(hipFree(d_rowRemapTable[i]));
        CHECK_HIP_STATUS(hipFree(d_colRemapTable[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "transform=sine_wave, interpolation=bilinear";
    printResult("RPP HIP Remap", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_TensorMin(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    // Output length: grayscale = n, RGB = n * 4 (R, G, B, overall per image)
    Rpp32u outputLength = isColor ? (num_images * 4) : num_images;

    vector<RpptDesc> srcDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);

    RpptROI *roiTensor;
    Rpp8u *d_minOutputs;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));
    CHECK_HIP_STATUS(hipMalloc(&d_minOutputs, outputLength * sizeof(Rpp8u)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);
        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            Rpp32u imgOutputLength = isColor ? 4 : 1;
            CHECK_RPP_STATUS(rppt_tensor_min(d_inputs[i], &srcDescs[i],
                                            d_minOutputs + (i * imgOutputLength), imgOutputLength,
                                            &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "TensorMin");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
    }
    CHECK_HIP_STATUS(hipFree(d_minOutputs));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP TensorMin", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_TensorMax(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    Rpp32u outputLength = isColor ? (num_images * 4) : num_images;

    vector<RpptDesc> srcDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);

    RpptROI *roiTensor;
    Rpp8u *d_maxOutputs;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));
    CHECK_HIP_STATUS(hipMalloc(&d_maxOutputs, outputLength * sizeof(Rpp8u)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);
        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            Rpp32u imgOutputLength = isColor ? 4 : 1;
            CHECK_RPP_STATUS(rppt_tensor_max(d_inputs[i], &srcDescs[i],
                                            d_maxOutputs + (i * imgOutputLength), imgOutputLength,
                                            &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "TensorMax");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
    }
    CHECK_HIP_STATUS(hipFree(d_maxOutputs));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP TensorMax", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_TensorSum(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    Rpp32u outputLength = isColor ? (num_images * 4) : num_images;

    vector<RpptDesc> srcDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);

    RpptROI *roiTensor;
    Rpp64u *d_sumOutputs;  // Note: tensor_sum returns Rpp64u for U8 input
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));
    CHECK_HIP_STATUS(hipMalloc(&d_sumOutputs, outputLength * sizeof(Rpp64u)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);
        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            Rpp32u imgOutputLength = isColor ? 4 : 1;
            CHECK_RPP_STATUS(rppt_tensor_sum(d_inputs[i], &srcDescs[i],
                                            d_sumOutputs + (i * imgOutputLength), imgOutputLength,
                                            &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "TensorSum");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
    }
    CHECK_HIP_STATUS(hipFree(d_sumOutputs));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP TensorSum", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_TensorMean(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    Rpp32u outputLength = isColor ? (num_images * 4) : num_images;

    vector<RpptDesc> srcDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);

    RpptROI *roiTensor;
    Rpp32f *d_meanOutputs;  // Note: tensor_mean returns Rpp32f
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));
    CHECK_HIP_STATUS(hipMalloc(&d_meanOutputs, outputLength * sizeof(Rpp32f)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);
        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            Rpp32u imgOutputLength = isColor ? 4 : 1;
            CHECK_RPP_STATUS(rppt_tensor_mean(d_inputs[i], &srcDescs[i],
                                             d_meanOutputs + (i * imgOutputLength), imgOutputLength,
                                             &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "TensorMean");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
    }
    CHECK_HIP_STATUS(hipFree(d_meanOutputs));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP TensorMean", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_TensorStddev(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    Rpp32u outputLength = isColor ? (num_images * 4) : num_images;

    vector<RpptDesc> srcDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);

    RpptROI *roiTensor;
    Rpp32f *d_meanOutputs;
    Rpp32f *d_stddevOutputs;
    Rpp32f *h_meanOutputs;  // Host buffer to compute mean first
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));
    CHECK_HIP_STATUS(hipHostMalloc(&h_meanOutputs, outputLength * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipMalloc(&d_meanOutputs, outputLength * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipMalloc(&d_stddevOutputs, outputLength * sizeof(Rpp32f)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);
        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    // First compute mean (required for stddev calculation)
    for (int i = 0; i < num_images; ++i) {
        Rpp32u imgOutputLength = isColor ? 4 : 1;
        CHECK_RPP_STATUS(rppt_tensor_mean(d_inputs[i], &srcDescs[i],
                                         d_meanOutputs + (i * imgOutputLength), imgOutputLength,
                                         &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "TensorMean (for stddev)");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));

    // Copy mean to host (stddev requires host mean tensor)
    CHECK_HIP_STATUS(hipMemcpy(h_meanOutputs, d_meanOutputs, outputLength * sizeof(Rpp32f), hipMemcpyDeviceToHost));

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            Rpp32u imgOutputLength = isColor ? 4 : 1;
            CHECK_RPP_STATUS(rppt_tensor_stddev(d_inputs[i], &srcDescs[i],
                                               d_stddevOutputs + (i * imgOutputLength), imgOutputLength,
                                               h_meanOutputs + (i * imgOutputLength),
                                               &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "TensorStddev");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
    }
    CHECK_HIP_STATUS(hipFree(d_meanOutputs));
    CHECK_HIP_STATUS(hipFree(d_stddevOutputs));
    CHECK_HIP_STATUS(hipHostFree(h_meanOutputs));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP TensorStddev", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_GaussianNoise(const vector<Mat>& imgs, bool isColor, float mean, float stdDev,
                                    rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    Rpp32f *meanTensor, *stdDevTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&meanTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&stdDevTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        meanTensor[i] = mean;
        stdDevTensor[i] = stdDev;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    // Use fixed seed for reproducibility
    Rpp32u seed = 12345;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_gaussian_noise(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                &meanTensor[i], &stdDevTensor[i], seed,
                                                &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "GaussianNoise");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(meanTensor));
    CHECK_HIP_STATUS(hipHostFree(stdDevTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "mean=" << mean << ", stdDev=" << stdDev;
    printResult("RPP HIP GaussianNoise", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_SaltAndPepperNoise(const vector<Mat>& imgs, bool isColor, float prob,
                                         rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    Rpp32f *noiseProbabilityTensor, *saltProbabilityTensor, *saltValueTensor, *pepperValueTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&noiseProbabilityTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&saltProbabilityTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&saltValueTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&pepperValueTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        noiseProbabilityTensor[i] = prob;
        saltProbabilityTensor[i] = 0.5f;  // 50% probability of salt when noise occurs
        saltValueTensor[i] = 1.0f;        // Salt value (normalized to 0-1 range)
        pepperValueTensor[i] = 0.0f;      // Pepper value (normalized to 0-1 range)
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    // Use fixed seed for reproducibility
    Rpp32u seed = 12345;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_salt_and_pepper_noise(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                       &noiseProbabilityTensor[i], &saltProbabilityTensor[i],
                                                       &saltValueTensor[i], &pepperValueTensor[i], seed,
                                                       &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "SaltAndPepperNoise");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(noiseProbabilityTensor));
    CHECK_HIP_STATUS(hipHostFree(saltProbabilityTensor));
    CHECK_HIP_STATUS(hipHostFree(saltValueTensor));
    CHECK_HIP_STATUS(hipHostFree(pepperValueTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "probability=" << prob;
    printResult("RPP HIP SaltAndPepperNoise", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_NoiseShot(const vector<Mat>& imgs, bool isColor, float shotNoiseParam,
                                rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors in pinned host memory
    Rpp32f *shotNoiseFactorTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&shotNoiseFactorTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Shot noise factor (controls Poisson noise intensity)
        shotNoiseFactorTensor[i] = shotNoiseParam;

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Allocate device memory
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    // Seed for random number generation
    Rpp32u seed = 12345;

    rppSetBatchSize(handle, 1);

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_shot_noise(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                            &shotNoiseFactorTensor[i], seed,
                                            &roiTensor[i], RpptRoiType::XYWH,
                                            handle, RPP_HIP_BACKEND),
                            "ShotNoise");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(shotNoiseFactorTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "factor=" << shotNoiseParam;
    printResult("RPP HIP NoiseShot", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Posterize(const vector<Mat>& imgs, bool isColor, int bits,
                                rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    Rpp8u *posterizeLevelBits;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&posterizeLevelBits, num_images * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        posterizeLevelBits[i] = (Rpp8u)bits;  // Number of bits to preserve (e.g., 4 bits)
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_posterize(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                          &posterizeLevelBits[i],
                                          &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Posterize");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(posterizeLevelBits));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "bits=" << bits;
    printResult("RPP HIP Posterize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Solarize(const vector<Mat>& imgs, bool isColor, int threshold,
                               rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    Rpp32f *thresholdTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&thresholdTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Threshold in normalized range (0.0-1.0) for U8 images
        thresholdTensor[i] = threshold / 255.0f;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_solarize(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                          &thresholdTensor[i],
                                          &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Solarize");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(thresholdTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "threshold=" << threshold;
    printResult("RPP HIP Solarize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_ColorCast(const vector<Mat>& imgs, bool isColor, float r, float g, float b,
                                rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    // ColorCast requires RGB images (3 channels)
    if (numChannels != 3) {
        cerr << "Warning: ColorCast requires RGB images (3 channels). Skipping." << endl;
        return;
    }

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    RpptRGB *rgbTensor;
    Rpp32f *alphaTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&rgbTensor, num_images * sizeof(RpptRGB)));
    CHECK_HIP_STATUS(hipHostMalloc(&alphaTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = RpptLayout::NHWC;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // RGB color cast values (0-255 range)
        rgbTensor[i].R = (Rpp8u)(r * 255.0f);
        rgbTensor[i].G = (Rpp8u)(g * 255.0f);
        rgbTensor[i].B = (Rpp8u)(b * 255.0f);
        alphaTensor[i] = 0.5f;  // Alpha blending factor (0.0-1.0)

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_color_cast(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                            &rgbTensor[i], &alphaTensor[i],
                                            &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "ColorCast");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(rgbTensor));
    CHECK_HIP_STATUS(hipHostFree(alphaTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "R=" << r << ", G=" << g << ", B=" << b;
    printResult("RPP HIP ColorCast", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_ColorTemperature(const vector<Mat>& imgs, bool isColor, int adjustment,
                                       rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    // ColorTemperature requires RGB images (3 channels)
    if (numChannels != 3) {
        cerr << "Warning: ColorTemperature requires RGB images (3 channels). Skipping." << endl;
        return;
    }

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    Rpp32s *adjustmentValueTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&adjustmentValueTensor, num_images * sizeof(Rpp32s)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = RpptLayout::NHWC;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Color temperature adjustment (-100 to 100, negative for cooler, positive for warmer)
        adjustmentValueTensor[i] = (Rpp32s)adjustment;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_color_temperature(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                    &adjustmentValueTensor[i],
                                                    &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "ColorTemperature");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(adjustmentValueTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "adjustment=" << adjustment;
    printResult("RPP HIP ColorTemperature", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_ColorTwist(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    // ColorTwist requires RGB images (3 channels)
    if (numChannels != 3) {
        cerr << "Warning: ColorTwist requires RGB images (3 channels). Skipping." << endl;
        return;
    }

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    Rpp32f *brightnessTensor, *contrastTensor, *hueTensor, *saturationTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&brightnessTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&contrastTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&hueTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&saturationTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = RpptLayout::NHWC;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // ColorTwist parameters: applies brightness, contrast, hue, and saturation adjustments
        brightnessTensor[i] = 1.0f;   // Brightness factor (1.0 = no change)
        contrastTensor[i] = 1.0f;     // Contrast factor (1.0 = no change)
        hueTensor[i] = 60.0f;         // Hue shift in degrees (-180 to 180)
        saturationTensor[i] = 1.3f;   // Saturation factor (1.0 = no change)

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_color_twist(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                             &brightnessTensor[i], &contrastTensor[i],
                                             &hueTensor[i], &saturationTensor[i],
                                             &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "ColorTwist");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(brightnessTensor));
    CHECK_HIP_STATUS(hipHostFree(contrastTensor));
    CHECK_HIP_STATUS(hipHostFree(hueTensor));
    CHECK_HIP_STATUS(hipHostFree(saturationTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "hue=60, saturation=1.3";
    printResult("RPP HIP ColorTwist", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Vignette(const vector<Mat>& imgs, bool isColor, float strength,
                               rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    Rpp32f *vignetteIntensityTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&vignetteIntensityTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Vignette intensity (0.0-1.0, where 1.0 is maximum darkening at corners)
        vignetteIntensityTensor[i] = strength;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_vignette(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                          &vignetteIntensityTensor[i],
                                          &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Vignette");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(vignetteIntensityTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "intensity=" << strength;
    printResult("RPP HIP Vignette", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_NonLinearBlend(const vector<Mat>& imgs, bool isColor, float strength,
                                     rppHandle_t handle, hipStream_t stream) {
    if (imgs.size() < 2) {
        cout << "NonLinearBlend requires at least 2 images. Skipping." << endl;
        return;
    }

    int num_images = (int)imgs.size() / 2;  // Process pairs of images
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs1(num_images);
    vector<Rpp8u*> d_inputs2(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors in pinned host memory
    Rpp32f *stdDevTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&stdDevTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        // Use pairs of input images
        const Mat& img1 = imgs[i * 2];
        const Mat& img2 = imgs[i * 2 + 1];

        RpptLayout layout = (isColor && img1.channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, img1.rows, img1.cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Standard deviation for Gaussian blending (controls blend smoothness)
        stdDevTensor[i] = strength;

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = img1.cols;
        roiTensor[i].xywhROI.roiHeight = img1.rows;

        // Allocate device memory
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs1[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_inputs2[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy first image data
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = img1.cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < img1.rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   img1.data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs1[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));

        // Copy second image data
        for (int row = 0; row < img2.rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   img2.data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs2[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));

        delete[] h_tempBuffer;
    }

    rppSetBatchSize(handle, 1);

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_non_linear_blend(d_inputs1[i], d_inputs2[i], &srcDescs[i],
                                                   d_outputs[i], &dstDescs[i],
                                                   &stdDevTensor[i],
                                                   &roiTensor[i], RpptRoiType::XYWH,
                                                   handle, RPP_HIP_BACKEND),
                            "NonLinearBlend");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs1[i]));
        CHECK_HIP_STATUS(hipFree(d_inputs2[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(stdDevTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "stdDev=" << strength;
    printResult("RPP HIP NonLinearBlend", num_images, isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Erase(const vector<Mat>& imgs, bool isColor, int numBoxes,
                            rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    RpptRoiLtrb *anchorBoxInfoTensor;
    Rpp8u *colorsTensor;
    Rpp32u *numBoxesTensor;
    RpptROI *roiTensor;

    int maxBoxesPerImage = numBoxes;
    CHECK_HIP_STATUS(hipHostMalloc(&anchorBoxInfoTensor, num_images * maxBoxesPerImage * sizeof(RpptRoiLtrb)));
    CHECK_HIP_STATUS(hipHostMalloc(&colorsTensor, num_images * maxBoxesPerImage * numChannels * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipHostMalloc(&numBoxesTensor, num_images * sizeof(Rpp32u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        numBoxesTensor[i] = numBoxes;

        // Define anchor boxes for erasing (simple pattern: divide image into regions)
        int width = imgs[i].cols;
        int height = imgs[i].rows;
        for (int box = 0; box < numBoxes; ++box) {
            int idx = i * maxBoxesPerImage + box;
            // Create boxes in different regions
            int boxWidth = width / (numBoxes + 1);
            int boxHeight = height / (numBoxes + 1);
            anchorBoxInfoTensor[idx].lt.x = (box + 1) * boxWidth / 2;
            anchorBoxInfoTensor[idx].lt.y = (box + 1) * boxHeight / 2;
            anchorBoxInfoTensor[idx].rb.x = anchorBoxInfoTensor[idx].lt.x + boxWidth;
            anchorBoxInfoTensor[idx].rb.y = anchorBoxInfoTensor[idx].lt.y + boxHeight;

            // Fill with black color
            for (int c = 0; c < numChannels; ++c) {
                colorsTensor[(idx * numChannels) + c] = 0;
            }
        }

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_erase(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                       &anchorBoxInfoTensor[i * maxBoxesPerImage],
                                       &colorsTensor[i * maxBoxesPerImage * numChannels],
                                       &numBoxesTensor[i],
                                       &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Erase");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(anchorBoxInfoTensor));
    CHECK_HIP_STATUS(hipHostFree(colorsTensor));
    CHECK_HIP_STATUS(hipHostFree(numBoxesTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "numBoxes=" << numBoxes;
    printResult("RPP HIP Erase", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_RandomErase(const vector<Mat>& imgs, bool isColor, int numBoxes, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);
    vector<Rpp8u*> d_noiseBuffers(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    RpptRoiLtrb *anchorBoxInfoTensor;
    RpptROI *roiTensor;

    Rpp32u maxEraseBoxes = numBoxes;  // Number of random erase regions
    CHECK_HIP_STATUS(hipHostMalloc(&anchorBoxInfoTensor, num_images * maxEraseBoxes * sizeof(RpptRoiLtrb)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Define random erase boxes
        int width = imgs[i].cols;
        int height = imgs[i].rows;

        for (Rpp32u box = 0; box < maxEraseBoxes; ++box) {
            int idx = i * maxEraseBoxes + box;
            // Create random-sized erase regions
            int boxWidth = width / (maxEraseBoxes + 2);
            int boxHeight = height / (maxEraseBoxes + 2);
            int xPos = (box * width) / (maxEraseBoxes + 1);
            int yPos = (box * height) / (maxEraseBoxes + 1);

            anchorBoxInfoTensor[idx].lt.x = std::max(0, xPos);
            anchorBoxInfoTensor[idx].lt.y = std::max(0, yPos);
            anchorBoxInfoTensor[idx].rb.x = std::min(width - 1, xPos + boxWidth);
            anchorBoxInfoTensor[idx].rb.y = std::min(height - 1, yPos + boxHeight);
        }

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Allocate noise buffer (filled with random values for erased regions)
        CHECK_HIP_STATUS(hipMalloc(&d_noiseBuffers[i], alignedBufferSize));

        // Fill noise buffer with random pattern (simple: use gradient for demo)
        Rpp8u* h_noiseBuffer = new Rpp8u[alignedBufferSize];
        for (size_t idx = 0; idx < alignedBufferSize; ++idx) {
            h_noiseBuffer[idx] = (idx % 256);  // Simple pattern
        }
        CHECK_HIP_STATUS(hipMemcpy(d_noiseBuffers[i], h_noiseBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_noiseBuffer;

        // Copy image data
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_random_erase(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                               &anchorBoxInfoTensor[i * maxEraseBoxes],
                                               d_noiseBuffers[i],
                                               &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "RandomErase");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
        CHECK_HIP_STATUS(hipFree(d_noiseBuffers[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(anchorBoxInfoTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "numBoxes=" << numBoxes;
    printResult("RPP HIP RandomErase", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_CoarseDropout(const vector<Mat>& imgs, bool isColor, Rpp32u maxBoxesPerImage,
                                    rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    RpptRoiLtrb *anchorBoxInfoTensor;
    Rpp32u *numBoxesTensor;
    RpptROI *roiTensor;

    CHECK_HIP_STATUS(hipHostMalloc(&anchorBoxInfoTensor, num_images * maxBoxesPerImage * sizeof(RpptRoiLtrb)));
    CHECK_HIP_STATUS(hipHostMalloc(&numBoxesTensor, num_images * sizeof(Rpp32u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    srand(DROPOUT_FIXED_SEED);
    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        int h = imgs[i].rows;
        int w = imgs[i].cols;

        // Random number of boxes (2 to maxBoxesPerImage) - matches HOST implementation
        Rpp32u numBoxes = 2 + (rand() % (maxBoxesPerImage - 1));
        numBoxesTensor[i] = numBoxes;

        for (Rpp32u b = 0; b < numBoxes; ++b) {
            int idx = i * maxBoxesPerImage + b;

            // Random box sizes (30 to 110 pixels) - matches HOST implementation
            int box_w = 30 + (rand() % 80);
            int box_h = 30 + (rand() % 80);
            int x = rand() % max(1, w - box_w);
            int y = rand() % max(1, h - box_h);

            anchorBoxInfoTensor[idx].lt.x = x;
            anchorBoxInfoTensor[idx].lt.y = y;
            anchorBoxInfoTensor[idx].rb.x = x + box_w;
            anchorBoxInfoTensor[idx].rb.y = y + box_h;
        }

        // Fill remaining boxes with zeros - matches HOST implementation
        for (Rpp32u b = numBoxes; b < maxBoxesPerImage; ++b) {
            int idx = i * maxBoxesPerImage + b;
            anchorBoxInfoTensor[idx].lt.x = 0;
            anchorBoxInfoTensor[idx].lt.y = 0;
            anchorBoxInfoTensor[idx].rb.x = 0;
            anchorBoxInfoTensor[idx].rb.y = 0;
        }

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_coarse_dropout(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                 &anchorBoxInfoTensor[i * maxBoxesPerImage],
                                                 &numBoxesTensor[i], maxBoxesPerImage,
                                                 &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "CoarseDropout");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(anchorBoxInfoTensor));
    CHECK_HIP_STATUS(hipHostFree(numBoxesTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "maxBoxes=" << maxBoxesPerImage;
    printResult("RPP HIP CoarseDropout", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_GridDropout(const vector<Mat>& imgs, bool isColor, int tileWidth, int tileHeight,
                                  rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Calculate boxes per image based on grid (matching BATCH implementation)
    int maxWidth = imgs[0].cols;
    int maxHeight = imgs[0].rows;
    int boxesInEachImage = (maxWidth / tileWidth) * (maxHeight / tileHeight);
    boxesInEachImage = std::max(boxesInEachImage, 16);

    Rpp32u maxHoleW = tileWidth / 2;
    Rpp32u maxHoleH = tileHeight / 2;

    // Allocate parameter tensors and ROIs in pinned host memory
    RpptRoiLtrb *anchorBoxInfoTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&anchorBoxInfoTensor, num_images * boxesInEachImage * sizeof(RpptRoiLtrb)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Create grid of boxes (matching BATCH implementation)
        int width = imgs[i].cols;
        int height = imgs[i].rows;

        int boxIdx = 0;
        for (int y = 0; y < height && boxIdx < boxesInEachImage; y += tileHeight) {
            for (int x = 0; x < width && boxIdx < boxesInEachImage; x += tileWidth) {
                int idx = i * boxesInEachImage + boxIdx;
                anchorBoxInfoTensor[idx].lt.x = x;
                anchorBoxInfoTensor[idx].lt.y = y;
                anchorBoxInfoTensor[idx].rb.x = std::min(x + tileWidth, width);
                anchorBoxInfoTensor[idx].rb.y = std::min(y + tileHeight, height);
                boxIdx++;
            }
        }

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_grid_dropout(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                               &anchorBoxInfoTensor[i * boxesInEachImage],
                                               boxesInEachImage, maxHoleW, maxHoleH,
                                               &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "GridDropout");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(anchorBoxInfoTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "tileWidth=" << tileWidth << ", tileHeight=" << tileHeight;
    printResult("RPP HIP GridDropout", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Gridmask(const vector<Mat>& imgs, bool isColor, int tileWidth, float ratio,
                               rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate ROIs in pinned host memory
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    // Gridmask parameters (same for all images in batch)
    Rpp32u gridTileWidth = (Rpp32u)tileWidth;
    Rpp32f gridRatio = ratio;  // Ratio of grid hole to tile (0.0-1.0)
    Rpp32f gridAngle = 0.0f;   // Rotation angle in degrees
    RpptUintVector2D translateVector;
    translateVector.x = 0;
    translateVector.y = 0;

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                       imgs[i].data + row * elementsPerRow,
                       elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    // CRITICAL FIX: Clear any pending HIP errors from previous operations before calling rppt_gridmask
    // The HIP_CHECK_LAUNCH_RETURN() macro in gridmask.cpp picks up stale errors from earlier HIP calls
    // Synchronize to ensure all previous operations complete, then clear the error state
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    (void)hipGetLastError();

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_gridmask(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                          gridTileWidth, gridRatio, gridAngle, translateVector,
                                          &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "Gridmask");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "tileWidth=" << tileWidth << ", ratio=" << ratio;
    printResult("RPP HIP Gridmask", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_ChannelDropout(const vector<Mat>& imgs, bool isColor, float dropoutProb,
                                     rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    // ChannelDropout requires RGB images (3 channels)
    if (numChannels != 3) {
        cerr << "Warning: ChannelDropout requires RGB images (3 channels). Skipping." << endl;
        return;
    }

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    Rpp8u *dropoutTensor;  // 3 channels: 0 or 1 for each channel (0=drop, 1=keep)
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&dropoutTensor, num_images * 3 * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = RpptLayout::NHWC;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Simple dropout pattern: randomly drop one channel based on probability
        // For demo: drop R channel if prob > 0.66, G if 0.33-0.66, B if < 0.33
        if (dropoutProb > 0.66f) {
            dropoutTensor[i * 3 + 0] = 0;  // Drop R
            dropoutTensor[i * 3 + 1] = 1;  // Keep G
            dropoutTensor[i * 3 + 2] = 1;  // Keep B
        } else if (dropoutProb > 0.33f) {
            dropoutTensor[i * 3 + 0] = 1;  // Keep R
            dropoutTensor[i * 3 + 1] = 0;  // Drop G
            dropoutTensor[i * 3 + 2] = 1;  // Keep B
        } else {
            dropoutTensor[i * 3 + 0] = 1;  // Keep R
            dropoutTensor[i * 3 + 1] = 1;  // Keep G
            dropoutTensor[i * 3 + 2] = 0;  // Drop B
        }

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_channel_dropout(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                  &dropoutTensor[i * 3],
                                                  &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "ChannelDropout");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(dropoutTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "dropoutProb=" << dropoutProb;
    printResult("RPP HIP ChannelDropout", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_CutoutDropout(const vector<Mat>& imgs, bool isColor, Rpp32u numBoxes,
                                    rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    RpptRoiLtrb *anchorBoxInfoTensor;
    Rpp8u *colorsTensor;
    Rpp32u *numBoxesTensor;
    RpptROI *roiTensor;

    Rpp32u maxBoxesPerImage = numBoxes;
    CHECK_HIP_STATUS(hipHostMalloc(&anchorBoxInfoTensor, num_images * maxBoxesPerImage * sizeof(RpptRoiLtrb)));
    CHECK_HIP_STATUS(hipHostMalloc(&colorsTensor, num_images * maxBoxesPerImage * numChannels * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipHostMalloc(&numBoxesTensor, num_images * sizeof(Rpp32u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        numBoxesTensor[i] = numBoxes;

        // Define cutout boxes (random regions to drop out)
        int width = imgs[i].cols;
        int height = imgs[i].rows;
        for (Rpp32u box = 0; box < numBoxes; ++box) {
            int idx = i * maxBoxesPerImage + box;
            // Create random-sized boxes in different positions
            int cutoutSize = std::min(width, height) / (numBoxes + 2);
            int xPos = (box * width) / (numBoxes + 1);
            int yPos = (box * height) / (numBoxes + 1);

            anchorBoxInfoTensor[idx].lt.x = xPos;
            anchorBoxInfoTensor[idx].lt.y = yPos;
            anchorBoxInfoTensor[idx].rb.x = std::min(xPos + cutoutSize, width - 1);
            anchorBoxInfoTensor[idx].rb.y = std::min(yPos + cutoutSize, height - 1);

            // Fill with gray color (128)
            for (int c = 0; c < numChannels; ++c) {
                colorsTensor[(idx * numChannels) + c] = 128;
            }
        }

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_cutout_dropout(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                 &anchorBoxInfoTensor[i * maxBoxesPerImage],
                                                 &colorsTensor[i * maxBoxesPerImage * numChannels],
                                                 &numBoxesTensor[i],
                                                 &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "CutoutDropout");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(anchorBoxInfoTensor));
    CHECK_HIP_STATUS(hipHostFree(colorsTensor));
    CHECK_HIP_STATUS(hipHostFree(numBoxesTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "numBoxes=" << numBoxes;
    printResult("RPP HIP CutoutDropout", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_JpegCompressionDistortion(const vector<Mat>& imgs, bool isColor, Rpp32s quality,
                                                rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors and ROIs in pinned host memory
    Rpp32s *qualityTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&qualityTensor, num_images * sizeof(Rpp32s)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = RpptLayout::NHWC;  // JPEG distortion works with NHWC layout
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        qualityTensor[i] = quality;  // JPEG quality factor (1-100)
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();  // Zero-initialized
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_jpeg_compression_distortion(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                             &qualityTensor[i],
                                                             &roiTensor[i], RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                            "JpegCompressionDistortion");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(qualityTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "quality=" << quality;
    printResult("RPP HIP JpegCompressionDistortion", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Copy(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_copy(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                      handle, RPP_HIP_BACKEND),
                            "Copy");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }

    ostringstream params;
    params << "layout=" << (isColor ? "NHWC" : "NCHW");
    printResult("RPP HIP Copy", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_ChannelPermute(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    if (!isColor) {
        cout << "ChannelPermute requires RGB images (3 channels). Skipping grayscale." << endl;
        return;
    }

    int num_images = (int)imgs.size();
    int numChannels = 3;  // RGB only

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate permutation tensor in pinned host memory
    Rpp32u *permutationTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&permutationTensor, num_images * 3 * sizeof(Rpp32u)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = RpptLayout::NHWC;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Set channel permutation (BGR to RGB: 2, 1, 0)
        permutationTensor[i * 3 + 0] = 2;  // R from B
        permutationTensor[i * 3 + 1] = 1;  // G from G
        permutationTensor[i * 3 + 2] = 0;  // B from R

        // Calculate buffer size based on aligned descriptor dimensions
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_channel_permute(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                                  &permutationTensor[i * 3],
                                                  handle, RPP_HIP_BACKEND),
                            "ChannelPermute");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(permutationTensor));

    ostringstream params;
    params << "permutation=BGR->RGB";
    printResult("RPP HIP ChannelPermute", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Slice(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptGenericDesc> srcDescs(num_images);
    vector<RpptGenericDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate anchor and shape tensors in pinned host memory
    Rpp32s *anchorTensor;
    Rpp32s *shapeTensor;
    Rpp32u *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&anchorTensor, num_images * 4 * sizeof(Rpp32s)));  // 4D tensor [N,H,W,C]
    CHECK_HIP_STATUS(hipHostMalloc(&shapeTensor, num_images * 4 * sizeof(Rpp32s)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(Rpp32u)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

        // Setup generic descriptor (4D tensor)
        srcDescs[i].numDims = 4;
        srcDescs[i].offsetInBytes = 0;
        srcDescs[i].dataType = RpptDataType::U8;
        srcDescs[i].layout = layout;

        // Set dimensions [N, H, W, C] or [N, C, H, W] based on layout
        int height = imgs[i].rows;
        int width = imgs[i].cols;
        int alignedWidth = ((width + 7) / 8) * 8;  // Align to 8

        if (layout == RpptLayout::NHWC) {
            srcDescs[i].dims[0] = 1;  // N
            srcDescs[i].dims[1] = height;  // H
            srcDescs[i].dims[2] = alignedWidth;  // W (aligned)
            srcDescs[i].dims[3] = numChannels;  // C
        } else {  // NCHW
            srcDescs[i].dims[0] = 1;  // N
            srcDescs[i].dims[1] = numChannels;  // C
            srcDescs[i].dims[2] = height;  // H
            srcDescs[i].dims[3] = alignedWidth;  // W (aligned)
        }

        // Calculate strides
        srcDescs[i].strides[3] = 1;
        srcDescs[i].strides[2] = srcDescs[i].dims[3];
        srcDescs[i].strides[1] = srcDescs[i].strides[2] * srcDescs[i].dims[2];
        srcDescs[i].strides[0] = srcDescs[i].strides[1] * srcDescs[i].dims[1];

        // Slice center region (50% of image)
        int sliceHeight = height / 2;
        int sliceWidth = width / 2;
        int startY = height / 4;
        int startX = width / 4;

        // Set anchor point [N=0, H=startY, W=startX, C=0]
        if (layout == RpptLayout::NHWC) {
            anchorTensor[i * 4 + 0] = 0;  // N
            anchorTensor[i * 4 + 1] = startY;  // H
            anchorTensor[i * 4 + 2] = startX;  // W
            anchorTensor[i * 4 + 3] = 0;  // C

            // Set shape [N=1, H=sliceHeight, W=sliceWidth, C=numChannels]
            shapeTensor[i * 4 + 0] = 1;  // N
            shapeTensor[i * 4 + 1] = sliceHeight;  // H
            shapeTensor[i * 4 + 2] = sliceWidth;  // W
            shapeTensor[i * 4 + 3] = numChannels;  // C

            // Setup destination descriptor
            dstDescs[i] = srcDescs[i];
            dstDescs[i].dims[1] = sliceHeight;
            dstDescs[i].dims[2] = sliceWidth;
        } else {  // NCHW
            anchorTensor[i * 4 + 0] = 0;  // N
            anchorTensor[i * 4 + 1] = 0;  // C
            anchorTensor[i * 4 + 2] = startY;  // H
            anchorTensor[i * 4 + 3] = startX;  // W

            // Set shape [N=1, C=numChannels, H=sliceHeight, W=sliceWidth]
            shapeTensor[i * 4 + 0] = 1;  // N
            shapeTensor[i * 4 + 1] = numChannels;  // C
            shapeTensor[i * 4 + 2] = sliceHeight;  // H
            shapeTensor[i * 4 + 3] = sliceWidth;  // W

            // Setup destination descriptor
            dstDescs[i] = srcDescs[i];
            dstDescs[i].dims[2] = sliceHeight;
            dstDescs[i].dims[3] = sliceWidth;
        }

        roiTensor[i] = 1;  // Single image

        // Calculate buffer sizes
        size_t srcBufferSize = srcDescs[i].strides[0] * srcDescs[i].dims[0] * sizeof(Rpp8u);
        size_t dstBufferSize = dstDescs[i].strides[0] * dstDescs[i].dims[0] * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], srcBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], dstBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[srcBufferSize]();
        int elementsPerRow = width * numChannels;

        for (int row = 0; row < height; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, srcBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_slice(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                       anchorTensor + i * 4, shapeTensor + i * 4,
                                       nullptr, false, &roiTensor[i],
                                       handle, RPP_HIP_BACKEND),
                            "Slice");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(anchorTensor));
    CHECK_HIP_STATUS(hipHostFree(shapeTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "slice=center_50%";
    printResult("RPP HIP Slice", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Transpose(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    // Allocate descriptors in pinned memory so they're GPU-accessible
    RpptGenericDesc *srcDescs, *dstDescs;
    CHECK_HIP_STATUS(hipHostMalloc(&srcDescs, num_images * sizeof(RpptGenericDesc)));
    CHECK_HIP_STATUS(hipHostMalloc(&dstDescs, num_images * sizeof(RpptGenericDesc)));

    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate permutation tensor and ROI tensor in pinned host memory
    // The rppt_transpose HIP kernel processes 3D sub-tensors (HWC) for each image,
    // excluding the batch dimension (N). Therefore:
    // 1. permTensor must have (numDims-1) = 3 elements per image
    // 2. roiTensor must have (numDims-1)*2 = 6 values per image
    Rpp32u *permTensor;
    Rpp32u *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&permTensor, num_images * 3 * sizeof(Rpp32u)));  // 3D permutation
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * 6 * sizeof(Rpp32u)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

        // Setup generic descriptor (4D tensor)
        srcDescs[i].numDims = 4;
        srcDescs[i].offsetInBytes = 0;
        srcDescs[i].dataType = RpptDataType::U8;
        srcDescs[i].layout = layout;

        // Set dimensions [N, H, W, C] or [N, C, H, W] based on layout
        int height = imgs[i].rows;
        int width = imgs[i].cols;
        int alignedWidth = ((width + 7) / 8) * 8;  // Align to 8

        if (layout == RpptLayout::NHWC) {
            // Source: NHWC [N, H, W, C]
            srcDescs[i].dims[0] = 1;  // N
            srcDescs[i].dims[1] = height;  // H
            srcDescs[i].dims[2] = width;  // W (use original width, not aligned)
            srcDescs[i].dims[3] = numChannels;  // C

            // Transpose H and W in the 3D sub-tensor (HWC): HWC -> WHC
            // Permutation for the 3D HWC sub-tensor (excluding N dimension)
            // [1, 0, 2] means: W goes to dim0, H goes to dim1, C stays at dim2
            permTensor[i * 3 + 0] = 1;  // output dim0 <- input dim1 (W)
            permTensor[i * 3 + 1] = 0;  // output dim1 <- input dim0 (H)
            permTensor[i * 3 + 2] = 2;  // output dim2 <- input dim2 (C)

            // Destination: NWHC [N, W, H, C] - swapped H and W
            dstDescs[i].numDims = 4;
            dstDescs[i].offsetInBytes = 0;
            dstDescs[i].dataType = RpptDataType::U8;
            dstDescs[i].layout = layout;
            dstDescs[i].dims[0] = 1;  // N
            dstDescs[i].dims[1] = width;  // W (now in dim1)
            dstDescs[i].dims[2] = height;  // H (now in dim2)
            dstDescs[i].dims[3] = numChannels;  // C
        } else {  // NCHW
            // Source: NCHW [N, C, H, W]
            srcDescs[i].dims[0] = 1;  // N
            srcDescs[i].dims[1] = numChannels;  // C
            srcDescs[i].dims[2] = height;  // H
            srcDescs[i].dims[3] = width;  // W (use original width, not aligned)

            // Transpose H and W in the 3D sub-tensor (CHW): CHW -> CWH
            // Permutation for the 3D CHW sub-tensor (excluding N dimension)
            // [0, 2, 1] means: C stays at dim0, W goes to dim1, H goes to dim2
            permTensor[i * 3 + 0] = 0;  // output dim0 <- input dim0 (C)
            permTensor[i * 3 + 1] = 2;  // output dim1 <- input dim2 (W)
            permTensor[i * 3 + 2] = 1;  // output dim2 <- input dim1 (H)

            // Destination: NCWH [N, C, W, H] - swapped H and W
            dstDescs[i].numDims = 4;
            dstDescs[i].offsetInBytes = 0;
            dstDescs[i].dataType = RpptDataType::U8;
            dstDescs[i].layout = layout;
            dstDescs[i].dims[0] = 1;  // N
            dstDescs[i].dims[1] = numChannels;  // C
            dstDescs[i].dims[2] = width;  // W (now in dim2)
            dstDescs[i].dims[3] = height;  // H (now in dim3)
        }

        // ROI tensor: (numDims-1)*2 = 6 values per image
        // Format: [dim0_start, dim1_start, dim2_start, dim0_size, dim1_size, dim2_size]
        int roiIdx = i * 6;
        if (layout == RpptLayout::NHWC) {
            // For HWC sub-tensor
            roiTensor[roiIdx + 0] = 0;         // h_start
            roiTensor[roiIdx + 1] = 0;         // w_start
            roiTensor[roiIdx + 2] = 0;         // c_start
            roiTensor[roiIdx + 3] = height;    // h_size
            roiTensor[roiIdx + 4] = width;     // w_size
            roiTensor[roiIdx + 5] = numChannels;  // c_size
        } else {  // NCHW
            // For CHW sub-tensor
            roiTensor[roiIdx + 0] = 0;         // c_start
            roiTensor[roiIdx + 1] = 0;         // h_start
            roiTensor[roiIdx + 2] = 0;         // w_start
            roiTensor[roiIdx + 3] = numChannels;  // c_size
            roiTensor[roiIdx + 4] = height;    // h_size
            roiTensor[roiIdx + 5] = width;     // w_size
        }

        // Calculate strides for source (with alignment)
        int alignedHeight = ((height + 7) / 8) * 8;  // Align height to 8 as well
        if (layout == RpptLayout::NHWC) {
            srcDescs[i].strides[3] = 1;
            srcDescs[i].strides[2] = numChannels;
            srcDescs[i].strides[1] = alignedWidth * numChannels;  // Use aligned width for stride
            srcDescs[i].strides[0] = srcDescs[i].strides[1] * height;
        } else {  // NCHW
            srcDescs[i].strides[3] = 1;
            srcDescs[i].strides[2] = alignedWidth;  // Use aligned width for stride
            srcDescs[i].strides[1] = srcDescs[i].strides[2] * height;
            srcDescs[i].strides[0] = srcDescs[i].strides[1] * numChannels;
        }

        // Calculate destination strides (H and W are swapped)
        if (layout == RpptLayout::NHWC) {
            // Destination NWHC: strides for [N, W, H, C]
            dstDescs[i].strides[3] = 1;  // C stride
            dstDescs[i].strides[2] = numChannels;  // H stride
            dstDescs[i].strides[1] = alignedHeight * numChannels;  // W stride (use aligned height since H and W swapped)
            dstDescs[i].strides[0] = dstDescs[i].strides[1] * width;  // N stride
        } else {  // NCHW
            // Destination NCWH: strides for [N, C, W, H]
            dstDescs[i].strides[3] = 1;  // H stride
            dstDescs[i].strides[2] = alignedHeight;  // W stride (use aligned height since H and W swapped)
            dstDescs[i].strides[1] = dstDescs[i].strides[2] * width;  // C stride
            dstDescs[i].strides[0] = dstDescs[i].strides[1] * numChannels;  // N stride
        }

        // Calculate buffer sizes
        size_t srcBufferSize = srcDescs[i].strides[0] * sizeof(Rpp8u);
        size_t dstBufferSize = dstDescs[i].strides[0] * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], srcBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], dstBufferSize));

        // Copy image data line by line to account for width alignment
        Rpp8u* h_tempBuffer = new Rpp8u[srcBufferSize]();
        int elementsPerRow = width * numChannels;

        if (layout == RpptLayout::NHWC) {
            for (int row = 0; row < height; ++row) {
                memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                       imgs[i].data + row * elementsPerRow,
                       elementsPerRow * sizeof(Rpp8u));
            }
        } else {  // NCHW
            for (int c = 0; c < numChannels; ++c) {
                for (int row = 0; row < height; ++row) {
                    for (int col = 0; col < width; ++col) {
                        int srcIdx = row * elementsPerRow + col * numChannels + c;
                        int dstIdx = c * (height * alignedWidth) + row * alignedWidth + col;
                        h_tempBuffer[dstIdx] = imgs[i].data[srcIdx];
                    }
                }
            }
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, srcBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_transpose(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                           permTensor + i * 3, roiTensor + i * 6,
                                           handle, RPP_HIP_BACKEND),
                            "Transpose");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(srcDescs));
    CHECK_HIP_STATUS(hipHostFree(dstDescs));
    CHECK_HIP_STATUS(hipHostFree(permTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "permutation=0,2,1,3";
    printResult("RPP HIP Transpose", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Normalize(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptGenericDesc> srcDescs(num_images);
    vector<RpptGenericDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate mean, stddev, and ROI tensors in pinned host memory
    Rpp32f *meanTensor;
    Rpp32f *stdDevTensor;
    Rpp32u *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&meanTensor, num_images * numChannels * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&stdDevTensor, num_images * numChannels * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(Rpp32u)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

        // Setup generic descriptor (4D tensor)
        srcDescs[i].numDims = 4;
        srcDescs[i].offsetInBytes = 0;
        srcDescs[i].dataType = RpptDataType::U8;
        srcDescs[i].layout = layout;

        int height = imgs[i].rows;
        int width = imgs[i].cols;
        int alignedWidth = ((width + 7) / 8) * 8;

        if (layout == RpptLayout::NHWC) {
            srcDescs[i].dims[0] = 1;
            srcDescs[i].dims[1] = height;
            srcDescs[i].dims[2] = width;
            srcDescs[i].dims[3] = numChannels;
            srcDescs[i].strides[3] = 1;
            srcDescs[i].strides[2] = numChannels;
            srcDescs[i].strides[1] = alignedWidth * numChannels;
            srcDescs[i].strides[0] = srcDescs[i].strides[1] * height;
        } else {  // NCHW
            srcDescs[i].dims[0] = 1;
            srcDescs[i].dims[1] = numChannels;
            srcDescs[i].dims[2] = height;
            srcDescs[i].dims[3] = width;
            srcDescs[i].strides[3] = 1;
            srcDescs[i].strides[2] = alignedWidth;
            srcDescs[i].strides[1] = srcDescs[i].strides[2] * height;
            srcDescs[i].strides[0] = srcDescs[i].strides[1] * numChannels;
        }

        dstDescs[i] = srcDescs[i];
        roiTensor[i] = 1;

        // Mean and stddev will be auto-computed (computeMeanStddev=3)
        // Initialize to 0 (will be filled by the function)
        for (int c = 0; c < numChannels; ++c) {
            meanTensor[i * numChannels + c] = 0.0f;
            stdDevTensor[i * numChannels + c] = 0.0f;
        }

        // Allocate device memory
        size_t bufferSize = srcDescs[i].strides[0] * sizeof(Rpp8u);
        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], bufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], bufferSize));

        // Copy image data
        Rpp8u* h_tempBuffer = new Rpp8u[bufferSize]();
        int elementsPerRow = width * numChannels;

        if (layout == RpptLayout::NHWC) {
            for (int row = 0; row < height; ++row) {
                memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                       imgs[i].data + row * elementsPerRow,
                       elementsPerRow * sizeof(Rpp8u));
            }
        } else {  // NCHW conversion
            for (int c = 0; c < numChannels; ++c) {
                for (int row = 0; row < height; ++row) {
                    for (int col = 0; col < width; ++col) {
                        int srcIdx = row * elementsPerRow + col * numChannels + c;
                        int dstIdx = c * (height * alignedWidth) + row * alignedWidth + col;
                        h_tempBuffer[dstIdx] = imgs[i].data[srcIdx];
                    }
                }
            }
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, bufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    // Scale and shift for normalization
    Rpp32f scale = 1.0f;
    Rpp32f shift = 0.0f;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            // axisMask = 0x7 (normalize across HWC dimensions)
            // computeMeanStddev = 3 (auto-compute both mean and stddev)
            // scale = 1.0, shift = 0.0
            CHECK_RPP_STATUS(rppt_normalize(d_inputs[i], &srcDescs[i], d_outputs[i], &dstDescs[i],
                                           0x7, meanTensor + i * numChannels,
                                           stdDevTensor + i * numChannels,
                                           3, scale, shift, &roiTensor[i],
                                           handle, RPP_HIP_BACKEND),
                            "Normalize");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(meanTensor));
    CHECK_HIP_STATUS(hipHostFree(stdDevTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "mode=auto_compute, scale=" << scale << ", shift=" << shift;
    printResult("RPP HIP Normalize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_FusedMultiplyAddScalar(const vector<Mat>& imgs, bool isColor, float mulVal, float addVal,
                                             rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // rppt_fused_multiply_add_scalar ONLY supports F32 datatype
    // Use 5D generic descriptor (NDHWC or NCDHW) with depth=1 for 2D images
    RpptGenericDesc genericDesc;
    genericDesc.numDims = 5;
    genericDesc.offsetInBytes = 0;
    genericDesc.dataType = RpptDataType::F32;

    vector<Rpp32f*> d_inputs(num_images);
    vector<Rpp32f*> d_outputs(num_images);

    // Allocate mul, add, and ROI tensors in pinned host memory
    Rpp32f *mulTensor;
    Rpp32f *addTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&mulTensor, num_images * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&addTensor, num_images * sizeof(Rpp32f)));

    RpptROI3DPtr roiGenericPtr;
    CHECK_HIP_STATUS(hipHostMalloc(&roiGenericPtr, num_images * sizeof(RpptROI3D)));

    int imageSize = height * width * numChannels;

    if (isColor) {
        // NDHWC layout
        genericDesc.layout = RpptLayout::NDHWC;
        genericDesc.dims[0] = 1;  // Process one image at a time
        genericDesc.dims[1] = 1;  // Depth = 1 for 2D images
        genericDesc.dims[2] = height;
        genericDesc.dims[3] = width;
        genericDesc.dims[4] = numChannels;
        genericDesc.strides[4] = 1;
        genericDesc.strides[3] = numChannels;
        genericDesc.strides[2] = width * numChannels;
        genericDesc.strides[1] = height * width * numChannels;
        genericDesc.strides[0] = height * width * numChannels;
    } else {
        // NCDHW layout for grayscale
        genericDesc.layout = RpptLayout::NCDHW;
        genericDesc.dims[0] = 1;  // Process one image at a time
        genericDesc.dims[1] = numChannels;
        genericDesc.dims[2] = 1;  // Depth = 1 for 2D images
        genericDesc.dims[3] = height;
        genericDesc.dims[4] = width;
        genericDesc.strides[4] = 1;
        genericDesc.strides[3] = width;
        genericDesc.strides[2] = height * width;
        genericDesc.strides[1] = height * width;
        genericDesc.strides[0] = numChannels * height * width;
    }

    for (int i = 0; i < num_images; ++i) {
        // Convert U8 to F32
        Mat imgF32;
        imgs[i].convertTo(imgF32, CV_32F);

        // Allocate device memory for F32
        size_t bufferSize = imageSize * sizeof(Rpp32f);
        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], bufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], bufferSize));

        // Copy F32 data to device
        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], imgF32.data, bufferSize, hipMemcpyHostToDevice));

        // Set multiply and add values
        mulTensor[i] = mulVal;
        addTensor[i] = addVal;

        // Set ROI
        roiGenericPtr[i].xyzwhdROI.xyz.x = 0;
        roiGenericPtr[i].xyzwhdROI.xyz.y = 0;
        roiGenericPtr[i].xyzwhdROI.xyz.z = 0;
        roiGenericPtr[i].xyzwhdROI.roiWidth = width;
        roiGenericPtr[i].xyzwhdROI.roiHeight = height;
        roiGenericPtr[i].xyzwhdROI.roiDepth = 1;
    }

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            // Output = (Input * mulTensor) + addTensor
            CHECK_RPP_STATUS(rppt_fused_multiply_add_scalar(d_inputs[i], &genericDesc,
                                                            d_outputs[i], &genericDesc,
                                                            &mulTensor[i], &addTensor[i],
                                                            &roiGenericPtr[i], RpptRoi3DType::XYZWHD,
                                                            handle, RPP_HIP_BACKEND),
                            "FusedMultiplyAddScalar");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(mulTensor));
    CHECK_HIP_STATUS(hipHostFree(addTensor));
    CHECK_HIP_STATUS(hipHostFree(roiGenericPtr));

    ostringstream params;
    params << "mul=" << mulVal << ", add=" << addVal;
    printResult("RPP HIP FusedMultiplyAddScalar", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_CropAndPatch(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    if (imgs.size() < 2) {
        cout << "CropAndPatch requires at least 2 images. Skipping." << endl;
        return;
    }

    int num_images = (int)imgs.size();  // Process all images (same as other implementations)
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs1(num_images);
    vector<Rpp8u*> d_inputs2(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate ROI tensors in pinned host memory
    RpptROI *roiTensorDst;
    RpptROI *cropRoiTensor;
    RpptROI *patchRoiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensorDst, num_images * sizeof(RpptROI)));
    CHECK_HIP_STATUS(hipHostMalloc(&cropRoiTensor, num_images * sizeof(RpptROI)));
    CHECK_HIP_STATUS(hipHostMalloc(&patchRoiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        // Crop from img[i], patch into img[i+1] (same as other implementations)
        const Mat& img1 = imgs[i];
        const Mat& img2 = (i < num_images - 1) ? imgs[i + 1] : imgs[i];

        RpptLayout layout = (isColor && img1.channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, img1.rows, img1.cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);
        dstDescs[i] = srcDescs[i];

        // Crop region from image1 (center quarter - 50% size at 25% offset)
        int cropWidth = img1.cols / 2;
        int cropHeight = img1.rows / 2;
        int cropX = img1.cols / 4;
        int cropY = img1.rows / 4;

        cropRoiTensor[i].xywhROI.xy.x = cropX;
        cropRoiTensor[i].xywhROI.xy.y = cropY;
        cropRoiTensor[i].xywhROI.roiWidth = cropWidth;
        cropRoiTensor[i].xywhROI.roiHeight = cropHeight;

        // Patch location in image2 (center quarter - same as crop, same as other implementations)
        patchRoiTensor[i].xywhROI.xy.x = cropX;
        patchRoiTensor[i].xywhROI.xy.y = cropY;
        patchRoiTensor[i].xywhROI.roiWidth = cropWidth;
        patchRoiTensor[i].xywhROI.roiHeight = cropHeight;

        // Full destination ROI
        roiTensorDst[i].xywhROI.xy.x = 0;
        roiTensorDst[i].xywhROI.xy.y = 0;
        roiTensorDst[i].xywhROI.roiWidth = img2.cols;
        roiTensorDst[i].xywhROI.roiHeight = img2.rows;

        // Allocate device memory
        size_t alignedBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs1[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_inputs2[i], alignedBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], alignedBufferSize));

        // Copy first image data (source for crop)
        Rpp8u* h_tempBuffer = new Rpp8u[alignedBufferSize]();
        int originalWidth = img1.cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < img1.rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   img1.data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs1[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));

        // Copy second image data (destination for patch)
        for (int row = 0; row < img2.rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   img2.data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
        CHECK_HIP_STATUS(hipMemcpy(d_inputs2[i], h_tempBuffer, alignedBufferSize, hipMemcpyHostToDevice));

        delete[] h_tempBuffer;
    }

    rppSetBatchSize(handle, 1);

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images - 1; ++i) {  // Process num_images-1 pairs (same as others)
            CHECK_RPP_STATUS(rppt_crop_and_patch(d_inputs1[i], d_inputs2[i], &srcDescs[i],
                                                 d_outputs[i], &dstDescs[i],
                                                 &roiTensorDst[i],
                                                 &cropRoiTensor[i], &patchRoiTensor[i],
                                                 RpptRoiType::XYWH,
                                                 handle, RPP_HIP_BACKEND),
                            "CropAndPatch");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs1[i]));
        CHECK_HIP_STATUS(hipFree(d_inputs2[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(roiTensorDst));
    CHECK_HIP_STATUS(hipHostFree(cropRoiTensor));
    CHECK_HIP_STATUS(hipHostFree(patchRoiTensor));

    ostringstream params;
    params << "crop=center_quarter, patch=center_quarter";
    printResult("RPP HIP CropAndPatch", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_CropMirrorNormalize(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors in pinned host memory
    Rpp32f *offsetTensor;
    Rpp32f *multiplierTensor;
    Rpp32u *mirrorTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&offsetTensor, num_images * numChannels * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&multiplierTensor, num_images * numChannels * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&mirrorTensor, num_images * sizeof(Rpp32u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * sizeof(RpptROI)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);

        // Crop to center half (same as other implementations)
        int cropWidth = imgs[i].cols / 2;
        int cropHeight = imgs[i].rows / 2;
        int cropX = (imgs[i].cols - cropWidth) / 2;
        int cropY = (imgs[i].rows - cropHeight) / 2;

        // Destination descriptor (cropped size)
        set_descriptor_dims_and_strides_local(&dstDescs[i], 1, cropHeight, cropWidth, numChannels, 0);
        dstDescs[i].layout = layout;
        dstDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&dstDescs[i]);

        // Set ROI (crop region)
        roiTensor[i].xywhROI.xy.x = cropX;
        roiTensor[i].xywhROI.xy.y = cropY;
        roiTensor[i].xywhROI.roiWidth = cropWidth;
        roiTensor[i].xywhROI.roiHeight = cropHeight;

        // Set normalization parameters (same as other implementations)
        // offset = -mean/stddev, multiplier = 1/stddev
        if (isColor) {
            Rpp32f mean[3] = {60.0f, 80.0f, 100.0f};
            Rpp32f stdDev[3] = {0.9f, 0.9f, 0.9f};
            for (int c = 0; c < 3; ++c) {
                offsetTensor[i * 3 + c] = -mean[c] / stdDev[c];
                multiplierTensor[i * 3 + c] = 1.0f / stdDev[c];
            }
        } else {
            offsetTensor[i] = -100.0f / 0.9f;
            multiplierTensor[i] = 1.0f / 0.9f;
        }

        // Mirror: horizontal flip (same as other implementations)
        mirrorTensor[i] = 1;

        // Allocate device memory
        size_t srcBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);
        size_t dstBufferSize = dstDescs[i].n * dstDescs[i].h * dstDescs[i].w * dstDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], srcBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], dstBufferSize));

        // Copy image data line by line
        Rpp8u* h_tempBuffer = new Rpp8u[srcBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, srcBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    rppSetBatchSize(handle, 1);

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_crop_mirror_normalize(d_inputs[i], &srcDescs[i],
                                                        d_outputs[i], &dstDescs[i],
                                                        offsetTensor + i * numChannels,
                                                        multiplierTensor + i * numChannels,
                                                        &mirrorTensor[i],
                                                        &roiTensor[i], RpptRoiType::XYWH,
                                                        handle, RPP_HIP_BACKEND),
                            "CropMirrorNormalize");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(offsetTensor));
    CHECK_HIP_STATUS(hipHostFree(multiplierTensor));
    CHECK_HIP_STATUS(hipHostFree(mirrorTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "crop=half, mirror=horizontal, mean=60/80/100, stddev=0.9";
    printResult("RPP HIP CropMirrorNormalize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_ResizeMirrorNormalize(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors in pinned host memory
    Rpp32f *meanTensor;
    Rpp32f *stdDevTensor;
    Rpp32u *mirrorTensor;
    RpptROI *roiTensor;
    RpptImagePatch *dstImgSizes;
    CHECK_HIP_STATUS(hipHostMalloc(&meanTensor, num_images * numChannels * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&stdDevTensor, num_images * numChannels * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&mirrorTensor, num_images * sizeof(Rpp32u)));
    // Allocate 256 ROI elements per image as workaround for kernel bug (launches 256 threads without bounds check)
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * 256 * sizeof(RpptROI)));
    CHECK_HIP_STATUS(hipHostMalloc(&dstImgSizes, num_images * sizeof(RpptImagePatch)));

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);

        // Resize to half (same as other implementations)
        int targetWidth = imgs[i].cols / 2;
        int targetHeight = imgs[i].rows / 2;

        // Destination descriptor (resized to half)
        set_descriptor_dims_and_strides_local(&dstDescs[i], 1, targetHeight, targetWidth, numChannels, 0);
        dstDescs[i].layout = layout;
        dstDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&dstDescs[i]);

        // Each image gets 256 ROI slots (only first is used, rest are buffer for buggy kernel)
        roiTensor[i * 256].xywhROI.xy.x = 0;
        roiTensor[i * 256].xywhROI.xy.y = 0;
        roiTensor[i * 256].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i * 256].xywhROI.roiHeight = imgs[i].rows;

        // Destination image size
        dstImgSizes[i].width = targetWidth;
        dstImgSizes[i].height = targetHeight;

        // Set normalization parameters (same as other implementations)
        if (isColor) {
            meanTensor[i * 3 + 0] = 60.0f;
            meanTensor[i * 3 + 1] = 80.0f;
            meanTensor[i * 3 + 2] = 100.0f;
            stdDevTensor[i * 3 + 0] = 1.0f;
            stdDevTensor[i * 3 + 1] = 1.0f;
            stdDevTensor[i * 3 + 2] = 1.0f;
        } else {
            meanTensor[i] = 100.0f;
            stdDevTensor[i] = 1.0f;
        }

        // Mirror: horizontal flip (same as other implementations)
        mirrorTensor[i] = 1;

        // Allocate device memory
        size_t srcBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);
        size_t dstBufferSize = dstDescs[i].n * dstDescs[i].h * dstDescs[i].w * dstDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], srcBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], dstBufferSize));

        // Copy image data line by line
        Rpp8u* h_tempBuffer = new Rpp8u[srcBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, srcBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    rppSetBatchSize(handle, 1);

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            // Pass pointer to the i-th 256-element ROI block
            CHECK_RPP_STATUS(rppt_resize_mirror_normalize(d_inputs[i], &srcDescs[i],
                                                         d_outputs[i], &dstDescs[i],
                                                         &dstImgSizes[i],
                                                         RpptInterpolationType::BILINEAR,
                                                         meanTensor + i * numChannels,
                                                         stdDevTensor + i * numChannels,
                                                         &mirrorTensor[i],
                                                         &roiTensor[i * 256], RpptRoiType::XYWH,
                                                         handle, RPP_HIP_BACKEND),
                            "ResizeMirrorNormalize");
        }
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(meanTensor));
    CHECK_HIP_STATUS(hipHostFree(stdDevTensor));
    CHECK_HIP_STATUS(hipHostFree(mirrorTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));
    CHECK_HIP_STATUS(hipHostFree(dstImgSizes));

    ostringstream params;
    params << "resize=half, mirror=horizontal, mean=60/80/100, stddev=1";
    printResult("RPP HIP ResizeMirrorNormalize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_ResizeCropMirror(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int num_images = (int)imgs.size();
    int numChannels = isColor ? 3 : 1;

    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<Rpp8u*> d_inputs(num_images);
    vector<Rpp8u*> d_outputs(num_images);

    // Allocate parameter tensors in pinned host memory
    Rpp32u *mirrorTensor;
    RpptROI *roiTensor;
    RpptImagePatch *dstImgSizes;
    CHECK_HIP_STATUS(hipHostMalloc(&mirrorTensor, num_images * sizeof(Rpp32u)));
    // Allocate 256 ROI elements per image as workaround for kernel bug (launches 256 threads without bounds check)
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, num_images * 256 * sizeof(RpptROI)));
    CHECK_HIP_STATUS(hipHostMalloc(&dstImgSizes, num_images * sizeof(RpptImagePatch)));

    // Target resize dimensions (224x224 common for neural networks)
    int targetWidth = 224;
    int targetHeight = 224;

    // Calculate crop parameters once (all images have same dimensions)
    int cropWidth = (imgs[0].cols * 4) / 5;
    int cropHeight = (imgs[0].rows * 4) / 5;
    int cropX = (imgs[0].cols - cropWidth) / 2;
    int cropY = (imgs[0].rows - cropHeight) / 2;

    for (int i = 0; i < num_images; ++i) {
        RpptLayout layout = (isColor && imgs[i].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;
        set_descriptor_dims_and_strides_local(&srcDescs[i], 1, imgs[i].rows, imgs[i].cols, numChannels, 0);
        srcDescs[i].layout = layout;
        srcDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&srcDescs[i]);

        // Destination descriptor (resized and cropped to 224x224)
        set_descriptor_dims_and_strides_local(&dstDescs[i], 1, targetHeight, targetWidth, numChannels, 0);
        dstDescs[i].layout = layout;
        dstDescs[i].dataType = RpptDataType::U8;
        update_strides_from_layout(&dstDescs[i]);

        // Each image gets 256 ROI slots (only first is used, rest are buffer for buggy kernel)
        roiTensor[i * 256].xywhROI.xy.x = cropX;
        roiTensor[i * 256].xywhROI.xy.y = cropY;
        roiTensor[i * 256].xywhROI.roiWidth = cropWidth;
        roiTensor[i * 256].xywhROI.roiHeight = cropHeight;

        // Destination image size (after resize)
        dstImgSizes[i].width = targetWidth;
        dstImgSizes[i].height = targetHeight;

        // Mirror: alternate images for demo
        mirrorTensor[i] = (i % 2);

        // Allocate device memory
        size_t srcBufferSize = srcDescs[i].n * srcDescs[i].h * srcDescs[i].w * srcDescs[i].c * sizeof(Rpp8u);
        size_t dstBufferSize = dstDescs[i].n * dstDescs[i].h * dstDescs[i].w * dstDescs[i].c * sizeof(Rpp8u);

        CHECK_HIP_STATUS(hipMalloc(&d_inputs[i], srcBufferSize));
        CHECK_HIP_STATUS(hipMalloc(&d_outputs[i], dstBufferSize));

        // Copy image data line by line
        Rpp8u* h_tempBuffer = new Rpp8u[srcBufferSize]();
        int originalWidth = imgs[i].cols;
        int alignedWidth = srcDescs[i].w;
        int elementsPerRow = originalWidth * numChannels;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(h_tempBuffer + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }

        CHECK_HIP_STATUS(hipMemcpy(d_inputs[i], h_tempBuffer, srcBufferSize, hipMemcpyHostToDevice));
        delete[] h_tempBuffer;
    }

    // NOTE: rppt_resize_crop_mirror has handle corruption bug similar to rppt_flip
    // Run only once to avoid memory corruption, then multiply time by PERF_RUNS
            perfMonitor.start(DeviceType::GPU, 0);
    for (int i = 0; i < num_images; ++i) {
        // Pass pointer to the i-th 256-element ROI block
        CHECK_RPP_STATUS(rppt_resize_crop_mirror(d_inputs[i], &srcDescs[i],
                                                 d_outputs[i], &dstDescs[i],
                                                 &dstImgSizes[i],
                                                 RpptInterpolationType::BILINEAR,
                                                 &mirrorTensor[i],
                                                 &roiTensor[i * 256], RpptRoiType::XYWH,
                                                 handle, RPP_HIP_BACKEND),
                        "ResizeCropMirror");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    // Multiply by HIP_PERF_RUNS to normalize with other benchmarks
    double singleRunTime = perfMonitor.getTotalTime();
    double adjustedTime = singleRunTime * HIP_PERF_RUNS;

    for (int i = 0; i < num_images; ++i) {
        CHECK_HIP_STATUS(hipFree(d_inputs[i]));
        CHECK_HIP_STATUS(hipFree(d_outputs[i]));
    }
    CHECK_HIP_STATUS(hipHostFree(mirrorTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));
    CHECK_HIP_STATUS(hipHostFree(dstImgSizes));

    ostringstream params;
    params << "crop=80%,resize=224x224,mirror";
    printResult("RPP HIP ResizeCropMirror", imgs.size(), isColor, adjustedTime, perfMonitor.getTotalEnergy(), params.str());
}

// ============================================================================
// BATCHED PROCESSING BENCHMARK FUNCTIONS
// ============================================================================
// These functions process all images in a single RPP API call with batch descriptors


// Helper macro for common batched buffer setup
#define BATCHED_BUFFER_SETUP(batchSize, isColor, srcDesc, dstDesc, maxHeight, maxWidth) \
    if ((batchSize) == 0) return; \
    int numChannels = (isColor) ? 3 : 1; \
    int maxHeight = imgs[0].rows; \
    int maxWidth = imgs[0].cols; \
    RpptLayout layout = ((isColor) && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW; \
    RpptDesc srcDesc, dstDesc; \
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0); \
    srcDesc.layout = layout; \
    srcDesc.dataType = RpptDataType::U8; \
    update_strides_from_layout(&srcDesc);

// ===== GEOMETRIC OPERATIONS =====

void benchmark_RPP_HIP_Flip_Batched(const vector<Mat>& imgs, bool isColor, int flipCode, rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;
    
    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.h * srcDesc.w * srcDesc.c * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32u *horizontalTensor, *verticalTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&horizontalTensor, batchSize * sizeof(Rpp32u)));
    CHECK_HIP_STATUS(hipHostMalloc(&verticalTensor, batchSize * sizeof(Rpp32u)));
    // Allocate 256 ROI elements per image as workaround for kernel bug (launches 256 threads without bounds check)
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * 256 * sizeof(RpptROI)));

    Rpp32u horizontalFlag = (flipCode == 1 || flipCode == -1) ? 1 : 0;
    Rpp32u verticalFlag = (flipCode == 0 || flipCode == -1) ? 1 : 0;

    for (int i = 0; i < batchSize; ++i) {
        horizontalTensor[i] = horizontalFlag;
        verticalTensor[i] = verticalFlag;
        // Each image gets 256 ROI slots (only first is used, rest are buffer for buggy kernel)
        roiTensor[i * 256].xywhROI.xy.x = 0;
        roiTensor[i * 256].xywhROI.xy.y = 0;
        roiTensor[i * 256].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i * 256].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[totalBufferSize]();
    int alignedWidth = srcDesc.w;

    for (int i = 0; i < batchSize; ++i) {
        int originalWidth = imgs[i].cols;
        int elementsPerRow = originalWidth * numChannels;
        Rpp8u* imgStart = h_tempBuffer + i * bufferSizePerImage;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(imgStart + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        // CRITICAL: Sync before resetting ROI to prevent CPU-GPU race
        // The GPU may still be reading roiTensor from the previous iteration
        if (k > 0) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
        }

        for (int i = 0; i < batchSize; ++i) {
            roiTensor[i * 256].xywhROI.xy.x = 0;
            roiTensor[i * 256].xywhROI.xy.y = 0;
            roiTensor[i * 256].xywhROI.roiWidth = imgs[i].cols;
            roiTensor[i * 256].xywhROI.roiHeight = imgs[i].rows;
        }

        CHECK_RPP_STATUS(rppt_flip(d_input, &srcDesc, d_output, &dstDesc,
                                   horizontalTensor, verticalTensor,
                                   roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "Flip");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(horizontalTensor));
    CHECK_HIP_STATUS(hipHostFree(verticalTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    string name = (flipCode == 1) ? "Horizontal" : (flipCode == 0) ? "Vertical" : "Both";
    printResult("RPP HIP BATCH Flip", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), "type=" + name);
}

void benchmark_RPP_HIP_Resize_Batched(const vector<Mat>& imgs, bool isColor, int dstW, int dstH,
                                     RpptInterpolationType interpType, const string& interpName,
                                     rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);

    set_descriptor_dims_and_strides_local(&dstDesc, batchSize, dstH, dstW, numChannels, 0);
    dstDesc.layout = layout;
    dstDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&dstDesc);

    size_t srcBufferSizePerImage = srcDesc.h * srcDesc.w * srcDesc.c * sizeof(Rpp8u);
    size_t dstBufferSizePerImage = dstDesc.h * dstDesc.w * dstDesc.c * sizeof(Rpp8u);
    size_t totalSrcBufferSize = srcBufferSizePerImage * batchSize;
    size_t totalDstBufferSize = dstBufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalSrcBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalDstBufferSize));

    RpptImagePatch *dstImgSizes;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&dstImgSizes, batchSize * sizeof(RpptImagePatch)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        dstImgSizes[i].width = dstW;
        dstImgSizes[i].height = dstH;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[totalSrcBufferSize]();
    int alignedWidth = srcDesc.w;

    for (int i = 0; i < batchSize; ++i) {
        int originalWidth = imgs[i].cols;
        int elementsPerRow = originalWidth * numChannels;
        Rpp8u* imgStart = h_tempBuffer + i * srcBufferSizePerImage;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(imgStart + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalSrcBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_resize(d_input, &srcDesc, d_output, &dstDesc,
                                     dstImgSizes, interpType, roiTensor, RpptRoiType::XYWH,
                                     handle, RPP_HIP_BACKEND),
                        "Resize");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(dstImgSizes));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "type=" << interpName << ", size=" << dstW << "x" << dstH;
    printResult("RPP HIP BATCH Resize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}


void benchmark_RPP_HIP_Crop_Batched(const vector<Mat>& imgs, bool isColor, int cropW, int cropH,
                                    rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);

    set_descriptor_dims_and_strides_local(&dstDesc, batchSize, cropH, cropW, numChannels, 0);
    dstDesc.layout = layout;
    dstDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&dstDesc);

    size_t srcBufferSizePerImage = srcDesc.h * srcDesc.w * srcDesc.c * sizeof(Rpp8u);
    size_t dstBufferSizePerImage = dstDesc.h * dstDesc.w * dstDesc.c * sizeof(Rpp8u);
    size_t totalSrcBufferSize = srcBufferSizePerImage * batchSize;
    size_t totalDstBufferSize = dstBufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalSrcBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalDstBufferSize));

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    int cropX = (maxWidth - cropW) / 2;
    int cropY = (maxHeight - cropH) / 2;

    for (int i = 0; i < batchSize; ++i) {
        roiTensor[i].xywhROI.xy.x = cropX;
        roiTensor[i].xywhROI.xy.y = cropY;
        roiTensor[i].xywhROI.roiWidth = cropW;
        roiTensor[i].xywhROI.roiHeight = cropH;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[totalSrcBufferSize]();
    int alignedWidth = srcDesc.w;

    for (int i = 0; i < batchSize; ++i) {
        int originalWidth = imgs[i].cols;
        int elementsPerRow = originalWidth * numChannels;
        Rpp8u* imgStart = h_tempBuffer + i * srcBufferSizePerImage;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(imgStart + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalSrcBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_crop(d_input, &srcDesc, d_output, &dstDesc,
                                   roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "Crop");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "size=" << cropW << "x" << cropH;
    printResult("RPP HIP BATCH Crop", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Rotate_Batched(const vector<Mat>& imgs, bool isColor, float angleDeg,
                                     rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.h * srcDesc.w * srcDesc.c * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32f *angleTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&angleTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        angleTensor[i] = angleDeg;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[totalBufferSize]();
    int alignedWidth = srcDesc.w;

    for (int i = 0; i < batchSize; ++i) {
        int originalWidth = imgs[i].cols;
        int elementsPerRow = originalWidth * numChannels;
        Rpp8u* imgStart = h_tempBuffer + i * bufferSizePerImage;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(imgStart + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_rotate(d_input, &srcDesc, d_output, &dstDesc,
                                     angleTensor, RpptInterpolationType::BILINEAR,
                                     roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "Rotate");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(angleTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "angle=" << angleDeg << "deg";
    printResult("RPP HIP BATCH Rotate", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

// ===== FILTER OPERATIONS =====

void benchmark_RPP_HIP_BoxFilter_Batched(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                        rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    Rpp32u kSize = static_cast<Rpp32u>(kernelSize);
    int offsetInBytes = 12 * (kernelSize / 2);
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.h * srcDesc.w * srcDesc.c * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[totalBufferSize]();
    int alignedWidth = srcDesc.w;

    for (int i = 0; i < batchSize; ++i) {
        int originalWidth = imgs[i].cols;
        int elementsPerRow = originalWidth * numChannels;
        Rpp8u* imgStart = h_tempBuffer + i * bufferSizePerImage;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(imgStart + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_box_filter(d_input, &srcDesc, d_output, &dstDesc,
                                         kSize, borderType, roiTensor, RpptRoiType::XYWH,
                                         handle, RPP_HIP_BACKEND),
                        "BoxFilter");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    
    ostringstream params;
    params << "kernel=" << kernelSize;
    printResult("RPP HIP BATCH BoxFilter", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_GaussianFilter_Batched(const vector<Mat>& imgs, bool isColor, int kernelSize, float sigma,
                                              rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    Rpp32u kSize = static_cast<Rpp32u>(kernelSize);
    int offsetInBytes = 12 * (kernelSize / 2);
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.h * srcDesc.w * srcDesc.c * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32f *stdDevTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&stdDevTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        stdDevTensor[i] = sigma;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[totalBufferSize]();
    int alignedWidth = srcDesc.w;

    for (int i = 0; i < batchSize; ++i) {
        int originalWidth = imgs[i].cols;
        int elementsPerRow = originalWidth * numChannels;
        Rpp8u* imgStart = h_tempBuffer + i * bufferSizePerImage;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(imgStart + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_gaussian_filter(d_input, &srcDesc, d_output, &dstDesc,
                                              stdDevTensor, kSize, borderType,
                                              roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "GaussianFilter");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(stdDevTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "kernel=" << kernelSize << ", sigma=" << sigma;
    printResult("RPP HIP BATCH GaussianFilter", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_MedianFilter_Batched(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                            rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    Rpp32u kSize = static_cast<Rpp32u>(kernelSize);
    int offsetInBytes = 12 * (kernelSize / 2);
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.h * srcDesc.w * srcDesc.c * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[totalBufferSize]();
    int alignedWidth = srcDesc.w;

    for (int i = 0; i < batchSize; ++i) {
        int originalWidth = imgs[i].cols;
        int elementsPerRow = originalWidth * numChannels;
        Rpp8u* imgStart = h_tempBuffer + i * bufferSizePerImage;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(imgStart + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_median_filter(d_input, &srcDesc, d_output, &dstDesc,
                                            kSize, borderType, roiTensor, RpptRoiType::XYWH,
                                            handle, RPP_HIP_BACKEND),
                        "MedianFilter");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    
    ostringstream params;
    params << "kernel=" << kernelSize;
    printResult("RPP HIP BATCH MedianFilter", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_SobelFilter_Batched(const vector<Mat>& imgs, bool isColor, int sobelType,
                                           rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int dstChannels = 1;  // Sobel filter always outputs grayscale
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u kernelSize = 3;  // Sobel uses 3x3 kernel
    RpptLayout srcLayout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = srcLayout;
    srcDesc.dataType = RpptDataType::U8;
    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);

    dstDesc.layout = RpptLayout::NCHW;  // Output is always grayscale (planar)
    dstDesc.dataType = RpptDataType::U8;
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, dstChannels, 0);

    size_t srcBufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t dstBufferSizePerImage = dstDesc.strides.nStride * sizeof(Rpp8u);
    size_t srcTotalBufferSize = srcBufferSizePerImage * batchSize;
    size_t dstTotalBufferSize = dstBufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, srcTotalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, dstTotalBufferSize));

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcTotalBufferSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcBufferSizePerImage;
        int elementsPerRow = imgs[i].cols * numChannels;

        if (srcLayout == RpptLayout::NHWC) {
            for (int row = 0; row < imgs[i].rows; ++row) {
                memcpy(imgStart + row * srcDesc.strides.hStride,
                       imgs[i].data + row * elementsPerRow,
                       elementsPerRow * sizeof(Rpp8u));
            }
        } else {
            // For NCHW, need to deinterleave channels
            for (int row = 0; row < imgs[i].rows; ++row) {
                for (int col = 0; col < imgs[i].cols; ++col) {
                    for (int ch = 0; ch < numChannels; ++ch) {
                        int srcIdx = row * elementsPerRow + col * numChannels + ch;
                        int dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col;
                        imgStart[dstIdx] = imgs[i].data[srcIdx];
                    }
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, srcTotalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_sobel_filter(d_input, &srcDesc, d_output, &dstDesc,
                                           sobelType, kernelSize, roiTensor, RpptRoiType::XYWH,
                                           handle, RPP_HIP_BACKEND),
                        "SobelFilter");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    
    ostringstream params;
    params << "type=" << sobelType;
    printResult("RPP HIP BATCH SobelFilter", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_HistogramEqualize_Batched(const vector<Mat>& imgs, bool isColor,
                                                 rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);

    dstDesc.layout = layout;
    dstDesc.dataType = RpptDataType::U8;
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, 0);

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[totalBufferSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * bufferSizePerImage;
        int elementsPerRow = imgs[i].cols * numChannels;

        if (layout == RpptLayout::NHWC) {
            for (int row = 0; row < imgs[i].rows; ++row) {
                memcpy(imgStart + row * srcDesc.strides.hStride,
                       imgs[i].data + row * elementsPerRow,
                       elementsPerRow * sizeof(Rpp8u));
            }
        } else {
            // For NCHW, need to deinterleave channels
            for (int row = 0; row < imgs[i].rows; ++row) {
                for (int col = 0; col < imgs[i].cols; ++col) {
                    for (int ch = 0; ch < numChannels; ++ch) {
                        int srcIdx = row * elementsPerRow + col * numChannels + ch;
                        int dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col;
                        imgStart[dstIdx] = imgs[i].data[srcIdx];
                    }
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_histogram_equalize(d_input, &srcDesc, d_output, &dstDesc,
                                                 roiTensor, RpptRoiType::XYWH,
                                                 handle, RPP_HIP_BACKEND),
                        "HistogramEqualize");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP BATCH HistogramEqualize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), "");
}


// ===== COLOR OPERATIONS =====

void benchmark_RPP_HIP_Hue_Batched(const vector<Mat>& imgs, bool isColor, float hueFactor,
                                   rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.h * srcDesc.w * srcDesc.c * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32f *hueTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&hueTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        hueTensor[i] = hueFactor;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[totalBufferSize]();
    int alignedWidth = srcDesc.w;

    for (int i = 0; i < batchSize; ++i) {
        int originalWidth = imgs[i].cols;
        int elementsPerRow = originalWidth * numChannels;
        Rpp8u* imgStart = h_tempBuffer + i * bufferSizePerImage;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(imgStart + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_hue(d_input, &srcDesc, d_output, &dstDesc,
                                  hueTensor, roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "Hue");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(hueTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "hue=" << hueFactor;
    printResult("RPP HIP BATCH Hue", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Saturation_Batched(const vector<Mat>& imgs, bool isColor, float satFactor,
                                          rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.h * srcDesc.w * srcDesc.c * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32f *satTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&satTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        satTensor[i] = satFactor;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[totalBufferSize]();
    int alignedWidth = srcDesc.w;

    for (int i = 0; i < batchSize; ++i) {
        int originalWidth = imgs[i].cols;
        int elementsPerRow = originalWidth * numChannels;
        Rpp8u* imgStart = h_tempBuffer + i * bufferSizePerImage;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(imgStart + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_saturation(d_input, &srcDesc, d_output, &dstDesc,
                                         satTensor, roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "Saturation");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(satTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "factor=" << satFactor;
    printResult("RPP HIP BATCH Saturation", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_ColorToGreyscale_Batched(const vector<Mat>& imgs, bool isColor,
                                                rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int srcChannels = 3;  // Always RGB input
    int dstChannels = 1;  // Grayscale output
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;

    RpptDesc srcDesc, dstDesc;
    // Use set_descriptor_dims_and_strides_local for proper alignment
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, srcChannels, 0);
    srcDesc.layout = RpptLayout::NHWC;  // Source is RGB in NHWC
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);

    set_descriptor_dims_and_strides_local(&dstDesc, batchSize, maxHeight, maxWidth, dstChannels, 0);
    dstDesc.layout = RpptLayout::NCHW;  // REQUIRED: rppt_color_to_greyscale requires NCHW for dst
    dstDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&dstDesc);

    size_t srcBufferSizePerImage = srcDesc.h * srcDesc.w * srcDesc.c * sizeof(Rpp8u);
    size_t dstBufferSizePerImage = dstDesc.h * dstDesc.w * dstDesc.c * sizeof(Rpp8u);
    size_t totalSrcBufferSize = srcBufferSizePerImage * batchSize;
    size_t totalDstBufferSize = dstBufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalSrcBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalDstBufferSize));

    Rpp8u* h_tempBuffer = new Rpp8u[totalSrcBufferSize]();
    int alignedWidth = srcDesc.w;

    for (int i = 0; i < batchSize; ++i) {
        int originalWidth = imgs[i].cols;
        int elementsPerRow = originalWidth * srcChannels;
        Rpp8u* imgStart = h_tempBuffer + i * srcBufferSizePerImage;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(imgStart + row * alignedWidth * srcChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalSrcBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    RpptSubpixelLayout srcSubpixelLayout = RpptSubpixelLayout::RGBtype;
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_color_to_greyscale(d_input, &srcDesc, d_output, &dstDesc,
                                                 srcSubpixelLayout,
                                                 handle, RPP_HIP_BACKEND),
                        "ColorToGreyscale");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));

    printResult("RPP HIP BATCH ColorToGreyscale", imgs.size(), true,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), "");
}

void benchmark_RPP_HIP_Brightness_Batched(const vector<Mat>& imgs, bool isColor, float alpha, float beta,
                                          rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.h * srcDesc.w * srcDesc.c * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32f *alphaTensor, *betaTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&alphaTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&betaTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        alphaTensor[i] = alpha;
        betaTensor[i] = beta;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[totalBufferSize]();
    int alignedWidth = srcDesc.w;

    for (int i = 0; i < batchSize; ++i) {
        int originalWidth = imgs[i].cols;
        int elementsPerRow = originalWidth * numChannels;
        Rpp8u* imgStart = h_tempBuffer + i * bufferSizePerImage;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(imgStart + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_brightness(d_input, &srcDesc, d_output, &dstDesc,
                                         alphaTensor, betaTensor,
                                         roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "Brightness");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(alphaTensor));
    CHECK_HIP_STATUS(hipHostFree(betaTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "alpha=" << alpha << ", beta=" << beta;
    printResult("RPP HIP BATCH Brightness", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Contrast_Batched(const vector<Mat>& imgs, bool isColor, float contrastFactor, float contrastCenter,
                                        rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.h * srcDesc.w * srcDesc.c * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32f *contrastFactorTensor, *contrastCenterTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&contrastFactorTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&contrastCenterTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        contrastFactorTensor[i] = contrastFactor;
        contrastCenterTensor[i] = contrastCenter;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[totalBufferSize]();
    int alignedWidth = srcDesc.w;

    for (int i = 0; i < batchSize; ++i) {
        int originalWidth = imgs[i].cols;
        int elementsPerRow = originalWidth * numChannels;
        Rpp8u* imgStart = h_tempBuffer + i * bufferSizePerImage;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(imgStart + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_contrast(d_input, &srcDesc, d_output, &dstDesc,
                                       contrastFactorTensor, contrastCenterTensor,
                                       roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "Contrast");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(contrastFactorTensor));
    CHECK_HIP_STATUS(hipHostFree(contrastCenterTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "factor=" << contrastFactor << ", center=" << contrastCenter;
    printResult("RPP HIP BATCH Contrast", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Emboss_Batched(const vector<Mat>& imgs, bool isColor, int kernelSize, float strength,
                                     rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    // REQUIRED: rppt_emboss needs offsetInBytes >= 12 * (kernelSize / 2) for edge handling
    int requiredOffset = 12 * (kernelSize / 2);

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, requiredOffset);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);

    // Destination descriptor with zero offset (output doesn't need the padding)
    set_descriptor_dims_and_strides_local(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    dstDesc.layout = layout;
    dstDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&dstDesc);

    // Buffer size calculation: account for src offset for edge handling
    size_t srcBufferSizePerImage = srcDesc.h * srcDesc.w * srcDesc.c * sizeof(Rpp8u);
    size_t dstBufferSizePerImage = dstDesc.h * dstDesc.w * dstDesc.c * sizeof(Rpp8u);
    size_t totalSrcBufferSize = (srcBufferSizePerImage * batchSize) + requiredOffset;  // Add offset for edge padding
    size_t totalDstBufferSize = dstBufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalSrcBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalDstBufferSize));

    Rpp32f *strengthTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&strengthTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        strengthTensor[i] = strength;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    // Allocate host buffer with offset for edge handling
    Rpp8u* h_tempBuffer = new Rpp8u[totalSrcBufferSize]();
    int alignedWidth = srcDesc.w;

    for (int i = 0; i < batchSize; ++i) {
        int originalWidth = imgs[i].cols;
        int elementsPerRow = originalWidth * numChannels;
        // Start copying after the offset to leave padding space for edge handling
        Rpp8u* imgStart = h_tempBuffer + requiredOffset + (i * srcBufferSizePerImage);

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(imgStart + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalSrcBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    // REQUIRED: rppt_emboss only supports REPLICATE border type (validated in implementation)
    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_emboss(d_input, &srcDesc, d_output, &dstDesc,
                                     strengthTensor, kernelSize,
                                     borderType, roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "Emboss");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(strengthTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "kernel=" << kernelSize << ", strength=" << strength;
    printResult("RPP HIP BATCH Emboss", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}


// ===== ARITHMETIC OPERATIONS =====

void benchmark_RPP_HIP_AddScalar_Batched(const vector<Mat>& imgs, bool isColor, float addVal,
                                        rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    // Scalar operations only support F32 datatype - set this BEFORE calculating dimensions/strides
    srcDesc.dataType = RpptDataType::F32;
    srcDesc.layout = layout;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    // Convert to GenericDesc for 3D arithmetic operations
    RpptGenericDesc srcGenDesc, dstGenDesc;
    convert_desc_to_generic_desc_3d(&srcDesc, &srcGenDesc);
    convert_desc_to_generic_desc_3d(&dstDesc, &dstGenDesc);

    // Allocate F32 buffers since scalar operations require F32 datatype
    // Use nStride (element count per image) for buffer size calculation
    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp32f);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp32f *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32f *addTensor;
    RpptROI *roiTensor;
    RpptROI3D *roi3dTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&addTensor, batchSize * numChannels * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));
    CHECK_HIP_STATUS(hipHostMalloc(&roi3dTensor, batchSize * sizeof(RpptROI3D)));

    for (int i = 0; i < batchSize; ++i) {
        for (int c = 0; c < numChannels; ++c) {
            addTensor[i * numChannels + c] = addVal;
        }
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    // Convert U8 image data to F32 (normalized to 0-255 range for consistency with OpenCV)
    // Use nStride for proper buffer indexing per image
    Rpp32f* h_tempBuffer = new Rpp32f[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp32f* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    // Calculate destination index based on layout
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else { // NCHW
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }

                    // Source index in original image (always NHWC for OpenCV Mat)
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = static_cast<Rpp32f>(imgs[i].data[srcIdx]);
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    // Convert ROI to ROI3D
    convert_roi_to_roi3d(roiTensor, roi3dTensor, batchSize);

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_add_scalar(d_input, &srcGenDesc, d_output, &dstGenDesc,
                                         addTensor, roi3dTensor, RpptRoi3DType::XYZWHD,
                                         handle, RPP_HIP_BACKEND),
                        "AddScalar");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(addTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));
    CHECK_HIP_STATUS(hipHostFree(roi3dTensor));

    
    ostringstream params;
    params << "value=" << (int)addVal;
    printResult("RPP HIP BATCH AddScalar", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_SubtractScalar_Batched(const vector<Mat>& imgs, bool isColor, float subVal,
                                              rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    // Scalar operations only support F32 datatype - set this BEFORE calculating dimensions/strides
    srcDesc.dataType = RpptDataType::F32;
    srcDesc.layout = layout;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    // Convert to GenericDesc for 3D arithmetic operations
    RpptGenericDesc srcGenDesc, dstGenDesc;
    convert_desc_to_generic_desc_3d(&srcDesc, &srcGenDesc);
    convert_desc_to_generic_desc_3d(&dstDesc, &dstGenDesc);

    // Allocate F32 buffers since scalar operations require F32 datatype
    // Use nStride (element count per image) for buffer size calculation
    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp32f);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp32f *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32f *subTensor;
    RpptROI *roiTensor;
    RpptROI3D *roi3dTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&subTensor, batchSize * numChannels * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));
    CHECK_HIP_STATUS(hipHostMalloc(&roi3dTensor, batchSize * sizeof(RpptROI3D)));

    for (int i = 0; i < batchSize; ++i) {
        for (int c = 0; c < numChannels; ++c) {
            subTensor[i * numChannels + c] = subVal;
        }
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    // Convert U8 image data to F32 (normalized to 0-255 range for consistency with OpenCV)
    // Use nStride for proper buffer indexing per image
    Rpp32f* h_tempBuffer = new Rpp32f[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp32f* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    // Calculate destination index based on layout
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else { // NCHW
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }

                    // Source index in original image (always NHWC for OpenCV Mat)
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = static_cast<Rpp32f>(imgs[i].data[srcIdx]);
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    // Convert ROI to ROI3D
    convert_roi_to_roi3d(roiTensor, roi3dTensor, batchSize);

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_subtract_scalar(d_input, &srcGenDesc, d_output, &dstGenDesc,
                                              subTensor, roi3dTensor, RpptRoi3DType::XYZWHD,
                                              handle, RPP_HIP_BACKEND),
                        "SubtractScalar");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(subTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));
    CHECK_HIP_STATUS(hipHostFree(roi3dTensor));

    
    ostringstream params;
    params << "value=" << (int)subVal;
    printResult("RPP HIP BATCH SubtractScalar", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_MultiplyScalar_Batched(const vector<Mat>& imgs, bool isColor, float mulVal,
                                              rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    // Scalar operations only support F32 datatype - set this BEFORE calculating dimensions/strides
    srcDesc.dataType = RpptDataType::F32;
    srcDesc.layout = layout;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    // Convert to GenericDesc for 3D arithmetic operations
    RpptGenericDesc srcGenDesc, dstGenDesc;
    convert_desc_to_generic_desc_3d(&srcDesc, &srcGenDesc);
    convert_desc_to_generic_desc_3d(&dstDesc, &dstGenDesc);

    // Allocate F32 buffers since scalar operations require F32 datatype
    // Use nStride (element count per image) for buffer size calculation
    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp32f);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp32f *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32f *mulTensor;
    RpptROI *roiTensor;
    RpptROI3D *roi3dTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&mulTensor, batchSize * numChannels * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));
    CHECK_HIP_STATUS(hipHostMalloc(&roi3dTensor, batchSize * sizeof(RpptROI3D)));

    for (int i = 0; i < batchSize; ++i) {
        for (int c = 0; c < numChannels; ++c) {
            mulTensor[i * numChannels + c] = mulVal;
        }
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    // Convert U8 image data to F32 (normalized to 0-255 range for consistency with OpenCV)
    // Use nStride for proper buffer indexing per image
    Rpp32f* h_tempBuffer = new Rpp32f[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp32f* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    // Calculate destination index based on layout
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else { // NCHW
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }

                    // Source index in original image (always NHWC for OpenCV Mat)
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = static_cast<Rpp32f>(imgs[i].data[srcIdx]);
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    // Convert ROI to ROI3D
    convert_roi_to_roi3d(roiTensor, roi3dTensor, batchSize);

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_multiply_scalar(d_input, &srcGenDesc, d_output, &dstGenDesc,
                                              mulTensor, roi3dTensor, RpptRoi3DType::XYZWHD,
                                              handle, RPP_HIP_BACKEND),
                        "MultiplyScalar");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(mulTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));
    CHECK_HIP_STATUS(hipHostFree(roi3dTensor));

    ostringstream params;
    params << "value=" << mulVal;
    printResult("RPP HIP BATCH MultiplyScalar", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_GaussianNoise_Batched(const vector<Mat>& imgs, bool isColor, float mean,
                                              float stddev, rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32f *meanTensor, *stddevTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&meanTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&stddevTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        meanTensor[i] = mean;
        stddevTensor[i] = stddev;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    unsigned long long seed = 12345;
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_gaussian_noise(d_input, &srcDesc, d_output, &dstDesc, meanTensor, stddevTensor,
                                             seed, roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "GaussianNoise");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(meanTensor));
    CHECK_HIP_STATUS(hipHostFree(stddevTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "mean=" << mean << ", stddev=" << stddev;
    printResult("RPP HIP BATCH GaussianNoise", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_SaltAndPepperNoise_Batched(const vector<Mat>& imgs, bool isColor, float noiseProb,
                                                   rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32f *noiseProbTensor, *saltProbTensor, *saltValueTensor, *pepperValueTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&noiseProbTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&saltProbTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&saltValueTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&pepperValueTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        noiseProbTensor[i] = noiseProb;
        saltProbTensor[i] = 0.5f;
        saltValueTensor[i] = 1.0f;
        pepperValueTensor[i] = 0.0f;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    Rpp32u seed = 12345;
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_salt_and_pepper_noise(d_input, &srcDesc, d_output, &dstDesc, noiseProbTensor,
                                                     saltProbTensor, saltValueTensor, pepperValueTensor,
                                                     seed, roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "SaltAndPepperNoise");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(noiseProbTensor));
    CHECK_HIP_STATUS(hipHostFree(saltProbTensor));
    CHECK_HIP_STATUS(hipHostFree(saltValueTensor));
    CHECK_HIP_STATUS(hipHostFree(pepperValueTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    
    ostringstream params;
    params << "probability=" << noiseProb;
    printResult("RPP HIP BATCH SaltAndPepperNoise", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_NoiseShot_Batched(const vector<Mat>& imgs, bool isColor, float shotNoiseFactor,
                                         rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32f *shotNoiseTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&shotNoiseTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        shotNoiseTensor[i] = shotNoiseFactor;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_shot_noise(d_input, &srcDesc, d_output, &dstDesc, shotNoiseTensor, 12345,
                                         roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "NoiseShot");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(shotNoiseTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    
    ostringstream params;
    params << "factor=" << shotNoiseFactor;
    printResult("RPP HIP BATCH NoiseShot", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_ColorCast_Batched(const vector<Mat>& imgs, bool isColor, float rShift, float gShift,
                                         float bShift, rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    RpptRGB *rgbTensor;
    Rpp32f *alphaTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&rgbTensor, batchSize * sizeof(RpptRGB)));
    CHECK_HIP_STATUS(hipHostMalloc(&alphaTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        rgbTensor[i].R = (Rpp8u)max(0.0f, min(255.0f, rShift));
        rgbTensor[i].G = (Rpp8u)max(0.0f, min(255.0f, gShift));
        rgbTensor[i].B = (Rpp8u)max(0.0f, min(255.0f, bShift));
        alphaTensor[i] = 1.0f;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_color_cast(d_input, &srcDesc, d_output, &dstDesc, rgbTensor, alphaTensor,
                                         roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "ColorCast");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(rgbTensor));
    CHECK_HIP_STATUS(hipHostFree(alphaTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "R=" << rShift << ", G=" << gShift << ", B=" << bShift;
    printResult("RPP HIP BATCH ColorCast", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_ColorTemperature_Batched(const vector<Mat>& imgs, bool isColor, int adjustmentValue,
                                                rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32s *adjustmentTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&adjustmentTensor, batchSize * sizeof(Rpp32s)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        adjustmentTensor[i] = adjustmentValue;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_color_temperature(d_input, &srcDesc, d_output, &dstDesc, adjustmentTensor,
                                                 roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "ColorTemperature");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(adjustmentTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    
    ostringstream params;
    params << "adjustment=" << adjustmentValue;
    printResult("RPP HIP BATCH ColorTemperature", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_ColorTwist_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle,
                                          hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0 || !isColor) return;

    int numChannels = 3;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = RpptLayout::NHWC;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32f *alpha, *beta, *hueShift, *satFactor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&alpha, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&beta, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&hueShift, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&satFactor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        alpha[i] = 1.0f;     // brightness (0 < brightness <= 20)
        beta[i] = 1.0f;      // contrast (0 < contrast <= 255)
        hueShift[i] = 60.0f; // hue (0 <= hue <= 359)
        satFactor[i] = 1.3f; // saturation (saturation >= 0)
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_color_twist(d_input, &srcDesc, d_output, &dstDesc, alpha, beta, hueShift,
                                          satFactor, roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "ColorTwist");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    ostringstream params;
    params << "hue=" << hueShift[0] << ", saturation=" << satFactor[0];
    printResult("RPP HIP BATCH ColorTwist", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(alpha));
    CHECK_HIP_STATUS(hipHostFree(beta));
    CHECK_HIP_STATUS(hipHostFree(hueShift));
    CHECK_HIP_STATUS(hipHostFree(satFactor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));
}

void benchmark_RPP_HIP_Vignette_Batched(const vector<Mat>& imgs, bool isColor, float vignetteIntensity,
                                        rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32f *intensityTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&intensityTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        intensityTensor[i] = vignetteIntensity;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_vignette(d_input, &srcDesc, d_output, &dstDesc, intensityTensor,
                                       roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "Vignette");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(intensityTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    
    ostringstream params;
    params << "intensity=" << vignetteIntensity;
    printResult("RPP HIP BATCH Vignette", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_NonLinearBlend_Batched(const vector<Mat>& imgs, bool isColor, float stdDev,
                                              rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize < 2) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input1, *d_input2, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input1, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_input2, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32f *stdDevTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&stdDevTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        stdDevTensor[i] = stdDev;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer1 = new Rpp8u[srcDesc.strides.nStride * batchSize]();
    Rpp8u* h_tempBuffer2 = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart1 = h_tempBuffer1 + i * srcDesc.strides.nStride;
        Rpp8u* imgStart2 = h_tempBuffer2 + i * srcDesc.strides.nStride;
        int idx2 = (i + 1) % batchSize;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx1 = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    int srcIdx2 = row * imgs[idx2].cols * numChannels + col * numChannels + ch;
                    imgStart1[dstIdx] = imgs[i].data[srcIdx1];
                    imgStart2[dstIdx] = imgs[idx2].data[srcIdx2];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input1, h_tempBuffer1, totalBufferSize, hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(d_input2, h_tempBuffer2, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer1;
    delete[] h_tempBuffer2;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_non_linear_blend(d_input1, d_input2, &srcDesc, d_output, &dstDesc, stdDevTensor,
                                               roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "NonLinearBlend");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input1));
    CHECK_HIP_STATUS(hipFree(d_input2));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(stdDevTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    
    ostringstream params;
    params << "stdDev=" << stdDev;
    printResult("RPP HIP BATCH NonLinearBlend", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Posterize_Batched(const vector<Mat>& imgs, bool isColor, int levelBits,
                                         rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp8u *posterizeLevelBits;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&posterizeLevelBits, batchSize * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        posterizeLevelBits[i] = (Rpp8u)levelBits;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_posterize(d_input, &srcDesc, d_output, &dstDesc, posterizeLevelBits,
                                        roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "Posterize");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(posterizeLevelBits));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    
    ostringstream params;
    params << "bits=" << levelBits;
    printResult("RPP HIP BATCH Posterize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Solarize_Batched(const vector<Mat>& imgs, bool isColor, int threshold,
                                        rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32f *thresholdTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&thresholdTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        thresholdTensor[i] = threshold / 255.0f;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }

        CHECK_RPP_STATUS(rppt_solarize(d_input, &srcDesc, d_output, &dstDesc, thresholdTensor,
                                       roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "Solarize");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(thresholdTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    
    ostringstream params;
    params << "threshold=" << threshold;
    printResult("RPP HIP BATCH Solarize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Glitch_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle,
                                      hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0 || !isColor) return;

    int numChannels = 3;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = RpptLayout::NHWC;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    RpptChannelOffsets *rgbOffsets;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&rgbOffsets, batchSize * sizeof(RpptChannelOffsets)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        rgbOffsets[i].r.x = 10;
        rgbOffsets[i].r.y = 10;
        rgbOffsets[i].g.x = 0;
        rgbOffsets[i].g.y = 0;
        rgbOffsets[i].b.x = 5;
        rgbOffsets[i].b.y = 5;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_glitch(d_input, &srcDesc, d_output, &dstDesc, rgbOffsets,
                                     roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "Glitch");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(rgbOffsets));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP BATCH Glitch", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_JpegCompressionDistortion_Batched(const vector<Mat>& imgs, bool isColor, Rpp32s quality,
                                                         rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32s *qualityTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&qualityTensor, batchSize * sizeof(Rpp32s)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        qualityTensor[i] = quality;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_jpeg_compression_distortion(d_input, &srcDesc, d_output, &dstDesc, qualityTensor,
                                                          roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "JpegCompressionDistortion");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    ostringstream params;
    params << "quality=" << quality;
    printResult("RPP HIP BATCH JpegCompressionDistortion", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(qualityTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));
}

void benchmark_RPP_HIP_TensorMin_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle,
                                         hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));

    Rpp32u outputLength = isColor ? (batchSize * 4) : batchSize;
    Rpp8u *minOutputs;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&minOutputs, outputLength * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_tensor_min(d_input, &srcDesc, minOutputs, outputLength,
                                         roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "TensorMin");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipHostFree(minOutputs));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP BATCH TensorMin", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_TensorMax_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle,
                                         hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));

    Rpp32u outputLength = isColor ? (batchSize * 4) : batchSize;
    Rpp8u *maxOutputs;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&maxOutputs, outputLength * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_tensor_max(d_input, &srcDesc, maxOutputs, outputLength,
                                         roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "TensorMax");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipHostFree(maxOutputs));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP BATCH TensorMax", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_TensorSum_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle,
                                         hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));

    Rpp32u outputLength = isColor ? (batchSize * 4) : batchSize;
    Rpp64u *sumOutputs;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&sumOutputs, outputLength * sizeof(Rpp64u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_tensor_sum(d_input, &srcDesc, sumOutputs, outputLength,
                                         roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "TensorSum");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipHostFree(sumOutputs));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP BATCH TensorSum", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_TensorMean_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle,
                                          hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));

    Rpp32u outputLength = isColor ? (batchSize * 4) : batchSize;
    Rpp32f *meanOutputs;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&meanOutputs, outputLength * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_tensor_mean(d_input, &srcDesc, meanOutputs, outputLength,
                                          roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "TensorMean");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipHostFree(meanOutputs));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP BATCH TensorMean", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_TensorStddev_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle,
                                            hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));

    Rpp32u outputLength = isColor ? (batchSize * 4) : batchSize;
    Rpp32f *stddevOutputs, *meanOutputs;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&stddevOutputs, outputLength * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&meanOutputs, outputLength * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    CHECK_RPP_STATUS(rppt_tensor_mean(d_input, &srcDesc, meanOutputs, outputLength,
                                      roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                    "TensorMean");

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_tensor_stddev(d_input, &srcDesc, stddevOutputs, outputLength, meanOutputs,
                                            roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "TensorStddev");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipHostFree(stddevOutputs));
    CHECK_HIP_STATUS(hipHostFree(meanOutputs));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP BATCH TensorStddev", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_Threshold_Batched(const vector<Mat>& imgs, bool isColor, float thresh,
                                         rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32f *minTensor, *maxTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&minTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&maxTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        minTensor[i] = thresh;
        maxTensor[i] = 255.0f;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;
        Mat grayImg;
        if (isColor)
            cvtColor(imgs[i], grayImg, COLOR_BGR2GRAY);
        else
            grayImg = imgs[i];

        for (int row = 0; row < grayImg.rows; ++row) {
            for (int col = 0; col < grayImg.cols; ++col) {
                int dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                int srcIdx = row * grayImg.cols + col;
                imgStart[dstIdx] = grayImg.data[srcIdx];
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_threshold(d_input, &srcDesc, d_output, &dstDesc, minTensor, maxTensor,
                                        roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "Threshold");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(minTensor));
    CHECK_HIP_STATUS(hipHostFree(maxTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    
    ostringstream params;
    params << "threshold=" << thresh;
    printResult("RPP HIP BATCH Threshold", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_WarpAffine_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle,
                                          hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32f *affineTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&affineTensor, batchSize * 6 * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        affineTensor[i * 6 + 0] = 1.0f;
        affineTensor[i * 6 + 1] = 0.1f;
        affineTensor[i * 6 + 2] = 10.0f;
        affineTensor[i * 6 + 3] = 0.1f;
        affineTensor[i * 6 + 4] = 1.0f;
        affineTensor[i * 6 + 5] = 10.0f;

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_warp_affine(d_input, &srcDesc, d_output, &dstDesc, affineTensor,
                                          RpptInterpolationType::BILINEAR, roiTensor,
                                          RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "WarpAffine");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(affineTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP BATCH WarpAffine", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_WarpPerspective_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle,
                                               hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32f *perspectiveTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&perspectiveTensor, batchSize * 9 * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        perspectiveTensor[i * 9 + 0] = 1.0f;
        perspectiveTensor[i * 9 + 1] = 0.1f;
        perspectiveTensor[i * 9 + 2] = 0.0f;
        perspectiveTensor[i * 9 + 3] = 0.1f;
        perspectiveTensor[i * 9 + 4] = 1.0f;
        perspectiveTensor[i * 9 + 5] = 0.0f;
        perspectiveTensor[i * 9 + 6] = 0.0f;
        perspectiveTensor[i * 9 + 7] = 0.0f;
        perspectiveTensor[i * 9 + 8] = 1.0f;

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_warp_perspective(d_input, &srcDesc, d_output, &dstDesc, perspectiveTensor,
                                               RpptInterpolationType::BILINEAR, roiTensor,
                                               RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "WarpPerspective");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(perspectiveTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP BATCH WarpPerspective", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_Fisheye_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle,
                                       hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_fisheye(d_input, &srcDesc, d_output, &dstDesc, roiTensor,
                                      RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "Fisheye");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP BATCH Fisheye", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_LensCorrection_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle,
                                              hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32f *cameraMatrix, *distortionCoeffs;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&cameraMatrix, batchSize * 9 * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&distortionCoeffs, batchSize * 8 * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        cameraMatrix[i * 9 + 0] = 534.07088364f;
        cameraMatrix[i * 9 + 1] = 0.0f;
        cameraMatrix[i * 9 + 2] = 341.53407554f;
        cameraMatrix[i * 9 + 3] = 0.0f;
        cameraMatrix[i * 9 + 4] = 534.11914595f;
        cameraMatrix[i * 9 + 5] = 232.94565259f;
        cameraMatrix[i * 9 + 6] = 0.0f;
        cameraMatrix[i * 9 + 7] = 0.0f;
        cameraMatrix[i * 9 + 8] = 1.0f;

        distortionCoeffs[i * 8 + 0] = -0.29297164f;
        distortionCoeffs[i * 8 + 1] = 0.10770696f;
        distortionCoeffs[i * 8 + 2] = 0.00131038f;
        distortionCoeffs[i * 8 + 3] = -0.0000311f;
        distortionCoeffs[i * 8 + 4] = 0.0434798f;
        distortionCoeffs[i * 8 + 5] = 0.0f;
        distortionCoeffs[i * 8 + 6] = 0.0f;
        distortionCoeffs[i * 8 + 7] = 0.0f;

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    RpptDesc tableDesc = srcDesc;
    tableDesc.c = 1;
    tableDesc.strides.nStride = maxHeight * maxWidth;
    tableDesc.strides.hStride = maxWidth;
    tableDesc.strides.wStride = 1;
    tableDesc.strides.cStride = 1;

    size_t tableSize = (size_t)maxHeight * (size_t)maxWidth * (size_t)batchSize * sizeof(Rpp32f);
    Rpp32f *d_rowRemapTable, *d_colRemapTable;
    CHECK_HIP_STATUS(hipMalloc(&d_rowRemapTable, tableSize));
    CHECK_HIP_STATUS(hipMalloc(&d_colRemapTable, tableSize));

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_lens_correction(d_input, &srcDesc, d_output, &dstDesc, d_rowRemapTable,
                                              d_colRemapTable, &tableDesc, cameraMatrix,
                                              distortionCoeffs, roiTensor, RpptRoiType::XYWH,
                                              handle, RPP_HIP_BACKEND),
                        "LensCorrection");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipFree(d_rowRemapTable));
    CHECK_HIP_STATUS(hipFree(d_colRemapTable));
    CHECK_HIP_STATUS(hipHostFree(cameraMatrix));
    CHECK_HIP_STATUS(hipHostFree(distortionCoeffs));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP BATCH LensCorrection", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HIP_GammaCorrection_Batched(const vector<Mat>& imgs, bool isColor, float gamma,
                                               rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32f *gammaTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&gammaTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        gammaTensor[i] = gamma;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_gamma_correction(d_input, &srcDesc, d_output, &dstDesc, gammaTensor,
                                               roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "GammaCorrection");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(gammaTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "gamma=" << gamma;
    printResult("RPP HIP BATCH GammaCorrection", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Exposure_Batched(const vector<Mat>& imgs, bool isColor, float exposureFactor,
                                        rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32f *exposureTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&exposureTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        exposureTensor[i] = exposureFactor;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;

        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_exposure(d_input, &srcDesc, d_output, &dstDesc, exposureTensor,
                                       roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "Exposure");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(exposureTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "factor=" << exposureFactor;
    printResult("RPP HIP BATCH Exposure", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Blend_Batched(const vector<Mat>& imgs, bool isColor, float alpha,
                                    rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0 || batchSize % 2 != 0) return;  // Need pairs of images

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.h * srcDesc.w * srcDesc.c * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input1, *d_input2, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input1, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_input2, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp32f *alphaTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&alphaTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        alphaTensor[i] = alpha;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer1 = new Rpp8u[totalBufferSize]();
    Rpp8u* h_tempBuffer2 = new Rpp8u[totalBufferSize]();
    int alignedWidth = srcDesc.w;

    for (int i = 0; i < batchSize; ++i) {
        int originalWidth = imgs[i].cols;
        int elementsPerRow = originalWidth * numChannels;
        Rpp8u* imgStart1 = h_tempBuffer1 + i * bufferSizePerImage;
        Rpp8u* imgStart2 = h_tempBuffer2 + i * bufferSizePerImage;
        int srcIdx = i % imgs.size();

        for (int row = 0; row < imgs[srcIdx].rows; ++row) {
            memcpy(imgStart1 + row * alignedWidth * numChannels,
                   imgs[srcIdx].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
            memcpy(imgStart2 + row * alignedWidth * numChannels,
                   imgs[srcIdx].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input1, h_tempBuffer1, totalBufferSize, hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(d_input2, h_tempBuffer2, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer1;
    delete[] h_tempBuffer2;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_blend(d_input1, d_input2, &srcDesc, d_output, &dstDesc,
                                    alphaTensor, roiTensor, RpptRoiType::XYWH,
                                    handle, RPP_HIP_BACKEND),
                        "Blend");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input1));
    CHECK_HIP_STATUS(hipFree(d_input2));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(alphaTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    ostringstream params;
    params << "alpha=" << alpha;
    printResult("RPP HIP BATCH Blend", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

// ===== MORPHOLOGICAL OPERATIONS =====

void benchmark_RPP_HIP_Erode_Batched(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                    rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    // REQUIRED: rppt_erode needs offsetInBytes >= 12 * (kernelSize / 2) for edge handling
    int requiredOffset = 12 * (kernelSize / 2);

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, requiredOffset);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);

    // Destination descriptor with zero offset (output doesn't need the padding)
    set_descriptor_dims_and_strides_local(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    dstDesc.layout = layout;
    dstDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&dstDesc);

    // Buffer size calculation: account for src offset for edge handling
    size_t srcBufferSizePerImage = srcDesc.h * srcDesc.w * srcDesc.c * sizeof(Rpp8u);
    size_t dstBufferSizePerImage = dstDesc.h * dstDesc.w * dstDesc.c * sizeof(Rpp8u);
    size_t totalSrcBufferSize = (srcBufferSizePerImage * batchSize) + requiredOffset;  // Add offset for edge padding
    size_t totalDstBufferSize = dstBufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalSrcBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalDstBufferSize));

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    // Allocate host buffer with offset for edge handling
    Rpp8u* h_tempBuffer = new Rpp8u[totalSrcBufferSize]();
    int alignedWidth = srcDesc.w;

    for (int i = 0; i < batchSize; ++i) {
        int originalWidth = imgs[i].cols;
        int elementsPerRow = originalWidth * numChannels;
        // Start copying after the offset to leave padding space for edge handling
        Rpp8u* imgStart = h_tempBuffer + requiredOffset + (i * srcBufferSizePerImage);

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(imgStart + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalSrcBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    Rpp32u kSize = static_cast<Rpp32u>(kernelSize);
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        // Pass roiTensor directly (first element of 256-element buffer per image)
        CHECK_RPP_STATUS(rppt_erode(d_input, &srcDesc, d_output, &dstDesc,
                                    kSize, roiTensor, RpptRoiType::XYWH,
                                    handle, RPP_HIP_BACKEND),
                        "Erode");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    
    ostringstream params;
    params << "kernel=" << kernelSize;
    printResult("RPP HIP BATCH Erode", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HIP_Dilate_Batched(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                     rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    // REQUIRED: rppt_dilate needs offsetInBytes >= 12 * (kernelSize / 2) for edge handling
    int requiredOffset = 12 * (kernelSize / 2);

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, requiredOffset);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);

    // Destination descriptor with zero offset (output doesn't need the padding)
    set_descriptor_dims_and_strides_local(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    dstDesc.layout = layout;
    dstDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&dstDesc);

    // Buffer size calculation: account for src offset for edge handling
    size_t srcBufferSizePerImage = srcDesc.h * srcDesc.w * srcDesc.c * sizeof(Rpp8u);
    size_t dstBufferSizePerImage = dstDesc.h * dstDesc.w * dstDesc.c * sizeof(Rpp8u);
    size_t totalSrcBufferSize = (srcBufferSizePerImage * batchSize) + requiredOffset;  // Add offset for edge padding
    size_t totalDstBufferSize = dstBufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalSrcBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalDstBufferSize));

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    // Allocate host buffer with offset for edge handling
    Rpp8u* h_tempBuffer = new Rpp8u[totalSrcBufferSize]();
    int alignedWidth = srcDesc.w;

    for (int i = 0; i < batchSize; ++i) {
        int originalWidth = imgs[i].cols;
        int elementsPerRow = originalWidth * numChannels;
        // Start copying after the offset to leave padding space for edge handling
        Rpp8u* imgStart = h_tempBuffer + requiredOffset + (i * srcBufferSizePerImage);

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(imgStart + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalSrcBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    Rpp32u kSize = static_cast<Rpp32u>(kernelSize);
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        // Pass roiTensor directly (first element of 256-element buffer per image)
        CHECK_RPP_STATUS(rppt_dilate(d_input, &srcDesc, d_output, &dstDesc,
                                     kSize, roiTensor, RpptRoiType::XYWH,
                                     handle, RPP_HIP_BACKEND),
                        "Dilate");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    
    ostringstream params;
    params << "kernel=" << kernelSize;
    printResult("RPP HIP BATCH Dilate", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}


// ===== BITWISE OPERATIONS =====

void benchmark_RPP_HIP_BitwiseAnd_Batched(const vector<Mat>& imgs, bool isColor,
                                         rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0 || batchSize % 2 != 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.h * srcDesc.w * srcDesc.c * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input1, *d_input2, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input1, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_input2, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i % imgs.size()].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i % imgs.size()].rows;
    }

    Rpp8u* h_tempBuffer1 = new Rpp8u[totalBufferSize]();
    Rpp8u* h_tempBuffer2 = new Rpp8u[totalBufferSize]();
    int alignedWidth = srcDesc.w;

    for (int i = 0; i < batchSize; ++i) {
        int srcIdx = i % imgs.size();
        int originalWidth = imgs[srcIdx].cols;
        int elementsPerRow = originalWidth * numChannels;
        Rpp8u* imgStart1 = h_tempBuffer1 + i * bufferSizePerImage;
        Rpp8u* imgStart2 = h_tempBuffer2 + i * bufferSizePerImage;

        for (int row = 0; row < imgs[srcIdx].rows; ++row) {
            memcpy(imgStart1 + row * alignedWidth * numChannels,
                   imgs[srcIdx].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
            memcpy(imgStart2 + row * alignedWidth * numChannels,
                   imgs[srcIdx].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input1, h_tempBuffer1, totalBufferSize, hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(d_input2, h_tempBuffer2, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer1;
    delete[] h_tempBuffer2;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_bitwise_and(d_input1, d_input2, &srcDesc, d_output, &dstDesc,
                                          roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "BitwiseAnd");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input1));
    CHECK_HIP_STATUS(hipFree(d_input2));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP BATCH BitwiseAnd", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), "");
}

void benchmark_RPP_HIP_BitwiseOr_Batched(const vector<Mat>& imgs, bool isColor,
                                        rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0 || batchSize % 2 != 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.h * srcDesc.w * srcDesc.c * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input1, *d_input2, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input1, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_input2, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i % imgs.size()].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i % imgs.size()].rows;
    }

    Rpp8u* h_tempBuffer1 = new Rpp8u[totalBufferSize]();
    Rpp8u* h_tempBuffer2 = new Rpp8u[totalBufferSize]();
    int alignedWidth = srcDesc.w;

    for (int i = 0; i < batchSize; ++i) {
        int srcIdx = i % imgs.size();
        int originalWidth = imgs[srcIdx].cols;
        int elementsPerRow = originalWidth * numChannels;
        Rpp8u* imgStart1 = h_tempBuffer1 + i * bufferSizePerImage;
        Rpp8u* imgStart2 = h_tempBuffer2 + i * bufferSizePerImage;

        for (int row = 0; row < imgs[srcIdx].rows; ++row) {
            memcpy(imgStart1 + row * alignedWidth * numChannels,
                   imgs[srcIdx].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
            memcpy(imgStart2 + row * alignedWidth * numChannels,
                   imgs[srcIdx].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input1, h_tempBuffer1, totalBufferSize, hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(d_input2, h_tempBuffer2, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer1;
    delete[] h_tempBuffer2;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_bitwise_or(d_input1, d_input2, &srcDesc, d_output, &dstDesc,
                                         roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "BitwiseOr");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input1));
    CHECK_HIP_STATUS(hipFree(d_input2));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP BATCH BitwiseOr", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), "");
}

void benchmark_RPP_HIP_BitwiseNot_Batched(const vector<Mat>& imgs, bool isColor,
                                         rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.h * srcDesc.w * srcDesc.c * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[totalBufferSize]();
    int alignedWidth = srcDesc.w;

    for (int i = 0; i < batchSize; ++i) {
        int originalWidth = imgs[i].cols;
        int elementsPerRow = originalWidth * numChannels;
        Rpp8u* imgStart = h_tempBuffer + i * bufferSizePerImage;

        for (int row = 0; row < imgs[i].rows; ++row) {
            memcpy(imgStart + row * alignedWidth * numChannels,
                   imgs[i].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_bitwise_not(d_input, &srcDesc, d_output, &dstDesc,
                                          roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "BitwiseNot");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP BATCH BitwiseNot", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), "");
}

void benchmark_RPP_HIP_BitwiseXor_Batched(const vector<Mat>& imgs, bool isColor,
                                         rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0 || batchSize % 2 != 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.h * srcDesc.w * srcDesc.c * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input1, *d_input2, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input1, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_input2, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i % imgs.size()].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i % imgs.size()].rows;
    }

    Rpp8u* h_tempBuffer1 = new Rpp8u[totalBufferSize]();
    Rpp8u* h_tempBuffer2 = new Rpp8u[totalBufferSize]();
    int alignedWidth = srcDesc.w;

    for (int i = 0; i < batchSize; ++i) {
        int srcIdx = i % imgs.size();
        int originalWidth = imgs[srcIdx].cols;
        int elementsPerRow = originalWidth * numChannels;
        Rpp8u* imgStart1 = h_tempBuffer1 + i * bufferSizePerImage;
        Rpp8u* imgStart2 = h_tempBuffer2 + i * bufferSizePerImage;

        for (int row = 0; row < imgs[srcIdx].rows; ++row) {
            memcpy(imgStart1 + row * alignedWidth * numChannels,
                   imgs[srcIdx].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
            memcpy(imgStart2 + row * alignedWidth * numChannels,
                   imgs[srcIdx].data + row * elementsPerRow,
                   elementsPerRow * sizeof(Rpp8u));
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input1, h_tempBuffer1, totalBufferSize, hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(d_input2, h_tempBuffer2, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer1;
    delete[] h_tempBuffer2;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_bitwise_xor(d_input1, d_input2, &srcDesc, d_output, &dstDesc,
                                          roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "BitwiseXor");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    CHECK_HIP_STATUS(hipFree(d_input1));
    CHECK_HIP_STATUS(hipFree(d_input2));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));

    printResult("RPP HIP BATCH BitwiseXor", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), "");
}


// ==================== Batched Dropout Augmentations ====================

void benchmark_RPP_HIP_Erase_Batched(const vector<Mat>& imgs, bool isColor, int numBoxes,
                                     rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    int maxBoxesPerImage = numBoxes;
    RpptRoiLtrb *anchorBoxInfoTensor;
    Rpp8u *colorsTensor;
    Rpp32u *numBoxesTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&anchorBoxInfoTensor, batchSize * maxBoxesPerImage * sizeof(RpptRoiLtrb)));
    CHECK_HIP_STATUS(hipHostMalloc(&colorsTensor, batchSize * maxBoxesPerImage * numChannels * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipHostMalloc(&numBoxesTensor, batchSize * sizeof(Rpp32u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        numBoxesTensor[i] = numBoxes;
        int width = imgs[i].cols;
        int height = imgs[i].rows;
        
        for (int box = 0; box < numBoxes; ++box) {
            int idx = i * maxBoxesPerImage + box;
            int boxWidth = width / (numBoxes + 1);
            int boxHeight = height / (numBoxes + 1);
            anchorBoxInfoTensor[idx].lt.x = (box + 1) * boxWidth / 2;
            anchorBoxInfoTensor[idx].lt.y = (box + 1) * boxHeight / 2;
            anchorBoxInfoTensor[idx].rb.x = anchorBoxInfoTensor[idx].lt.x + boxWidth;
            anchorBoxInfoTensor[idx].rb.y = anchorBoxInfoTensor[idx].lt.y + boxHeight;

            for (int c = 0; c < numChannels; ++c) {
                colorsTensor[(idx * numChannels) + c] = 0;
            }
        }

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;
        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_erase(d_input, &srcDesc, d_output, &dstDesc,
                                    anchorBoxInfoTensor, colorsTensor, numBoxesTensor,
                                    roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "Erase");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    ostringstream params;
    params << "numBoxes=" << numBoxes;
    printResult("RPP HIP BATCH Erase", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(anchorBoxInfoTensor));
    CHECK_HIP_STATUS(hipHostFree(colorsTensor));
    CHECK_HIP_STATUS(hipHostFree(numBoxesTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));
}

void benchmark_RPP_HIP_RandomErase_Batched(const vector<Mat>& imgs, bool isColor, int numBoxes,
                                          rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output, *d_noiseBuffer;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));
    
    // Noise buffer size (255x255 tiled noise pattern per image per channel)
    size_t noiseBufferSize = batchSize * 255 * 255 * numChannels * sizeof(Rpp8u);
    CHECK_HIP_STATUS(hipMalloc(&d_noiseBuffer, noiseBufferSize));

    RpptRoiLtrb *anchorBoxInfoTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&anchorBoxInfoTensor, batchSize * numBoxes * sizeof(RpptRoiLtrb)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    // Fill noise buffer with random pattern
    Rpp8u* h_noiseBuffer = new Rpp8u[noiseBufferSize]();
    for (size_t i = 0; i < noiseBufferSize; ++i) {
        h_noiseBuffer[i] = rand() % 256;
    }
    CHECK_HIP_STATUS(hipMemcpy(d_noiseBuffer, h_noiseBuffer, noiseBufferSize, hipMemcpyHostToDevice));
    delete[] h_noiseBuffer;

    for (int i = 0; i < batchSize; ++i) {
        int width = imgs[i].cols;
        int height = imgs[i].rows;

        for (int b = 0; b < numBoxes; ++b) {
            int idx = i * numBoxes + b;
            // Define erase boxes at different positions
            int boxWidth = width / 4;
            int boxHeight = height / 4;
            int xOffset = (b % 2) * (width / 2);
            int yOffset = (b / 2) * (height / 2);
            anchorBoxInfoTensor[idx].lt.x = xOffset + (width / 4 - boxWidth) / 2;
            anchorBoxInfoTensor[idx].lt.y = yOffset + (height / 4 - boxHeight) / 2;
            anchorBoxInfoTensor[idx].rb.x = anchorBoxInfoTensor[idx].lt.x + boxWidth;
            anchorBoxInfoTensor[idx].rb.y = anchorBoxInfoTensor[idx].lt.y + boxHeight;
        }

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;
        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_random_erase(d_input, &srcDesc, d_output, &dstDesc,
                                           anchorBoxInfoTensor, d_noiseBuffer,
                                           roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "RandomErase");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    ostringstream params;
    params << "numBoxes=" << numBoxes;
    printResult("RPP HIP BATCH RandomErase", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipFree(d_noiseBuffer));
    CHECK_HIP_STATUS(hipHostFree(anchorBoxInfoTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));
}

void benchmark_RPP_HIP_CoarseDropout_Batched(const vector<Mat>& imgs, bool isColor, Rpp32u maxBoxesPerImage,
                                             rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    RpptRoiLtrb *anchorBoxInfoTensor;
    Rpp32u *numBoxesTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&anchorBoxInfoTensor, batchSize * maxBoxesPerImage * sizeof(RpptRoiLtrb)));
    CHECK_HIP_STATUS(hipHostMalloc(&numBoxesTensor, batchSize * sizeof(Rpp32u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    // Initialize dropout boxes - same logic as single-image implementation
    srand(DROPOUT_FIXED_SEED);
    for (int i = 0; i < batchSize; ++i) {
        int h = imgs[i].rows;
        int w = imgs[i].cols;

        // Random number of boxes (2 to maxBoxesPerImage) - matches single-image implementation
        Rpp32u numBoxes = 2 + (rand() % (maxBoxesPerImage - 1));
        numBoxesTensor[i] = numBoxes;

        for (Rpp32u j = 0; j < numBoxes; ++j) {
            int idx = i * maxBoxesPerImage + j;

            // Random box sizes (30 to 110 pixels) - matches single-image implementation
            int box_w = 30 + (rand() % 80);
            int box_h = 30 + (rand() % 80);
            int x = rand() % max(1, w - box_w);
            int y = rand() % max(1, h - box_h);

            anchorBoxInfoTensor[idx].lt.x = x;
            anchorBoxInfoTensor[idx].lt.y = y;
            anchorBoxInfoTensor[idx].rb.x = x + box_w;
            anchorBoxInfoTensor[idx].rb.y = y + box_h;
        }

        // Fill remaining boxes with zeros - matches single-image implementation
        for (Rpp32u j = numBoxes; j < maxBoxesPerImage; ++j) {
            int idx = i * maxBoxesPerImage + j;
            anchorBoxInfoTensor[idx].lt.x = 0;
            anchorBoxInfoTensor[idx].lt.y = 0;
            anchorBoxInfoTensor[idx].rb.x = 0;
            anchorBoxInfoTensor[idx].rb.y = 0;
        }

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;
        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_coarse_dropout(d_input, &srcDesc, d_output, &dstDesc,
                                             anchorBoxInfoTensor, numBoxesTensor, maxBoxesPerImage,
                                             roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "CoarseDropout");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    ostringstream params;
    params << "maxBoxes=" << maxBoxesPerImage;
    printResult("RPP HIP BATCH CoarseDropout", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(anchorBoxInfoTensor));
    CHECK_HIP_STATUS(hipHostFree(numBoxesTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));
}

void benchmark_RPP_HIP_GridDropout_Batched(const vector<Mat>& imgs, bool isColor, int tileWidth, int tileHeight,
                                           rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    // Calculate boxes per image based on grid
    int boxesInEachImage = (maxWidth / tileWidth) * (maxHeight / tileHeight);
    boxesInEachImage = max(boxesInEachImage, 16);

    RpptRoiLtrb *anchorBoxInfoTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&anchorBoxInfoTensor, batchSize * boxesInEachImage * sizeof(RpptRoiLtrb)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    Rpp32u maxHoleW = tileWidth / 2;
    Rpp32u maxHoleH = tileHeight / 2;

    for (int i = 0; i < batchSize; ++i) {
        int width = imgs[i].cols;
        int height = imgs[i].rows;
        
        // Create grid of boxes
        int boxIdx = 0;
        for (int y = 0; y < height && boxIdx < boxesInEachImage; y += tileHeight) {
            for (int x = 0; x < width && boxIdx < boxesInEachImage; x += tileWidth) {
                int idx = i * boxesInEachImage + boxIdx;
                anchorBoxInfoTensor[idx].lt.x = x;
                anchorBoxInfoTensor[idx].lt.y = y;
                anchorBoxInfoTensor[idx].rb.x = min(x + tileWidth, width);
                anchorBoxInfoTensor[idx].rb.y = min(y + tileHeight, height);
                boxIdx++;
            }
        }

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;
        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_grid_dropout(d_input, &srcDesc, d_output, &dstDesc,
                                           anchorBoxInfoTensor, boxesInEachImage, maxHoleW, maxHoleH,
                                           roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "GridDropout");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    ostringstream params;
    params << "tileWidth=" << tileWidth << ", tileHeight=" << tileHeight;
    printResult("RPP HIP BATCH GridDropout", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(anchorBoxInfoTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));
}

void benchmark_RPP_HIP_Gridmask_Batched(const vector<Mat>& imgs, bool isColor, int tileWidth, float ratio,
                                       rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    // Use actual dimensions without padding to avoid mismatch with ROI dimensions
    srcDesc.numDims = 4;
    srcDesc.offsetInBytes = 0;
    srcDesc.n = batchSize;
    srcDesc.h = maxHeight;
    srcDesc.w = maxWidth;  // Use actual width, not padded
    srcDesc.c = numChannels;
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    // Validate tileWidth constraint: must be <= min(width, height)
    Rpp32u minDim = min(maxWidth, maxHeight);
    Rpp32u tileWidthVal = min((Rpp32u)tileWidth, minDim);
    Rpp32f gridRatio = ratio;
    Rpp32f gridAngle = 0.5f;  // Grid rotation angle in radians
    RpptUintVector2D translateVector;
    translateVector.x = 0;
    translateVector.y = 0;

    // Allocate ROI tensor in device memory (not pinned host memory)
    RpptROI *d_roiTensor;
    CHECK_HIP_STATUS(hipMalloc(&d_roiTensor, batchSize * sizeof(RpptROI)));

    // Create temporary host buffer
    RpptROI *h_roiTensor = new RpptROI[batchSize];
    for (int i = 0; i < batchSize; ++i) {
        h_roiTensor[i].xywhROI.xy.x = 0;
        h_roiTensor[i].xywhROI.xy.y = 0;
        h_roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        h_roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    // Copy ROI data to device
    CHECK_HIP_STATUS(hipMemcpy(d_roiTensor, h_roiTensor, batchSize * sizeof(RpptROI), hipMemcpyHostToDevice));

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;
        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    // CRITICAL FIX: Clear any pending HIP errors from previous operations before calling rppt_gridmask
    // The HIP_CHECK_LAUNCH_RETURN() macro in gridmask.cpp picks up stale errors from earlier HIP calls
    // Synchronize to ensure all previous operations complete, then clear the error state
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    (void)hipGetLastError();

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_gridmask(d_input, &srcDesc, d_output, &dstDesc,
                                       tileWidthVal, gridRatio, gridAngle, translateVector,
                                       d_roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "Gridmask");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    ostringstream params;
    params << "tileWidth=" << tileWidth << ", ratio=" << ratio;
    printResult("RPP HIP BATCH Gridmask", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipFree(d_roiTensor));
    delete[] h_roiTensor;
}

void benchmark_RPP_HIP_ChannelDropout_Batched(const vector<Mat>& imgs, bool isColor, float dropoutProb,
                                              rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0 || !isColor) return;

    int numChannels = 3;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = RpptLayout::NHWC;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    Rpp8u *dropoutTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&dropoutTensor, batchSize * numChannels * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        // Randomly dropout channels based on probability
        for (int c = 0; c < numChannels; ++c) {
            float randVal = (float)rand() / RAND_MAX;
            dropoutTensor[i * numChannels + c] = (randVal < dropoutProb) ? 1 : 0;
        }

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;
        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_channel_dropout(d_input, &srcDesc, d_output, &dstDesc,
                                              dropoutTensor,
                                              roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "ChannelDropout");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    ostringstream params;
    params << "dropoutProb=" << dropoutProb;
    printResult("RPP HIP BATCH ChannelDropout", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(dropoutTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));
}

void benchmark_RPP_HIP_CutoutDropout_Batched(const vector<Mat>& imgs, bool isColor, Rpp32u numBoxes,
                                             rppHandle_t handle, hipStream_t stream) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    RpptLayout layout = (isColor && imgs[0].channels() == 3) ? RpptLayout::NHWC : RpptLayout::NCHW;

    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, 0);
    srcDesc.layout = layout;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    dstDesc = srcDesc;

    size_t bufferSizePerImage = srcDesc.strides.nStride * sizeof(Rpp8u);
    size_t totalBufferSize = bufferSizePerImage * batchSize;

    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, totalBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, totalBufferSize));

    RpptRoiLtrb *anchorBoxInfoTensor;
    Rpp8u *colorsTensor;
    Rpp32u *numBoxesTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&anchorBoxInfoTensor, batchSize * numBoxes * sizeof(RpptRoiLtrb)));
    CHECK_HIP_STATUS(hipHostMalloc(&colorsTensor, batchSize * numBoxes * numChannels * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipHostMalloc(&numBoxesTensor, batchSize * sizeof(Rpp32u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    for (int i = 0; i < batchSize; ++i) {
        numBoxesTensor[i] = numBoxes;
        int width = imgs[i].cols;
        int height = imgs[i].rows;
        
        // Create cutout boxes (random locations)
        for (Rpp32u box = 0; box < numBoxes; ++box) {
            int idx = i * numBoxes + box;
            int boxW = width / (numBoxes + 2);
            int boxH = height / (numBoxes + 2);
            int x = (rand() % (width - boxW));
            int y = (rand() % (height - boxH));
            
            anchorBoxInfoTensor[idx].lt.x = x;
            anchorBoxInfoTensor[idx].lt.y = y;
            anchorBoxInfoTensor[idx].rb.x = x + boxW;
            anchorBoxInfoTensor[idx].rb.y = y + boxH;

            // Fill with gray (128)
            for (int c = 0; c < numChannels; ++c) {
                colorsTensor[(idx * numChannels) + c] = 128;
            }
        }

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp8u* h_tempBuffer = new Rpp8u[srcDesc.strides.nStride * batchSize]();

    for (int i = 0; i < batchSize; ++i) {
        Rpp8u* imgStart = h_tempBuffer + i * srcDesc.strides.nStride;
        for (int row = 0; row < imgs[i].rows; ++row) {
            for (int col = 0; col < imgs[i].cols; ++col) {
                for (int ch = 0; ch < numChannels; ++ch) {
                    int dstIdx;
                    if (layout == RpptLayout::NHWC) {
                        dstIdx = row * srcDesc.strides.hStride + col * srcDesc.strides.wStride + ch * srcDesc.strides.cStride;
                    } else {
                        dstIdx = ch * srcDesc.strides.cStride + row * srcDesc.strides.hStride + col * srcDesc.strides.wStride;
                    }
                    int srcIdx = row * imgs[i].cols * numChannels + col * numChannels + ch;
                    imgStart[dstIdx] = imgs[i].data[srcIdx];
                }
            }
        }
    }

    CHECK_HIP_STATUS(hipMemcpy(d_input, h_tempBuffer, totalBufferSize, hipMemcpyHostToDevice));
    delete[] h_tempBuffer;

    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_cutout_dropout(d_input, &srcDesc, d_output, &dstDesc,
                                             anchorBoxInfoTensor, colorsTensor, numBoxesTensor,
                                             roiTensor, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "CutoutDropout");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    ostringstream params;
    params << "numBoxes=" << numBoxes;
    printResult("RPP HIP BATCH CutoutDropout", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(anchorBoxInfoTensor));
    CHECK_HIP_STATUS(hipHostFree(colorsTensor));
    CHECK_HIP_STATUS(hipHostFree(numBoxesTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));
}

// Copy Batched (HIP)
void benchmark_RPP_HIP_Copy_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int batchSize = imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Set up descriptors FIRST to get correct buffer size with padding
    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, height, width, channels, 0);
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    set_descriptor_dims_and_strides_local(&dstDesc, batchSize, height, width, channels, 0);
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.dataType = RpptDataType::U8;

    // Calculate buffer size from descriptor (includes padding)
    size_t bufferSize = srcDesc.n * srcDesc.strides.nStride;

    // Allocate HOST buffer for input with padded size
    Rpp8u* h_input = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));
    // Copy image data with proper stride handling
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* dstRow = h_input + i * srcDesc.strides.nStride;
        for (int h = 0; h < height; h++) {
            memcpy(dstRow + h * srcDesc.strides.hStride,
                   imgs[i].data + h * width * channels,
                   width * channels);
        }
    }

    // Allocate HIP device buffers
    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, bufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, bufferSize));
    CHECK_HIP_STATUS(hipMemcpy(d_input, h_input, bufferSize, hipMemcpyHostToDevice));

    // Benchmark loop
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_copy((RppPtr_t)d_input, &srcDesc, (RppPtr_t)d_output, &dstDesc,
                                   handle, RPP_HIP_BACKEND),
                        "Copy");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    ostringstream params;
    params << "layout=NHWC";
    printResult("RPP HIP BATCH Copy", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    free(h_input);
}

// Slice Batched (HIP)
void benchmark_RPP_HIP_Slice_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int batchSize = imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;
    size_t bufferSize = batchSize * height * width * channels;

    // Allocate HOST buffer for input
    Rpp8u* h_input = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));
    for (int i = 0; i < batchSize; i++) {
        memcpy(h_input + i * height * width * channels, imgs[i].data, height * width * channels);
    }

    // Allocate HIP device buffers
    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, bufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, bufferSize));
    CHECK_HIP_STATUS(hipMemcpy(d_input, h_input, bufferSize, hipMemcpyHostToDevice));

    // Set up generic descriptors
    RpptGenericDesc srcGenericDesc, dstGenericDesc;
    srcGenericDesc.numDims = 4;
    srcGenericDesc.offsetInBytes = 0;
    srcGenericDesc.dataType = RpptDataType::U8;
    srcGenericDesc.dims[0] = batchSize;
    srcGenericDesc.dims[1] = height;
    srcGenericDesc.dims[2] = width;
    srcGenericDesc.dims[3] = channels;
    srcGenericDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NHWC;
    for (int i = 0; i < 4; i++) srcGenericDesc.strides[i] = 1;
    srcGenericDesc.strides[2] = channels;
    srcGenericDesc.strides[1] = width * channels;
    srcGenericDesc.strides[0] = height * width * channels;
    dstGenericDesc = srcGenericDesc;

    // Anchor and shape tensors in pinned memory
    Rpp32s *anchorTensor, *shapeTensor;
    Rpp32u *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&anchorTensor, batchSize * 4 * sizeof(Rpp32s)));
    CHECK_HIP_STATUS(hipHostMalloc(&shapeTensor, batchSize * 4 * sizeof(Rpp32s)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * 4 * sizeof(Rpp32u)));

    for (int i = 0; i < batchSize; i++) {
        anchorTensor[i * 4 + 0] = 0;
        anchorTensor[i * 4 + 1] = 0;
        anchorTensor[i * 4 + 2] = 0;
        anchorTensor[i * 4 + 3] = 0;
        shapeTensor[i * 4 + 0] = 1;
        shapeTensor[i * 4 + 1] = height;
        shapeTensor[i * 4 + 2] = width;
        shapeTensor[i * 4 + 3] = channels;
        roiTensor[i * 4 + 0] = 1;
        roiTensor[i * 4 + 1] = height;
        roiTensor[i * 4 + 2] = width;
        roiTensor[i * 4 + 3] = channels;
    }

    Rpp8u fillValue = 0;

    // Benchmark loop
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_slice((RppPtr_t)d_input, &srcGenericDesc, (RppPtr_t)d_output, &dstGenericDesc,
                                    anchorTensor, shapeTensor, (RppPtr_t)&fillValue, false, roiTensor,
                                    handle, RPP_HIP_BACKEND),
                        "Slice");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    ostringstream params;
    params << "slice=center_50%";
    printResult("RPP HIP BATCH Slice", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(anchorTensor));
    CHECK_HIP_STATUS(hipHostFree(shapeTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));
    free(h_input);
}

// ChannelPermute Batched (HIP)
void benchmark_RPP_HIP_ChannelPermute_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    // ChannelPermute requires 3 channels
    if (!isColor) {
        cout << "RPP HIP ChannelPermute (Batched) - Skipped (requires RGB images)" << endl;
        return;
    }

    int batchSize = imgs.size();
    int channels = 3;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Set up descriptors FIRST
    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, height, width, channels, 0);
    srcDesc.layout = RpptLayout::NHWC;
    srcDesc.dataType = RpptDataType::U8;
    set_descriptor_dims_and_strides_local(&dstDesc, batchSize, height, width, channels, 0);
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.dataType = RpptDataType::U8;

    // Calculate buffer size from descriptor
    size_t bufferSize = srcDesc.n * srcDesc.strides.nStride;

    // Allocate HOST buffer with padding
    Rpp8u* h_input = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* dstRow = h_input + i * srcDesc.strides.nStride;
        for (int h = 0; h < height; h++) {
            memcpy(dstRow + h * srcDesc.strides.hStride,
                   imgs[i].data + h * width * channels,
                   width * channels);
        }
    }

    // Allocate HIP device buffers
    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, bufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, bufferSize));
    CHECK_HIP_STATUS(hipMemcpy(d_input, h_input, bufferSize, hipMemcpyHostToDevice));

    // Permutation tensor: BGR to RGB (2,1,0) in pinned memory
    Rpp32u *permutationTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&permutationTensor, batchSize * channels * sizeof(Rpp32u)));
    for (int i = 0; i < batchSize; i++) {
        if (isColor) {
            permutationTensor[i * 3 + 0] = 2;
            permutationTensor[i * 3 + 1] = 1;
            permutationTensor[i * 3 + 2] = 0;
        } else {
            permutationTensor[i] = 0;
        }
    }

    // Benchmark loop
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_channel_permute((RppPtr_t)d_input, &srcDesc, (RppPtr_t)d_output, &dstDesc,
                                              permutationTensor, handle, RPP_HIP_BACKEND),
                        "ChannelPermute");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    ostringstream params;
    if (isColor) params << "permutation=BGR->RGB";
    printResult("RPP HIP BATCH ChannelPermute", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(permutationTensor));
    free(h_input);
}

// Transpose Batched (HIP)
void benchmark_RPP_HIP_Transpose_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int batchSize = imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;
    size_t inputBufferSize = batchSize * height * width * channels;
    size_t outputBufferSize = batchSize * width * height * channels;

    // Allocate HOST buffer for input
    Rpp8u* h_input = (Rpp8u*)calloc(inputBufferSize, sizeof(Rpp8u));
    for (int i = 0; i < batchSize; i++) {
        memcpy(h_input + i * height * width * channels, imgs[i].data, height * width * channels);
    }

    // Allocate HIP device buffers
    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, inputBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, outputBufferSize));
    CHECK_HIP_STATUS(hipMemcpy(d_input, h_input, inputBufferSize, hipMemcpyHostToDevice));

    // Set up generic descriptors in pinned memory (like reference implementation)
    RpptGenericDesc *srcGenericDesc, *dstGenericDesc;
    CHECK_HIP_STATUS(hipHostMalloc(&srcGenericDesc, sizeof(RpptGenericDesc)));
    CHECK_HIP_STATUS(hipHostMalloc(&dstGenericDesc, sizeof(RpptGenericDesc)));

    srcGenericDesc->numDims = 4;
    srcGenericDesc->offsetInBytes = 0;
    srcGenericDesc->dataType = RpptDataType::U8;
    srcGenericDesc->dims[0] = batchSize;
    srcGenericDesc->dims[1] = height;
    srcGenericDesc->dims[2] = width;
    srcGenericDesc->dims[3] = channels;
    srcGenericDesc->layout = isColor ? RpptLayout::NHWC : RpptLayout::NHWC;
    for (int i = 0; i < 4; i++) srcGenericDesc->strides[i] = 1;
    srcGenericDesc->strides[2] = channels;
    srcGenericDesc->strides[1] = width * channels;
    srcGenericDesc->strides[0] = height * width * channels;

    // Destination has swapped H and W
    dstGenericDesc->numDims = 4;
    dstGenericDesc->offsetInBytes = 0;
    dstGenericDesc->dataType = RpptDataType::U8;
    dstGenericDesc->dims[0] = batchSize;
    dstGenericDesc->dims[1] = width;
    dstGenericDesc->dims[2] = height;
    dstGenericDesc->dims[3] = channels;
    dstGenericDesc->layout = isColor ? RpptLayout::NHWC : RpptLayout::NHWC;
    for (int i = 0; i < 4; i++) dstGenericDesc->strides[i] = 1;
    dstGenericDesc->strides[2] = channels;
    dstGenericDesc->strides[1] = height * channels;
    dstGenericDesc->strides[0] = width * height * channels;

    // BUG FIX: The rppt_transpose HIP kernel processes 3D sub-tensors (HWC) for each image in the batch,
    // excluding the batch dimension (N). Therefore:
    // 1. permTensor must have (numDims-1) = 3 elements, not 4
    // 2. roiTensor must have (numDims-1)*2 = 6 values per image, not 8
    //
    // This matches the HOST implementation behavior and reference test setup.
    Rpp32u *permTensor, *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&permTensor, 3 * sizeof(Rpp32u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * 6 * sizeof(Rpp32u)));

    // Permutation for HWC sub-tensor: swap H and W dimensions
    // Input NHWC [batch, H, W, C] -> Output NWHC [batch, W, H, C]
    // For the 3D HWC sub-tensor, this is WHC, so permutation is [1, 0, 2]:
    //   output_dim_0 (W) <- input_dim_1 (W)
    //   output_dim_1 (H) <- input_dim_0 (H)
    //   output_dim_2 (C) <- input_dim_2 (C)
    permTensor[0] = 1;
    permTensor[1] = 0;
    permTensor[2] = 2;

    // ROI tensor: (numDims-1)*2 = 6 values per image
    // Format: [h_start, w_start, c_start, h_size, w_size, c_size]
    for (int i = 0; i < batchSize; i++) {
        int idx = i * 6;
        roiTensor[idx + 0] = 0;         // h_start
        roiTensor[idx + 1] = 0;         // w_start
        roiTensor[idx + 2] = 0;         // c_start
        roiTensor[idx + 3] = height;    // h_size
        roiTensor[idx + 4] = width;     // w_size
        roiTensor[idx + 5] = channels;  // c_size
    }

    // Benchmark loop
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_transpose((RppPtr_t)d_input, srcGenericDesc, (RppPtr_t)d_output, dstGenericDesc,
                                        permTensor, roiTensor, handle, RPP_HIP_BACKEND),
                        "Transpose");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    ostringstream params;
    params << "permutation=0,2,1,3";
    printResult("RPP HIP BATCH Transpose", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(srcGenericDesc));
    CHECK_HIP_STATUS(hipHostFree(dstGenericDesc));
    CHECK_HIP_STATUS(hipHostFree(permTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));
    free(h_input);
}

// LUT Batched (HIP)
void benchmark_RPP_HIP_LUT_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int batchSize = imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;
    size_t inputBufferSize = batchSize * height * width * channels;

    // Allocate HOST buffer for input
    Rpp8u* h_input = (Rpp8u*)calloc(inputBufferSize, sizeof(Rpp8u));
    for (int i = 0; i < batchSize; i++) {
        memcpy(h_input + i * height * width * channels, imgs[i].data, height * width * channels);
    }

    // Allocate HIP device buffers
    Rpp8u *d_input, *d_output, *d_lut;
    CHECK_HIP_STATUS(hipMalloc(&d_input, inputBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, inputBufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_lut, 256 * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipMemcpy(d_input, h_input, inputBufferSize, hipMemcpyHostToDevice));

    // Create and copy LUT
    Rpp8u h_lut[256];
    for (int i = 0; i < 256; i++) h_lut[i] = 255 - i;
    CHECK_HIP_STATUS(hipMemcpy(d_lut, h_lut, 256 * sizeof(Rpp8u), hipMemcpyHostToDevice));

    // Set up descriptors in pinned memory
    RpptDesc *srcDesc, *dstDesc;
    CHECK_HIP_STATUS(hipHostMalloc(&srcDesc, sizeof(RpptDesc)));
    CHECK_HIP_STATUS(hipHostMalloc(&dstDesc, sizeof(RpptDesc)));

    srcDesc->layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc->dataType = RpptDataType::U8;
    set_descriptor_dims_and_strides(srcDesc, batchSize, height, width, channels, 0);

    dstDesc->layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc->dataType = RpptDataType::U8;
    set_descriptor_dims_and_strides(dstDesc, batchSize, height, width, channels, 0);

    // Setup ROI in pinned memory
    RpptROI* roiTensorPtrSrc;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensorPtrSrc, batchSize * sizeof(RpptROI)));
    for (int i = 0; i < batchSize; i++) {
        roiTensorPtrSrc[i].xywhROI.xy.x = 0;
        roiTensorPtrSrc[i].xywhROI.xy.y = 0;
        roiTensorPtrSrc[i].xywhROI.roiWidth = width;
        roiTensorPtrSrc[i].xywhROI.roiHeight = height;
    }

    // Benchmark loop
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_lut((RppPtr_t)d_input, srcDesc, (RppPtr_t)d_output, dstDesc,
                                  (RppPtr_t)d_lut, roiTensorPtrSrc, RpptRoiType::XYWH,
                                  handle, RPP_HIP_BACKEND),
                        "LUT");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    ostringstream params;
    params << "lut=inverse";
    printResult("RPP HIP BATCH LUT", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipFree(d_lut));
    CHECK_HIP_STATUS(hipHostFree(srcDesc));
    CHECK_HIP_STATUS(hipHostFree(dstDesc));
    CHECK_HIP_STATUS(hipHostFree(roiTensorPtrSrc));
    free(h_input);
}

// Magnitude Batched (HIP)
void benchmark_RPP_HIP_Magnitude_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int batchSize = imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;
    size_t bufferSize = batchSize * height * width * channels;

    // Allocate HOST buffers
    Rpp8u* h_input = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));
    for (int i = 0; i < batchSize; i++) {
        memcpy(h_input + i * height * width * channels, imgs[i].data, height * width * channels);
    }

    // Allocate HIP device buffers
    Rpp8u *d_input1, *d_input2, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input1, bufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_input2, bufferSize));
    CHECK_HIP_STATUS(hipMalloc(&d_output, bufferSize));
    CHECK_HIP_STATUS(hipMemcpy(d_input1, h_input, bufferSize, hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(d_input2, h_input, bufferSize, hipMemcpyHostToDevice));

    // Set up descriptors in pinned memory
    RpptDesc *srcDesc, *dstDesc;
    CHECK_HIP_STATUS(hipHostMalloc(&srcDesc, sizeof(RpptDesc)));
    CHECK_HIP_STATUS(hipHostMalloc(&dstDesc, sizeof(RpptDesc)));

    srcDesc->layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc->dataType = RpptDataType::U8;
    set_descriptor_dims_and_strides(srcDesc, batchSize, height, width, channels, 0);

    dstDesc->layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc->dataType = RpptDataType::U8;
    set_descriptor_dims_and_strides(dstDesc, batchSize, height, width, channels, 0);

    // Setup ROI in pinned memory
    RpptROI* roiTensorPtrSrc;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensorPtrSrc, batchSize * sizeof(RpptROI)));
    for (int i = 0; i < batchSize; i++) {
        roiTensorPtrSrc[i].xywhROI.xy.x = 0;
        roiTensorPtrSrc[i].xywhROI.xy.y = 0;
        roiTensorPtrSrc[i].xywhROI.roiWidth = width;
        roiTensorPtrSrc[i].xywhROI.roiHeight = height;
    }

    // Benchmark loop
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_magnitude((RppPtr_t)d_input1, (RppPtr_t)d_input2, srcDesc,
                                        (RppPtr_t)d_output, dstDesc,
                                        roiTensorPtrSrc, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "Magnitude");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    ostringstream params;
    params << "inputs=2";
    printResult("RPP HIP BATCH Magnitude", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    CHECK_HIP_STATUS(hipFree(d_input1));
    CHECK_HIP_STATUS(hipFree(d_input2));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(srcDesc));
    CHECK_HIP_STATUS(hipHostFree(dstDesc));
    CHECK_HIP_STATUS(hipHostFree(roiTensorPtrSrc));
    free(h_input);
}

// FusedMultiplyAddScalar Batched (HIP)
void benchmark_RPP_HIP_FusedMultiplyAddScalar_Batched(const vector<Mat>& imgs, bool isColor,
                                                       float mulVal, float addVal,
                                                       rppHandle_t handle, hipStream_t stream) {
    int batchSize = imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;
    size_t bufferSize = batchSize * height * width * channels;

    // Allocate HOST buffer and convert to F32
    Rpp32f* h_inputF32 = (Rpp32f*)calloc(bufferSize, sizeof(Rpp32f));
    for (int i = 0; i < batchSize; i++) {
        for (size_t j = 0; j < height * width * channels; j++) {
            h_inputF32[i * height * width * channels + j] = (Rpp32f)imgs[i].data[j];
        }
    }

    // Allocate HIP device buffers (F32 required)
    Rpp32f *d_inputF32, *d_outputF32;
    Rpp32f *d_mulTensor, *d_addTensor;
    CHECK_HIP_STATUS(hipMalloc(&d_inputF32, bufferSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipMalloc(&d_outputF32, bufferSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipMalloc(&d_mulTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipMalloc(&d_addTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipMemcpy(d_inputF32, h_inputF32, bufferSize * sizeof(Rpp32f), hipMemcpyHostToDevice));

    // Setup mul and add tensors in pinned memory
    Rpp32f* mulTensor;
    Rpp32f* addTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&mulTensor, batchSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&addTensor, batchSize * sizeof(Rpp32f)));
    for (int i = 0; i < batchSize; i++) {
        mulTensor[i] = mulVal;
        addTensor[i] = addVal;
    }

    // Set up 5D generic descriptors in pinned memory (NDHWC with depth=1 for 2D images)
    RpptGenericDesc *srcDesc, *dstDesc;
    CHECK_HIP_STATUS(hipHostMalloc(&srcDesc, sizeof(RpptGenericDesc)));
    CHECK_HIP_STATUS(hipHostMalloc(&dstDesc, sizeof(RpptGenericDesc)));

    int layoutType = isColor ? 1 : 0;  // 1=NDHWC, 0=NCDHW
    set_generic_descriptor(srcDesc, batchSize, width, height, 1, channels, 0, layoutType);
    set_generic_descriptor(dstDesc, batchSize, width, height, 1, channels, 0, layoutType);

    // Setup ROI3D in pinned memory
    RpptROI3D* roiTensorPtrSrc;
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensorPtrSrc, batchSize * sizeof(RpptROI3D)));
    for (int i = 0; i < batchSize; i++) {
        roiTensorPtrSrc[i].xyzwhdROI.xyz.x = 0;
        roiTensorPtrSrc[i].xyzwhdROI.xyz.y = 0;
        roiTensorPtrSrc[i].xyzwhdROI.xyz.z = 0;
        roiTensorPtrSrc[i].xyzwhdROI.roiWidth = width;
        roiTensorPtrSrc[i].xyzwhdROI.roiHeight = height;
        roiTensorPtrSrc[i].xyzwhdROI.roiDepth = 1;
    }

    // Benchmark loop
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_fused_multiply_add_scalar((RppPtr_t)d_inputF32, srcDesc,
                                                         (RppPtr_t)d_outputF32, dstDesc,
                                                         mulTensor, addTensor,
                                                         roiTensorPtrSrc, RpptRoi3DType::XYZWHD,
                                                         handle, RPP_HIP_BACKEND),
                        "FusedMultiplyAddScalar");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    ostringstream params;
    params << "mul=" << mulVal << ", add=" << addVal;
    printResult("RPP HIP BATCH FusedMultiplyAddScalar", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    CHECK_HIP_STATUS(hipFree(d_inputF32));
    CHECK_HIP_STATUS(hipFree(d_outputF32));
    CHECK_HIP_STATUS(hipFree(d_mulTensor));
    CHECK_HIP_STATUS(hipFree(d_addTensor));
    CHECK_HIP_STATUS(hipHostFree(srcDesc));
    CHECK_HIP_STATUS(hipHostFree(dstDesc));
    CHECK_HIP_STATUS(hipHostFree(roiTensorPtrSrc));
    CHECK_HIP_STATUS(hipHostFree(mulTensor));
    CHECK_HIP_STATUS(hipHostFree(addTensor));
    free(h_inputF32);
}

// Remap Batched (HIP)
void benchmark_RPP_HIP_Remap_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int batchSize = imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Setup descriptors (on host for init_remap)
    RpptDesc srcDesc, dstDesc, tableDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, height, width, channels, 0);
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    set_descriptor_dims_and_strides_local(&dstDesc, batchSize, height, width, channels, 0);
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&dstDesc);

    // Calculate sizes using strides (accounts for proper layout)
    size_t bufferSize = (size_t)srcDesc.n * srcDesc.strides.nStride;
    size_t tableSize = (size_t)srcDesc.n * srcDesc.h * srcDesc.w;

    // Allocate HOST buffer
    Rpp8u* h_input = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgStart = h_input + i * srcDesc.strides.nStride;
        for (int h = 0; h < height; h++) {
            Rpp8u* rowStart = imgStart + h * srcDesc.strides.hStride;
            const Rpp8u* srcRow = imgs[i].data + h * width * channels;
            memcpy(rowStart, srcRow, width * channels * sizeof(Rpp8u));
        }
    }

    // Setup ROI (on host for init_remap)
    RpptROI* roiTensorPtrSrc = (RpptROI*)calloc(batchSize, sizeof(RpptROI));
    for (int i = 0; i < batchSize; i++) {
        roiTensorPtrSrc[i].xywhROI.xy.x = 0;
        roiTensorPtrSrc[i].xywhROI.xy.y = 0;
        roiTensorPtrSrc[i].xywhROI.roiWidth = width;
        roiTensorPtrSrc[i].xywhROI.roiHeight = height;
    }

    // Allocate and initialize remap tables on host
    Rpp32f* h_rowRemapTable = (Rpp32f*)calloc(tableSize, sizeof(Rpp32f));
    Rpp32f* h_colRemapTable = (Rpp32f*)calloc(tableSize, sizeof(Rpp32f));
    tableDesc = srcDesc;
    init_remap(&tableDesc, &srcDesc, roiTensorPtrSrc, h_rowRemapTable, h_colRemapTable);

    // Allocate HIP device buffers
    Rpp8u *d_input, *d_output;
    Rpp32f *d_rowRemapTable, *d_colRemapTable;
    CHECK_HIP_STATUS(hipMalloc(&d_input, bufferSize * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipMalloc(&d_output, bufferSize * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipMalloc(&d_rowRemapTable, tableSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipMalloc(&d_colRemapTable, tableSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipMemcpy(d_input, h_input, bufferSize * sizeof(Rpp8u), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(d_rowRemapTable, h_rowRemapTable, tableSize * sizeof(Rpp32f), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(d_colRemapTable, h_colRemapTable, tableSize * sizeof(Rpp32f), hipMemcpyHostToDevice));

    // Copy descriptors and ROI to pinned memory
    RpptDesc *d_srcDesc, *d_dstDesc, *d_tableDesc;
    RpptROI* d_roiTensorPtrSrc;
    CHECK_HIP_STATUS(hipHostMalloc(&d_srcDesc, sizeof(RpptDesc)));
    CHECK_HIP_STATUS(hipHostMalloc(&d_dstDesc, sizeof(RpptDesc)));
    CHECK_HIP_STATUS(hipHostMalloc(&d_tableDesc, sizeof(RpptDesc)));
    CHECK_HIP_STATUS(hipHostMalloc(&d_roiTensorPtrSrc, batchSize * sizeof(RpptROI)));
    *d_srcDesc = srcDesc;
    *d_dstDesc = dstDesc;
    *d_tableDesc = tableDesc;
    memcpy(d_roiTensorPtrSrc, roiTensorPtrSrc, batchSize * sizeof(RpptROI));

    RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;

    // Benchmark loop
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_remap((RppPtr_t)d_input, d_srcDesc, (RppPtr_t)d_output, d_dstDesc,
                                    d_rowRemapTable, d_colRemapTable, d_tableDesc, interpolationType,
                                    d_roiTensorPtrSrc, RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "Remap");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    ostringstream params;
    params << "transform=sine_wave, interpolation=bilinear";
    printResult("RPP HIP BATCH Remap", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipFree(d_rowRemapTable));
    CHECK_HIP_STATUS(hipFree(d_colRemapTable));
    CHECK_HIP_STATUS(hipHostFree(d_srcDesc));
    CHECK_HIP_STATUS(hipHostFree(d_dstDesc));
    CHECK_HIP_STATUS(hipHostFree(d_tableDesc));
    CHECK_HIP_STATUS(hipHostFree(d_roiTensorPtrSrc));
    free(h_input);
    free(h_rowRemapTable);
    free(h_colRemapTable);
    free(roiTensorPtrSrc);
}

// Phase Batched (HIP)
void benchmark_RPP_HIP_Phase_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int batchSize = imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Setup descriptors (on host for init)
    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, height, width, channels, 0);
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    set_descriptor_dims_and_strides_local(&dstDesc, batchSize, height, width, channels, 0);
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&dstDesc);

    // Calculate sizes using strides (accounts for proper layout)
    size_t bufferSize = (size_t)srcDesc.n * srcDesc.strides.nStride;

    // Allocate HOST buffers
    Rpp8u* h_input1 = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));
    Rpp8u* h_input2 = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));

    // Fill input buffers
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgStart1 = h_input1 + i * srcDesc.strides.nStride;
        Rpp8u* imgStart2 = h_input2 + i * srcDesc.strides.nStride;

        for (int h = 0; h < height; h++) {
            Rpp8u* rowStart1 = imgStart1 + h * srcDesc.strides.hStride;
            Rpp8u* rowStart2 = imgStart2 + h * srcDesc.strides.hStride;
            const Rpp8u* srcRow = imgs[i].data + h * width * channels;

            // Copy first input
            memcpy(rowStart1, srcRow, width * channels * sizeof(Rpp8u));

            // Create second input (shifted for phase calculation)
            for (int w = 0; w < width * channels; w++) {
                rowStart2[w] = srcRow[w] >> 1;
            }
        }
    }

    // Allocate HIP device buffers
    Rpp8u *d_input1, *d_input2, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input1, bufferSize * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipMalloc(&d_input2, bufferSize * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipMalloc(&d_output, bufferSize * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipMemcpy(d_input1, h_input1, bufferSize * sizeof(Rpp8u), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(d_input2, h_input2, bufferSize * sizeof(Rpp8u), hipMemcpyHostToDevice));

    // Copy descriptors and ROI to pinned memory
    RpptDesc *d_srcDesc, *d_dstDesc;
    RpptROI* d_roiTensorPtrSrc;
    CHECK_HIP_STATUS(hipHostMalloc(&d_srcDesc, sizeof(RpptDesc)));
    CHECK_HIP_STATUS(hipHostMalloc(&d_dstDesc, sizeof(RpptDesc)));
    CHECK_HIP_STATUS(hipHostMalloc(&d_roiTensorPtrSrc, batchSize * sizeof(RpptROI)));
    *d_srcDesc = srcDesc;
    *d_dstDesc = dstDesc;
    for (int i = 0; i < batchSize; i++) {
        d_roiTensorPtrSrc[i].xywhROI.xy.x = 0;
        d_roiTensorPtrSrc[i].xywhROI.xy.y = 0;
        d_roiTensorPtrSrc[i].xywhROI.roiWidth = width;
        d_roiTensorPtrSrc[i].xywhROI.roiHeight = height;
    }

    // Benchmark loop
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_phase((RppPtr_t)d_input1, (RppPtr_t)d_input2, d_srcDesc,
                                    (RppPtr_t)d_output, d_dstDesc,
                                    d_roiTensorPtrSrc, RpptRoiType::XYWH,
                                    handle, RPP_HIP_BACKEND),
                        "Phase");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    ostringstream params;
    params << "inputs=2";
    printResult("RPP HIP BATCH Phase", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    CHECK_HIP_STATUS(hipFree(d_input1));
    CHECK_HIP_STATUS(hipFree(d_input2));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(d_srcDesc));
    CHECK_HIP_STATUS(hipHostFree(d_dstDesc));
    CHECK_HIP_STATUS(hipHostFree(d_roiTensorPtrSrc));
    free(h_input1);
    free(h_input2);
}

// Normalize Batched (HIP)
void benchmark_RPP_HIP_Normalize_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int batchSize = imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;
    int nDim = 3;  // 3D per-image: H, W, C

    // Create generic descriptor for 4D tensor (NHWC)
    RpptGenericDesc *genericDesc;
    CHECK_HIP_STATUS(hipHostMalloc(&genericDesc, sizeof(RpptGenericDesc)));
    genericDesc->numDims = nDim + 1;  // 4D: N, H, W, C
    genericDesc->offsetInBytes = 0;
    genericDesc->dataType = RpptDataType::U8;
    genericDesc->layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    genericDesc->dims[0] = batchSize;   // N (batch)
    genericDesc->dims[1] = height;      // H
    genericDesc->dims[2] = width;       // W
    genericDesc->dims[3] = channels;    // C

    // Compute strides for NHWC layout
    genericDesc->strides[3] = 1;                          // C stride
    genericDesc->strides[2] = channels;                   // W stride
    genericDesc->strides[1] = width * channels;           // H stride
    genericDesc->strides[0] = height * width * channels;  // N stride

    // axisMask: normalize over H and W
    Rpp32u axisMask = 0x7;

    // Allocate mean, stddev, and ROI tensors in pinned memory
    Rpp32u meanStddevSize = batchSize * channels;
    Rpp32f *meanTensor, *stdDevTensor;
    Rpp32u *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&meanTensor, meanStddevSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&stdDevTensor, meanStddevSize * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(Rpp32u)));

    // Initialize ROI tensor - simplified format matching non-batched
    for (int i = 0; i < batchSize; i++) {
        roiTensor[i] = 1;
    }

    // computeMeanStddev: 3 = compute both
    Rpp8u computeMeanStddev = 3;
    Rpp32f scale = 1.0f;
    Rpp32f shift = 0.0f;

    // Allocate HOST buffers
    size_t imageSize = height * width * channels;
    Rpp8u* h_inputBuffer = (Rpp8u*)calloc(batchSize * imageSize, sizeof(Rpp8u));
    for (int i = 0; i < batchSize; i++)
        memcpy(h_inputBuffer + i * imageSize, imgs[i].data, imageSize);

    // Allocate HIP device buffers
    Rpp8u *d_inputBuffer, *d_outputBuffer;
    CHECK_HIP_STATUS(hipMalloc(&d_inputBuffer, batchSize * imageSize * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipMalloc(&d_outputBuffer, batchSize * imageSize * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipMemcpy(d_inputBuffer, h_inputBuffer, batchSize * imageSize * sizeof(Rpp8u), hipMemcpyHostToDevice));

    // Benchmark loop
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_normalize((RppPtr_t)d_inputBuffer, genericDesc,
                                        (RppPtr_t)d_outputBuffer, genericDesc,
                                        axisMask, meanTensor, stdDevTensor, computeMeanStddev,
                                        scale, shift, roiTensor,
                                        handle, RPP_HIP_BACKEND),
                        "Normalize");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    ostringstream params;
    params << "mode=auto_compute, scale=" << scale << ", shift=" << shift;
    printResult("RPP HIP BATCH Normalize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    CHECK_HIP_STATUS(hipFree(d_inputBuffer));
    CHECK_HIP_STATUS(hipFree(d_outputBuffer));
    CHECK_HIP_STATUS(hipHostFree(genericDesc));
    CHECK_HIP_STATUS(hipHostFree(meanTensor));
    CHECK_HIP_STATUS(hipHostFree(stdDevTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));
    free(h_inputBuffer);
}

// CropAndPatch Batched (HIP)
void benchmark_RPP_HIP_CropAndPatch_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int batchSize = imgs.size();
    if (batchSize < 2) {
        cout << "CropAndPatch requires at least 2 images. Skipping." << endl;
        return;
    }
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Setup descriptors
    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, height, width, channels, 0);
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    set_descriptor_dims_and_strides_local(&dstDesc, batchSize, height, width, channels, 0);
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&dstDesc);

    // Calculate buffer size using strides
    size_t bufferSize = (size_t)srcDesc.n * srcDesc.strides.nStride;

    // Allocate HOST buffers
    Rpp8u* h_input1 = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));
    Rpp8u* h_input2 = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));

    // Fill input buffers
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgStart1 = h_input1 + i * srcDesc.strides.nStride;
        Rpp8u* imgStart2 = h_input2 + i * srcDesc.strides.nStride;
        int nextIdx = (i + 1) % batchSize;

        for (int h = 0; h < height; h++) {
            Rpp8u* rowStart1 = imgStart1 + h * srcDesc.strides.hStride;
            Rpp8u* rowStart2 = imgStart2 + h * srcDesc.strides.hStride;
            const Rpp8u* srcRow1 = imgs[i].data + h * width * channels;
            const Rpp8u* srcRow2 = imgs[nextIdx].data + h * width * channels;

            memcpy(rowStart1, srcRow1, width * channels * sizeof(Rpp8u));
            memcpy(rowStart2, srcRow2, width * channels * sizeof(Rpp8u));
        }
    }

    // Allocate HIP device buffers
    Rpp8u *d_input1, *d_input2, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input1, bufferSize * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipMalloc(&d_input2, bufferSize * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipMalloc(&d_output, bufferSize * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipMemcpy(d_input1, h_input1, bufferSize * sizeof(Rpp8u), hipMemcpyHostToDevice));
    CHECK_HIP_STATUS(hipMemcpy(d_input2, h_input2, bufferSize * sizeof(Rpp8u), hipMemcpyHostToDevice));

    // Copy descriptors and ROIs to pinned memory
    RpptDesc *d_srcDesc, *d_dstDesc;
    RpptROI *dstRoiTensor, *cropRoiTensor, *patchRoiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&d_srcDesc, sizeof(RpptDesc)));
    CHECK_HIP_STATUS(hipHostMalloc(&d_dstDesc, sizeof(RpptDesc)));
    CHECK_HIP_STATUS(hipHostMalloc(&dstRoiTensor, batchSize * sizeof(RpptROI)));
    CHECK_HIP_STATUS(hipHostMalloc(&cropRoiTensor, batchSize * sizeof(RpptROI)));
    CHECK_HIP_STATUS(hipHostMalloc(&patchRoiTensor, batchSize * sizeof(RpptROI)));

    *d_srcDesc = srcDesc;
    *d_dstDesc = dstDesc;

    for (int i = 0; i < batchSize; i++) {
        dstRoiTensor[i].xywhROI.xy.x = 0;
        dstRoiTensor[i].xywhROI.xy.y = 0;
        dstRoiTensor[i].xywhROI.roiWidth = width;
        dstRoiTensor[i].xywhROI.roiHeight = height;

        cropRoiTensor[i].xywhROI.xy.x = width / 4;
        cropRoiTensor[i].xywhROI.xy.y = height / 4;
        cropRoiTensor[i].xywhROI.roiWidth = width / 2;
        cropRoiTensor[i].xywhROI.roiHeight = height / 2;

        patchRoiTensor[i] = cropRoiTensor[i];
    }

    // Benchmark loop
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_crop_and_patch((RppPtr_t)d_input1, (RppPtr_t)d_input2, d_srcDesc,
                                             (RppPtr_t)d_output, d_dstDesc,
                                             dstRoiTensor, cropRoiTensor, patchRoiTensor,
                                             RpptRoiType::XYWH, handle, RPP_HIP_BACKEND),
                        "CropAndPatch");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    ostringstream params;
    params << "crop=center_quarter, patch=center_quarter";
    printResult("RPP HIP BATCH CropAndPatch", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    CHECK_HIP_STATUS(hipFree(d_input1));
    CHECK_HIP_STATUS(hipFree(d_input2));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(d_srcDesc));
    CHECK_HIP_STATUS(hipHostFree(d_dstDesc));
    CHECK_HIP_STATUS(hipHostFree(dstRoiTensor));
    CHECK_HIP_STATUS(hipHostFree(cropRoiTensor));
    CHECK_HIP_STATUS(hipHostFree(patchRoiTensor));
    free(h_input1);
    free(h_input2);
}

// CropMirrorNormalize Batched (HIP)
void benchmark_RPP_HIP_CropMirrorNormalize_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int batchSize = imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    int dstHeight = height / 2;
    int dstWidth = width / 2;

    // Setup descriptors
    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, height, width, channels, 0);
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    set_descriptor_dims_and_strides_local(&dstDesc, batchSize, dstHeight, dstWidth, channels, 0);
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&dstDesc);

    size_t srcBufferSize = (size_t)srcDesc.n * srcDesc.strides.nStride;
    size_t dstBufferSize = (size_t)dstDesc.n * dstDesc.strides.nStride;

    // Allocate HOST buffer
    Rpp8u* h_input = (Rpp8u*)calloc(srcBufferSize, sizeof(Rpp8u));
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgStart = h_input + i * srcDesc.strides.nStride;
        for (int h = 0; h < height; h++) {
            Rpp8u* rowStart = imgStart + h * srcDesc.strides.hStride;
            const Rpp8u* srcRow = imgs[i].data + h * width * channels;
            memcpy(rowStart, srcRow, width * channels * sizeof(Rpp8u));
        }
    }

    // Allocate HIP device buffers
    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, srcBufferSize * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipMalloc(&d_output, dstBufferSize * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipMemcpy(d_input, h_input, srcBufferSize * sizeof(Rpp8u), hipMemcpyHostToDevice));

    // Allocate pinned memory for descriptors and parameters
    RpptDesc *d_srcDesc, *d_dstDesc;
    Rpp32f *offsetTensor, *multiplierTensor;
    Rpp32u *mirrorTensor;
    RpptROI *cropRoiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&d_srcDesc, sizeof(RpptDesc)));
    CHECK_HIP_STATUS(hipHostMalloc(&d_dstDesc, sizeof(RpptDesc)));
    CHECK_HIP_STATUS(hipHostMalloc(&offsetTensor, batchSize * channels * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&multiplierTensor, batchSize * channels * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&mirrorTensor, batchSize * sizeof(Rpp32u)));
    CHECK_HIP_STATUS(hipHostMalloc(&cropRoiTensor, batchSize * sizeof(RpptROI)));

    *d_srcDesc = srcDesc;
    *d_dstDesc = dstDesc;

    // Set normalization parameters
    if (isColor) {
        Rpp32f mean[3] = {60.0f, 80.0f, 100.0f};
        Rpp32f stdDev[3] = {0.9f, 0.9f, 0.9f};
        for (int i = 0; i < batchSize; i++) {
            for (int c = 0; c < 3; c++) {
                offsetTensor[i * 3 + c] = -mean[c] / stdDev[c];
                multiplierTensor[i * 3 + c] = 1.0f / stdDev[c];
            }
            mirrorTensor[i] = 1;
        }
    } else {
        Rpp32f mean = 100.0f;
        Rpp32f stdDev = 0.9f;
        for (int i = 0; i < batchSize; i++) {
            offsetTensor[i] = -mean / stdDev;
            multiplierTensor[i] = 1.0f / stdDev;
            mirrorTensor[i] = 1;
        }
    }

    for (int i = 0; i < batchSize; i++) {
        cropRoiTensor[i].xywhROI.xy.x = width / 4;
        cropRoiTensor[i].xywhROI.xy.y = height / 4;
        cropRoiTensor[i].xywhROI.roiWidth = dstWidth;
        cropRoiTensor[i].xywhROI.roiHeight = dstHeight;
    }

    // Benchmark loop
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_crop_mirror_normalize((RppPtr_t)d_input, d_srcDesc,
                                                    (RppPtr_t)d_output, d_dstDesc,
                                                    offsetTensor, multiplierTensor, mirrorTensor,
                                                    cropRoiTensor, RpptRoiType::XYWH,
                                                    handle, RPP_HIP_BACKEND),
                        "CropMirrorNormalize");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    ostringstream params;
    params << "crop=half, mirror=horizontal, mean=60/80/100, stddev=0.9";
    printResult("RPP HIP BATCH CropMirrorNormalize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(d_srcDesc));
    CHECK_HIP_STATUS(hipHostFree(d_dstDesc));
    CHECK_HIP_STATUS(hipHostFree(offsetTensor));
    CHECK_HIP_STATUS(hipHostFree(multiplierTensor));
    CHECK_HIP_STATUS(hipHostFree(mirrorTensor));
    CHECK_HIP_STATUS(hipHostFree(cropRoiTensor));
    free(h_input);
}

// ResizeMirrorNormalize Batched (HIP)
void benchmark_RPP_HIP_ResizeMirrorNormalize_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int batchSize = imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    int dstHeight = height / 2;
    int dstWidth = width / 2;

    // Setup descriptors
    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, height, width, channels, 0);
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    set_descriptor_dims_and_strides_local(&dstDesc, batchSize, dstHeight, dstWidth, channels, 0);
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&dstDesc);

    size_t srcBufferSize = (size_t)srcDesc.n * srcDesc.strides.nStride;
    size_t dstBufferSize = (size_t)dstDesc.n * dstDesc.strides.nStride;

    // Allocate HOST buffer
    Rpp8u* h_input = (Rpp8u*)calloc(srcBufferSize, sizeof(Rpp8u));
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgStart = h_input + i * srcDesc.strides.nStride;
        for (int h = 0; h < height; h++) {
            Rpp8u* rowStart = imgStart + h * srcDesc.strides.hStride;
            const Rpp8u* srcRow = imgs[i].data + h * width * channels;
            memcpy(rowStart, srcRow, width * channels * sizeof(Rpp8u));
        }
    }

    // Allocate HIP device buffers
    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, srcBufferSize * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipMalloc(&d_output, dstBufferSize * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipMemcpy(d_input, h_input, srcBufferSize * sizeof(Rpp8u), hipMemcpyHostToDevice));

    // Allocate pinned memory
    RpptDesc *d_srcDesc, *d_dstDesc;
    RpptImagePatch *dstImgSizes;
    Rpp32f *meanTensor, *stdDevTensor;
    Rpp32u *mirrorTensor;
    RpptROI *roiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&d_srcDesc, sizeof(RpptDesc)));
    CHECK_HIP_STATUS(hipHostMalloc(&d_dstDesc, sizeof(RpptDesc)));
    CHECK_HIP_STATUS(hipHostMalloc(&dstImgSizes, batchSize * sizeof(RpptImagePatch)));
    CHECK_HIP_STATUS(hipHostMalloc(&meanTensor, batchSize * channels * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&stdDevTensor, batchSize * channels * sizeof(Rpp32f)));
    CHECK_HIP_STATUS(hipHostMalloc(&mirrorTensor, batchSize * sizeof(Rpp32u)));
    CHECK_HIP_STATUS(hipHostMalloc(&roiTensor, batchSize * sizeof(RpptROI)));

    *d_srcDesc = srcDesc;
    *d_dstDesc = dstDesc;

    for (int i = 0; i < batchSize; i++) {
        dstImgSizes[i].width = dstWidth;
        dstImgSizes[i].height = dstHeight;

        if (isColor) {
            meanTensor[i * 3 + 0] = 60.0f;
            meanTensor[i * 3 + 1] = 80.0f;
            meanTensor[i * 3 + 2] = 100.0f;
            stdDevTensor[i * 3 + 0] = 1.0f;
            stdDevTensor[i * 3 + 1] = 1.0f;
            stdDevTensor[i * 3 + 2] = 1.0f;
        } else {
            meanTensor[i] = 100.0f;
            stdDevTensor[i] = 1.0f;
        }
        mirrorTensor[i] = 1;

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = dstWidth;
        roiTensor[i].xywhROI.roiHeight = dstHeight;
    }

    // Benchmark loop
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_resize_mirror_normalize((RppPtr_t)d_input, d_srcDesc,
                                                      (RppPtr_t)d_output, d_dstDesc,
                                                      dstImgSizes, RpptInterpolationType::BILINEAR,
                                                      meanTensor, stdDevTensor, mirrorTensor,
                                                      roiTensor, RpptRoiType::XYWH,
                                                      handle, RPP_HIP_BACKEND),
                        "ResizeMirrorNormalize");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    ostringstream params;
    params << "resize=half, mirror=horizontal, mean=60/80/100, stddev=1";
    printResult("RPP HIP BATCH ResizeMirrorNormalize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(d_srcDesc));
    CHECK_HIP_STATUS(hipHostFree(d_dstDesc));
    CHECK_HIP_STATUS(hipHostFree(dstImgSizes));
    CHECK_HIP_STATUS(hipHostFree(meanTensor));
    CHECK_HIP_STATUS(hipHostFree(stdDevTensor));
    CHECK_HIP_STATUS(hipHostFree(mirrorTensor));
    CHECK_HIP_STATUS(hipHostFree(roiTensor));
    free(h_input);
}

// ResizeCropMirror Batched (HIP)
void benchmark_RPP_HIP_ResizeCropMirror_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream) {
    int batchSize = imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Target resize dimensions (224x224 to match non-batched version for fair comparison)
    int targetWidth = 224;
    int targetHeight = 224;

    // Calculate center 80% crop region (same as non-batched)
    int cropWidth = (width * 4) / 5;
    int cropHeight = (height * 4) / 5;
    int cropX = (width - cropWidth) / 2;
    int cropY = (height - cropHeight) / 2;

    // Setup descriptors
    RpptDesc srcDesc, dstDesc;
    set_descriptor_dims_and_strides_local(&srcDesc, batchSize, height, width, channels, 0);
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&srcDesc);
    set_descriptor_dims_and_strides_local(&dstDesc, batchSize, targetHeight, targetWidth, channels, 0);
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.dataType = RpptDataType::U8;
    update_strides_from_layout(&dstDesc);

    size_t srcBufferSize = (size_t)srcDesc.n * srcDesc.strides.nStride;
    size_t dstBufferSize = (size_t)dstDesc.n * dstDesc.strides.nStride;

    // Allocate HOST buffer
    Rpp8u* h_input = (Rpp8u*)calloc(srcBufferSize, sizeof(Rpp8u));
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgStart = h_input + i * srcDesc.strides.nStride;
        for (int h = 0; h < height; h++) {
            Rpp8u* rowStart = imgStart + h * srcDesc.strides.hStride;
            const Rpp8u* srcRow = imgs[i].data + h * width * channels;
            memcpy(rowStart, srcRow, width * channels * sizeof(Rpp8u));
        }
    }

    // Allocate HIP device buffers
    Rpp8u *d_input, *d_output;
    CHECK_HIP_STATUS(hipMalloc(&d_input, srcBufferSize * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipMalloc(&d_output, dstBufferSize * sizeof(Rpp8u)));
    CHECK_HIP_STATUS(hipMemcpy(d_input, h_input, srcBufferSize * sizeof(Rpp8u), hipMemcpyHostToDevice));

    // Allocate pinned memory
    RpptDesc *d_srcDesc, *d_dstDesc;
    RpptImagePatch *dstImgSizes;
    Rpp32u *mirrorTensor;
    RpptROI *dstRoiTensor;
    CHECK_HIP_STATUS(hipHostMalloc(&d_srcDesc, sizeof(RpptDesc)));
    CHECK_HIP_STATUS(hipHostMalloc(&d_dstDesc, sizeof(RpptDesc)));
    CHECK_HIP_STATUS(hipHostMalloc(&dstImgSizes, batchSize * sizeof(RpptImagePatch)));
    CHECK_HIP_STATUS(hipHostMalloc(&mirrorTensor, batchSize * sizeof(Rpp32u)));
    // Allocate 256 ROI elements per image as workaround for kernel bug (same as non-batched)
    CHECK_HIP_STATUS(hipHostMalloc(&dstRoiTensor, batchSize * 256 * sizeof(RpptROI)));

    *d_srcDesc = srcDesc;
    *d_dstDesc = dstDesc;

    // Initialize ROI tensor to zero first (hipHostMalloc doesn't zero memory!)
    memset(dstRoiTensor, 0, batchSize * 256 * sizeof(RpptROI));

    for (int i = 0; i < batchSize; i++) {
        // Destination image size (after resize)
        dstImgSizes[i].width = targetWidth;
        dstImgSizes[i].height = targetHeight;
        mirrorTensor[i] = 1;
        // Each image gets 256 ROI slots (only first is used, rest are buffer for buggy kernel)
        dstRoiTensor[i * 256].xywhROI.xy.x = cropX;
        dstRoiTensor[i * 256].xywhROI.xy.y = cropY;
        dstRoiTensor[i * 256].xywhROI.roiWidth = cropWidth;
        dstRoiTensor[i * 256].xywhROI.roiHeight = cropHeight;
    }

    // Benchmark loop
    for (int k = 0; k < HIP_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            CHECK_HIP_STATUS(hipStreamSynchronize(stream));
            perfMonitor.start(DeviceType::GPU, 0);
        }
        CHECK_RPP_STATUS(rppt_resize_crop_mirror((RppPtr_t)d_input, d_srcDesc,
                                                 (RppPtr_t)d_output, d_dstDesc,
                                                 dstImgSizes, RpptInterpolationType::BILINEAR,
                                                 mirrorTensor, dstRoiTensor, RpptRoiType::XYWH,
                                                 handle, RPP_HIP_BACKEND),
                        "ResizeCropMirror");
    }
    CHECK_HIP_STATUS(hipStreamSynchronize(stream));
    perfMonitor.stop();

    ostringstream params;
    params << "crop=80%,resize=224x224,mirror";
    printResult("RPP HIP BATCH ResizeCropMirror", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    CHECK_HIP_STATUS(hipFree(d_input));
    CHECK_HIP_STATUS(hipFree(d_output));
    CHECK_HIP_STATUS(hipHostFree(d_srcDesc));
    CHECK_HIP_STATUS(hipHostFree(d_dstDesc));
    CHECK_HIP_STATUS(hipHostFree(dstImgSizes));
    CHECK_HIP_STATUS(hipHostFree(mirrorTensor));
    CHECK_HIP_STATUS(hipHostFree(dstRoiTensor));
    free(h_input);
}
