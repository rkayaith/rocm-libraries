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

// Macro for checking RPP API status
#define CHECK_RPP_STATUS(call, func_name)                                                       \
    do {                                                                                        \
        RppStatus status = (call);                                                              \
        if (status != RPP_SUCCESS) {                                                            \
            std::cerr << "RPP " << func_name << " failed with status: " << status << std::endl; \
        }                                                                                       \
    } while (0)

// ==================== RPP BENCHMARK FUNCTIONS ====================

void benchmark_RPP_HOST_Brightness(const vector<Mat>& imgs, bool isColor, float alpha, float beta,
                              rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_brightness(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], &alpha,
                                &beta, &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "brightness");
        }
    }
    perfMonitor.stop();
    ostringstream params;
    params << "alpha=" << alpha << ", beta=" << beta;
    printResult("RPP HOST Brightness", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_GammaCorrection(const vector<Mat>& imgs, bool isColor, float gamma,
                                   rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_gamma_correction(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], &gamma,
                                      &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "gamma_correction");
        }
    }
    perfMonitor.stop();

    ostringstream params;
    params << "gamma=" << gamma;
    printResult("RPP HOST GammaCorrection", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Blend(const vector<Mat>& imgs, bool isColor, float alpha, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images), imgs2(imgs.size());
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        imgs[i].convertTo(imgs2[i], -1, 0.8, 30);
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_blend(imgs[i].data, imgs2[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                           &alpha, &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "blend");
        }
    }
    perfMonitor.stop();

    ostringstream params;
    params << "alpha=" << alpha;
    printResult("RPP HOST Blend", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Contrast(const vector<Mat>& imgs, bool isColor, float contrastFactor,
                            float contrastCenter, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_contrast(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                           &contrastFactor, &contrastCenter, &rois[i],
                                           RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "contrast");
        }
    }
    perfMonitor.stop();

    ostringstream params;
    params << "factor=" << contrastFactor << ", center=" << contrastCenter;
    printResult("RPP HOST Contrast", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Exposure(const vector<Mat>& imgs, bool isColor, float exposureFactor,
                            rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_exposure(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                           &exposureFactor, &rois[i], RpptRoiType::XYWH, handle,
                                           RPP_HOST_BACKEND),
                             "exposure");
        }
    }
    perfMonitor.stop();

    ostringstream params;
    params << "factor=" << exposureFactor;
    printResult("RPP HOST Exposure", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Hue(const vector<Mat>& imgs, float hueDelta, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], RpptLayout::NHWC);
        dstDescs[i] = createRppDescriptor(out[i], RpptLayout::NHWC);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_hue(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], &hueDelta, &rois[i],
                         RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "hue");
        }
    }
    perfMonitor.stop();

    ostringstream params;
    params << "hue=" << hueDelta;
    printResult("RPP HOST Hue", imgs.size(), true, perfMonitor.getTotalTime(),
                perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Saturation(const vector<Mat>& imgs, float satFactor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], RpptLayout::NHWC);
        dstDescs[i] = createRppDescriptor(out[i], RpptLayout::NHWC);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_saturation(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], &satFactor,
                                &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "saturation");
        }
    }
    perfMonitor.stop();

    ostringstream params;
    params << "factor=" << satFactor;
    printResult("RPP HOST Saturation", imgs.size(), true,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_ColorToGreyscale(const vector<Mat>& imgs, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].rows, imgs[i].cols, CV_8UC1);
        srcDescs[i] = createRppDescriptor(imgs[i], RpptLayout::NHWC);
        dstDescs[i] = createRppDescriptor(out[i], RpptLayout::NCHW);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_color_to_greyscale(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                        RpptSubpixelLayout::BGRtype, handle, RPP_HOST_BACKEND),
                "color_to_greyscale");
        }
    }
    perfMonitor.stop();
    printResult("RPP HOST ColorToGreyscale", imgs.size(), true,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_ColorJitter(const vector<Mat>& imgs, float brightness, float contrast,
                               float saturation, float hue, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], RpptLayout::NHWC);
        dstDescs[i] = createRppDescriptor(out[i], RpptLayout::NHWC);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_color_jitter(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                  &brightness, &contrast, &saturation, &hue, &rois[i],
                                  RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "color_jitter");
        }
    }
    perfMonitor.stop();
    ostringstream params;
    params << "brightness=" << brightness << ", contrast=" << contrast
           << ", saturation=" << saturation << ", hue=" << hue;
    printResult("RPP HOST ColorJitter", imgs.size(), true,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_BoxFilter(const vector<Mat>& imgs, bool isColor, int kernelSize,
                             rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_box_filter(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], kernelSize,
                                borderType, &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "box_filter");
        }
    }
    perfMonitor.stop();
    ostringstream params;
    params << "kernel=" << kernelSize;
    printResult("RPP HOST BoxFilter", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_MedianFilter(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_median_filter(imgs[i].data, &srcDescs[i], out[i].data,
                                                &dstDescs[i], kernelSize, borderType, &rois[i],
                                                RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "median_filter");
        }
    }
    perfMonitor.stop();
    ostringstream params;
    params << "kernel=" << kernelSize;
    printResult("RPP HOST MedianFilter", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_GaussianFilter(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                  double sigma, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    float stdDev = static_cast<float>(sigma);
    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_gaussian_filter(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], &stdDev,
                                     kernelSize, borderType, &rois[i], RpptRoiType::XYWH, handle,
                                     RPP_HOST_BACKEND),
                "gaussian_filter");
        }
    }
    perfMonitor.stop();
    ostringstream params;
    params << "kernel=" << kernelSize << ", sigma=" << sigma;
    printResult("RPP HOST GaussianFilter", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_SobelFilter(const vector<Mat>& imgs, bool isColor, int sobelType,
                               rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), CV_8UC1);
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    Rpp32u kernelSize = 3;  // Sobel uses 3x3 kernel
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_sobel_filter(imgs[i].data, &srcDescs[i], out[i].data,
                                               &dstDescs[i], sobelType, kernelSize, &rois[i],
                                               RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "sobel_filter");
        }
    }
    perfMonitor.stop();

    ostringstream params;
    params << "type=" << sobelType;
    printResult("RPP HOST SobelFilter", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Emboss(const vector<Mat>& imgs, bool isColor, int kernelSize, float strength,
                          rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    vector<Rpp32f> strengthTensor(num_images, strength);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_emboss(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                         strengthTensor.data(), kernelSize, borderType, &rois[i],
                                         RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "emboss");
        }
    }
    perfMonitor.stop();
    ostringstream params;
    params << "kernel=" << kernelSize << ", strength=" << strength;
    printResult("RPP HOST Emboss", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Crop(const vector<Mat>& imgs, bool isColor, int cropWidth, int cropHeight,
                        rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(cropHeight, cropWidth, imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i].xywhROI.xy.x = (imgs[i].cols - cropWidth) / 2;
        rois[i].xywhROI.xy.y = (imgs[i].rows - cropHeight) / 2;
        rois[i].xywhROI.roiWidth = cropWidth;
        rois[i].xywhROI.roiHeight = cropHeight;
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_crop(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                       &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "crop");
        }
    }
    perfMonitor.stop();
    ostringstream params;
    params << "size=" << cropWidth << "x" << cropHeight;
    printResult("RPP HOST Crop", imgs.size(), isColor, perfMonitor.getTotalTime(),
                perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Resize(const vector<Mat>& imgs, bool isColor, int dstW, int dstH,
                          RpptInterpolationType interpType, const string& interpName,
                          rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);
    RpptImagePatch dstImgSize;
    dstImgSize.width = dstW;
    dstImgSize.height = dstH;

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(dstH, dstW, imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_resize(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], &dstImgSize,
                            interpType, &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "resize");
        }
    }
    perfMonitor.stop();
    ostringstream params;
    params << "type=" << interpName << ", size=" << dstW << "x" << dstH;
    printResult("RPP HOST Resize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Flip(const vector<Mat>& imgs, bool isColor, int flipCode, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    Rpp32u horizontalFlag = (flipCode == 1 || flipCode == -1) ? 1 : 0;
    Rpp32u verticalFlag = (flipCode == 0 || flipCode == -1) ? 1 : 0;

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_flip(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], &horizontalFlag,
                          &verticalFlag, &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "flip");
        }
    }
    perfMonitor.stop();
    string name = (flipCode == 1) ? "Horizontal" : (flipCode == 0) ? "Vertical" : "Both";
    ostringstream params;
    params << "type=" << name;
    printResult("RPP HOST Flip", imgs.size(), isColor, perfMonitor.getTotalTime(),
                perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Rotate(const vector<Mat>& imgs, bool isColor, float angleDeg,
                          rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_rotate(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                         &angleDeg, RpptInterpolationType::BILINEAR, &rois[i],
                                         RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "rotate");
        }
    }
    perfMonitor.stop();
    ostringstream params;
    params << "angle=" << angleDeg << "deg";
    printResult("RPP HOST Rotate", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_WarpAffine(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    float affine[6] = {1.0f, 0.1f, 10.0f, 0.1f, 1.0f, 10.0f};
    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_warp_affine(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                              affine, RpptInterpolationType::BILINEAR, &rois[i],
                                              RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "warp_affine");
        }
    }
    perfMonitor.stop();
    printResult("RPP HOST WarpAffine", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_Fisheye(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_fisheye(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                          &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "fisheye");
        }
    }
    perfMonitor.stop();
    printResult("RPP HOST Fisheye", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_LensCorrection(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    // Camera matrix and distortion coefficients (from reference implementation)
    struct CameraMatrix {
        Rpp32f data[9];
    };
    struct DistortionCoeffs {
        Rpp32f data[8];
    };

    vector<CameraMatrix> cameraMatrices(num_images);
    vector<DistortionCoeffs> distortionCoeffs(num_images);

    // Sample camera calibration parameters (from reference)
    CameraMatrix sampleCameraMatrix = {
        {534.07088364f, 0.0f, 341.53407554f, 0.0f, 534.11914595f, 232.94565259f, 0.0f, 0.0f, 1.0f}};
    DistortionCoeffs sampleDistortion = {
        {-0.29297164f, 0.10770696f, 0.00131038f, -0.0000311f, 0.0434798f, 0.0f, 0.0f, 0.0f}};

    for (int i = 0; i < num_images; ++i) {
        cameraMatrices[i] = sampleCameraMatrix;
        distortionCoeffs[i] = sampleDistortion;
    }

    // Table descriptor for remap tables
    RpptDesc tableDesc;
    if (num_images > 0) {
        tableDesc = createRppDescriptor(imgs[0], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        tableDesc.c = 1;
        tableDesc.strides.nStride = imgs[0].rows * imgs[0].cols;
        tableDesc.strides.hStride = imgs[0].cols;
        tableDesc.strides.wStride = tableDesc.strides.cStride = 1;
    }

    // Allocate remap tables
    size_t tableSize = num_images * imgs[0].rows * imgs[0].cols;
    vector<Rpp32f> rowRemapTable(tableSize);
    vector<Rpp32f> colRemapTable(tableSize);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_lens_correction(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                     rowRemapTable.data(), colRemapTable.data(), &tableDesc,
                                     cameraMatrices[i].data, distortionCoeffs[i].data, &rois[i],
                                     RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "lens_correction");
        }
    }
    perfMonitor.stop();
    printResult("RPP HOST LensCorrection", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_Erode(const vector<Mat>& imgs, bool isColor, int kernelSize,
                         rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_erode_host(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                             kernelSize, &rois[i], RpptRoiType::XYWH, handle),
                             "erode_host");
        }
    }
    perfMonitor.stop();
    ostringstream params;
    params << "kernel=" << kernelSize;
    printResult("RPP HOST Erode", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Dilate(const vector<Mat>& imgs, bool isColor, int kernelSize,
                          rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_dilate_host(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                              kernelSize, &rois[i], RpptRoiType::XYWH, handle),
                             "dilate_host");
        }
    }
    perfMonitor.stop();
    ostringstream params;
    params << "kernel=" << kernelSize;
    printResult("RPP HOST Dilate", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_AddScalar(const vector<Mat>& imgs, bool isColor, float addVal,
                             rppHandle_t handle) {
    int num_images = (int)imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Convert to F32 and create batch buffer
    int imageSize = height * width * channels;
    vector<Rpp32f> inputBuffer(num_images * imageSize);
    vector<Rpp32f> outputBuffer(num_images * imageSize);

    for (int i = 0; i < num_images; ++i) {
        Mat imgF32;
        imgs[i].convertTo(imgF32, CV_32F);
        memcpy(inputBuffer.data() + i * imageSize, imgF32.data, imageSize * sizeof(Rpp32f));
    }

    // Create 5D generic descriptor (NDHWC or NCDHW) with depth=1 for 2D images
    RpptGenericDesc genericDesc;
    genericDesc.numDims = 5;
    genericDesc.offsetInBytes = 0;
    genericDesc.dataType = RpptDataType::F32;

    if (isColor) {
        // NDHWC layout
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
        // NCDHW layout for grayscale
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

    // Create ROI3D array for the batch
    vector<RpptROI3D> roi3ds(num_images);
    for (int i = 0; i < num_images; ++i) {
        roi3ds[i].xyzwhdROI.xyz.x = 0;
        roi3ds[i].xyzwhdROI.xyz.y = 0;
        roi3ds[i].xyzwhdROI.xyz.z = 0;
        roi3ds[i].xyzwhdROI.roiWidth = width;
        roi3ds[i].xyzwhdROI.roiHeight = height;
        roi3ds[i].xyzwhdROI.roiDepth = 1;
    }

    // Create addTensor array (one value per image)
    vector<Rpp32f> addTensor(num_images, addVal);

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }

        CHECK_RPP_STATUS(rppt_add_scalar(inputBuffer.data(), &srcGenericDesc, outputBuffer.data(),
                                         &dstGenericDesc, addTensor.data(), roi3ds.data(),
                                         RpptRoi3DType::XYZWHD, handle, RPP_HOST_BACKEND),
                         "add_scalar");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "value=" << addVal;
    printResult("RPP HOST AddScalar", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_SubtractScalar(const vector<Mat>& imgs, bool isColor, float subVal,
                                  rppHandle_t handle) {
    int num_images = (int)imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Convert to F32 and create batch buffer
    int imageSize = height * width * channels;
    vector<Rpp32f> inputBuffer(num_images * imageSize);
    vector<Rpp32f> outputBuffer(num_images * imageSize);

    for (int i = 0; i < num_images; ++i) {
        Mat imgF32;
        imgs[i].convertTo(imgF32, CV_32F);
        memcpy(inputBuffer.data() + i * imageSize, imgF32.data, imageSize * sizeof(Rpp32f));
    }

    // Create 5D generic descriptor (NDHWC or NCDHW) with depth=1 for 2D images
    RpptGenericDesc genericDesc;
    genericDesc.numDims = 5;
    genericDesc.offsetInBytes = 0;
    genericDesc.dataType = RpptDataType::F32;

    if (isColor) {
        // NDHWC layout
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
        // NCDHW layout for grayscale
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

    // Create ROI3D array for the batch
    vector<RpptROI3D> roi3ds(num_images);
    for (int i = 0; i < num_images; ++i) {
        roi3ds[i].xyzwhdROI.xyz.x = 0;
        roi3ds[i].xyzwhdROI.xyz.y = 0;
        roi3ds[i].xyzwhdROI.xyz.z = 0;
        roi3ds[i].xyzwhdROI.roiWidth = width;
        roi3ds[i].xyzwhdROI.roiHeight = height;
        roi3ds[i].xyzwhdROI.roiDepth = 1;
    }

    // Create subtractTensor array (one value per image)
    vector<Rpp32f> subtractTensor(num_images, subVal);

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }

        CHECK_RPP_STATUS(
            rppt_subtract_scalar(inputBuffer.data(), &srcGenericDesc, outputBuffer.data(),
                                 &dstGenericDesc, subtractTensor.data(), roi3ds.data(),
                                 RpptRoi3DType::XYZWHD, handle, RPP_HOST_BACKEND),
            "subtract_scalar");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "value=" << subVal;
    printResult("RPP HOST SubtractScalar", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_MultiplyScalar(const vector<Mat>& imgs, bool isColor, float mulVal,
                                  rppHandle_t handle) {
    int num_images = (int)imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Convert to F32 and create batch buffer
    int imageSize = height * width * channels;
    vector<Rpp32f> inputBuffer(num_images * imageSize);
    vector<Rpp32f> outputBuffer(num_images * imageSize);

    for (int i = 0; i < num_images; ++i) {
        Mat imgF32;
        imgs[i].convertTo(imgF32, CV_32F);
        memcpy(inputBuffer.data() + i * imageSize, imgF32.data, imageSize * sizeof(Rpp32f));
    }

    // Create 5D generic descriptor (NDHWC or NCDHW) with depth=1 for 2D images
    RpptGenericDesc genericDesc;
    genericDesc.numDims = 5;
    genericDesc.offsetInBytes = 0;
    genericDesc.dataType = RpptDataType::F32;

    if (isColor) {
        // NDHWC layout
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
        // NCDHW layout for grayscale
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

    // Create ROI3D array for the batch
    vector<RpptROI3D> roi3ds(num_images);
    for (int i = 0; i < num_images; ++i) {
        roi3ds[i].xyzwhdROI.xyz.x = 0;
        roi3ds[i].xyzwhdROI.xyz.y = 0;
        roi3ds[i].xyzwhdROI.xyz.z = 0;
        roi3ds[i].xyzwhdROI.roiWidth = width;
        roi3ds[i].xyzwhdROI.roiHeight = height;
        roi3ds[i].xyzwhdROI.roiDepth = 1;
    }

    // Create multiplyTensor array (one value per image)
    vector<Rpp32f> multiplyTensor(num_images, mulVal);

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }

        CHECK_RPP_STATUS(
            rppt_multiply_scalar(inputBuffer.data(), &srcGenericDesc, outputBuffer.data(),
                                 &dstGenericDesc, multiplyTensor.data(), roi3ds.data(),
                                 RpptRoi3DType::XYZWHD, handle, RPP_HOST_BACKEND),
            "multiply_scalar");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "value=" << mulVal;
    printResult("RPP HOST MultiplyScalar", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_BitwiseAnd(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images), imgs2(imgs.size());
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        imgs[i].copyTo(imgs2[i]);
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_bitwise_and(imgs[i].data, imgs2[i].data, &srcDescs[i],
                                              out[i].data, &dstDescs[i], &rois[i],
                                              RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "bitwise_and");
        }
    }
    perfMonitor.stop();
    printResult("RPP HOST BitwiseAnd", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_BitwiseOr(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images), imgs2(imgs.size());
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        imgs[i].copyTo(imgs2[i]);
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_bitwise_or(imgs[i].data, imgs2[i].data, &srcDescs[i], out[i].data,
                                             &dstDescs[i], &rois[i], RpptRoiType::XYWH, handle,
                                             RPP_HOST_BACKEND),
                             "bitwise_or");
        }
    }
    perfMonitor.stop();
    printResult("RPP HOST BitwiseOr", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_BitwiseNot(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_bitwise_not(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], &rois[i],
                                 RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "bitwise_not");
        }
    }
    perfMonitor.stop();
    printResult("RPP HOST BitwiseNot", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_TensorMin(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    // For grayscale: output length = n, For RGB: output length = n * 4
    Rpp32u outputLength = isColor ? (num_images * 4) : num_images;
    // Use Rpp8u for U8 input images (tensor_min returns U8 for U8 input)
    vector<Rpp8u> minOutputs(outputLength);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            Rpp32u imgOutputLength = isColor ? 4 : 1;
            CHECK_RPP_STATUS(rppt_tensor_min(imgs[i].data, &srcDescs[i],
                                             &minOutputs[i * imgOutputLength], imgOutputLength,
                                             &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "tensor_min");
        }
    }
    perfMonitor.stop();
    printResult("RPP HOST TensorMin", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_TensorMax(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    // For grayscale: output length = n, For RGB: output length = n * 4
    Rpp32u outputLength = isColor ? (num_images * 4) : num_images;
    // Use Rpp8u for U8 input images (tensor_max returns U8 for U8 input)
    vector<Rpp8u> maxOutputs(outputLength);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            Rpp32u imgOutputLength = isColor ? 4 : 1;
            CHECK_RPP_STATUS(rppt_tensor_max(imgs[i].data, &srcDescs[i],
                                             &maxOutputs[i * imgOutputLength], imgOutputLength,
                                             &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "tensor_max");
        }
    }
    perfMonitor.stop();
    printResult("RPP HOST TensorMax", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_TensorSum(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    // For grayscale: output length = n, For RGB: output length = n * 4
    Rpp32u outputLength = isColor ? (num_images * 4) : num_images;
    // Use Rpp64u for U8 input images (tensor_sum returns U64 for U8 input)
    vector<Rpp64u> sumOutputs(outputLength);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            Rpp32u imgOutputLength = isColor ? 4 : 1;
            CHECK_RPP_STATUS(rppt_tensor_sum(imgs[i].data, &srcDescs[i],
                                             &sumOutputs[i * imgOutputLength], imgOutputLength,
                                             &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "tensor_sum");
        }
    }
    perfMonitor.stop();
    printResult("RPP HOST TensorSum", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_TensorMean(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    // For grayscale: output length = n, For RGB: output length = n * 4
    Rpp32u outputLength = isColor ? (num_images * 4) : num_images;
    vector<Rpp32f> meanOutputs(outputLength);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            Rpp32u imgOutputLength = isColor ? 4 : 1;
            CHECK_RPP_STATUS(
                rppt_tensor_mean(imgs[i].data, &srcDescs[i], &meanOutputs[i * imgOutputLength],
                                 imgOutputLength, &rois[i], RpptRoiType::XYWH, handle,
                                 RPP_HOST_BACKEND),
                "tensor_mean");
        }
    }
    perfMonitor.stop();
    printResult("RPP HOST TensorMean", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_TensorStddev(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    // For grayscale: output length = n, For RGB: output length = n * 4
    Rpp32u outputLength = isColor ? (num_images * 4) : num_images;
    vector<Rpp32f> stddevOutputs(outputLength);
    vector<Rpp32f> meanOutputs(outputLength);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptROI> rois(num_images);

    // First compute mean for stddev calculation
    for (int i = 0; i < num_images; ++i) {
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);

        // Compute mean first (stddev requires mean)
        Rpp32u imgOutputLength = isColor ? 4 : 1;
        CHECK_RPP_STATUS(rppt_tensor_mean(imgs[i].data, &srcDescs[i],
                                          &meanOutputs[i * imgOutputLength], imgOutputLength,
                                          &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                         "tensor_mean");
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            Rpp32u imgOutputLength = isColor ? 4 : 1;
            CHECK_RPP_STATUS(
                rppt_tensor_stddev(imgs[i].data, &srcDescs[i], &stddevOutputs[i * imgOutputLength],
                                   imgOutputLength, &meanOutputs[i * imgOutputLength], &rois[i],
                                   RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "tensor_stddev");
        }
    }
    perfMonitor.stop();
    printResult("RPP HOST TensorStddev", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_Threshold(const vector<Mat>& imgs, bool isColor, double thresh,
                             rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<Mat> grayImgs(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    Rpp32f minVal = static_cast<Rpp32f>(thresh);
    Rpp32f maxVal = 255.0f;

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), CV_8UC1);
        if (isColor)
            cvtColor(imgs[i], grayImgs[i], COLOR_BGR2GRAY);
        else
            grayImgs[i] = imgs[i];

        srcDescs[i] = createRppDescriptor(grayImgs[i], RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], RpptLayout::NCHW);
        rois[i] = createFullImageROI(grayImgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_threshold(grayImgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], &minVal,
                               &maxVal, &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "threshold");
        }
    }
    perfMonitor.stop();

    ostringstream params;
    params << "threshold=" << thresh;
    printResult("RPP HOST Threshold", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_GaussianNoise(const vector<Mat>& imgs, bool isColor, float mean, float stddev,
                                 rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    unsigned long long seed = 12345;
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_gaussian_noise(imgs[i].data, &srcDescs[i], out[i].data,
                                                 &dstDescs[i], &mean, &stddev, seed, &rois[i],
                                                 RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "gaussian_noise");
        }
    }
    perfMonitor.stop();

    ostringstream params;
    params << "mean=" << mean << ", stddev=" << stddev;
    printResult("RPP HOST GaussianNoise", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_SaltAndPepperNoise(const vector<Mat>& imgs, bool isColor, float noiseProb,
                                      rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    Rpp32u seed = 12345;
    Rpp32f saltProb = 0.5f;     // 50% of noise is salt (white)
    Rpp32f saltValue = 1.0f;    // Normalized value for salt (0.0 to 1.0 range)
    Rpp32f pepperValue = 0.0f;  // Normalized value for pepper (0.0 to 1.0 range)

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_salt_and_pepper_noise(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                           &noiseProb, &saltProb, &saltValue, &pepperValue, seed,
                                           &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "salt_and_pepper_noise");
        }
    }
    perfMonitor.stop();

    ostringstream params;
    params << "probability=" << noiseProb;
    printResult("RPP HOST SaltAndPepperNoise", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Copy(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_copy(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                       handle, RPP_HOST_BACKEND),
                             "copy");
        }
    }
    perfMonitor.stop();
    ostringstream params;
    params << "layout=NHWC";
    printResult("RPP HOST Copy", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_BitwiseXor(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            int idx1 = i;
            int idx2 = (i + 1) % num_images;
            CHECK_RPP_STATUS(rppt_bitwise_xor(imgs[idx1].data, imgs[idx2].data, &srcDescs[idx1],
                                              out[i].data, &dstDescs[i], &rois[i],
                                              RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "bitwise_xor");
        }
    }
    perfMonitor.stop();
    printResult("RPP HOST BitwiseXor", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_HistogramEqualize(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_histogram_equalize(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                        &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "histogram_equalize");
        }
    }
    perfMonitor.stop();
    printResult("RPP HOST HistogramEqualize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_Transpose(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // For image transpose, we swap H and W dimensions
    // Input layout: NHWC (batch, height, width, channels)
    // Output layout: NWHC (batch, width, height, channels)
    // Permutation: [0, 2, 1, 3] - keep N and C, swap H and W

    // Create generic descriptors for 4D tensor (NHWC)
    RpptGenericDesc srcGenericDesc, dstGenericDesc;
    srcGenericDesc.numDims = 4;
    srcGenericDesc.offsetInBytes = 0;
    srcGenericDesc.dataType = RpptDataType::U8;
    srcGenericDesc.layout = RpptLayout::NHWC;
    srcGenericDesc.dims[0] = num_images;  // N
    srcGenericDesc.dims[1] = height;      // H
    srcGenericDesc.dims[2] = width;       // W
    srcGenericDesc.dims[3] = channels;    // C

    // Compute strides for input (NHWC)
    srcGenericDesc.strides[3] = 1;                          // C stride
    srcGenericDesc.strides[2] = channels;                   // W stride
    srcGenericDesc.strides[1] = width * channels;           // H stride
    srcGenericDesc.strides[0] = height * width * channels;  // N stride

    // Output has swapped H and W
    dstGenericDesc.numDims = 4;
    dstGenericDesc.offsetInBytes = 0;
    dstGenericDesc.dataType = RpptDataType::U8;
    dstGenericDesc.layout = RpptLayout::NHWC;
    dstGenericDesc.dims[0] = num_images;  // N
    dstGenericDesc.dims[1] = width;       // W (swapped)
    dstGenericDesc.dims[2] = height;      // H (swapped)
    dstGenericDesc.dims[3] = channels;    // C

    // Compute strides for output (NWHC)
    dstGenericDesc.strides[3] = 1;                          // C stride
    dstGenericDesc.strides[2] = channels;                   // H stride (now at position 2)
    dstGenericDesc.strides[1] = height * channels;          // W stride (now at position 1)
    dstGenericDesc.strides[0] = width * height * channels;  // N stride

    // Permutation tensor: [0, 2, 1, 3] means swap dimensions 1 and 2 (H and W)
    Rpp32u permTensor[4] = {0, 2, 1, 3};

    // ROI tensor: for each image, specify the full ROI as [start_coords, size]
    // For 4D: [n_start, h_start, w_start, c_start, n_size, h_size, w_size, c_size]
    vector<Rpp32u> roiTensor(num_images * 8);
    for (int i = 0; i < num_images; ++i) {
        int idx = i * 8;
        roiTensor[idx + 0] = 0;         // n_start
        roiTensor[idx + 1] = 0;         // h_start
        roiTensor[idx + 2] = 0;         // w_start
        roiTensor[idx + 3] = 0;         // c_start
        roiTensor[idx + 4] = 1;         // n_size (process 1 image at a time)
        roiTensor[idx + 5] = height;    // h_size
        roiTensor[idx + 6] = width;     // w_size
        roiTensor[idx + 7] = channels;  // c_size
    }

    // Concatenate all images into a batch buffer
    int imageSize = height * width * channels;
    vector<Rpp8u> inputBuffer(num_images * imageSize);
    vector<Rpp8u> outputBuffer(num_images * imageSize);

    for (int i = 0; i < num_images; ++i)
        memcpy(inputBuffer.data() + i * imageSize, imgs[i].data, imageSize);

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }

        CHECK_RPP_STATUS(
            rppt_transpose(inputBuffer.data(), &srcGenericDesc, outputBuffer.data(),
                           &dstGenericDesc, permTensor, roiTensor.data(), handle, RPP_HOST_BACKEND),
            "transpose");
    }
    perfMonitor.stop();
    ostringstream params;
    params << "permutation=0,2,1,3";
    printResult("RPP HOST Transpose", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_LUT(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    // Create LUT - simple inversion table
    Rpp8u lut[256];
    for (int i = 0; i < 256; ++i) lut[i] = 255 - i;

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_lut(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], lut,
                                      &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "lut");
        }
    }
    perfMonitor.stop();
    ostringstream params;
    params << "lut=inverse";
    printResult("RPP HOST LUT", imgs.size(), isColor, perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Magnitude(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<Mat> grad_x(num_images);
    vector<Mat> grad_y(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    // Compute gradients first (Sobel)
    for (int i = 0; i < num_images; ++i) {
        Mat gray = isColor ? Mat() : imgs[i];
        if (isColor) cvtColor(imgs[i], gray, COLOR_BGR2GRAY);

        Sobel(gray, grad_x[i], CV_32F, 1, 0, 3);
        Sobel(gray, grad_y[i], CV_32F, 0, 1, 3);

        out[i] = Mat::zeros(imgs[i].size(), CV_32F);
        srcDescs[i] = createRppDescriptor(grad_x[i], RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_magnitude(grad_x[i].data, grad_y[i].data, &srcDescs[i], out[i].data,
                               &dstDescs[i], &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "magnitude");
        }
    }
    perfMonitor.stop();
    ostringstream params;
    params << "inputs=2";
    printResult("RPP HOST Magnitude", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Phase(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<Mat> grad_x(num_images);
    vector<Mat> grad_y(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    // Compute gradients first (Sobel)
    for (int i = 0; i < num_images; ++i) {
        Mat gray = isColor ? Mat() : imgs[i];
        if (isColor) cvtColor(imgs[i], gray, COLOR_BGR2GRAY);

        Sobel(gray, grad_x[i], CV_32F, 1, 0, 3);
        Sobel(gray, grad_y[i], CV_32F, 0, 1, 3);

        out[i] = Mat::zeros(imgs[i].size(), CV_32F);
        srcDescs[i] = createRppDescriptor(grad_x[i], RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_phase(grad_x[i].data, grad_y[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                           &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "phase");
        }
    }
    perfMonitor.stop();
    ostringstream params;
    params << "inputs=2";
    printResult("RPP HOST Phase", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Normalize(const vector<Mat>& imgs, bool isColor,
                                         rppHandle_t handle) {
    int num_images = (int)imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;
    int nDim = 3;  // 3D per-image: H, W, C

    // Create output buffers for each image
    vector<Mat> outputImages(num_images);
    for (int i = 0; i < num_images; ++i)
        outputImages[i] = Mat::zeros(imgs[i].size(), imgs[i].type());

    // Create generic descriptor for SINGLE IMAGE (batchSize=1)
    RpptGenericDesc srcGenericDesc, dstGenericDesc;
    srcGenericDesc.numDims = nDim + 1;  // 4D: N, H, W, C
    srcGenericDesc.offsetInBytes = 0;
    srcGenericDesc.dataType = RpptDataType::U8;
    srcGenericDesc.layout = RpptLayout::NHWC;
    srcGenericDesc.dims[0] = 1;  // *** BATCH SIZE = 1 (single image) ***
    srcGenericDesc.dims[1] = height;
    srcGenericDesc.dims[2] = width;
    srcGenericDesc.dims[3] = channels;
    srcGenericDesc.strides[3] = 1;
    srcGenericDesc.strides[2] = channels;
    srcGenericDesc.strides[1] = width * channels;
    srcGenericDesc.strides[0] = height * width * channels;

    dstGenericDesc = srcGenericDesc;

    // axisMask: Bit 0=H, Bit 1=W, Bit 2=C
    // axisMask=3 (0b011) -> normalize over H and W, keep C separate
    Rpp32u axisMask = 3;

    // ROI for SINGLE IMAGE
    vector<Rpp32u> roiTensor(nDim * 2);  // Only 6 values for one image
    roiTensor[0] = 0;                    // h_start
    roiTensor[1] = 0;                    // w_start
    roiTensor[2] = 0;                    // c_start
    roiTensor[3] = height;               // h_size
    roiTensor[4] = width;                // w_size
    roiTensor[5] = channels;             // c_size

    // Mean/stddev for SINGLE IMAGE (one value per channel)
    vector<Rpp32f> meanTensor(channels, 0.0f);
    vector<Rpp32f> stdDevTensor(channels, 0.0f);

    Rpp8u computeMeanStddev = 3;  // Compute both internally
    Rpp32f scale = 1.0f;
    Rpp32f shift = 0.0f;

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_normalize(imgs[i].data, &srcGenericDesc, outputImages[i].data, &dstGenericDesc,
                               axisMask, meanTensor.data(), stdDevTensor.data(), computeMeanStddev,
                               scale, shift, roiTensor.data(), handle, RPP_HOST_BACKEND),
                "normalize");
        }
    }
    perfMonitor.stop();

    ostringstream params;
    params << "mode=auto_compute, scale=" << scale << ", shift=" << shift;
    printResult("RPP HOST Normalize", num_images, isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_WarpPerspective(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    // Create perspective transform matrix (simple example)
    float perspectiveMatrix[9] = {1.0f, 0.1f, 0.0f, 0.1f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_warp_perspective(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                      perspectiveMatrix, RpptInterpolationType::BILINEAR, &rois[i],
                                      RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "warp_perspective");
        }
    }
    perfMonitor.stop();
    printResult("RPP HOST WarpPerspective", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_Remap(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    // Create simple identity remap (with slight distortion)
    int h = imgs[0].rows;
    int w = imgs[0].cols;
    vector<Rpp32f> mapX(num_images * h * w);
    vector<Rpp32f> mapY(num_images * h * w);

    // Create table descriptor
    RpptDesc tableDesc;
    tableDesc.n = num_images;
    tableDesc.h = h;
    tableDesc.w = w;
    tableDesc.c = 1;
    tableDesc.strides.nStride = h * w;
    tableDesc.strides.hStride = w;
    tableDesc.strides.wStride = 1;
    tableDesc.strides.cStride = 1;

    for (int i = 0; i < num_images; ++i) {
        Rpp32f* mapXImg = mapX.data() + i * h * w;
        Rpp32f* mapYImg = mapY.data() + i * h * w;

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int idx = y * w + x;
                mapYImg[idx] = y + sin(x * 0.01f) * 5.0f;
                mapXImg[idx] = x + cos(y * 0.01f) * 5.0f;
            }
        }

        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_remap(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                        mapY.data() + i * h * w, mapX.data() + i * h * w,
                                        &tableDesc, RpptInterpolationType::BILINEAR, &rois[i],
                                        RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "remap");
        }
    }
    perfMonitor.stop();
    ostringstream params;
    params << "transform=sine_wave, interpolation=bilinear";
    printResult("RPP HOST Remap", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_FusedMultiplyAddScalar(const vector<Mat>& imgs, bool isColor, Rpp32f mul,
                                          Rpp32f add, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Convert to F32 and create batch buffer
    int imageSize = height * width * channels;
    vector<Rpp32f> inputBuffer(num_images * imageSize);
    vector<Rpp32f> outputBuffer(num_images * imageSize);

    for (int i = 0; i < num_images; ++i) {
        Mat imgF32;
        imgs[i].convertTo(imgF32, CV_32F);
        memcpy(inputBuffer.data() + i * imageSize, imgF32.data, imageSize * sizeof(Rpp32f));
    }

    // Create 5D generic descriptor (NDHWC or NCDHW) with depth=1 for 2D images
    RpptGenericDesc genericDesc;
    genericDesc.numDims = 5;
    genericDesc.offsetInBytes = 0;
    genericDesc.dataType = RpptDataType::F32;

    if (isColor) {
        // NDHWC layout
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
        // NCDHW layout for grayscale
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

    // Create mul and add tensors (one value per image in batch)
    vector<Rpp32f> mulTensor(num_images, mul);
    vector<Rpp32f> addTensor(num_images, add);

    // Create ROI tensor using XYZWHD format (full image ROI for each image)
    vector<RpptROI3D> roiTensor(num_images);
    for (int i = 0; i < num_images; ++i) {
        roiTensor[i].xyzwhdROI.xyz.x = 0;
        roiTensor[i].xyzwhdROI.xyz.y = 0;
        roiTensor[i].xyzwhdROI.xyz.z = 0;
        roiTensor[i].xyzwhdROI.roiWidth = width;
        roiTensor[i].xyzwhdROI.roiHeight = height;
        roiTensor[i].xyzwhdROI.roiDepth = 1;
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }

        CHECK_RPP_STATUS(rppt_fused_multiply_add_scalar(
                             inputBuffer.data(), &genericDesc, outputBuffer.data(), &genericDesc,
                             mulTensor.data(), addTensor.data(), roiTensor.data(),
                             RpptRoi3DType::XYZWHD, handle, RPP_HOST_BACKEND),
                         "fused_multiply_add_scalar");
    }
    perfMonitor.stop();
    ostringstream params;
    params << "mul=" << mul << ", add=" << add;
    printResult("RPP HOST FusedMultiplyAddScalar", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Posterize(const vector<Mat>& imgs, bool isColor, Rpp32u bits,
                             rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);
    vector<Rpp8u> bitsTensor(num_images, (Rpp8u)bits);  // FIXED: was Rpp32u

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_posterize(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                            bitsTensor.data(), &rois[i], RpptRoiType::XYWH, handle,
                                            RPP_HOST_BACKEND),
                             "posterize");
        }
    }
    perfMonitor.stop();
    stringstream ss;
    ss << "bits=" << bits;
    printResult("RPP HOST Posterize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), ss.str());
}

void benchmark_RPP_HOST_Solarize(const vector<Mat>& imgs, bool isColor, Rpp8u threshold,
                            rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);
    vector<Rpp32f> thresholdTensor(num_images, threshold / 255.0f);  // FIXED: was Rpp8u

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_solarize(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                           thresholdTensor.data(), &rois[i], RpptRoiType::XYWH,
                                           handle, RPP_HOST_BACKEND),
                             "solarize");
        }
    }
    perfMonitor.stop();
    ostringstream params;
    params << "threshold=" << (int)threshold;
    printResult("RPP HOST Solarize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_NoiseShot(const vector<Mat>& imgs, bool isColor, Rpp32f shot_noise_factor,
                             rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);
    vector<Rpp32f> shotNoiseFactor(num_images, shot_noise_factor);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_shot_noise(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                shotNoiseFactor.data(), 12345,  // FIXED: added seed parameter
                                &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "shot_noise");
        }
    }
    perfMonitor.stop();
    stringstream ss;
    ss << "factor=" << shot_noise_factor;
    printResult("RPP HOST NoiseShot", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), ss.str());
}

void benchmark_RPP_HOST_Gridmask(const vector<Mat>& imgs, bool isColor, Rpp32u tileWidth,
                            Rpp32f gridRatio, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    RpptUintVector2D translateVector = {0, 0};  // No translation

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_gridmask(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], tileWidth,
                              gridRatio, 0.0f, translateVector,  // Uses scalars!
                              &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "gridmask");
        }
    }
    perfMonitor.stop();
    stringstream ss;
    ss << "tileWidth=" << tileWidth << ", ratio=" << gridRatio;
    printResult("RPP HOST Gridmask", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), ss.str());
}

void benchmark_RPP_HOST_ColorCast(const vector<Mat>& imgs, bool isColor, Rpp32f rShift, Rpp32f gShift,
                             Rpp32f bShift, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);
    vector<RpptRGB> rgbTensor(num_images);
    vector<Rpp32f> alphaTensor(num_images, 1.0f);

    for (int i = 0; i < num_images; ++i) {
        rgbTensor[i].R = (Rpp8u)max(0.0f, min(255.0f, rShift));
        rgbTensor[i].G = (Rpp8u)max(0.0f, min(255.0f, gShift));
        rgbTensor[i].B = (Rpp8u)max(0.0f, min(255.0f, bShift));

        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_color_cast(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                             rgbTensor.data(), alphaTensor.data(), &rois[i],
                                             RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "color_cast");
        }
    }
    perfMonitor.stop();
    stringstream ss;
    ss << "R=" << rShift << ", G=" << gShift << ", B=" << bShift;
    printResult("RPP HOST ColorCast", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), ss.str());
}

void benchmark_RPP_HOST_ColorTemperature(const vector<Mat>& imgs, bool isColor, Rpp32s adjustmentValue,
                                    rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);
    vector<Rpp32s> adjustmentTensor(num_images, adjustmentValue);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_color_temperature(imgs[i].data, &srcDescs[i], out[i].data,
                                                    &dstDescs[i], adjustmentTensor.data(), &rois[i],
                                                    RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "color_temperature");
        }
    }
    perfMonitor.stop();
    stringstream ss;
    ss << "adjustment=" << adjustmentValue;
    printResult("RPP HOST ColorTemperature", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), ss.str());
}

void benchmark_RPP_HOST_Vignette(const vector<Mat>& imgs, bool isColor, Rpp32f vignetteIntensity,
                            rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);
    vector<Rpp32f> intensityTensor(num_images, vignetteIntensity);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_vignette(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                           intensityTensor.data(), &rois[i], RpptRoiType::XYWH,
                                           handle, RPP_HOST_BACKEND),
                             "vignette");
        }
    }
    perfMonitor.stop();
    stringstream ss;
    ss << "intensity=" << vignetteIntensity;
    printResult("RPP HOST Vignette", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), ss.str());
}

void benchmark_RPP_HOST_NonLinearBlend(const vector<Mat>& imgs, bool isColor, Rpp32f stdDev,
                                  rppHandle_t handle) {
    int num_images = (int)imgs.size();
    if (num_images < 2) {
        cout << "NonLinearBlend requires at least 2 images. Skipping." << endl;
        return;
    }

    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);
    vector<Rpp32f> stdDevTensor(num_images, stdDev);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images - 1; ++i) {
            CHECK_RPP_STATUS(
                rppt_non_linear_blend(imgs[i].data, imgs[i + 1].data, &srcDescs[i], out[i].data,
                                      &dstDescs[i], stdDevTensor.data(), &rois[i],
                                      RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "non_linear_blend");
        }
    }
    perfMonitor.stop();
    stringstream ss;
    ss << "stdDev=" << stdDev;
    printResult("RPP HOST NonLinearBlend", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), ss.str());
}

void benchmark_RPP_HOST_Erase(const vector<Mat>& imgs, bool isColor, Rpp32u boxesPerImage,
                         rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);
    int channels = isColor ? 3 : 1;

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            int h = imgs[i].rows;
            int w = imgs[i].cols;

            // Create fresh arrays for each image
            vector<RpptRoiLtrb> anchorBoxInfoTensor(boxesPerImage);
            vector<Rpp32f> colorBuffer(boxesPerImage * channels);
            Rpp32u numBoxes = boxesPerImage;

            // Create random boxes
            for (Rpp32u b = 0; b < boxesPerImage; ++b) {
                int box_w = 50 + (rand() % 100);
                int box_h = 50 + (rand() % 100);
                int x = rand() % max(1, w - box_w);
                int y = rand() % max(1, h - box_h);

                anchorBoxInfoTensor[b].lt.x = x;
                anchorBoxInfoTensor[b].lt.y = y;
                anchorBoxInfoTensor[b].rb.x = x + box_w;
                anchorBoxInfoTensor[b].rb.y = y + box_h;

                // Fill color (gray)
                for (int c = 0; c < channels; ++c) colorBuffer[b * channels + c] = 128.0f;
            }

            CHECK_RPP_STATUS(rppt_erase(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                        anchorBoxInfoTensor.data(), colorBuffer.data(), &numBoxes,
                                        &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "erase");
        }
    }
    perfMonitor.stop();
    ostringstream params;
    params << "numBoxes=" << boxesPerImage;
    printResult("RPP HOST Erase", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_CoarseDropout(const vector<Mat>& imgs, bool isColor, Rpp32u maxBoxesPerImage,
                                 rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            int h = imgs[i].rows;
            int w = imgs[i].cols;

            // Random number of boxes (2 to maxBoxesPerImage)
            Rpp32u numBoxes = 2 + (rand() % (maxBoxesPerImage - 1));

            // Create fresh box array for each image
            vector<RpptRoiLtrb> anchorBoxInfoTensor(maxBoxesPerImage);

            for (Rpp32u b = 0; b < numBoxes; ++b) {
                int box_w = 30 + (rand() % 80);
                int box_h = 30 + (rand() % 80);
                int x = rand() % max(1, w - box_w);
                int y = rand() % max(1, h - box_h);

                anchorBoxInfoTensor[b].lt.x = x;
                anchorBoxInfoTensor[b].lt.y = y;
                anchorBoxInfoTensor[b].rb.x = x + box_w;
                anchorBoxInfoTensor[b].rb.y = y + box_h;
            }

            // Fill remaining boxes with zeros
            for (Rpp32u b = numBoxes; b < maxBoxesPerImage; ++b) {
                anchorBoxInfoTensor[b].lt.x = 0;
                anchorBoxInfoTensor[b].lt.y = 0;
                anchorBoxInfoTensor[b].rb.x = 0;
                anchorBoxInfoTensor[b].rb.y = 0;
            }

            CHECK_RPP_STATUS(
                rppt_coarse_dropout(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                    anchorBoxInfoTensor.data(), &numBoxes, maxBoxesPerImage,
                                    &rois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "coarse_dropout");
        }
    }
    perfMonitor.stop();
    stringstream ss;
    ss << "maxBoxes=" << maxBoxesPerImage;
    printResult("RPP HOST CoarseDropout", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), ss.str());
}

void benchmark_RPP_HOST_GridDropout(const vector<Mat>& imgs, bool isColor, Rpp32u numGridsPerRow,
                               Rpp32u numGridsPerColumn, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    Rpp32u boxesInEachImage = numGridsPerRow * numGridsPerColumn;

    // Pre-create output buffers and descriptors
    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    Rpp32u totalBoxes = num_images * boxesInEachImage;
    Rpp32f holeRatio = 0.4f;
    int seed = 12345;  // Fixed seed for reproducibility

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        // Initialize anchor boxes for entire batch using proper helper function
        vector<RpptRoiLtrb> anchorBoxInfoTensor(totalBoxes);
        Rpp32u maxHoleW = 0, maxHoleH = 0;

        init_grid_dropout_boxes(num_images, anchorBoxInfoTensor.data(), rois.data(),
                                numGridsPerColumn, numGridsPerRow, maxHoleW, maxHoleH, holeRatio,
                                seed);

        // Process each image individually
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_grid_dropout(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                  anchorBoxInfoTensor.data() + i * boxesInEachImage,
                                  boxesInEachImage, maxHoleW, maxHoleH, &rois[i], RpptRoiType::XYWH,
                                  handle, RPP_HOST_BACKEND),
                "grid_dropout");
        }
    }
    perfMonitor.stop();
    ostringstream params;
    params << "tileWidth=" << numGridsPerRow << ", tileHeight=" << numGridsPerColumn;
    printResult("RPP HOST GridDropout", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_RandomErase(const vector<Mat>& imgs, bool isColor, int numBoxes, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);
    int channels = isColor ? 3 : 1;

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            int h = imgs[i].rows;
            int w = imgs[i].cols;

            for (int b = 0; b < numBoxes; ++b) {
                // Random box for this image
                int box_w = 50 + (rand() % 150);
                int box_h = 50 + (rand() % 150);
                int x = rand() % max(1, w - box_w);
                int y = rand() % max(1, h - box_h);

                // Create fresh box for each image
                RpptRoiLtrb anchorBox;
                anchorBox.lt.x = x;
                anchorBox.lt.y = y;
                anchorBox.rb.x = x + box_w;
                anchorBox.rb.y = y + box_h;

                // Create noise buffer
                int bufferSize = box_w * box_h * channels;
                vector<Rpp32f> colorBuffer(bufferSize);
                for (int j = 0; j < bufferSize; ++j) colorBuffer[j] = (rand() % 256) / 255.0f;

                CHECK_RPP_STATUS(
                    rppt_random_erase(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i], &anchorBox,
                                      colorBuffer.data(), &rois[i], RpptRoiType::XYWH, handle,
                                      RPP_HOST_BACKEND),
                    "random_erase");
            }
        }
    }
    perfMonitor.stop();
    ostringstream params;
    params << "numBoxes=" << numBoxes;
    printResult("RPP HOST RandomErase", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_ColorTwist(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    if (!isColor) {
        cout << "ColorTwist requires RGB images. Skipping." << endl;
        return;
    }

    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    // Color twist parameters: hue shift
    vector<Rpp32f> alpha(num_images, 1.0f);
    vector<Rpp32f> beta(num_images, 0.0f);
    vector<Rpp32f> hueShift(num_images, 60.0f);  // 60 degrees
    vector<Rpp32f> saturationFactor(num_images, 1.3f);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_color_twist(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                              alpha.data(), beta.data(), hueShift.data(),
                                              saturationFactor.data(), &rois[i], RpptRoiType::XYWH,
                                              handle, RPP_HOST_BACKEND),
                             "color_twist");
        }
    }
    perfMonitor.stop();
    ostringstream params;
    params << "hue=" << hueShift[0] << ", saturation=" << saturationFactor[0];
    printResult("RPP HOST ColorTwist", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_CropAndPatch(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    if (num_images < 2) {
        cout << "CropAndPatch requires at least 2 images. Skipping." << endl;
        return;
    }

    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> dstRois(num_images);
    vector<RpptROI> cropRois(num_images);
    vector<RpptROI> patchRois(num_images);

    // Setup: crop from one image, patch into another
    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstRois[i] = createFullImageROI(imgs[i]);

        int h = imgs[i].rows;
        int w = imgs[i].cols;

        // Crop region (center quarter)
        cropRois[i].xywhROI.xy.x = w / 4;
        cropRois[i].xywhROI.xy.y = h / 4;
        cropRois[i].xywhROI.roiWidth = w / 2;
        cropRois[i].xywhROI.roiHeight = h / 2;

        // Patch region (same as crop)
        patchRois[i] = cropRois[i];
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images - 1; ++i) {
            // Crop from img[i], patch into img[i+1]
            CHECK_RPP_STATUS(
                rppt_crop_and_patch(imgs[i].data, imgs[i + 1].data, &srcDescs[i], out[i].data,
                                    &dstDescs[i], &dstRois[i], &cropRois[i], &patchRois[i],
                                    RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "crop_and_patch");
        }
    }
    perfMonitor.stop();

    ostringstream params;
    params << "crop=center_quarter, patch=center_quarter";
    printResult("RPP HOST CropAndPatch", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_CropMirrorNormalize(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> dstRois(num_images);
    int channels = isColor ? 3 : 1;

    // Normalization parameters
    vector<Rpp32f> offset(num_images * channels);
    vector<Rpp32f> multiplier(num_images * channels);
    vector<Rpp32u> mirror(num_images, 1);  // 1 = horizontal flip

    if (isColor) {
        Rpp32f mean[3] = {60.0f, 80.0f, 100.0f};
        Rpp32f stdDev[3] = {0.9f, 0.9f, 0.9f};
        for (int i = 0, j = 0; i < num_images; i++, j += 3) {
            for (int c = 0; c < 3; c++) {
                offset[j + c] = -mean[c] / stdDev[c];
                multiplier[j + c] = 1.0f / stdDev[c];
            }
        }
    } else {
        Rpp32f mean = 100.0f;
        Rpp32f stdDev = 0.9f;
        for (int i = 0; i < num_images; i++) {
            offset[i] = -mean / stdDev;
            multiplier[i] = 1.0f / stdDev;
        }
    }

    for (int i = 0; i < num_images; ++i) {
        int h = imgs[i].rows;
        int w = imgs[i].cols;

        // Output size (half of input as crop)
        out[i] = Mat::zeros(h / 2, w / 2, imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);

        // Destination ROI (where to crop from source)
        dstRois[i].xywhROI.xy.x = w / 4;
        dstRois[i].xywhROI.xy.y = h / 4;
        dstRois[i].xywhROI.roiWidth = w / 2;
        dstRois[i].xywhROI.roiHeight = h / 2;
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_crop_mirror_normalize(
                                 imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                 offset.data(), multiplier.data(), mirror.data(), &dstRois[i],
                                 RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "crop_mirror_normalize");
        }
    }
    perfMonitor.stop();

    ostringstream params;
    params << "crop=half, mirror=horizontal, mean=60/80/100, stddev=0.9";
    printResult("RPP HOST CropMirrorNormalize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_ResizeMirrorNormalize(const vector<Mat>& imgs, bool isColor,
                                         rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> dstRois(num_images);
    vector<RpptImagePatch> dstImgSizes(num_images);
    int channels = isColor ? 3 : 1;

    // Normalization parameters
    vector<Rpp32f> mean(num_images * channels);
    vector<Rpp32f> stdDev(num_images * channels);
    vector<Rpp32u> mirror(num_images, 1);  // Horizontal flip

    for (int i = 0, j = 0; i < num_images; i++, j += channels) {
        if (isColor) {
            mean[j] = 60.0f;
            stdDev[j] = 1.0f;
            mean[j + 1] = 80.0f;
            stdDev[j + 1] = 1.0f;
            mean[j + 2] = 100.0f;
            stdDev[j + 2] = 1.0f;
        } else {
            mean[i] = 100.0f;
            stdDev[i] = 1.0f;
        }
    }

    for (int i = 0; i < num_images; ++i) {
        int h = imgs[i].rows;
        int w = imgs[i].cols;

        // Resize to half
        dstImgSizes[i].width = w / 2;
        dstImgSizes[i].height = h / 2;

        out[i] = Mat::zeros(h / 2, w / 2, imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);

        dstRois[i].xywhROI.xy.x = 0;
        dstRois[i].xywhROI.xy.y = 0;
        dstRois[i].xywhROI.roiWidth = w / 2;
        dstRois[i].xywhROI.roiHeight = h / 2;
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_resize_mirror_normalize(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                             dstImgSizes.data(), RpptInterpolationType::BILINEAR,
                                             mean.data(), stdDev.data(), mirror.data(), &dstRois[i],
                                             RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "resize_mirror_normalize");
        }
    }
    perfMonitor.stop();

    ostringstream params;
    params << "resize=half, mirror=horizontal, mean=60/80/100, stddev=1";
    printResult("RPP HOST ResizeMirrorNormalize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_ResizeCropMirror(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> dstRois(num_images);
    vector<RpptImagePatch> dstImgSizes(num_images);
    vector<Rpp32u> mirror(num_images, 1);  // Horizontal flip

    // Target parameters (same as all other versions)
    int targetWidth = 224;
    int targetHeight = 224;

    // Calculate crop parameters once (all images have same dimensions)
    int cropWidth = (imgs[0].cols * 4) / 5;
    int cropHeight = (imgs[0].rows * 4) / 5;
    int cropX = (imgs[0].cols - cropWidth) / 2;
    int cropY = (imgs[0].rows - cropHeight) / 2;

    for (int i = 0; i < num_images; ++i) {
        // Resize to target size
        dstImgSizes[i].width = targetWidth;
        dstImgSizes[i].height = targetHeight;

        // ROI defines source crop region
        dstRois[i].xywhROI.xy.x = cropX;
        dstRois[i].xywhROI.xy.y = cropY;
        dstRois[i].xywhROI.roiWidth = cropWidth;
        dstRois[i].xywhROI.roiHeight = cropHeight;

        out[i] = Mat::zeros(targetHeight, targetWidth, imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_resize_crop_mirror(
                                 imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                 dstImgSizes.data(), RpptInterpolationType::BILINEAR, mirror.data(),
                                 &dstRois[i], RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "resize_crop_mirror");
        }
    }
    perfMonitor.stop();
    ostringstream params;
    params << "crop=80%,resize=224x224,mirror";
    printResult("RPP HOST ResizeCropMirror", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_RICAP(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    if (num_images < 4) {
        cout << "RICAP requires at least 4 images. Skipping." << endl;
        return;
    }

    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);

    int maxWidth = imgs[0].cols;
    int maxHeight = imgs[0].rows;

    vector<Rpp32u> permutationTensor(num_images * 4);
    RpptROI roiPtrInputCropRegion[4];

    init_ricap_boxes(maxWidth, maxHeight, num_images, permutationTensor.data(),
                     roiPtrInputCropRegion);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_ricap(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                        permutationTensor.data(), roiPtrInputCropRegion,
                                        RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "ricap");
        }
    }
    perfMonitor.stop();
    printResult("RPP HOST RICAP", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), "4way_cutmix");
}

// Forward declaration of helper functions from dropout_helpers.cpp
void generate_channel_dropout_mask(Rpp8u* dropoutTensor, Rpp32f* dropoutProbability, int batchSize,
                                   int channels, int seed);
void init_cutout_dropout(int batchSize, int maxBoxesPerImage, Rpp32u* numOfBoxes,
                         RpptRoiLtrb* anchorBoxInfoTensor, RpptROIPtr roiTensorPtrSrc, int channels,
                         int BitDepthTestMode, int seed, int dropoutType, void* colorBuffer);

void benchmark_RPP_HOST_ChannelDropout(const vector<Mat>& imgs, bool isColor, float dropoutProb,
                                  rppHandle_t handle) {
    if (!isColor) {
        cout << "RPP HOST ChannelDropout - skipped (requires RGB)" << endl;
        return;
    }

    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    // Generate channel dropout mask
    int channels = 3;
    vector<Rpp32f> dropoutProbability(num_images, dropoutProb);
    vector<Rpp8u> dropoutTensor(num_images * channels);
    Rpp32u seed = 12345;

    generate_channel_dropout_mask(dropoutTensor.data(), dropoutProbability.data(), num_images,
                                  channels, seed);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(rppt_channel_dropout(imgs[i].data, &srcDescs[i], out[i].data,
                                                  &dstDescs[i], dropoutTensor.data(), &rois[i],
                                                  RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                             "channel_dropout");
        }
    }
    perfMonitor.stop();

    ostringstream params;
    params << "dropoutProb=" << dropoutProb;
    printResult("RPP HOST ChannelDropout", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_CutoutDropout(const vector<Mat>& imgs, bool isColor, Rpp32u numBoxes,
                                 rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    Rpp32u boxesInEachImage = numBoxes;
    Rpp32u seed = 12345;
    int channels = isColor ? 3 : 1;

    vector<RpptRoiLtrb> anchorBoxInfoTensor(num_images * boxesInEachImage);
    vector<Rpp32u> numBoxesTensor(num_images * boxesInEachImage, numBoxes);
    vector<Rpp8u> colorBuffer(num_images * boxesInEachImage * channels);

    for (int i = 0; i < num_images; ++i) {
        rois[i] = createFullImageROI(imgs[i]);
    }

    init_cutout_dropout(num_images, boxesInEachImage, numBoxesTensor.data(),
                        anchorBoxInfoTensor.data(), rois.data(), channels, 0 /* U8_TO_U8 */, seed,
                        1 /* cutout type */, colorBuffer.data());

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_cutout_dropout(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                    anchorBoxInfoTensor.data(), colorBuffer.data(),
                                    numBoxesTensor.data(), &rois[i], RpptRoiType::XYWH, handle,
                                    RPP_HOST_BACKEND),
                "cutout_dropout");
        }
    }
    perfMonitor.stop();

    ostringstream params;
    params << "numBoxes=" << numBoxes;
    printResult("RPP HOST CutoutDropout", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_JpegCompressionDistortion(const vector<Mat>& imgs, bool isColor, Rpp32s quality,
                                             rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);
    vector<RpptROI> rois(num_images);

    vector<Rpp32s> qualityTensor(num_images, quality);

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        rois[i] = createFullImageROI(imgs[i]);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_jpeg_compression_distortion(imgs[i].data, &srcDescs[i], out[i].data,
                                                 &dstDescs[i], qualityTensor.data(), &rois[i],
                                                 RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                "jpeg_compression_distortion");
        }
    }
    perfMonitor.stop();

    ostringstream params;
    params << "quality=" << quality;
    printResult("RPP HOST JpegCompressionDistortion", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_ChannelPermute(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    if (!isColor) {
        cout << "RPP HOST ChannelPermute - skipped (requires RGB)" << endl;
        return;
    }

    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);
    vector<RpptDesc> srcDescs(num_images);
    vector<RpptDesc> dstDescs(num_images);

    // Permutation tensor: BGR to RGB (swap channels 0 and 2)
    // For each image, specify permutation [2, 1, 0] to swap B and R channels
    vector<Rpp32u> permutationTensor(num_images * 3);
    for (int i = 0; i < num_images; i++) {
        permutationTensor[i * 3 + 0] = 2;  // R <- B
        permutationTensor[i * 3 + 1] = 1;  // G <- G
        permutationTensor[i * 3 + 2] = 0;  // B <- R
    }

    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        srcDescs[i] = createRppDescriptor(imgs[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
        dstDescs[i] = createRppDescriptor(out[i], isColor ? RpptLayout::NHWC : RpptLayout::NCHW);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        for (int i = 0; i < num_images; ++i) {
            CHECK_RPP_STATUS(
                rppt_channel_permute(imgs[i].data, &srcDescs[i], out[i].data, &dstDescs[i],
                                     permutationTensor.data(), handle, RPP_HOST_BACKEND),
                "channel_permute");
        }
    }
    perfMonitor.stop();
    ostringstream params;
    params << "permutation=BGR->RGB";
    printResult("RPP HOST ChannelPermute", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Slice(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int num_images = (int)imgs.size();
    vector<Mat> out(num_images);

    int channels = isColor ? 3 : 1;
    int numDims = 3;  // H, W, C (excluding batch) for NHWC layout

    // Slice parameters: extract center region (half the size)
    vector<Rpp32s> anchorTensor(num_images * numDims);
    vector<Rpp32s> shapeTensor(num_images * numDims);
    vector<Rpp32u> roiTensor(num_images * numDims * 2);
    vector<RpptROI> rois(num_images);

    // Create output buffer (same size as input, but we'll only use the sliced portion)
    for (int i = 0; i < num_images; ++i) {
        out[i] = Mat::zeros(imgs[i].size(), imgs[i].type());
        rois[i] = createFullImageROI(imgs[i]);
    }

    // Create generic descriptor for the batch
    RpptGenericDesc genericDesc;
    genericDesc.numDims = 4;  // N, H, W, C for NHWC
    genericDesc.offsetInBytes = 0;
    genericDesc.dataType = RpptDataType::U8;
    genericDesc.layout = RpptLayout::NHWC;

    genericDesc.dims[0] = num_images;    // Batch size
    genericDesc.dims[1] = imgs[0].rows;  // Height
    genericDesc.dims[2] = imgs[0].cols;  // Width
    genericDesc.dims[3] = channels;      // Channels

    // Set strides for NHWC layout
    genericDesc.strides[0] = imgs[0].rows * imgs[0].cols * channels;  // Batch stride
    genericDesc.strides[1] = imgs[0].cols * channels;                 // Height stride
    genericDesc.strides[2] = channels;                                // Width stride
    genericDesc.strides[3] = 1;                                       // Channel stride

    // Initialize anchor, shape, and ROI tensors using the same logic as test suite
    for (int i = 0; i < num_images; ++i) {
        int h = imgs[i].rows;
        int w = imgs[i].cols;

        int idx1 = i * 3;  // 3 dims per image (H, W, C)
        int idx2 = i * 6;  // 6 values per image (3 pairs for ROI)

        // For NHWC: order is [H, W, C]
        // Anchor: starting position (extract from center, starting at 1/4 of dimensions)
        roiTensor[idx2 + 0] = anchorTensor[idx1 + 0] = h / 4;  // Y anchor
        roiTensor[idx2 + 1] = anchorTensor[idx1 + 1] = w / 4;  // X anchor
        roiTensor[idx2 + 2] = anchorTensor[idx1 + 2] = 0;      // C anchor (all channels)

        // ROI bounds
        roiTensor[idx2 + 3] = h;         // H max
        roiTensor[idx2 + 4] = w;         // W max
        roiTensor[idx2 + 5] = channels;  // C max

        // Shape: size to extract (half the image size)
        shapeTensor[idx1 + 0] = h / 2;     // H shape
        shapeTensor[idx1 + 1] = w / 2;     // W shape
        shapeTensor[idx1 + 2] = channels;  // C shape (all channels)
    }

    Rpp8u fillValue = 0;
    bool enablePadding = false;

    // Concatenate all images into a single buffer
    size_t imageSize = imgs[0].rows * imgs[0].cols * channels;
    vector<Rpp8u> inputBuffer(num_images * imageSize);
    vector<Rpp8u> outputBuffer(num_images * imageSize);

    for (int i = 0; i < num_images; ++i) {
        memcpy(inputBuffer.data() + i * imageSize, imgs[i].data, imageSize);
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(
            rppt_slice(inputBuffer.data(), &genericDesc, outputBuffer.data(), &genericDesc,
                       anchorTensor.data(), shapeTensor.data(), &fillValue, enablePadding,
                       roiTensor.data(), handle, RPP_HOST_BACKEND),
            "slice");
    }
    perfMonitor.stop();

    // Copy results back to output mats (only the sliced portion)
    for (int i = 0; i < num_images; ++i) {
        int sliceH = imgs[i].rows / 2;
        int sliceW = imgs[i].cols / 2;
        int startY = imgs[i].rows / 4;
        int startX = imgs[i].cols / 4;

        // Extract the sliced region from output buffer
        for (int y = 0; y < sliceH; ++y) {
            for (int x = 0; x < sliceW; ++x) {
                for (int c = 0; c < channels; ++c) {
                    size_t outIdx = i * imageSize + (y * imgs[i].cols + x) * channels + c;
                    size_t matIdx = ((startY + y) * imgs[i].cols + (startX + x)) * channels + c;
                    out[i].data[matIdx] = outputBuffer[outIdx];
                }
            }
        }
    }

    ostringstream params;
    params << "slice=center_50%";
    printResult("RPP HOST Slice", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

// ==================== BATCHED RPP HOST BENCHMARK FUNCTIONS ====================
// Following reference implementation from Tensor_image_host.cpp
// Memory allocation: calloc/free (not hipHostMalloc)
// Descriptor setup: set_descriptor_dims_and_strides from rpp_test_suite_image.h

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

void benchmark_RPP_HOST_Flip_Batched(const vector<Mat>& imgs, bool isColor, int flipCode, rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;
    
    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32u *horizontalTensor = static_cast<Rpp32u*>(calloc(batchSize, sizeof(Rpp32u)));
    Rpp32u *verticalTensor = static_cast<Rpp32u*>(calloc(batchSize, sizeof(Rpp32u)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        horizontalTensor[i] = (flipCode == 1 || flipCode == -1) ? 1 : 0;
        verticalTensor[i] = (flipCode == 0 || flipCode == -1) ? 1 : 0;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_flip(input, &srcDesc, output, &dstDesc, horizontalTensor, verticalTensor, 
                                   roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND), "Flip");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(horizontalTensor);
    free(verticalTensor);
    free(roiTensor);

    string name = (flipCode == 1) ? "Horizontal" : (flipCode == 0) ? "Vertical" : "Both";
    ostringstream params;
    params << "type=" << name;
    printResult("RPP HOST BATCH Flip", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

// This file contains the batched implementations to match HIP backend

#include "benchmarks_common.h"

void benchmark_RPP_HOST_Resize_Batched(const vector<Mat>& imgs, bool isColor, int dstW, int dstH,
                                      RpptInterpolationType interpType, const string& interpName,
                                      rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.dataType = RpptDataType::U8;
    set_descriptor_dims_and_strides(&dstDesc, batchSize, dstH, dstW, numChannels, offsetInBytes);

    Rpp64u srcBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp64u dstBufferSize = (Rpp64u)dstDesc.h * (Rpp64u)dstDesc.w * (Rpp64u)dstDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(srcBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(dstBufferSize, sizeof(Rpp8u)));

    RpptImagePatch *dstImgSizes = static_cast<RpptImagePatch*>(calloc(batchSize, sizeof(RpptImagePatch)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        dstImgSizes[i].width = dstW;
        dstImgSizes[i].height = dstH;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_resize(input, &srcDesc, output, &dstDesc, dstImgSizes, interpType,
                                     roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND), "Resize");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(dstImgSizes);
    free(roiTensor);

    ostringstream params;
    params << "type=" << interpName << ", size=" << dstW << "x" << dstH;
    printResult("RPP HOST BATCH Resize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Crop_Batched(const vector<Mat>& imgs, bool isColor, int cropW, int cropH,
                                     rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.dataType = RpptDataType::U8;
    set_descriptor_dims_and_strides(&dstDesc, batchSize, cropH, cropW, numChannels, offsetInBytes);

    Rpp64u srcBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp64u dstBufferSize = (Rpp64u)dstDesc.h * (Rpp64u)dstDesc.w * (Rpp64u)dstDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(srcBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(dstBufferSize, sizeof(Rpp8u)));

    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    int cropX = (maxWidth - cropW) / 2;
    int cropY = (maxHeight - cropH) / 2;

    for (int i = 0; i < batchSize; i++) {
        roiTensor[i].xywhROI.xy.x = cropX;
        roiTensor[i].xywhROI.xy.y = cropY;
        roiTensor[i].xywhROI.roiWidth = cropW;
        roiTensor[i].xywhROI.roiHeight = cropH;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_crop(input, &srcDesc, output, &dstDesc,
                                   roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND), "Crop");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(roiTensor);

    ostringstream params;
    params << "size=" << cropW << "x" << cropH;
    printResult("RPP HOST BATCH Crop", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Rotate_Batched(const vector<Mat>& imgs, bool isColor, float angleDeg,
                                       rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32f *angleTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        angleTensor[i] = angleDeg;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_rotate(input, &srcDesc, output, &dstDesc, angleTensor,
                                     RpptInterpolationType::BILINEAR, roiTensor, RpptRoiType::XYWH,
                                     handle, RPP_HOST_BACKEND), "Rotate");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(angleTensor);
    free(roiTensor);

    ostringstream params;
    params << "angle=" << angleDeg << "deg";
    printResult("RPP HOST BATCH Rotate", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_BoxFilter_Batched(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                          rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_box_filter(input, &srcDesc, output, &dstDesc, kernelSize, borderType,
                                         roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND), "BoxFilter");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(roiTensor);

    
    ostringstream params;
    params << "kernel=" << kernelSize;
    printResult("RPP HOST BATCH BoxFilter", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_GaussianFilter_Batched(const vector<Mat>& imgs, bool isColor, int kernelSize, float sigma,
                                               rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    int offsetInBytes = 12 * (kernelSize / 2);

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32f *stdDevTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        stdDevTensor[i] = sigma;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_gaussian_filter(input, &srcDesc, output, &dstDesc, stdDevTensor, kernelSize,
                                              borderType, roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "GaussianFilter");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(stdDevTensor);
    free(roiTensor);


    ostringstream params;
    params << "kernel=" << kernelSize << ", sigma=" << sigma;
    printResult("RPP HOST BATCH GaussianFilter", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_MedianFilter_Batched(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                             rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_median_filter(input, &srcDesc, output, &dstDesc, kernelSize, borderType,
                                            roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "MedianFilter");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(roiTensor);

    
    ostringstream params;
    params << "kernel=" << kernelSize;
    printResult("RPP HOST BATCH MedianFilter", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_SobelFilter_Batched(const vector<Mat>& imgs, bool isColor, int sobelType,
                                            rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int dstChannels = 1;  // Sobel filter always outputs grayscale
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;
    Rpp32u kernelSize = 3;  // Sobel uses 3x3 kernel

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = RpptLayout::NCHW;  // Output is always grayscale (planar)
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, dstChannels, offsetInBytes);

    Rpp64u srcBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp64u dstBufferSize = (Rpp64u)dstDesc.h * (Rpp64u)dstDesc.w * (Rpp64u)dstDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(srcBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(dstBufferSize, sizeof(Rpp8u)));

    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_sobel_filter(input, &srcDesc, output, &dstDesc, sobelType, kernelSize,
                                           roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "SobelFilter");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(roiTensor);

    
    ostringstream params;
    params << "type=" << sobelType;
    printResult("RPP HOST BATCH SobelFilter", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_HistogramEqualize_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_histogram_equalize(input, &srcDesc, output, &dstDesc,
                                                 roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "HistogramEqualize");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(roiTensor);

    printResult("RPP HOST BATCH HistogramEqualize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_Hue_Batched(const vector<Mat>& imgs, bool isColor, float hueDelta, rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = 3;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = RpptLayout::NHWC;
    dstDesc.layout = RpptLayout::NHWC;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32f *hueTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        hueTensor[i] = hueDelta;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_hue(input, &srcDesc, output, &dstDesc, hueTensor,
                                  roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND), "Hue");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(hueTensor);
    free(roiTensor);

    ostringstream params;
    params << "hue=" << hueDelta;
    printResult("RPP HOST BATCH Hue", imgs.size(), true,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Saturation_Batched(const vector<Mat>& imgs, bool isColor, float satFactor, rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = 3;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = RpptLayout::NHWC;
    dstDesc.layout = RpptLayout::NHWC;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32f *saturationTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        saturationTensor[i] = satFactor;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_saturation(input, &srcDesc, output, &dstDesc, saturationTensor,
                                        roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND), "Saturation");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(saturationTensor);
    free(roiTensor);

    ostringstream params;
    params << "factor=" << satFactor;
    printResult("RPP HOST BATCH Saturation", imgs.size(), true,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_ColorToGreyscale_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int srcChannels = 3;
    int dstChannels = 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = RpptLayout::NHWC;
    srcDesc.dataType = RpptDataType::U8;
    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, srcChannels, offsetInBytes);

    dstDesc.layout = RpptLayout::NCHW;
    dstDesc.dataType = RpptDataType::U8;
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, dstChannels, offsetInBytes);

    Rpp64u srcBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp64u dstBufferSize = (Rpp64u)dstDesc.h * (Rpp64u)dstDesc.w * (Rpp64u)dstDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(srcBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(dstBufferSize, sizeof(Rpp8u)));

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * srcChannels, imgs[i].cols * srcChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_color_to_greyscale(input, &srcDesc, output, &dstDesc, RpptSubpixelLayout::BGRtype,
                                                 handle, RPP_HOST_BACKEND), "ColorToGreyscale");
    }
    perfMonitor.stop();

    free(input);
    free(output);

    printResult("RPP HOST BATCH ColorToGreyscale", imgs.size(), true,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_Brightness_Batched(const vector<Mat>& imgs, bool isColor, float alpha, float beta,
                                           rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32f *alphaTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    Rpp32f *betaTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        alphaTensor[i] = alpha;
        betaTensor[i] = beta;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_brightness(input, &srcDesc, output, &dstDesc, alphaTensor, betaTensor,
                                        roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND), "Brightness");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(alphaTensor);
    free(betaTensor);
    free(roiTensor);

    ostringstream params;
    params << "alpha=" << alpha << ", beta=" << beta;
    printResult("RPP HOST BATCH Brightness", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Contrast_Batched(const vector<Mat>& imgs, bool isColor, float contrastFactor, float contrastCenter,
                                        rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32f *contrastFactorTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    Rpp32f *contrastCenterTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        contrastFactorTensor[i] = contrastFactor;
        contrastCenterTensor[i] = contrastCenter;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_contrast(input, &srcDesc, output, &dstDesc, contrastFactorTensor, contrastCenterTensor,
                                       roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND), "Contrast");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(contrastFactorTensor);
    free(contrastCenterTensor);
    free(roiTensor);

    ostringstream params;
    params << "factor=" << contrastFactor << ", center=" << contrastCenter;
    printResult("RPP HOST BATCH Contrast", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Emboss_Batched(const vector<Mat>& imgs, bool isColor, int kernelSize, float strength,
                                       rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32f *strengthTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        strengthTensor[i] = strength;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    RpptImageBorderType borderType = RpptImageBorderType::REPLICATE;
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_emboss(input, &srcDesc, output, &dstDesc, strengthTensor, kernelSize, borderType,
                                     roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND), "Emboss");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(strengthTensor);
    free(roiTensor);

    ostringstream params;
    params << "kernel=" << kernelSize << ", strength=" << strength;
    printResult("RPP HOST BATCH Emboss", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_AddScalar_Batched(const vector<Mat>& imgs, bool isColor, float addVal,
                                         rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    // Scalar operations only support F32 datatype, not U8
    srcDesc.dataType = RpptDataType::F32;
    dstDesc.dataType = RpptDataType::F32;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    // Convert to GenericDesc for 3D arithmetic operations
    RpptGenericDesc srcGenDesc, dstGenDesc;
    convert_desc_to_generic_desc_3d(&srcDesc, &srcGenDesc);
    convert_desc_to_generic_desc_3d(&dstDesc, &dstGenDesc);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    // Allocate F32 buffers since scalar operations require F32 datatype
    Rpp32f *input = static_cast<Rpp32f*>(calloc(ioBufferSize, sizeof(Rpp32f)));
    Rpp32f *output = static_cast<Rpp32f*>(calloc(ioBufferSize, sizeof(Rpp32f)));

    Rpp32f *addTensor = static_cast<Rpp32f*>(calloc(batchSize * numChannels, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));
    RpptROI3D *roi3dTensor = static_cast<RpptROI3D*>(calloc(batchSize, sizeof(RpptROI3D)));

    for (int i = 0; i < batchSize; i++) {
        for (int c = 0; c < numChannels; c++) {
            addTensor[i * numChannels + c] = addVal;
        }
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    // Convert U8 image data to F32 (normalized to 0-255 range for consistency with OpenCV)
    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp32f* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            for (int col = 0; col < imgs[i].cols * numChannels; col++) {
                imgPtr[row * srcDesc.strides.hStride + col] =
                    static_cast<Rpp32f>(imgs[i].data[row * imgs[i].cols * numChannels + col]);
            }
        }
    }

    // Convert ROI to ROI3D
    convert_roi_to_roi3d(roiTensor, roi3dTensor, batchSize);

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_add_scalar(input, &srcGenDesc, output, &dstGenDesc, addTensor,
                                        roi3dTensor, RpptRoi3DType::XYZWHD, handle, RPP_HOST_BACKEND),
                        "AddScalar");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(addTensor);
    free(roiTensor);
    free(roi3dTensor);

    
    ostringstream params;
    params << "value=" << (int)addVal;
    printResult("RPP HOST BATCH AddScalar", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_SubtractScalar_Batched(const vector<Mat>& imgs, bool isColor, float subVal,
                                               rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    // Scalar operations only support F32 datatype, not U8
    srcDesc.dataType = RpptDataType::F32;
    dstDesc.dataType = RpptDataType::F32;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    // Convert to GenericDesc for 3D arithmetic operations
    RpptGenericDesc srcGenDesc, dstGenDesc;
    convert_desc_to_generic_desc_3d(&srcDesc, &srcGenDesc);
    convert_desc_to_generic_desc_3d(&dstDesc, &dstGenDesc);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    // Allocate F32 buffers since scalar operations require F32 datatype
    Rpp32f *input = static_cast<Rpp32f*>(calloc(ioBufferSize, sizeof(Rpp32f)));
    Rpp32f *output = static_cast<Rpp32f*>(calloc(ioBufferSize, sizeof(Rpp32f)));

    Rpp32f *subTensor = static_cast<Rpp32f*>(calloc(batchSize * numChannels, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));
    RpptROI3D *roi3dTensor = static_cast<RpptROI3D*>(calloc(batchSize, sizeof(RpptROI3D)));

    for (int i = 0; i < batchSize; i++) {
        for (int c = 0; c < numChannels; c++) {
            subTensor[i * numChannels + c] = subVal;
        }
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    // Convert U8 image data to F32 (normalized to 0-255 range for consistency with OpenCV)
    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp32f* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            for (int col = 0; col < imgs[i].cols * numChannels; col++) {
                imgPtr[row * srcDesc.strides.hStride + col] =
                    static_cast<Rpp32f>(imgs[i].data[row * imgs[i].cols * numChannels + col]);
            }
        }
    }

    // Convert ROI to ROI3D
    convert_roi_to_roi3d(roiTensor, roi3dTensor, batchSize);

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_subtract_scalar(input, &srcGenDesc, output, &dstGenDesc, subTensor,
                                              roi3dTensor, RpptRoi3DType::XYZWHD, handle, RPP_HOST_BACKEND),
                        "SubtractScalar");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(subTensor);
    free(roiTensor);
    free(roi3dTensor);

    
    ostringstream params;
    params << "value=" << (int)subVal;
    printResult("RPP HOST BATCH SubtractScalar", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_MultiplyScalar_Batched(const vector<Mat>& imgs, bool isColor, float mulVal,
                                               rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    // Scalar operations only support F32 datatype, not U8
    srcDesc.dataType = RpptDataType::F32;
    dstDesc.dataType = RpptDataType::F32;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    // Convert to GenericDesc for 3D arithmetic operations
    RpptGenericDesc srcGenDesc, dstGenDesc;
    convert_desc_to_generic_desc_3d(&srcDesc, &srcGenDesc);
    convert_desc_to_generic_desc_3d(&dstDesc, &dstGenDesc);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    // Allocate F32 buffers since scalar operations require F32 datatype
    Rpp32f *input = static_cast<Rpp32f*>(calloc(ioBufferSize, sizeof(Rpp32f)));
    Rpp32f *output = static_cast<Rpp32f*>(calloc(ioBufferSize, sizeof(Rpp32f)));

    Rpp32f *mulTensor = static_cast<Rpp32f*>(calloc(batchSize * numChannels, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));
    RpptROI3D *roi3dTensor = static_cast<RpptROI3D*>(calloc(batchSize, sizeof(RpptROI3D)));

    for (int i = 0; i < batchSize; i++) {
        for (int c = 0; c < numChannels; c++) {
            mulTensor[i * numChannels + c] = mulVal;
        }
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    // Convert U8 image data to F32 (normalized to 0-255 range for consistency with OpenCV)
    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp32f* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            for (int col = 0; col < imgs[i].cols * numChannels; col++) {
                imgPtr[row * srcDesc.strides.hStride + col] =
                    static_cast<Rpp32f>(imgs[i].data[row * imgs[i].cols * numChannels + col]);
            }
        }
    }

    // Convert ROI to ROI3D
    convert_roi_to_roi3d(roiTensor, roi3dTensor, batchSize);

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_multiply_scalar(input, &srcGenDesc, output, &dstGenDesc, mulTensor,
                                              roi3dTensor, RpptRoi3DType::XYZWHD, handle, RPP_HOST_BACKEND),
                        "MultiplyScalar");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(mulTensor);
    free(roiTensor);
    free(roi3dTensor);

    
    ostringstream params;
    params << "value=" << mulVal;
    printResult("RPP HOST BATCH MultiplyScalar", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_GaussianNoise_Batched(const vector<Mat>& imgs, bool isColor, float mean,
                                               float stddev, rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32f *meanTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    Rpp32f *stddevTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        meanTensor[i] = mean;
        stddevTensor[i] = stddev;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    unsigned long long seed = 12345;
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_gaussian_noise(input, &srcDesc, output, &dstDesc, meanTensor, stddevTensor,
                                             seed, roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "GaussianNoise");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(meanTensor);
    free(stddevTensor);
    free(roiTensor);

    
    ostringstream params;
    params << "mean=" << mean << ", stddev=" << stddev;
    printResult("RPP HOST BATCH GaussianNoise", imgs.size(), isColor,
                perfMonitor.getTotalTime(),
                perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_SaltAndPepperNoise_Batched(const vector<Mat>& imgs, bool isColor, float noiseProb,
                                                    rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32f *noiseProbTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    Rpp32f *saltProbTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    Rpp32f *saltValueTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    Rpp32f *pepperValueTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        noiseProbTensor[i] = noiseProb;
        saltProbTensor[i] = 0.5f;
        saltValueTensor[i] = 1.0f;
        pepperValueTensor[i] = 0.0f;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    Rpp32u seed = 12345;
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_salt_and_pepper_noise(input, &srcDesc, output, &dstDesc, noiseProbTensor,
                                                     saltProbTensor, saltValueTensor, pepperValueTensor,
                                                     seed, roiTensor, RpptRoiType::XYWH, handle,
                                                     RPP_HOST_BACKEND),
                        "SaltAndPepperNoise");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(noiseProbTensor);
    free(saltProbTensor);
    free(saltValueTensor);
    free(pepperValueTensor);
    free(roiTensor);

    
    ostringstream params;
    params << "probability=" << noiseProb;
    printResult("RPP HOST BATCH SaltAndPepperNoise", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_NoiseShot_Batched(const vector<Mat>& imgs, bool isColor, float shotNoiseFactor,
                                          rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32f *shotNoiseTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        shotNoiseTensor[i] = shotNoiseFactor;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_shot_noise(input, &srcDesc, output, &dstDesc, shotNoiseTensor, 12345,
                                         roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "NoiseShot");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(shotNoiseTensor);
    free(roiTensor);

    
    ostringstream params;
    params << "factor=" << shotNoiseFactor;
    printResult("RPP HOST BATCH NoiseShot", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_ColorCast_Batched(const vector<Mat>& imgs, bool isColor, float rShift, float gShift,
                                          float bShift, rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    RpptRGB *rgbTensor = static_cast<RpptRGB*>(calloc(batchSize, sizeof(RpptRGB)));
    Rpp32f *alphaTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        rgbTensor[i].R = (Rpp8u)max(0.0f, min(255.0f, rShift));
        rgbTensor[i].G = (Rpp8u)max(0.0f, min(255.0f, gShift));
        rgbTensor[i].B = (Rpp8u)max(0.0f, min(255.0f, bShift));
        alphaTensor[i] = 1.0f;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_color_cast(input, &srcDesc, output, &dstDesc, rgbTensor, alphaTensor,
                                         roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "ColorCast");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(rgbTensor);
    free(alphaTensor);
    free(roiTensor);

    ostringstream params;
    params << "R=" << rShift << ", G=" << gShift << ", B=" << bShift;
    printResult("RPP HOST BATCH ColorCast", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_ColorTemperature_Batched(const vector<Mat>& imgs, bool isColor, int adjustmentValue,
                                                  rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32s *adjustmentTensor = static_cast<Rpp32s*>(calloc(batchSize, sizeof(Rpp32s)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        adjustmentTensor[i] = adjustmentValue;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_color_temperature(input, &srcDesc, output, &dstDesc, adjustmentTensor,
                                                 roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "ColorTemperature");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(adjustmentTensor);
    free(roiTensor);

    
    ostringstream params;
    params << "adjustment=" << adjustmentValue;
    printResult("RPP HOST BATCH ColorTemperature", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_ColorTwist_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0 || !isColor) return;

    int numChannels = 3;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = RpptLayout::NHWC;
    dstDesc.layout = RpptLayout::NHWC;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32f *alpha = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    Rpp32f *beta = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    Rpp32f *hueShift = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    Rpp32f *satFactor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        alpha[i] = 1.0f;     // brightness (0 < brightness <= 20)
        beta[i] = 1.0f;      // contrast (0 < contrast <= 255)
        hueShift[i] = 60.0f; // hue (0 <= hue <= 359)
        satFactor[i] = 1.3f; // saturation (saturation >= 0)
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_color_twist(input, &srcDesc, output, &dstDesc, alpha, beta, hueShift,
                                          satFactor, roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "ColorTwist");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "hue=" << hueShift[0] << ", saturation=" << satFactor[0];
    printResult("RPP HOST BATCH ColorTwist", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    free(input);
    free(output);
    free(alpha);
    free(beta);
    free(hueShift);
    free(satFactor);
    free(roiTensor);
}

void benchmark_RPP_HOST_Vignette_Batched(const vector<Mat>& imgs, bool isColor, float vignetteIntensity,
                                         rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32f *intensityTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        intensityTensor[i] = vignetteIntensity;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_vignette(input, &srcDesc, output, &dstDesc, intensityTensor,
                                       roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "Vignette");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(intensityTensor);
    free(roiTensor);

    
    ostringstream params;
    params << "intensity=" << vignetteIntensity;
    printResult("RPP HOST BATCH Vignette", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_NonLinearBlend_Batched(const vector<Mat>& imgs, bool isColor, float stdDev,
                                               rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize < 2) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input1 = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *input2 = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32f *stdDevTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        stdDevTensor[i] = stdDev;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr1 = input1 + i * bufferSizePerImage;
        Rpp8u* imgPtr2 = input2 + i * bufferSizePerImage;
        int idx2 = (i + 1) % batchSize;

        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr1 + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
            memcpy(imgPtr2 + row * srcDesc.strides.hStride, imgs[idx2].data + row * imgs[idx2].cols * numChannels, imgs[idx2].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_non_linear_blend(input1, input2, &srcDesc, output, &dstDesc, stdDevTensor,
                                               roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "NonLinearBlend");
    }
    perfMonitor.stop();

    free(input1);
    free(input2);
    free(output);
    free(stdDevTensor);
    free(roiTensor);

    
    ostringstream params;
    params << "stdDev=" << stdDev;
    printResult("RPP HOST BATCH NonLinearBlend", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Posterize_Batched(const vector<Mat>& imgs, bool isColor, int levelBits,
                                          rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp8u *posterizeLevelBits = static_cast<Rpp8u*>(calloc(batchSize, sizeof(Rpp8u)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        posterizeLevelBits[i] = (Rpp8u)levelBits;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_posterize(input, &srcDesc, output, &dstDesc, posterizeLevelBits,
                                        roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "Posterize");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(posterizeLevelBits);
    free(roiTensor);

    
    ostringstream params;
    params << "bits=" << levelBits;
    printResult("RPP HOST BATCH Posterize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Solarize_Batched(const vector<Mat>& imgs, bool isColor, int threshold,
                                         rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32f *thresholdTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        thresholdTensor[i] = threshold / 255.0f;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_solarize(input, &srcDesc, output, &dstDesc, thresholdTensor,
                                       roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "Solarize");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(thresholdTensor);
    free(roiTensor);

    
    ostringstream params;
    params << "threshold=" << threshold;
    printResult("RPP HOST BATCH Solarize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Glitch_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0 || !isColor) return;

    int numChannels = 3;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = RpptLayout::NHWC;
    dstDesc.layout = RpptLayout::NHWC;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    RpptChannelOffsets *rgbOffsets = static_cast<RpptChannelOffsets*>(calloc(batchSize, sizeof(RpptChannelOffsets)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
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

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_glitch(input, &srcDesc, output, &dstDesc, rgbOffsets,
                                     roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "Glitch");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(rgbOffsets);
    free(roiTensor);

    printResult("RPP HOST BATCH Glitch", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_JpegCompressionDistortion_Batched(const vector<Mat>& imgs, bool isColor, Rpp32s quality,
                                                          rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32s *qualityTensor = static_cast<Rpp32s*>(calloc(batchSize, sizeof(Rpp32s)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        qualityTensor[i] = quality;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_jpeg_compression_distortion(input, &srcDesc, output, &dstDesc, qualityTensor,
                                                          roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "JpegCompressionDistortion");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "quality=" << quality;
    printResult("RPP HOST BATCH JpegCompressionDistortion", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    free(input);
    free(output);
    free(qualityTensor);
    free(roiTensor);
}

void benchmark_RPP_HOST_TensorMin_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32u outputLength = isColor ? (batchSize * 4) : batchSize;
    Rpp8u *minOutputs = static_cast<Rpp8u*>(calloc(outputLength, sizeof(Rpp8u)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_tensor_min(input, &srcDesc, minOutputs, outputLength,
                                         roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "TensorMin");
    }
    perfMonitor.stop();

    free(input);
    free(minOutputs);
    free(roiTensor);

    printResult("RPP HOST BATCH TensorMin", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_TensorMax_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32u outputLength = isColor ? (batchSize * 4) : batchSize;
    Rpp8u *maxOutputs = static_cast<Rpp8u*>(calloc(outputLength, sizeof(Rpp8u)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_tensor_max(input, &srcDesc, maxOutputs, outputLength,
                                         roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "TensorMax");
    }
    perfMonitor.stop();

    free(input);
    free(maxOutputs);
    free(roiTensor);

    printResult("RPP HOST BATCH TensorMax", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_TensorSum_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32u outputLength = isColor ? (batchSize * 4) : batchSize;
    Rpp64u *sumOutputs = static_cast<Rpp64u*>(calloc(outputLength, sizeof(Rpp64u)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_tensor_sum(input, &srcDesc, sumOutputs, outputLength,
                                         roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "TensorSum");
    }
    perfMonitor.stop();

    free(input);
    free(sumOutputs);
    free(roiTensor);

    printResult("RPP HOST BATCH TensorSum", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_TensorMean_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32u outputLength = isColor ? (batchSize * 4) : batchSize;
    Rpp32f *meanOutputs = static_cast<Rpp32f*>(calloc(outputLength, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_tensor_mean(input, &srcDesc, meanOutputs, outputLength,
                                          roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "TensorMean");
    }
    perfMonitor.stop();

    free(input);
    free(meanOutputs);
    free(roiTensor);

    printResult("RPP HOST BATCH TensorMean", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_TensorStddev_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32u outputLength = isColor ? (batchSize * 4) : batchSize;
    Rpp32f *stddevOutputs = static_cast<Rpp32f*>(calloc(outputLength, sizeof(Rpp32f)));
    Rpp32f *meanOutputs = static_cast<Rpp32f*>(calloc(outputLength, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    CHECK_RPP_STATUS(rppt_tensor_mean(input, &srcDesc, meanOutputs, outputLength,
                                      roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                    "TensorMean");

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_tensor_stddev(input, &srcDesc, stddevOutputs, outputLength, meanOutputs,
                                            roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "TensorStddev");
    }
    perfMonitor.stop();

    free(input);
    free(stddevOutputs);
    free(meanOutputs);
    free(roiTensor);

    printResult("RPP HOST BATCH TensorStddev", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_Threshold_Batched(const vector<Mat>& imgs, bool isColor, float thresh,
                                          rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = RpptLayout::NCHW;
    dstDesc.layout = RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32f *minTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    Rpp32f *maxTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        minTensor[i] = thresh;
        maxTensor[i] = 255.0f;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        Mat grayImg;
        if (isColor)
            cvtColor(imgs[i], grayImg, COLOR_BGR2GRAY);
        else
            grayImg = imgs[i];

        for (int row = 0; row < grayImg.rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, grayImg.data + row * grayImg.cols, grayImg.cols);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_threshold(input, &srcDesc, output, &dstDesc, minTensor, maxTensor,
                                        roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "Threshold");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(minTensor);
    free(maxTensor);
    free(roiTensor);

    
    ostringstream params;
    params << "threshold=" << thresh;
    printResult("RPP HOST BATCH Threshold", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_WarpAffine_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32f *affineTensor = static_cast<Rpp32f*>(calloc(batchSize * 6, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
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

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_warp_affine(input, &srcDesc, output, &dstDesc, affineTensor,
                                          RpptInterpolationType::BILINEAR, roiTensor,
                                          RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "WarpAffine");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(affineTensor);
    free(roiTensor);

    printResult("RPP HOST BATCH WarpAffine", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_WarpPerspective_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32f *perspectiveTensor = static_cast<Rpp32f*>(calloc(batchSize * 9, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
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

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_warp_perspective(input, &srcDesc, output, &dstDesc, perspectiveTensor,
                                               RpptInterpolationType::BILINEAR, roiTensor,
                                               RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "WarpPerspective");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(perspectiveTensor);
    free(roiTensor);

    printResult("RPP HOST BATCH WarpPerspective", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_Fisheye_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_fisheye(input, &srcDesc, output, &dstDesc, roiTensor,
                                      RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "Fisheye");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(roiTensor);

    printResult("RPP HOST BATCH Fisheye", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_LensCorrection_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    Rpp32f *cameraMatrix = static_cast<Rpp32f*>(calloc(batchSize * 9, sizeof(Rpp32f)));
    Rpp32f *distortionCoeffs = static_cast<Rpp32f*>(calloc(batchSize * 8, sizeof(Rpp32f)));

    for (int i = 0; i < batchSize; i++) {
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

    Rpp64u tableSize = (Rpp64u)maxHeight * (Rpp64u)maxWidth * (Rpp64u)batchSize;
    Rpp32f *rowRemapTable = static_cast<Rpp32f*>(calloc(tableSize, sizeof(Rpp32f)));
    Rpp32f *colRemapTable = static_cast<Rpp32f*>(calloc(tableSize, sizeof(Rpp32f)));

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_lens_correction(input, &srcDesc, output, &dstDesc, rowRemapTable,
                                              colRemapTable, &tableDesc, cameraMatrix,
                                              distortionCoeffs, roiTensor, RpptRoiType::XYWH,
                                              handle, RPP_HOST_BACKEND),
                        "LensCorrection");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(roiTensor);
    free(cameraMatrix);
    free(distortionCoeffs);
    free(rowRemapTable);
    free(colRemapTable);

    printResult("RPP HOST BATCH LensCorrection", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_GammaCorrection_Batched(const vector<Mat>& imgs, bool isColor, float gamma,
                                                rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32f *gammaTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        gammaTensor[i] = gamma;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_gamma_correction(input, &srcDesc, output, &dstDesc, gammaTensor,
                                               roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "GammaCorrection");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(gammaTensor);
    free(roiTensor);

    ostringstream params;
    params << "gamma=" << gamma;
    printResult("RPP HOST BATCH GammaCorrection", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Exposure_Batched(const vector<Mat>& imgs, bool isColor, float exposureFactor,
                                         rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32f *exposureTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        exposureTensor[i] = exposureFactor;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_exposure(input, &srcDesc, output, &dstDesc, exposureTensor,
                                       roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "Exposure");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(exposureTensor);
    free(roiTensor);

    ostringstream params;
    params << "factor=" << exposureFactor;
    printResult("RPP HOST BATCH Exposure", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Blend_Batched(const vector<Mat>& imgs, bool isColor, float alpha, rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input1 = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *input2 = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32f *alphaTensor = static_cast<Rpp32f*>(calloc(batchSize, sizeof(Rpp32f)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        alphaTensor[i] = alpha;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr1 = input1 + i * bufferSizePerImage;
        Rpp8u* imgPtr2 = input2 + i * bufferSizePerImage;

        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr1 + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }

        // Create a slightly modified version for input2
        Mat img2;
        imgs[i].convertTo(img2, -1, 0.8, 30);
        for (int row = 0; row < img2.rows; row++) {
            memcpy(imgPtr2 + row * srcDesc.strides.hStride, img2.data + row * img2.cols * numChannels, img2.cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_blend(input1, input2, &srcDesc, output, &dstDesc, alphaTensor,
                                    roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND), "Blend");
    }
    perfMonitor.stop();

    free(input1);
    free(input2);
    free(output);
    free(alphaTensor);
    free(roiTensor);


    ostringstream params;
    params << "alpha=" << alpha;
    printResult("RPP HOST BATCH Blend", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Erode_Batched(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                      rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_erode_host(input, &srcDesc, output, &dstDesc, kernelSize,
                                        roiTensor, RpptRoiType::XYWH, handle), "Erode");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(roiTensor);

    
    ostringstream params;
    params << "kernel=" << kernelSize;
    printResult("RPP HOST BATCH Erode", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_Dilate_Batched(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                       rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_dilate_host(input, &srcDesc, output, &dstDesc, kernelSize,
                                          roiTensor, RpptRoiType::XYWH, handle), "Dilate");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(roiTensor);

    
    ostringstream params;
    params << "kernel=" << kernelSize;
    printResult("RPP HOST BATCH Dilate", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());
}

void benchmark_RPP_HOST_BitwiseAnd_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input1 = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *input2 = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr1 = input1 + i * bufferSizePerImage;
        Rpp8u* imgPtr2 = input2 + i * bufferSizePerImage;

        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr1 + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
            memcpy(imgPtr2 + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_bitwise_and(input1, input2, &srcDesc, output, &dstDesc,
                                          roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "BitwiseAnd");
    }
    perfMonitor.stop();

    free(input1);
    free(input2);
    free(output);
    free(roiTensor);

    printResult("RPP HOST BATCH BitwiseAnd", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_BitwiseOr_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input1 = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *input2 = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr1 = input1 + i * bufferSizePerImage;
        Rpp8u* imgPtr2 = input2 + i * bufferSizePerImage;

        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr1 + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
            memcpy(imgPtr2 + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_bitwise_or(input1, input2, &srcDesc, output, &dstDesc,
                                         roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "BitwiseOr");
    }
    perfMonitor.stop();

    free(input1);
    free(input2);
    free(output);
    free(roiTensor);

    printResult("RPP HOST BATCH BitwiseOr", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_BitwiseNot_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_bitwise_not(input, &srcDesc, output, &dstDesc,
                                          roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "BitwiseNot");
    }
    perfMonitor.stop();

    free(input);
    free(output);
    free(roiTensor);

    printResult("RPP HOST BATCH BitwiseNot", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

void benchmark_RPP_HOST_BitwiseXor_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input1 = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *input2 = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr1 = input1 + i * bufferSizePerImage;
        Rpp8u* imgPtr2 = input2 + i * bufferSizePerImage;

        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr1 + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
            memcpy(imgPtr2 + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_bitwise_xor(input1, input2, &srcDesc, output, &dstDesc,
                                          roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "BitwiseXor");
    }
    perfMonitor.stop();

    free(input1);
    free(input2);
    free(output);
    free(roiTensor);

    printResult("RPP HOST BATCH BitwiseXor", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy());
}

// ==================== DROPOUT AUGMENTATIONS - BATCHED (WORKING SUBSET) ====================

void benchmark_RPP_HOST_GridDropout_Batched(const vector<Mat>& imgs, bool isColor, int tileWidth, int tileHeight,
                                           rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    int numBoxesW = maxWidth / tileWidth;
    int numBoxesH = maxHeight / tileHeight;
    Rpp32u boxesInEachImage = numBoxesW * numBoxesH;

    // Limit boxes to prevent excessive memory allocation
    if (boxesInEachImage > 1000) boxesInEachImage = 1000;

    RpptRoiLtrb *anchorBoxInfoTensor = static_cast<RpptRoiLtrb*>(calloc(batchSize * boxesInEachImage, sizeof(RpptRoiLtrb)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    if (!anchorBoxInfoTensor || !roiTensor) {
        std::cerr << "Failed to allocate memory for GridDropout" << std::endl;
        if (input) free(input);
        if (output) free(output);
        if (anchorBoxInfoTensor) free(anchorBoxInfoTensor);
        if (roiTensor) free(roiTensor);
        return;
    }

    Rpp32u maxHoleW = tileWidth / 2;
    Rpp32u maxHoleH = tileHeight / 2;

    for (int i = 0; i < batchSize; i++) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        int boxIdx = 0;
        for (int y = 0; y < numBoxesH && boxIdx < (int)boxesInEachImage; y++) {
            for (int x = 0; x < numBoxesW && boxIdx < (int)boxesInEachImage; x++, boxIdx++) {
                int idx = i * boxesInEachImage + boxIdx;
                anchorBoxInfoTensor[idx].lt.x = x * tileWidth;
                anchorBoxInfoTensor[idx].lt.y = y * tileHeight;
                anchorBoxInfoTensor[idx].rb.x = (x + 1) * tileWidth;
                anchorBoxInfoTensor[idx].rb.y = (y + 1) * tileHeight;
            }
        }
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_grid_dropout(input, &srcDesc, output, &dstDesc, anchorBoxInfoTensor,
                                          boxesInEachImage, maxHoleW, maxHoleH, roiTensor,
                                          RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "GridDropout");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "tileWidth=" << tileWidth << ", tileHeight=" << tileHeight;
    printResult("RPP HOST BATCH GridDropout", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    free(input);
    free(output);
    free(anchorBoxInfoTensor);
    free(roiTensor);
}

void benchmark_RPP_HOST_Gridmask_Batched(const vector<Mat>& imgs, bool isColor, int tileWidth, float ratio,
                                        rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    Rpp32u tileWidthVal = min((Rpp32u)tileWidth, (Rpp32u)min(maxWidth, maxHeight));
    Rpp32f gridRatio = ratio;
    Rpp32f gridAngle = 0.5f;
    RpptUintVector2D translateVector;
    translateVector.x = 0;
    translateVector.y = 0;

    for (int i = 0; i < batchSize; i++) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_gridmask(input, &srcDesc, output, &dstDesc, tileWidthVal, gridRatio,
                                      gridAngle, translateVector, roiTensor, RpptRoiType::XYWH,
                                      handle, RPP_HOST_BACKEND),
                        "Gridmask");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "tileWidth=" << tileWidth << ", ratio=" << ratio;
    printResult("RPP HOST BATCH Gridmask", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    free(input);
    free(output);
    free(roiTensor);
}

void benchmark_RPP_HOST_ChannelDropout_Batched(const vector<Mat>& imgs, bool isColor, float dropoutProb,
                                               rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp8u *dropoutTensor = static_cast<Rpp8u*>(calloc(batchSize * numChannels, sizeof(Rpp8u)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    srand(DROPOUT_FIXED_SEED);
    for (int i = 0; i < batchSize; i++) {
        for (int c = 0; c < numChannels; c++) {
            dropoutTensor[i * numChannels + c] = ((float)rand() / RAND_MAX) > dropoutProb ? 1 : 0;
        }
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_channel_dropout(input, &srcDesc, output, &dstDesc, dropoutTensor,
                                             roiTensor, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "ChannelDropout");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "dropoutProb=" << dropoutProb;
    printResult("RPP HOST BATCH ChannelDropout", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    free(input);
    free(output);
    free(dropoutTensor);
    free(roiTensor);
}

void benchmark_RPP_HOST_CutoutDropout_Batched(const vector<Mat>& imgs, bool isColor, int numBoxes,
                                             rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32u totalBoxes = batchSize * numBoxes;
    RpptRoiLtrb *anchorBoxInfoTensor = static_cast<RpptRoiLtrb*>(calloc(totalBoxes, sizeof(RpptRoiLtrb)));
    Rpp32u *numOfBoxes = static_cast<Rpp32u*>(calloc(batchSize, sizeof(Rpp32u)));
    Rpp8u *colorsTensor = static_cast<Rpp8u*>(calloc(totalBoxes * numChannels, sizeof(Rpp8u)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    for (int i = 0; i < batchSize; i++) {
        numOfBoxes[i] = numBoxes;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        for (int j = 0; j < numBoxes; j++) {
            int idx = i * numBoxes + j;
            int left = imgs[i].cols / 4;
            int top = imgs[i].rows / 4;
            int right = left + imgs[i].cols / 2;
            int bottom = top + imgs[i].rows / 2;

            anchorBoxInfoTensor[idx].lt.x = left;
            anchorBoxInfoTensor[idx].lt.y = top;
            anchorBoxInfoTensor[idx].rb.x = right;
            anchorBoxInfoTensor[idx].rb.y = bottom;

            for (int c = 0; c < numChannels; c++) {
                colorsTensor[idx * numChannels + c] = 0;
            }
        }
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_cutout_dropout(input, &srcDesc, output, &dstDesc, anchorBoxInfoTensor,
                                            colorsTensor, numOfBoxes, roiTensor, RpptRoiType::XYWH,
                                            handle, RPP_HOST_BACKEND),
                        "CutoutDropout");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "numBoxes=" << numBoxes;
    printResult("RPP HOST BATCH CutoutDropout", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    free(input);
    free(output);
    free(anchorBoxInfoTensor);
    free(numOfBoxes);
    free(colorsTensor);
    free(roiTensor);
}
// ==================== REMAINING DROPOUT AUGMENTATIONS ====================

void benchmark_RPP_HOST_Erase_Batched(const vector<Mat>& imgs, bool isColor, int numBoxes,
                                     rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32u totalBoxes = batchSize * numBoxes;
    RpptRoiLtrb *anchorBoxInfoTensor = static_cast<RpptRoiLtrb*>(calloc(totalBoxes, sizeof(RpptRoiLtrb)));
    Rpp32u *numBoxesTensor = static_cast<Rpp32u*>(calloc(batchSize, sizeof(Rpp32u)));
    Rpp8u *colorsTensor = static_cast<Rpp8u*>(calloc(totalBoxes * numChannels, sizeof(Rpp8u)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    // Initialize anchor boxes and colors
    for (int i = 0; i < batchSize; i++) {
        numBoxesTensor[i] = numBoxes;
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        for (int j = 0; j < numBoxes; j++) {
            int idx = i * numBoxes + j;
            int left = 50 + j * 100;
            int top = 50 + j * 100;
            int right = left + 80;
            int bottom = top + 80;

            anchorBoxInfoTensor[idx].lt.x = left;
            anchorBoxInfoTensor[idx].lt.y = top;
            anchorBoxInfoTensor[idx].rb.x = right;
            anchorBoxInfoTensor[idx].rb.y = bottom;

            // Set erase color to gray (128,128,128)
            for (int c = 0; c < numChannels; c++) {
                colorsTensor[idx * numChannels + c] = 128;
            }
        }
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_erase(input, &srcDesc, output, &dstDesc, anchorBoxInfoTensor,
                                   (RppPtr_t)colorsTensor, numBoxesTensor, roiTensor,
                                   RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "Erase");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "numBoxes=" << numBoxes;
    printResult("RPP HOST BATCH Erase", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    free(input);
    free(output);
    free(anchorBoxInfoTensor);
    free(numBoxesTensor);
    free(colorsTensor);
    free(roiTensor);
}

void benchmark_RPP_HOST_RandomErase_Batched(const vector<Mat>& imgs, bool isColor, int numBoxes,
                                           rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    Rpp32u totalBoxes = batchSize * numBoxes;
    RpptRoiLtrb *anchorBoxInfoTensor = static_cast<RpptRoiLtrb*>(calloc(totalBoxes, sizeof(RpptRoiLtrb)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    // Random erase uses a noise buffer of fixed size
    Rpp32u noiseBufferSize = RANDOM_ERASE_NOISE_BUFFER_SIDE * RANDOM_ERASE_NOISE_BUFFER_SIDE * numChannels;
    Rpp8u *noiseBuf = static_cast<Rpp8u*>(calloc(noiseBufferSize, sizeof(Rpp8u)));

    // Fill noise buffer with random values
    srand(DROPOUT_FIXED_SEED);
    for (Rpp32u i = 0; i < noiseBufferSize; i++) {
        noiseBuf[i] = rand() % 256;
    }

    // Initialize anchor boxes
    for (int i = 0; i < batchSize; i++) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        for (int j = 0; j < numBoxes; j++) {
            int idx = i * numBoxes + j;
            int left = 50 + j * 100;
            int top = 50 + j * 100;
            int right = left + 80;
            int bottom = top + 80;

            anchorBoxInfoTensor[idx].lt.x = left;
            anchorBoxInfoTensor[idx].lt.y = top;
            anchorBoxInfoTensor[idx].rb.x = right;
            anchorBoxInfoTensor[idx].rb.y = bottom;
        }
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_random_erase(input, &srcDesc, output, &dstDesc, anchorBoxInfoTensor,
                                          (RppPtr_t)noiseBuf, roiTensor, RpptRoiType::XYWH,
                                          handle, RPP_HOST_BACKEND),
                        "RandomErase");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "numBoxes=" << numBoxes;
    printResult("RPP HOST BATCH RandomErase", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    free(input);
    free(output);
    free(anchorBoxInfoTensor);
    free(noiseBuf);
    free(roiTensor);
}

void benchmark_RPP_HOST_CoarseDropout_Batched(const vector<Mat>& imgs, bool isColor, Rpp32u maxBoxesPerImage,
                                             rppHandle_t handle) {
    int batchSize = (int)imgs.size();
    if (batchSize == 0) return;

    int numChannels = isColor ? 3 : 1;
    int maxHeight = imgs[0].rows;
    int maxWidth = imgs[0].cols;
    Rpp32u offsetInBytes = 0;

    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.dataType = RpptDataType::U8;

    set_descriptor_dims_and_strides(&srcDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, maxHeight, maxWidth, numChannels, offsetInBytes);

    Rpp64u ioBufferSize = (Rpp64u)srcDesc.h * (Rpp64u)srcDesc.w * (Rpp64u)srcDesc.c * (Rpp64u)batchSize;
    Rpp8u *input = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));
    Rpp8u *output = static_cast<Rpp8u*>(calloc(ioBufferSize, sizeof(Rpp8u)));

    RpptRoiLtrb *anchorBoxInfoTensor = static_cast<RpptRoiLtrb*>(calloc(batchSize * maxBoxesPerImage, sizeof(RpptRoiLtrb)));
    Rpp32u *numBoxesTensor = static_cast<Rpp32u*>(calloc(batchSize, sizeof(Rpp32u)));
    RpptROI *roiTensor = static_cast<RpptROI*>(calloc(batchSize, sizeof(RpptROI)));

    // Initialize dropout boxes - same logic as single-image implementation
    srand(DROPOUT_FIXED_SEED);
    for (int i = 0; i < batchSize; i++) {
        int h = imgs[i].rows;
        int w = imgs[i].cols;

        // Random number of boxes (2 to maxBoxesPerImage) - matches single-image implementation
        Rpp32u numBoxes = 2 + (rand() % (maxBoxesPerImage - 1));
        numBoxesTensor[i] = numBoxes;

        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = imgs[i].cols;
        roiTensor[i].xywhROI.roiHeight = imgs[i].rows;

        for (Rpp32u j = 0; j < numBoxes; j++) {
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
        for (Rpp32u j = numBoxes; j < maxBoxesPerImage; j++) {
            int idx = i * maxBoxesPerImage + j;
            anchorBoxInfoTensor[idx].lt.x = 0;
            anchorBoxInfoTensor[idx].lt.y = 0;
            anchorBoxInfoTensor[idx].rb.x = 0;
            anchorBoxInfoTensor[idx].rb.y = 0;
        }
    }

    Rpp64u bufferSizePerImage = srcDesc.strides.nStride;
    for (int i = 0; i < batchSize; i++) {
        Rpp8u* imgPtr = input + i * bufferSizePerImage;
        for (int row = 0; row < imgs[i].rows; row++) {
            memcpy(imgPtr + row * srcDesc.strides.hStride, imgs[i].data + row * imgs[i].cols * numChannels, imgs[i].cols * numChannels);
        }
    }

    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }
        CHECK_RPP_STATUS(rppt_coarse_dropout(input, &srcDesc, output, &dstDesc, anchorBoxInfoTensor,
                                            numBoxesTensor, maxBoxesPerImage, roiTensor,
                                            RpptRoiType::XYWH, handle, RPP_HOST_BACKEND),
                        "CoarseDropout");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "maxBoxes=" << maxBoxesPerImage;
    printResult("RPP HOST BATCH CoarseDropout", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    free(input);
    free(output);
    free(anchorBoxInfoTensor);
    free(numBoxesTensor);
    free(roiTensor);
}

// Copy Batched (HOST)
void benchmark_RPP_HOST_Copy_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Set up descriptors with width padding for HOST FIRST
    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.dataType = RpptDataType::U8;
    int widthPadded = ((width / 8) * 8) + 8;
    set_descriptor_dims_and_strides(&srcDesc, batchSize, height, widthPadded, channels, 0);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, height, widthPadded, channels, 0);

    // Allocate HOST buffers using descriptor dimensions (CRITICAL: desc.w is modified by set_descriptor_dims_and_strides)
    size_t bufferSize = (size_t)srcDesc.n * srcDesc.h * srcDesc.w * srcDesc.c;
    Rpp8u* input = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));
    Rpp8u* output = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));

    // Fill input buffer from OpenCV Mats using descriptor strides
    for (int i = 0; i < batchSize; i++) {
        for (int h = 0; h < height; h++) {
            memcpy(input + i * srcDesc.strides.nStride + h * srcDesc.strides.hStride,
                   imgs[i].data + h * width * channels,
                   width * channels);
        }
    }

    // Benchmark loop
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }        
        RppStatus status = rppt_copy((RppPtr_t)input, &srcDesc, (RppPtr_t)output, &dstDesc, handle, RPP_HOST_BACKEND);
        CHECK_RPP_STATUS(status, "RPP HOST Copy failed with status");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "layout=NHWC";
    printResult("RPP HOST BATCH Copy", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    free(input);
    free(output);
}

// Slice Batched (HOST)
void benchmark_RPP_HOST_Slice_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Allocate HOST buffers
    Rpp8u* input = (Rpp8u*)calloc(batchSize * height * width * channels, sizeof(Rpp8u));
    Rpp8u* output = (Rpp8u*)calloc(batchSize * height * width * channels, sizeof(Rpp8u));

    // Fill input buffer from OpenCV Mats
    for (int i = 0; i < batchSize; i++) {
        memcpy(input + i * height * width * channels, imgs[i].data, height * width * channels);
    }

    // Set up generic descriptors for slice
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

    // BUG FIX: The rppt_slice kernel operates on (numDims-1) dimensional sub-tensors,
    // excluding the batch dimension. For 4D NHWC tensors, it processes 3D HWC sub-tensors.
    // Therefore:
    // - anchorTensor: batchSize * 3 elements (H, W, C anchors, NO batch dimension)
    // - shapeTensor: batchSize * 3 elements (H, W, C shapes, NO batch dimension)
    // - roiTensor: batchSize * 6 elements ([H_begin, W_begin, C_begin, H_length, W_length, C_length])
    //
    // The previous code incorrectly used 4-element tensors including the batch dimension,
    // causing out-of-bounds memory access and segmentation faults when batchSize >= 86.

    Rpp32s* anchorTensor = (Rpp32s*)calloc(batchSize * 3, sizeof(Rpp32s));
    Rpp32s* shapeTensor = (Rpp32s*)calloc(batchSize * 3, sizeof(Rpp32s));
    for (int i = 0; i < batchSize; i++) {
        // Slice from (0, 0, 0) in HWC space
        anchorTensor[i * 3 + 0] = 0;  // height anchor
        anchorTensor[i * 3 + 1] = 0;  // width anchor
        anchorTensor[i * 3 + 2] = 0;  // channel anchor

        // Slice the full HWC dimensions (no change to original image)
        shapeTensor[i * 3 + 0] = height;
        shapeTensor[i * 3 + 1] = width;
        shapeTensor[i * 3 + 2] = channels;
    }

    Rpp8u fillValue = 0;
    Rpp32u* roiTensor = (Rpp32u*)calloc(batchSize * 6, sizeof(Rpp32u));
    for (int i = 0; i < batchSize; i++) {
        // ROI format: [begin_H, begin_W, begin_C, length_H, length_W, length_C]
        roiTensor[i * 6 + 0] = 0;        // height begin
        roiTensor[i * 6 + 1] = 0;        // width begin
        roiTensor[i * 6 + 2] = 0;        // channel begin
        roiTensor[i * 6 + 3] = height;   // height length
        roiTensor[i * 6 + 4] = width;    // width length
        roiTensor[i * 6 + 5] = channels; // channel length
    }

    // Benchmark loop
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }        
        RppStatus status = rppt_slice((RppPtr_t)input, &srcGenericDesc, (RppPtr_t)output, &dstGenericDesc,
                                      anchorTensor, shapeTensor, (RppPtr_t)&fillValue, false, roiTensor,
                                      handle, RPP_HOST_BACKEND);
        CHECK_RPP_STATUS(status, "RPP HOST Slice failed with status");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "slice=center_50%";
    printResult("RPP HOST BATCH Slice", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    free(input);
    free(output);
    free(anchorTensor);
    free(shapeTensor);
    free(roiTensor);
}

// ChannelPermute Batched (HOST)
void benchmark_RPP_HOST_ChannelPermute_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    // ChannelPermute requires 3 channels
    if (!isColor) {
        cout << "RPP HOST ChannelPermute (Batched) - Skipped (requires RGB images)" << endl;
        return;
    }

    int batchSize = imgs.size();
    int channels = 3;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Set up descriptors with width padding for HOST FIRST
    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = RpptLayout::NHWC;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.layout = RpptLayout::NHWC;
    dstDesc.dataType = RpptDataType::U8;
    int widthPadded = ((width / 8) * 8) + 8;
    set_descriptor_dims_and_strides(&srcDesc, batchSize, height, widthPadded, channels, 0);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, height, widthPadded, channels, 0);

    // Allocate HOST buffers using descriptor dimensions (CRITICAL: desc.w is modified by set_descriptor_dims_and_strides)
    size_t bufferSize = (size_t)srcDesc.n * srcDesc.h * srcDesc.w * srcDesc.c;
    Rpp8u* input = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));
    Rpp8u* output = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));

    // Fill input buffer from OpenCV Mats using descriptor strides
    for (int i = 0; i < batchSize; i++) {
        for (int h = 0; h < height; h++) {
            memcpy(input + i * srcDesc.strides.nStride + h * srcDesc.strides.hStride,
                   imgs[i].data + h * width * channels,
                   width * channels);
        }
    }

    // Permutation tensor: BGR to RGB (2,1,0)
    Rpp32u* permutationTensor = new Rpp32u[batchSize * 3];
    for (int i = 0; i < batchSize; i++) {
        permutationTensor[i * 3 + 0] = 2;  // B channel
        permutationTensor[i * 3 + 1] = 1;  // G channel
        permutationTensor[i * 3 + 2] = 0;  // R channel
    }

    // Benchmark loop
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }        
        RppStatus status = rppt_channel_permute((RppPtr_t)input, &srcDesc, (RppPtr_t)output, &dstDesc,
                                                permutationTensor, handle, RPP_HOST_BACKEND);
        CHECK_RPP_STATUS(status, "RPP HOST ChannelPermute failed with status");
    }
    perfMonitor.stop();

    ostringstream params;
    if (isColor) params << "permutation=BGR->RGB";
    printResult("RPP HOST BATCH ChannelPermute", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    free(input);
    free(output);
    delete[] permutationTensor;
}

// Transpose Batched (HOST)
void benchmark_RPP_HOST_Transpose_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Allocate HOST buffers
    Rpp8u* input = (Rpp8u*)calloc(batchSize * height * width * channels, sizeof(Rpp8u));
    Rpp8u* output = (Rpp8u*)calloc(batchSize * width * height * channels, sizeof(Rpp8u));

    // Fill input buffer from OpenCV Mats
    for (int i = 0; i < batchSize; i++) {
        memcpy(input + i * height * width * channels, imgs[i].data, height * width * channels);
    }

    // Set up generic descriptors for transpose
    RpptGenericDesc srcGenericDesc, dstGenericDesc;
    srcGenericDesc.numDims = 4;
    srcGenericDesc.offsetInBytes = 0;
    srcGenericDesc.dataType = RpptDataType::U8;
    srcGenericDesc.dims[0] = batchSize;
    srcGenericDesc.dims[1] = height;
    srcGenericDesc.dims[2] = width;
    srcGenericDesc.dims[3] = channels;
    srcGenericDesc.layout = RpptLayout::NHWC;
    srcGenericDesc.strides[3] = 1;
    srcGenericDesc.strides[2] = channels;
    srcGenericDesc.strides[1] = width * channels;
    srcGenericDesc.strides[0] = height * width * channels;

    // Destination has swapped H and W dimensions
    dstGenericDesc.numDims = 4;
    dstGenericDesc.offsetInBytes = 0;
    dstGenericDesc.dataType = RpptDataType::U8;
    dstGenericDesc.dims[0] = batchSize;
    dstGenericDesc.dims[1] = width;   // swapped
    dstGenericDesc.dims[2] = height;  // swapped
    dstGenericDesc.dims[3] = channels;
    dstGenericDesc.layout = RpptLayout::NHWC;
    dstGenericDesc.strides[3] = 1;
    dstGenericDesc.strides[2] = channels;
    dstGenericDesc.strides[1] = height * channels;  // swapped stride
    dstGenericDesc.strides[0] = width * height * channels;

    // Permutation tensor: swap H and W axes (0,2,1,3)
    Rpp32u permTensor[4] = {0, 2, 1, 3};

    // CRITICAL: ROI tensor must have 8 values per image for 4D transpose!
    // Format: [n_start, h_start, w_start, c_start, n_size, h_size, w_size, c_size]
    Rpp32u* roiTensor = (Rpp32u*)calloc(batchSize * 8, sizeof(Rpp32u));
    for (int i = 0; i < batchSize; i++) {
        int idx = i * 8;
        roiTensor[idx + 0] = 0;         // n_start
        roiTensor[idx + 1] = 0;         // h_start
        roiTensor[idx + 2] = 0;         // w_start
        roiTensor[idx + 3] = 0;         // c_start
        roiTensor[idx + 4] = 1;         // n_size (process 1 image at a time)
        roiTensor[idx + 5] = height;    // h_size
        roiTensor[idx + 6] = width;     // w_size
        roiTensor[idx + 7] = channels;  // c_size
    }

    // Benchmark loop
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }        
        RppStatus status = rppt_transpose((RppPtr_t)input, &srcGenericDesc, (RppPtr_t)output, &dstGenericDesc,
                                          permTensor, roiTensor, handle, RPP_HOST_BACKEND);
        CHECK_RPP_STATUS(status, "RPP HOST Transpose failed with status");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "permutation=0,2,1,3";
    printResult("RPP HOST BATCH Transpose", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    free(input);
    free(output);
    free(roiTensor);
}
// LUT Batched (HOST)
void benchmark_RPP_HOST_LUT_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Set up descriptors with width padding for HOST
    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.dataType = RpptDataType::U8;
    int widthPadded = ((width / 8) * 8) + 8;
    set_descriptor_dims_and_strides(&srcDesc, batchSize, height, widthPadded, channels, 0);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, height, widthPadded, channels, 0);

    // Allocate HOST buffers using descriptor dimensions
    size_t bufferSize = (size_t)srcDesc.n * srcDesc.h * srcDesc.w * srcDesc.c;
    Rpp8u* input = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));
    Rpp8u* output = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));

    // Fill input buffer from OpenCV Mats using descriptor strides
    for (int i = 0; i < batchSize; i++) {
        for (int h = 0; h < height; h++) {
            memcpy(input + i * srcDesc.strides.nStride + h * srcDesc.strides.hStride,
                   imgs[i].data + h * width * channels,
                   width * channels);
        }
    }

    // Create LUT - simple inversion table
    Rpp8u lut[256];
    for (int i = 0; i < 256; i++) lut[i] = 255 - i;

    // Setup ROI
    RpptROI* roiTensorPtrSrc = (RpptROI*)calloc(batchSize, sizeof(RpptROI));
    for (int i = 0; i < batchSize; i++) {
        roiTensorPtrSrc[i].xywhROI.xy.x = 0;
        roiTensorPtrSrc[i].xywhROI.xy.y = 0;
        roiTensorPtrSrc[i].xywhROI.roiWidth = width;
        roiTensorPtrSrc[i].xywhROI.roiHeight = height;
    }

    // Benchmark loop
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }        
        RppStatus status = rppt_lut((RppPtr_t)input, &srcDesc, (RppPtr_t)output, &dstDesc,
                                    lut, roiTensorPtrSrc, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND);
        CHECK_RPP_STATUS(status, "RPP HOST LUT failed with status");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "lut=inverse";
    printResult("RPP HOST BATCH LUT", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    free(input);
    free(output);
    free(roiTensorPtrSrc);
}

// Magnitude Batched (HOST)
void benchmark_RPP_HOST_Magnitude_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Set up descriptors with width padding for HOST
    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.dataType = RpptDataType::U8;
    int widthPadded = ((width / 8) * 8) + 8;
    set_descriptor_dims_and_strides(&srcDesc, batchSize, height, widthPadded, channels, 0);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, height, widthPadded, channels, 0);

    // Allocate HOST buffers
    size_t bufferSize = (size_t)srcDesc.n * srcDesc.h * srcDesc.w * srcDesc.c;
    Rpp8u* input1 = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));
    Rpp8u* input2 = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));
    Rpp8u* output = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));

    // Fill input buffers (simulating gradient X and Y)
    for (int i = 0; i < batchSize; i++) {
        for (int h = 0; h < height; h++) {
            memcpy(input1 + i * srcDesc.strides.nStride + h * srcDesc.strides.hStride,
                   imgs[i].data + h * width * channels,
                   width * channels);
            memcpy(input2 + i * srcDesc.strides.nStride + h * srcDesc.strides.hStride,
                   imgs[i].data + h * width * channels,
                   width * channels);
        }
    }

    // Setup ROI
    RpptROI* roiTensorPtrSrc = (RpptROI*)calloc(batchSize, sizeof(RpptROI));
    for (int i = 0; i < batchSize; i++) {
        roiTensorPtrSrc[i].xywhROI.xy.x = 0;
        roiTensorPtrSrc[i].xywhROI.xy.y = 0;
        roiTensorPtrSrc[i].xywhROI.roiWidth = width;
        roiTensorPtrSrc[i].xywhROI.roiHeight = height;
    }

    // Benchmark loop
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }        
        RppStatus status = rppt_magnitude((RppPtr_t)input1, (RppPtr_t)input2, &srcDesc,
                                          (RppPtr_t)output, &dstDesc,
                                          roiTensorPtrSrc, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND);
        CHECK_RPP_STATUS(status, "RPP HOST Magnitude failed with status");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "inputs=2";
    printResult("RPP HOST BATCH Magnitude", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    free(input1);
    free(input2);
    free(output);
    free(roiTensorPtrSrc);
}

// FusedMultiplyAddScalar Batched (HOST)
void benchmark_RPP_HOST_FusedMultiplyAddScalar_Batched(const vector<Mat>& imgs, bool isColor,
                                                        float mulVal, float addVal, rppHandle_t handle) {
    int batchSize = imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Set up 5D generic descriptors (NDHWC with depth=1 for 2D images)
    RpptGenericDesc srcDesc, dstDesc;
    int layoutType = isColor ? 1 : 0;  // 1=NDHWC, 0=NCDHW
    set_generic_descriptor(&srcDesc, batchSize, width, height, 1, channels, 0, layoutType);
    set_generic_descriptor(&dstDesc, batchSize, width, height, 1, channels, 0, layoutType);

    // Allocate HOST buffers (F32 required for fused_multiply_add_scalar)
    size_t bufferSize = (size_t)batchSize * height * width * channels;
    Rpp32f* inputF32 = (Rpp32f*)calloc(bufferSize, sizeof(Rpp32f));
    Rpp32f* outputF32 = (Rpp32f*)calloc(bufferSize, sizeof(Rpp32f));

    // Fill input buffer - convert U8 to F32
    for (int i = 0; i < batchSize; i++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int c = 0; c < channels; c++) {
                    int idx = i * height * width * channels + h * width * channels + w * channels + c;
                    inputF32[idx] = (Rpp32f)imgs[i].data[h * width * channels + w * channels + c];
                }
            }
        }
    }

    // Setup mul and add tensors
    Rpp32f* mulTensor = (Rpp32f*)calloc(batchSize, sizeof(Rpp32f));
    Rpp32f* addTensor = (Rpp32f*)calloc(batchSize, sizeof(Rpp32f));
    for (int i = 0; i < batchSize; i++) {
        mulTensor[i] = mulVal;
        addTensor[i] = addVal;
    }

    // Setup ROI3D
    RpptROI3D* roiTensorPtrSrc = (RpptROI3D*)calloc(batchSize, sizeof(RpptROI3D));
    for (int i = 0; i < batchSize; i++) {
        roiTensorPtrSrc[i].xyzwhdROI.xyz.x = 0;
        roiTensorPtrSrc[i].xyzwhdROI.xyz.y = 0;
        roiTensorPtrSrc[i].xyzwhdROI.xyz.z = 0;
        roiTensorPtrSrc[i].xyzwhdROI.roiWidth = width;
        roiTensorPtrSrc[i].xyzwhdROI.roiHeight = height;
        roiTensorPtrSrc[i].xyzwhdROI.roiDepth = 1;
    }

    // Benchmark loop
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }        
        RppStatus status = rppt_fused_multiply_add_scalar((RppPtr_t)inputF32, &srcDesc,
                                                           (RppPtr_t)outputF32, &dstDesc,
                                                           mulTensor, addTensor,
                                                           roiTensorPtrSrc, RpptRoi3DType::XYZWHD,
                                                           handle, RPP_HOST_BACKEND);
        CHECK_RPP_STATUS(status, "RPP HOST FusedMultiplyAddScalar failed with status");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "mul=" << mulVal << ", add=" << addVal;
    printResult("RPP HOST BATCH FusedMultiplyAddScalar", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    free(inputF32);
    free(outputF32);
    free(mulTensor);
    free(addTensor);
    free(roiTensorPtrSrc);
}

// Remap Batched (HOST)
void benchmark_RPP_HOST_Remap_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Set up descriptors with width padding for HOST
    RpptDesc srcDesc, dstDesc, tableDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.dataType = RpptDataType::U8;
    int widthPadded = ((width / 8) * 8) + 8;
    set_descriptor_dims_and_strides(&srcDesc, batchSize, height, widthPadded, channels, 0);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, height, widthPadded, channels, 0);

    // Allocate HOST buffers
    size_t bufferSize = (size_t)srcDesc.n * srcDesc.h * srcDesc.w * srcDesc.c;
    Rpp8u* input = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));
    Rpp8u* output = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));

    // Fill input buffer
    for (int i = 0; i < batchSize; i++) {
        for (int h = 0; h < height; h++) {
            memcpy(input + i * srcDesc.strides.nStride + h * srcDesc.strides.hStride,
                   imgs[i].data + h * width * channels,
                   width * channels);
        }
    }

    // Setup ROI
    RpptROI* roiTensorPtrSrc = (RpptROI*)calloc(batchSize, sizeof(RpptROI));
    for (int i = 0; i < batchSize; i++) {
        roiTensorPtrSrc[i].xywhROI.xy.x = 0;
        roiTensorPtrSrc[i].xywhROI.xy.y = 0;
        roiTensorPtrSrc[i].xywhROI.roiWidth = width;
        roiTensorPtrSrc[i].xywhROI.roiHeight = height;
    }

    // Allocate and initialize remap tables (use padded width from descriptor)
    size_t tableSize = (size_t)srcDesc.n * srcDesc.h * srcDesc.w;
    Rpp32f* rowRemapTable = (Rpp32f*)calloc(tableSize, sizeof(Rpp32f));
    Rpp32f* colRemapTable = (Rpp32f*)calloc(tableSize, sizeof(Rpp32f));

    // Setup table descriptor using init_remap helper
    tableDesc = srcDesc;
    init_remap(&tableDesc, &srcDesc, roiTensorPtrSrc, rowRemapTable, colRemapTable);

    RpptInterpolationType interpolationType = RpptInterpolationType::BILINEAR;

    // Benchmark loop
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }        
        RppStatus status = rppt_remap((RppPtr_t)input, &srcDesc, (RppPtr_t)output, &dstDesc,
                                      rowRemapTable, colRemapTable, &tableDesc, interpolationType,
                                      roiTensorPtrSrc, RpptRoiType::XYWH, handle, RPP_HOST_BACKEND);
        CHECK_RPP_STATUS(status, "RPP HOST Remap failed with status");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "transform=sine_wave, interpolation=bilinear";
    printResult("RPP HOST BATCH Remap", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    free(input);
    free(output);
    free(rowRemapTable);
    free(colRemapTable);
    free(roiTensorPtrSrc);
}

// Phase Batched (HOST)
void benchmark_RPP_HOST_Phase_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Set up descriptors with width padding for HOST
    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.dataType = RpptDataType::U8;
    int widthPadded = ((width / 8) * 8) + 8;
    set_descriptor_dims_and_strides(&srcDesc, batchSize, height, widthPadded, channels, 0);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, height, widthPadded, channels, 0);

    // Allocate HOST buffers
    size_t bufferSize = (size_t)srcDesc.n * srcDesc.h * srcDesc.w * srcDesc.c;
    Rpp8u* input1 = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));
    Rpp8u* input2 = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));
    Rpp8u* output = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));

    // Fill input buffers
    for (int i = 0; i < batchSize; i++) {
        for (int h = 0; h < height; h++) {
            memcpy(input1 + i * srcDesc.strides.nStride + h * srcDesc.strides.hStride,
                   imgs[i].data + h * width * channels,
                   width * channels);
            // Create second input (shifted for phase calculation)
            for (int w = 0; w < width * channels; w++) {
                input2[i * srcDesc.strides.nStride + h * srcDesc.strides.hStride + w] =
                    imgs[i].data[h * width * channels + w] >> 1;
            }
        }
    }

    // Setup ROI
    RpptROI* roiTensorPtrSrc = (RpptROI*)calloc(batchSize, sizeof(RpptROI));
    for (int i = 0; i < batchSize; i++) {
        roiTensorPtrSrc[i].xywhROI.xy.x = 0;
        roiTensorPtrSrc[i].xywhROI.xy.y = 0;
        roiTensorPtrSrc[i].xywhROI.roiWidth = width;
        roiTensorPtrSrc[i].xywhROI.roiHeight = height;
    }

    // Benchmark loop
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }        
        RppStatus status = rppt_phase((RppPtr_t)input1, (RppPtr_t)input2, &srcDesc,
                                      (RppPtr_t)output, &dstDesc,
                                      roiTensorPtrSrc, RpptRoiType::XYWH,
                                      handle, RPP_HOST_BACKEND);
        CHECK_RPP_STATUS(status, "RPP HOST Phase failed with status");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "inputs=2";
    printResult("RPP HOST BATCH Phase", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    free(input1);
    free(input2);
    free(output);
    free(roiTensorPtrSrc);
}

// Normalize Batched (HOST)
void benchmark_RPP_HOST_Normalize_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;
    int nDim = 3;  // 3D per-image: H, W, C

    // Create generic descriptor for 4D tensor (NHWC)
    RpptGenericDesc genericDesc;
    genericDesc.numDims = nDim + 1;  // 4D: N, H, W, C
    genericDesc.offsetInBytes = 0;
    genericDesc.dataType = RpptDataType::U8;
    genericDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    genericDesc.dims[0] = batchSize;   // N (batch)
    genericDesc.dims[1] = height;      // H
    genericDesc.dims[2] = width;       // W
    genericDesc.dims[3] = channels;    // C

    // Compute strides for NHWC layout
    genericDesc.strides[3] = 1;                          // C stride
    genericDesc.strides[2] = channels;                   // W stride
    genericDesc.strides[1] = width * channels;           // H stride
    genericDesc.strides[0] = height * width * channels;  // N stride

    // axisMask: normalize over H and W (bit 0=H, bit 1=W)
    Rpp32u axisMask = 3;

    // Allocate mean and stddev tensors (one per channel per image)
    Rpp32u meanStddevSize = batchSize * channels;
    Rpp32f* meanTensor = (Rpp32f*)calloc(meanStddevSize, sizeof(Rpp32f));
    Rpp32f* stdDevTensor = (Rpp32f*)calloc(meanStddevSize, sizeof(Rpp32f));

    // computeMeanStddev: 3 = compute both mean and stddev internally
    Rpp8u computeMeanStddev = 3;

    // Scale and shift for normalization
    Rpp32f scale = 1.0f;
    Rpp32f shift = 0.0f;

    // ROI tensor: nDim * 2 values per batch item
    Rpp32u* roiTensor = (Rpp32u*)calloc(batchSize * nDim * 2, sizeof(Rpp32u));
    for (int i = 0; i < batchSize; i++) {
        int idx = i * (nDim * 2);
        roiTensor[idx + 0] = 0;         // h_start
        roiTensor[idx + 1] = 0;         // w_start
        roiTensor[idx + 2] = 0;         // c_start
        roiTensor[idx + 3] = height;    // h_size
        roiTensor[idx + 4] = width;     // w_size
        roiTensor[idx + 5] = channels;  // c_size
    }

    // Allocate input/output buffers
    size_t imageSize = height * width * channels;
    Rpp8u* inputBuffer = (Rpp8u*)calloc(batchSize * imageSize, sizeof(Rpp8u));
    Rpp8u* outputBuffer = (Rpp8u*)calloc(batchSize * imageSize, sizeof(Rpp8u));

    for (int i = 0; i < batchSize; i++)
        memcpy(inputBuffer + i * imageSize, imgs[i].data, imageSize);

    // Benchmark loop
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }        
        RppStatus status = rppt_normalize((RppPtr_t)inputBuffer, &genericDesc,
                                          (RppPtr_t)outputBuffer, &genericDesc,
                                          axisMask, meanTensor, stdDevTensor, computeMeanStddev,
                                          scale, shift, roiTensor,
                                          handle, RPP_HOST_BACKEND);
        CHECK_RPP_STATUS(status, "RPP HOST Normalize failed with status");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "mode=auto_compute, scale=" << scale << ", shift=" << shift;
    printResult("RPP HOST BATCH Normalize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    free(inputBuffer);
    free(outputBuffer);
    free(meanTensor);
    free(stdDevTensor);
    free(roiTensor);
}

// CropAndPatch Batched (HOST)
void benchmark_RPP_HOST_CropAndPatch_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = imgs.size();
    if (batchSize < 2) {
        cout << "CropAndPatch requires at least 2 images. Skipping." << endl;
        return;
    }
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Set up descriptors with width padding for HOST
    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.dataType = RpptDataType::U8;
    int widthPadded = ((width / 8) * 8) + 8;
    set_descriptor_dims_and_strides(&srcDesc, batchSize, height, widthPadded, channels, 0);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, height, widthPadded, channels, 0);

    // Allocate HOST buffers
    size_t bufferSize = (size_t)srcDesc.n * srcDesc.h * srcDesc.w * srcDesc.c;
    Rpp8u* input1 = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));
    Rpp8u* input2 = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));
    Rpp8u* output = (Rpp8u*)calloc(bufferSize, sizeof(Rpp8u));

    // Fill input buffers
    for (int i = 0; i < batchSize; i++) {
        for (int h = 0; h < height; h++) {
            memcpy(input1 + i * srcDesc.strides.nStride + h * srcDesc.strides.hStride,
                   imgs[i].data + h * width * channels,
                   width * channels);
            // Use next image for second input (wraparound)
            int nextIdx = (i + 1) % batchSize;
            memcpy(input2 + i * srcDesc.strides.nStride + h * srcDesc.strides.hStride,
                   imgs[nextIdx].data + h * width * channels,
                   width * channels);
        }
    }

    // Setup ROIs
    RpptROI* dstRoiTensor = (RpptROI*)calloc(batchSize, sizeof(RpptROI));
    RpptROI* cropRoiTensor = (RpptROI*)calloc(batchSize, sizeof(RpptROI));
    RpptROI* patchRoiTensor = (RpptROI*)calloc(batchSize, sizeof(RpptROI));

    for (int i = 0; i < batchSize; i++) {
        // Full image output
        dstRoiTensor[i].xywhROI.xy.x = 0;
        dstRoiTensor[i].xywhROI.xy.y = 0;
        dstRoiTensor[i].xywhROI.roiWidth = width;
        dstRoiTensor[i].xywhROI.roiHeight = height;

        // Crop region (center quarter)
        cropRoiTensor[i].xywhROI.xy.x = width / 4;
        cropRoiTensor[i].xywhROI.xy.y = height / 4;
        cropRoiTensor[i].xywhROI.roiWidth = width / 2;
        cropRoiTensor[i].xywhROI.roiHeight = height / 2;

        // Patch region (same as crop)
        patchRoiTensor[i] = cropRoiTensor[i];
    }

    // Benchmark loop
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }        
        RppStatus status = rppt_crop_and_patch((RppPtr_t)input1, (RppPtr_t)input2, &srcDesc,
                                               (RppPtr_t)output, &dstDesc,
                                               dstRoiTensor, cropRoiTensor, patchRoiTensor,
                                               RpptRoiType::XYWH, handle, RPP_HOST_BACKEND);
        CHECK_RPP_STATUS(status, "RPP HOST CropAndPatch failed with status");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "crop=center_quarter, patch=center_quarter";
    printResult("RPP HOST BATCH CropAndPatch", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    free(input1);
    free(input2);
    free(output);
    free(dstRoiTensor);
    free(cropRoiTensor);
    free(patchRoiTensor);
}

// CropMirrorNormalize Batched (HOST)
void benchmark_RPP_HOST_CropMirrorNormalize_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Output is cropped to half
    int dstHeight = height / 2;
    int dstWidth = width / 2;

    // Set up descriptors
    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.dataType = RpptDataType::U8;
    int widthPadded = ((width / 8) * 8) + 8;
    int dstWidthPadded = ((dstWidth / 8) * 8) + 8;
    set_descriptor_dims_and_strides(&srcDesc, batchSize, height, widthPadded, channels, 0);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, dstHeight, dstWidthPadded, channels, 0);

    // Allocate buffers
    size_t srcBufferSize = (size_t)srcDesc.n * srcDesc.h * srcDesc.w * srcDesc.c;
    size_t dstBufferSize = (size_t)dstDesc.n * dstDesc.h * dstDesc.w * dstDesc.c;
    Rpp8u* input = (Rpp8u*)calloc(srcBufferSize, sizeof(Rpp8u));
    Rpp8u* output = (Rpp8u*)calloc(dstBufferSize, sizeof(Rpp8u));

    // Fill input buffer
    for (int i = 0; i < batchSize; i++) {
        for (int h = 0; h < height; h++) {
            memcpy(input + i * srcDesc.strides.nStride + h * srcDesc.strides.hStride,
                   imgs[i].data + h * width * channels,
                   width * channels);
        }
    }

    // Normalization parameters
    Rpp32f* offsetTensor = (Rpp32f*)calloc(batchSize * channels, sizeof(Rpp32f));
    Rpp32f* multiplierTensor = (Rpp32f*)calloc(batchSize * channels, sizeof(Rpp32f));
    Rpp32u* mirrorTensor = (Rpp32u*)calloc(batchSize, sizeof(Rpp32u));

    if (isColor) {
        Rpp32f mean[3] = {60.0f, 80.0f, 100.0f};
        Rpp32f stdDev[3] = {0.9f, 0.9f, 0.9f};
        for (int i = 0; i < batchSize; i++) {
            for (int c = 0; c < 3; c++) {
                offsetTensor[i * 3 + c] = -mean[c] / stdDev[c];
                multiplierTensor[i * 3 + c] = 1.0f / stdDev[c];
            }
            mirrorTensor[i] = 1;  // Horizontal flip
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

    // Setup crop ROI
    RpptROI* cropRoiTensor = (RpptROI*)calloc(batchSize, sizeof(RpptROI));
    for (int i = 0; i < batchSize; i++) {
        cropRoiTensor[i].xywhROI.xy.x = width / 4;
        cropRoiTensor[i].xywhROI.xy.y = height / 4;
        cropRoiTensor[i].xywhROI.roiWidth = dstWidth;
        cropRoiTensor[i].xywhROI.roiHeight = dstHeight;
    }

    // Benchmark loop
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }        
        RppStatus status = rppt_crop_mirror_normalize((RppPtr_t)input, &srcDesc,
                                                      (RppPtr_t)output, &dstDesc,
                                                      offsetTensor, multiplierTensor, mirrorTensor,
                                                      cropRoiTensor, RpptRoiType::XYWH,
                                                      handle, RPP_HOST_BACKEND);
        CHECK_RPP_STATUS(status, "RPP HOST CropMirrorNormalize failed with status");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "crop=half, mirror=horizontal, mean=60/80/100, stddev=0.9";
    printResult("RPP HOST BATCH CropMirrorNormalize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    free(input);
    free(output);
    free(offsetTensor);
    free(multiplierTensor);
    free(mirrorTensor);
    free(cropRoiTensor);
}

// ResizeMirrorNormalize Batched (HOST)
void benchmark_RPP_HOST_ResizeMirrorNormalize_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Resize to half
    int dstHeight = height / 2;
    int dstWidth = width / 2;

    // Set up descriptors
    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.dataType = RpptDataType::U8;
    int widthPadded = ((width / 8) * 8) + 8;
    int dstWidthPadded = ((dstWidth / 8) * 8) + 8;
    set_descriptor_dims_and_strides(&srcDesc, batchSize, height, widthPadded, channels, 0);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, dstHeight, dstWidthPadded, channels, 0);

    // Allocate buffers
    size_t srcBufferSize = (size_t)srcDesc.n * srcDesc.h * srcDesc.w * srcDesc.c;
    size_t dstBufferSize = (size_t)dstDesc.n * dstDesc.h * dstDesc.w * dstDesc.c;
    Rpp8u* input = (Rpp8u*)calloc(srcBufferSize, sizeof(Rpp8u));
    Rpp8u* output = (Rpp8u*)calloc(dstBufferSize, sizeof(Rpp8u));

    // Fill input buffer
    for (int i = 0; i < batchSize; i++) {
        for (int h = 0; h < height; h++) {
            memcpy(input + i * srcDesc.strides.nStride + h * srcDesc.strides.hStride,
                   imgs[i].data + h * width * channels,
                   width * channels);
        }
    }

    // Destination image sizes
    RpptImagePatch* dstImgSizes = (RpptImagePatch*)calloc(batchSize, sizeof(RpptImagePatch));
    for (int i = 0; i < batchSize; i++) {
        dstImgSizes[i].width = dstWidth;
        dstImgSizes[i].height = dstHeight;
    }

    // Normalization parameters
    Rpp32f* meanTensor = (Rpp32f*)calloc(batchSize * channels, sizeof(Rpp32f));
    Rpp32f* stdDevTensor = (Rpp32f*)calloc(batchSize * channels, sizeof(Rpp32f));
    Rpp32u* mirrorTensor = (Rpp32u*)calloc(batchSize, sizeof(Rpp32u));

    for (int i = 0; i < batchSize; i++) {
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
        mirrorTensor[i] = 1;  // Horizontal flip
    }

    // Setup ROI
    RpptROI* roiTensor = (RpptROI*)calloc(batchSize, sizeof(RpptROI));
    for (int i = 0; i < batchSize; i++) {
        roiTensor[i].xywhROI.xy.x = 0;
        roiTensor[i].xywhROI.xy.y = 0;
        roiTensor[i].xywhROI.roiWidth = dstWidth;
        roiTensor[i].xywhROI.roiHeight = dstHeight;
    }

    // Benchmark loop
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }        
        RppStatus status = rppt_resize_mirror_normalize((RppPtr_t)input, &srcDesc,
                                                        (RppPtr_t)output, &dstDesc,
                                                        dstImgSizes, RpptInterpolationType::BILINEAR,
                                                        meanTensor, stdDevTensor, mirrorTensor,
                                                        roiTensor, RpptRoiType::XYWH,
                                                        handle, RPP_HOST_BACKEND);
        CHECK_RPP_STATUS(status, "RPP HOST ResizeMirrorNormalize failed with status");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "resize=half, mirror=horizontal, mean=60/80/100, stddev=1";
    printResult("RPP HOST BATCH ResizeMirrorNormalize", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    free(input);
    free(output);
    free(dstImgSizes);
    free(meanTensor);
    free(stdDevTensor);
    free(mirrorTensor);
    free(roiTensor);
}

// ResizeCropMirror Batched (HOST)
void benchmark_RPP_HOST_ResizeCropMirror_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle) {
    int batchSize = imgs.size();
    int channels = isColor ? 3 : 1;
    int height = imgs[0].rows;
    int width = imgs[0].cols;

    // Target parameters (same as all other versions)
    int targetWidth = 224;
    int targetHeight = 224;

    // Calculate center 80% crop region
    int cropWidth = (width * 4) / 5;
    int cropHeight = (height * 4) / 5;
    int cropX = (width - cropWidth) / 2;
    int cropY = (height - cropHeight) / 2;

    // Set up descriptors
    RpptDesc srcDesc, dstDesc;
    srcDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    srcDesc.dataType = RpptDataType::U8;
    dstDesc.layout = isColor ? RpptLayout::NHWC : RpptLayout::NCHW;
    dstDesc.dataType = RpptDataType::U8;
    int widthPadded = ((width / 8) * 8) + 8;
    int dstWidthPadded = ((targetWidth / 8) * 8) + 8;
    set_descriptor_dims_and_strides(&srcDesc, batchSize, height, widthPadded, channels, 0);
    set_descriptor_dims_and_strides(&dstDesc, batchSize, targetHeight, dstWidthPadded, channels, 0);

    // Allocate buffers
    size_t srcBufferSize = (size_t)srcDesc.n * srcDesc.h * srcDesc.w * srcDesc.c;
    size_t dstBufferSize = (size_t)dstDesc.n * dstDesc.h * dstDesc.w * dstDesc.c;
    Rpp8u* input = (Rpp8u*)calloc(srcBufferSize, sizeof(Rpp8u));
    Rpp8u* output = (Rpp8u*)calloc(dstBufferSize, sizeof(Rpp8u));

    // Fill input buffer
    for (int i = 0; i < batchSize; i++) {
        for (int h = 0; h < height; h++) {
            memcpy(input + i * srcDesc.strides.nStride + h * srcDesc.strides.hStride,
                   imgs[i].data + h * width * channels,
                   width * channels);
        }
    }

    // Resize target size
    RpptImagePatch* dstImgSizes = (RpptImagePatch*)calloc(batchSize, sizeof(RpptImagePatch));
    for (int i = 0; i < batchSize; i++) {
        dstImgSizes[i].width = targetWidth;
        dstImgSizes[i].height = targetHeight;
    }

    // Mirror flag
    Rpp32u* mirrorTensor = (Rpp32u*)calloc(batchSize, sizeof(Rpp32u));
    for (int i = 0; i < batchSize; i++) {
        mirrorTensor[i] = 1;  // Horizontal flip
    }

    // Setup ROIs - defines center 80% crop from source
    RpptROI* dstRoiTensor = (RpptROI*)calloc(batchSize, sizeof(RpptROI));
    for (int i = 0; i < batchSize; i++) {
        dstRoiTensor[i].xywhROI.xy.x = cropX;
        dstRoiTensor[i].xywhROI.xy.y = cropY;
        dstRoiTensor[i].xywhROI.roiWidth = cropWidth;
        dstRoiTensor[i].xywhROI.roiHeight = cropHeight;
    }

    // Benchmark loop
    for (int k = 0; k < HOST_TOTAL_RUNS; ++k) {
        if (k == WARMUP_RUNS) {
            perfMonitor.start(DeviceType::CPU, 0);
        }        
        RppStatus status = rppt_resize_crop_mirror((RppPtr_t)input, &srcDesc,
                                                   (RppPtr_t)output, &dstDesc,
                                                   dstImgSizes, RpptInterpolationType::BILINEAR,
                                                   mirrorTensor, dstRoiTensor, RpptRoiType::XYWH,
                                                   handle, RPP_HOST_BACKEND);
        CHECK_RPP_STATUS(status, "RPP HOST ResizeCropMirror failed with status");
    }
    perfMonitor.stop();

    ostringstream params;
    params << "crop=80%,resize=224x224,mirror";
    printResult("RPP HOST BATCH ResizeCropMirror", imgs.size(), isColor,
                perfMonitor.getTotalTime(), perfMonitor.getTotalEnergy(), params.str());

    free(input);
    free(output);
    free(dstImgSizes);
    free(mirrorTensor);
    free(dstRoiTensor);
}
