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

#include <cstring>

#include "benchmarks_common.h"

void printUsage(const char* programName) {
    cout << "Usage: " << programName << " [OPTIONS]\n" << endl;
    cout << "Options:" << endl;
    cout << "  -t, --threads <N>        Number of threads to use (default: auto-detect)" << endl;
    cout << "  -n, --num-runs <N>       Number of benchmark runs (default: 100)" << endl;
    cout << "  -w, --warmup-runs <N>    Number of warmup runs (default: 50)" << endl;
    cout << "  -g, --gray-path <PATH>   Path to grayscale images (default: "
         << DEFAULT_GRAY_IMAGE_PATH << ")" << endl;
    cout << "  -r, --rgb-path <PATH>    Path to RGB images (default: " << DEFAULT_RGB_IMAGE_PATH
         << ")" << endl;
    cout << "  -h, --help               Display this help message" << endl;
    cout << "\nExamples:" << endl;
    cout << "  " << programName
         << "                           # Auto-detect threads, 100 runs (default)" << endl;
    cout << "  " << programName << " --threads 64              # Use 64 threads" << endl;
    cout << "  " << programName << " -t 32 -n 50               # Use 32 threads with 50 runs"
         << endl;
    cout << "  " << programName << " -t 32 -n 50 -w 25         # Use 32 threads, 50 runs, 25 warmup runs"
         << endl;
    cout << "  " << programName << " -t 32 -g ./my_images/     # Use 32 threads with custom dataset"
         << endl;
    cout << endl;
}

int main(int argc, char* argv[]) {
    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) {
            if (i + 1 < argc) {
                NUM_THREADS = atoi(argv[++i]);
                if (NUM_THREADS <= 0) {
                    cerr << "Error: Thread count must be a positive integer" << endl;
                    return 1;
                }
            } else {
                cerr << "Error: --threads requires a value" << endl;
                printUsage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--num-runs") == 0) {
            if (i + 1 < argc) {
                PERF_RUNS = atoi(argv[++i]);
                if (PERF_RUNS <= 0) {
                    cerr << "Error: Number of runs must be a positive integer" << endl;
                    return 1;
                }
            } else {
                cerr << "Error: --num-runs requires a value" << endl;
                printUsage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--warmup-runs") == 0) {
            if (i + 1 < argc) {
                WARMUP_RUNS = atoi(argv[++i]);
                if (WARMUP_RUNS < 0) {
                    cerr << "Error: Number of warmup runs must be a non-negative integer" << endl;
                    return 1;
                }
            } else {
                cerr << "Error: --warmup-runs requires a value" << endl;
                printUsage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "-g") == 0 || strcmp(argv[i], "--gray-path") == 0) {
            if (i + 1 < argc) {
                GRAY_IMAGE_PATH = argv[++i];
            } else {
                cerr << "Error: --gray-path requires a value" << endl;
                printUsage(argv[0]);
                return 1;
            }
        } else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--rgb-path") == 0) {
            if (i + 1 < argc) {
                RGB_IMAGE_PATH = argv[++i];
            } else {
                cerr << "Error: --rgb-path requires a value" << endl;
                printUsage(argv[0]);
                return 1;
            }
        } else {
            cerr << "Error: Unknown option: " << argv[i] << endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    // Auto-detect thread count if not specified
    int maxAvailableThreads = omp_get_max_threads();
    if (NUM_THREADS == 0) {
        NUM_THREADS = maxAvailableThreads;
        cout << "Auto-detected " << NUM_THREADS << " available threads" << endl;
    }

    // Control OpenCV threading to match RPP HOST configuration for fair comparison
    cv::setNumThreads(NUM_THREADS);

    int batchSizeGray = 0, maxWidthGray = 0, maxHeightGray = 0;
    int batchSizeRGB = 0, maxWidthRGB = 0, maxHeightRGB = 0;

    vector<Mat> imgsGray =
        loadBatchImages(GRAY_IMAGE_PATH, batchSizeGray, maxWidthGray, maxHeightGray, false);
    vector<Mat> imgsRGB =
        loadBatchImages(RGB_IMAGE_PATH, batchSizeRGB, maxWidthRGB, maxHeightRGB, true);

    if (imgsGray.empty() && imgsRGB.empty()) {
        cerr << "No images found in the dataset directories!" << endl;
        return -1;
    }

    // Initialize global image metadata for Excel output
    if (!imgsGray.empty()) {
        ostringstream oss;
        oss << maxWidthGray << "x" << maxHeightGray;
        grayImageSize = oss.str();
        grayImageDtype = getDtypeString(imgsGray[0].type());
        grayBatchSize = batchSizeGray;
    }
    if (!imgsRGB.empty()) {
        ostringstream oss;
        oss << maxWidthRGB << "x" << maxHeightRGB;
        rgbImageSize = oss.str();
        rgbImageDtype = getDtypeString(imgsRGB[0].type());
        rgbBatchSize = batchSizeRGB;
    }

    // For Single image processing benchmarking
    // Initialize RPP HOST backend handle
    rppHandle_t handleHost;
    rppStatus_t status;
    status = rppCreate(&handleHost, 1, NUM_THREADS, nullptr, RPP_HOST_BACKEND);
    if (status != rppStatusSuccess) {
        cerr << "Error: Failed to initialize RPP HOST handle with " << NUM_THREADS
             << " threads (Status: " << status << ")" << endl;
        return 1;
    }

    // Initialize RPP HIP backend handle
    rppHandle_t handleHip;
    hipStream_t stream;
    hipError_t hipErr = hipStreamCreate(&stream);
    if (hipErr != hipSuccess) {
        cerr << "Error: Failed to create HIP stream (Error: " << hipErr << ")" << endl;
        rppDestroy(handleHost, RPP_HOST_BACKEND);
        return 1;
    }
    status = rppCreate(&handleHip, 1, 0, stream, RPP_HIP_BACKEND);
    if (status != rppStatusSuccess) {
        cerr << "Error: Failed to initialize RPP HIP handle (Status: " << status << ")" << endl;
        (void)hipStreamDestroy(stream);
        rppDestroy(handleHost, RPP_HOST_BACKEND);
        return 1;
    }

    // For Batched images processing benchmarking
    // Use maximum batch size to handle both grayscale and RGB images
    int maxBatchSize = max(batchSizeGray, batchSizeRGB);

    // Initialize RPP HOST backend handle
    rppHandle_t handleHostBatched;
    status = rppCreate(&handleHostBatched, maxBatchSize, maxAvailableThreads, nullptr, RPP_HOST_BACKEND);
    if (status != rppStatusSuccess) {
        cerr << "Error: Failed to initialize Batched RPP HOST handle with batch size " << maxBatchSize
             << " and " << maxAvailableThreads << " threads (Status: " << status << ")" << endl;
        rppDestroy(handleHost, RPP_HOST_BACKEND);
        rppDestroy(handleHip, RPP_HIP_BACKEND);
        (void)hipStreamDestroy(stream);
        return 1;
    }

    // Initialize RPP HIP backend handle
    rppHandle_t handleHipBatched;
    hipStream_t streamBatched;
    hipErr = hipStreamCreate(&streamBatched);
    if (hipErr != hipSuccess) {
        cerr << "Error: Failed to create batched HIP stream (Error: " << hipErr << ")" << endl;
        rppDestroy(handleHostBatched, RPP_HOST_BACKEND);
        rppDestroy(handleHost, RPP_HOST_BACKEND);
        rppDestroy(handleHip, RPP_HIP_BACKEND);
        (void)hipStreamDestroy(stream);
        return 1;
    }
    status = rppCreate(&handleHipBatched, maxBatchSize, 0, streamBatched, RPP_HIP_BACKEND);
    if (status != rppStatusSuccess) {
        cerr << "Error: Failed to initialize Batched RPP HIP handle with batch size " << maxBatchSize
             << " (Status: " << status << ")" << endl;
        (void)hipStreamDestroy(streamBatched);
        rppDestroy(handleHostBatched, RPP_HOST_BACKEND);
        rppDestroy(handleHost, RPP_HOST_BACKEND);
        rppDestroy(handleHip, RPP_HIP_BACKEND);
        (void)hipStreamDestroy(stream);
        return 1;
    }

    // Parameters
    const float alpha = 1.2f;
    const float beta = 20.0f;
    const float blendAlpha = 0.5f;
    const float gamma = 1.5f;
    const float hueDelta = 10.f;
    const float satFactor = 1.2f;
    const float contrastFactor = 1.3f;
    const float contrastCenter = 128.f;
    const float exposureFactor = 1.4f;
    const float addVal = 10.f;
    const float subVal = 10.f;
    const float mulVal = 1.2f;
    const float noiseProb = 0.05f;
    const float noiseMean = 0.f;
    const float noiseStd = 15.f;
    const int resizeW = 960;
    const int resizeH = 540;
    const int cropW = 800;
    const int cropH = 600;
    const int filterKernel = 3;
    const int medianKernel = 3;
    const double gaussSigma = 5.0;
    const float angleDeg = 45.f;
    const double threshVal = 128.0;
    const int morphKernel = 3;

    cout << "\n========================================" << endl;
    cout << "OPENCV vs RPP HOST vs RPP HIP BENCHMARKING" << endl;
    cout << "Competitive Analysis: All Kernels" << endl;
    cout << "========================================" << endl;

    // Print version and system information
    cout << "\n--- System Information ---" << endl;
    cout << "OpenCV Version: " << CV_VERSION << endl;
    cout << "RPP Version: " << getRPPVersion() << endl;
    cout << "ROCm Version: " << getROCmVersion() << endl;
    cout << "\n--- Host Information ---" << endl;
    cout << "OS: " << getOSInfo() << endl;
    cout << "CPU: " << getCPUInfo() << endl;
    cout << "GPU: " << getGPUInfo() << endl;
    cout << "Memory: " << getMemoryInfo() << endl;
    cout << "\n--- Benchmark Configuration ---" << endl;
    cout << "Number of Threads: " << NUM_THREADS << " (max available: " << maxAvailableThreads
         << ")" << endl;
    cout << "Number of Runs: " << PERF_RUNS << endl;
    cout << "Number of Warmup Runs: " << WARMUP_RUNS << endl;

    // Initialize TOTAL_RUNS after command-line arguments are processed
    TOTAL_RUNS = WARMUP_RUNS + PERF_RUNS;

    cout << "Grayscale Dataset: " << GRAY_IMAGE_PATH << endl;
    cout << "RGB Dataset: " << RGB_IMAGE_PATH << endl;
    cout << "========================================" << endl;

    // ==================== GRAYSCALE ====================
    if (!imgsGray.empty()) {
        cout << "\n========== GRAYSCALE IMAGES ==========" << endl;

        cout << "\n--- Color Augmentations ---" << endl;
        benchmark_OpenCV_Brightness(imgsGray, false, alpha, beta);
        benchmark_RPP_HOST_Brightness(imgsGray, false, alpha, beta, handleHost);
        benchmark_RPP_HIP_Brightness(imgsGray, false, alpha, beta, handleHip, stream);
        benchmark_RPP_HOST_Brightness_Batched(imgsGray, false, alpha, beta, handleHostBatched);
        benchmark_RPP_HIP_Brightness_Batched(imgsGray, false, alpha, beta, handleHipBatched, streamBatched);
        
        benchmark_OpenCV_GammaCorrection(imgsGray, false, gamma);
        benchmark_RPP_HOST_GammaCorrection(imgsGray, false, gamma, handleHost);
        benchmark_RPP_HIP_GammaCorrection(imgsGray, false, gamma, handleHip, stream);
        benchmark_RPP_HOST_GammaCorrection_Batched(imgsGray, false, gamma, handleHostBatched);
        benchmark_RPP_HIP_GammaCorrection_Batched(imgsGray, false, gamma, handleHipBatched, streamBatched);

        benchmark_OpenCV_Contrast(imgsGray, false, contrastFactor, contrastCenter);
        benchmark_RPP_HOST_Contrast(imgsGray, false, contrastFactor, contrastCenter, handleHost);
        benchmark_RPP_HIP_Contrast(imgsGray, false, contrastFactor, contrastCenter, handleHip, stream);
        benchmark_RPP_HOST_Contrast_Batched(imgsGray, false, contrastFactor, contrastCenter, handleHostBatched);
        benchmark_RPP_HIP_Contrast_Batched(imgsGray, false, contrastFactor, contrastCenter, handleHipBatched, streamBatched);

        benchmark_OpenCV_Exposure(imgsGray, false, exposureFactor);
        benchmark_RPP_HOST_Exposure(imgsGray, false, exposureFactor, handleHost);
        benchmark_RPP_HIP_Exposure(imgsGray, false, exposureFactor, handleHip, stream);
        benchmark_RPP_HOST_Exposure_Batched(imgsGray, false, exposureFactor, handleHostBatched);
        benchmark_RPP_HIP_Exposure_Batched(imgsGray, false, exposureFactor, handleHipBatched, streamBatched);

        cout << "\n--- Filter Augmentations ---" << endl;
        benchmark_OpenCV_BoxFilter(imgsGray, false, filterKernel);
        benchmark_RPP_HOST_BoxFilter(imgsGray, false, filterKernel, handleHost);
        benchmark_RPP_HIP_BoxFilter(imgsGray, false, filterKernel, handleHip, stream);
        benchmark_RPP_HOST_BoxFilter_Batched(imgsGray, false, filterKernel, handleHostBatched);
        benchmark_RPP_HIP_BoxFilter_Batched(imgsGray, false, filterKernel, handleHipBatched, streamBatched);

        benchmark_OpenCV_MedianFilter(imgsGray, false, medianKernel);
        benchmark_RPP_HOST_MedianFilter(imgsGray, false, medianKernel, handleHost);
        benchmark_RPP_HIP_MedianFilter(imgsGray, false, medianKernel, handleHip, stream);
        benchmark_RPP_HOST_MedianFilter_Batched(imgsGray, false, medianKernel, handleHostBatched);
        benchmark_RPP_HIP_MedianFilter_Batched(imgsGray, false, medianKernel, handleHipBatched, streamBatched);

        benchmark_OpenCV_GaussianFilter(imgsGray, false, filterKernel, gaussSigma);
        benchmark_RPP_HOST_GaussianFilter(imgsGray, false, filterKernel, gaussSigma, handleHost);
        benchmark_RPP_HIP_GaussianFilter(imgsGray, false, filterKernel, gaussSigma, handleHip, stream);
        benchmark_RPP_HOST_GaussianFilter_Batched(imgsGray, false, filterKernel, gaussSigma, handleHostBatched);
        benchmark_RPP_HIP_GaussianFilter_Batched(imgsGray, false, filterKernel, gaussSigma, handleHipBatched, streamBatched);

        benchmark_OpenCV_SobelFilter(imgsGray, false, 0);
        benchmark_RPP_HOST_SobelFilter(imgsGray, false, 0, handleHost);
        benchmark_RPP_HIP_SobelFilter(imgsGray, false, 0, handleHip, stream);
        benchmark_RPP_HOST_SobelFilter_Batched(imgsGray, false, 0, handleHostBatched);
        benchmark_RPP_HIP_SobelFilter_Batched(imgsGray, false, 0, handleHipBatched, streamBatched);

        benchmark_OpenCV_SobelFilter(imgsGray, false, 1);
        benchmark_RPP_HOST_SobelFilter(imgsGray, false, 1, handleHost);
        benchmark_RPP_HIP_SobelFilter(imgsGray, false, 1, handleHip, stream);
        benchmark_RPP_HOST_SobelFilter_Batched(imgsGray, false, 1, handleHostBatched);
        benchmark_RPP_HIP_SobelFilter_Batched(imgsGray, false, 1, handleHipBatched, streamBatched);

        benchmark_OpenCV_SobelFilter(imgsGray, false, 2);
        benchmark_RPP_HOST_SobelFilter(imgsGray, false, 2, handleHost);
        benchmark_RPP_HIP_SobelFilter(imgsGray, false, 2, handleHip, stream);
        benchmark_RPP_HOST_SobelFilter_Batched(imgsGray, false, 2, handleHostBatched);
        benchmark_RPP_HIP_SobelFilter_Batched(imgsGray, false, 2, handleHipBatched, streamBatched);

        benchmark_OpenCV_Emboss(imgsGray, false, 3, 1.0f);
        benchmark_RPP_HOST_Emboss(imgsGray, false, 3, 1.0f, handleHost);
        benchmark_RPP_HIP_Emboss(imgsGray, false, 3, 1.0f, handleHip, stream);
        benchmark_RPP_HOST_Emboss_Batched(imgsGray, false, 3, 1.0f, handleHostBatched);
        benchmark_RPP_HIP_Emboss_Batched(imgsGray, false, 3, 1.0f, handleHipBatched, streamBatched);

        cout << "\n--- Geometric Augmentations ---" << endl;
        benchmark_OpenCV_Crop(imgsGray, false, cropW, cropH);
        benchmark_RPP_HOST_Crop(imgsGray, false, cropW, cropH, handleHost);
        benchmark_RPP_HIP_Crop(imgsGray, false, cropW, cropH, handleHip, stream);
        benchmark_RPP_HOST_Crop_Batched(imgsGray, false, cropW, cropH, handleHostBatched);
        benchmark_RPP_HIP_Crop_Batched(imgsGray, false, cropW, cropH, handleHipBatched, streamBatched);

        benchmark_OpenCV_Resize(imgsGray, false, resizeW, resizeH, INTER_NEAREST, "Nearest");
        benchmark_RPP_HOST_Resize(imgsGray, false, resizeW, resizeH,
                             RpptInterpolationType::NEAREST_NEIGHBOR, "Nearest", handleHost);
        benchmark_RPP_HIP_Resize(imgsGray, false, resizeW, resizeH,
                             RpptInterpolationType::NEAREST_NEIGHBOR, "Nearest", handleHip, stream);
        benchmark_RPP_HOST_Resize_Batched(imgsGray, false, resizeW, resizeH,
                             RpptInterpolationType::NEAREST_NEIGHBOR, "Nearest", handleHostBatched);
        benchmark_RPP_HIP_Resize_Batched(imgsGray, false, resizeW, resizeH,
                             RpptInterpolationType::NEAREST_NEIGHBOR, "Nearest", handleHipBatched, streamBatched);

        benchmark_OpenCV_Resize(imgsGray, false, resizeW, resizeH, INTER_LINEAR, "Bilinear");
        benchmark_RPP_HOST_Resize(imgsGray, false, resizeW, resizeH, RpptInterpolationType::BILINEAR,
                             "Bilinear", handleHost);
        benchmark_RPP_HIP_Resize(imgsGray, false, resizeW, resizeH, RpptInterpolationType::BILINEAR,
                             "Bilinear", handleHip, stream);
        benchmark_RPP_HOST_Resize_Batched(imgsGray, false, resizeW, resizeH, RpptInterpolationType::BILINEAR,
                             "Bilinear", handleHostBatched);
        benchmark_RPP_HIP_Resize_Batched(imgsGray, false, resizeW, resizeH, RpptInterpolationType::BILINEAR,
                             "Bilinear", handleHipBatched, streamBatched);

        benchmark_OpenCV_Resize(imgsGray, false, resizeW, resizeH, INTER_CUBIC, "Bicubic");
        benchmark_RPP_HOST_Resize(imgsGray, false, resizeW, resizeH, RpptInterpolationType::BICUBIC,
                             "Bicubic", handleHost);
        benchmark_RPP_HIP_Resize(imgsGray, false, resizeW, resizeH, RpptInterpolationType::BICUBIC,
                             "Bicubic", handleHip, stream);
        benchmark_RPP_HOST_Resize_Batched(imgsGray, false, resizeW, resizeH, RpptInterpolationType::BICUBIC,
                             "Bicubic", handleHostBatched);
        benchmark_RPP_HIP_Resize_Batched(imgsGray, false, resizeW, resizeH, RpptInterpolationType::BICUBIC,
                             "Bicubic", handleHipBatched, streamBatched);

        benchmark_OpenCV_Flip(imgsGray, false, 1);
        benchmark_RPP_HOST_Flip(imgsGray, false, 1, handleHost);
        benchmark_RPP_HIP_Flip(imgsGray, false, 1, handleHip, stream);
        benchmark_RPP_HOST_Flip_Batched(imgsGray, false, 1, handleHostBatched);
        benchmark_RPP_HIP_Flip_Batched(imgsGray, false, 1, handleHipBatched, streamBatched);

        // benchmark_OpenCV_Flip(imgsGray, false, 0);
        // benchmark_RPP_HOST_Flip(imgsGray, false, 0, handleHost);
        // benchmark_RPP_HIP_Flip(imgsGray, false, 0, handleHip, stream);
        // benchmark_RPP_HOST_Flip_Batched(imgsGray, false, 0, handleHostBatched);
        // benchmark_RPP_HIP_Flip_Batched(imgsGray, false, 0, handleHipBatched, streamBatched);

        // benchmark_OpenCV_Flip(imgsGray, false, -1);
        // benchmark_RPP_HOST_Flip(imgsGray, false, -1, handleHost);
        // benchmark_RPP_HIP_Flip(imgsGray, false, -1, handleHip, stream);
        // benchmark_RPP_HOST_Flip_Batched(imgsGray, false, -1, handleHostBatched);
        // benchmark_RPP_HIP_Flip_Batched(imgsGray, false, -1, handleHipBatched, streamBatched);

        benchmark_OpenCV_Rotate(imgsGray, false, angleDeg);
        benchmark_RPP_HOST_Rotate(imgsGray, false, angleDeg, handleHost);
        benchmark_RPP_HIP_Rotate(imgsGray, false, angleDeg, handleHip, stream);
        benchmark_RPP_HOST_Rotate_Batched(imgsGray, false, angleDeg, handleHostBatched);
        benchmark_RPP_HIP_Rotate_Batched(imgsGray, false, angleDeg, handleHipBatched, streamBatched);

        benchmark_OpenCV_WarpAffine(imgsGray, false);
        benchmark_RPP_HOST_WarpAffine(imgsGray, false, handleHost);
        benchmark_RPP_HIP_WarpAffine(imgsGray, false, handleHip, stream);
        benchmark_RPP_HOST_WarpAffine_Batched(imgsGray, false, handleHostBatched);
        benchmark_RPP_HIP_WarpAffine_Batched(imgsGray, false, handleHipBatched, streamBatched);

        benchmark_OpenCV_WarpPerspective(imgsGray, false);
        benchmark_RPP_HOST_WarpPerspective(imgsGray, false, handleHost);
        benchmark_RPP_HIP_WarpPerspective(imgsGray, false, handleHip, stream);
        benchmark_RPP_HOST_WarpPerspective_Batched(imgsGray, false, handleHostBatched);
        benchmark_RPP_HIP_WarpPerspective_Batched(imgsGray, false, handleHipBatched, streamBatched);

        benchmark_OpenCV_Fisheye(imgsGray, false);
        benchmark_RPP_HOST_Fisheye(imgsGray, false, handleHost);
        benchmark_RPP_HIP_Fisheye(imgsGray, false, handleHip, stream);
        benchmark_RPP_HOST_Fisheye_Batched(imgsGray, false, handleHostBatched);
        benchmark_RPP_HIP_Fisheye_Batched(imgsGray, false, handleHipBatched, streamBatched);

        benchmark_OpenCV_LensCorrection(imgsGray, false);
        benchmark_RPP_HOST_LensCorrection(imgsGray, false, handleHost);
        benchmark_RPP_HIP_LensCorrection(imgsGray, false, handleHip, stream);
        benchmark_RPP_HOST_LensCorrection_Batched(imgsGray, false, handleHostBatched);
        benchmark_RPP_HIP_LensCorrection_Batched(imgsGray, false, handleHipBatched, streamBatched);

        cout << "\n--- Morphological Operations ---" << endl;
        benchmark_OpenCV_Erode(imgsGray, false, morphKernel);
        benchmark_RPP_HOST_Erode(imgsGray, false, morphKernel, handleHost);
        benchmark_RPP_HIP_Erode(imgsGray, false, morphKernel, handleHip, stream);
        benchmark_RPP_HOST_Erode_Batched(imgsGray, false, morphKernel, handleHostBatched);
        benchmark_RPP_HIP_Erode_Batched(imgsGray, false, morphKernel, handleHipBatched, streamBatched);

        benchmark_OpenCV_Dilate(imgsGray, false, morphKernel);
        benchmark_RPP_HOST_Dilate(imgsGray, false, morphKernel, handleHost);
        benchmark_RPP_HIP_Dilate(imgsGray, false, morphKernel, handleHip, stream);
        benchmark_RPP_HOST_Dilate_Batched(imgsGray, false, morphKernel, handleHostBatched);
        benchmark_RPP_HIP_Dilate_Batched(imgsGray, false, morphKernel, handleHipBatched, streamBatched);

        cout << "\n--- Arithmetic Operations ---" << endl;
        benchmark_OpenCV_AddScalar(imgsGray, false, addVal);
        benchmark_RPP_HOST_AddScalar(imgsGray, false, addVal, handleHost);
        benchmark_RPP_HIP_AddScalar(imgsGray, false, addVal, handleHip, stream);
        benchmark_RPP_HOST_AddScalar_Batched(imgsGray, false, addVal, handleHostBatched);
        benchmark_RPP_HIP_AddScalar_Batched(imgsGray, false, addVal, handleHipBatched, streamBatched);

        benchmark_OpenCV_SubtractScalar(imgsGray, false, subVal);
        benchmark_RPP_HOST_SubtractScalar(imgsGray, false, subVal, handleHost);
        benchmark_RPP_HIP_SubtractScalar(imgsGray, false, subVal, handleHip, stream);
        benchmark_RPP_HOST_SubtractScalar_Batched(imgsGray, false, subVal, handleHostBatched);
        benchmark_RPP_HIP_SubtractScalar_Batched(imgsGray, false, subVal, handleHipBatched, streamBatched);

        benchmark_OpenCV_MultiplyScalar(imgsGray, false, mulVal);
        benchmark_RPP_HOST_MultiplyScalar(imgsGray, false, mulVal, handleHost);
        benchmark_RPP_HIP_MultiplyScalar(imgsGray, false, mulVal, handleHip, stream);
        benchmark_RPP_HOST_MultiplyScalar_Batched(imgsGray, false, mulVal, handleHostBatched);
        benchmark_RPP_HIP_MultiplyScalar_Batched(imgsGray, false, mulVal, handleHipBatched, streamBatched);

        benchmark_OpenCV_Blend(imgsGray, false, blendAlpha);
        benchmark_RPP_HOST_Blend(imgsGray, false, blendAlpha, handleHost);
        benchmark_RPP_HIP_Blend(imgsGray, false, blendAlpha, handleHip, stream);
        benchmark_RPP_HOST_Blend_Batched(imgsGray, false, blendAlpha, handleHostBatched);
        benchmark_RPP_HIP_Blend_Batched(imgsGray, false, blendAlpha, handleHipBatched, streamBatched);

        cout << "\n--- Bitwise Operations ---" << endl;
        benchmark_OpenCV_BitwiseAnd(imgsGray, false);
        benchmark_RPP_HOST_BitwiseAnd(imgsGray, false, handleHost);
        benchmark_RPP_HIP_BitwiseAnd(imgsGray, false, handleHip, stream);
        benchmark_RPP_HOST_BitwiseAnd_Batched(imgsGray, false, handleHostBatched);
        benchmark_RPP_HIP_BitwiseAnd_Batched(imgsGray, false, handleHipBatched, streamBatched);

        benchmark_OpenCV_BitwiseOr(imgsGray, false);
        benchmark_RPP_HOST_BitwiseOr(imgsGray, false, handleHost);
        benchmark_RPP_HIP_BitwiseOr(imgsGray, false, handleHip, stream);
        benchmark_RPP_HOST_BitwiseOr_Batched(imgsGray, false, handleHostBatched);
        benchmark_RPP_HIP_BitwiseOr_Batched(imgsGray, false, handleHipBatched, streamBatched);

        benchmark_OpenCV_BitwiseNot(imgsGray, false);
        benchmark_RPP_HOST_BitwiseNot(imgsGray, false, handleHost);
        benchmark_RPP_HIP_BitwiseNot(imgsGray, false, handleHip, stream);
        benchmark_RPP_HOST_BitwiseNot_Batched(imgsGray, false, handleHostBatched);
        benchmark_RPP_HIP_BitwiseNot_Batched(imgsGray, false, handleHipBatched, streamBatched);

        benchmark_OpenCV_BitwiseXor(imgsGray, false);
        benchmark_RPP_HOST_BitwiseXor(imgsGray, false, handleHost);
        benchmark_RPP_HIP_BitwiseXor(imgsGray, false, handleHip, stream);
        benchmark_RPP_HOST_BitwiseXor_Batched(imgsGray, false, handleHostBatched);
        benchmark_RPP_HIP_BitwiseXor_Batched(imgsGray, false, handleHipBatched, streamBatched);

        cout << "\n--- Statistical Operations ---" << endl;
        benchmark_OpenCV_TensorMin(imgsGray, false);
        benchmark_RPP_HOST_TensorMin(imgsGray, false, handleHost);
        benchmark_RPP_HIP_TensorMin(imgsGray, false, handleHip, stream);
        benchmark_RPP_HOST_TensorMin_Batched(imgsGray, false, handleHostBatched);
        benchmark_RPP_HIP_TensorMin_Batched(imgsGray, false, handleHipBatched, streamBatched);

        benchmark_OpenCV_TensorMax(imgsGray, false);
        benchmark_RPP_HOST_TensorMax(imgsGray, false, handleHost);
        benchmark_RPP_HIP_TensorMax(imgsGray, false, handleHip, stream);
        benchmark_RPP_HOST_TensorMax_Batched(imgsGray, false, handleHostBatched);
        benchmark_RPP_HIP_TensorMax_Batched(imgsGray, false, handleHipBatched, streamBatched);

        benchmark_OpenCV_TensorSum(imgsGray, false);
        benchmark_RPP_HOST_TensorSum(imgsGray, false, handleHost);
        benchmark_RPP_HIP_TensorSum(imgsGray, false, handleHip, stream);
        benchmark_RPP_HOST_TensorSum_Batched(imgsGray, false, handleHostBatched);
        benchmark_RPP_HIP_TensorSum_Batched(imgsGray, false, handleHipBatched, streamBatched);

        benchmark_OpenCV_TensorMean(imgsGray, false);
        benchmark_RPP_HOST_TensorMean(imgsGray, false, handleHost);
        benchmark_RPP_HIP_TensorMean(imgsGray, false, handleHip, stream);
        benchmark_RPP_HOST_TensorMean_Batched(imgsGray, false, handleHostBatched);
        benchmark_RPP_HIP_TensorMean_Batched(imgsGray, false, handleHipBatched, streamBatched);

        benchmark_OpenCV_TensorStddev(imgsGray, false);
        benchmark_RPP_HOST_TensorStddev(imgsGray, false, handleHost);
        benchmark_RPP_HIP_TensorStddev(imgsGray, false, handleHip, stream);
        benchmark_RPP_HOST_TensorStddev_Batched(imgsGray, false, handleHostBatched);
        benchmark_RPP_HIP_TensorStddev_Batched(imgsGray, false, handleHipBatched, streamBatched);

        benchmark_OpenCV_Threshold(imgsGray, false, threshVal);
        benchmark_RPP_HOST_Threshold(imgsGray, false, threshVal, handleHost);
        benchmark_RPP_HIP_Threshold(imgsGray, false, threshVal, handleHip, stream);
        benchmark_RPP_HOST_Threshold_Batched(imgsGray, false, threshVal, handleHostBatched);
        benchmark_RPP_HIP_Threshold_Batched(imgsGray, false, threshVal, handleHipBatched, streamBatched);

        cout << "\n--- Effects Augmentations ---" << endl;
        benchmark_OpenCV_GaussianNoise(imgsGray, false, noiseMean, noiseStd);
        benchmark_RPP_HOST_GaussianNoise(imgsGray, false, noiseMean, noiseStd, handleHost);
        benchmark_RPP_HIP_GaussianNoise(imgsGray, false, noiseMean, noiseStd, handleHip, stream);
        benchmark_RPP_HOST_GaussianNoise_Batched(imgsGray, false, noiseMean, noiseStd, handleHostBatched);
        benchmark_RPP_HIP_GaussianNoise_Batched(imgsGray, false, noiseMean, noiseStd, handleHipBatched, streamBatched);

        benchmark_OpenCV_SaltAndPepperNoise(imgsGray, false, noiseProb);
        benchmark_RPP_HOST_SaltAndPepperNoise(imgsGray, false, noiseProb, handleHost);
        benchmark_RPP_HIP_SaltAndPepperNoise(imgsGray, false, noiseProb, handleHip, stream);
        benchmark_RPP_HOST_SaltAndPepperNoise_Batched(imgsGray, false, noiseProb, handleHostBatched);
        benchmark_RPP_HIP_SaltAndPepperNoise_Batched(imgsGray, false, noiseProb, handleHipBatched, streamBatched);

        benchmark_OpenCV_JpegCompressionDistortion(imgsGray, false, 50);
        benchmark_RPP_HOST_JpegCompressionDistortion(imgsGray, false, 50, handleHost);
        benchmark_RPP_HIP_JpegCompressionDistortion(imgsGray, false, 50, handleHip, stream);
        benchmark_RPP_HOST_JpegCompressionDistortion_Batched(imgsGray, false, 50, handleHostBatched);
        benchmark_RPP_HIP_JpegCompressionDistortion_Batched(imgsGray, false, 50, handleHipBatched, streamBatched);

        cout << "\n--- Data Operations ---" << endl;
        benchmark_OpenCV_Copy(imgsGray, false);
        benchmark_RPP_HOST_Copy(imgsGray, false, handleHost);
        benchmark_RPP_HIP_Copy(imgsGray, false, handleHip, stream);
        benchmark_RPP_HOST_Copy_Batched(imgsGray, false, handleHostBatched);
        benchmark_RPP_HIP_Copy_Batched(imgsGray, false, handleHipBatched, streamBatched);

        benchmark_OpenCV_Slice(imgsGray, false);
        benchmark_RPP_HOST_Slice(imgsGray, false, handleHost);
        benchmark_RPP_HIP_Slice(imgsGray, false, handleHip, stream);
        benchmark_RPP_HOST_Slice_Batched(imgsGray, false, handleHostBatched);
        benchmark_RPP_HIP_Slice_Batched(imgsGray, false, handleHipBatched, streamBatched);

        benchmark_OpenCV_Transpose(imgsGray, false);
        benchmark_RPP_HOST_Transpose(imgsGray, false, handleHost);
        benchmark_RPP_HIP_Transpose(imgsGray, false, handleHip, stream);
        benchmark_RPP_HOST_Transpose_Batched(imgsGray, false, handleHostBatched);
        benchmark_RPP_HIP_Transpose_Batched(imgsGray, false, handleHipBatched, streamBatched);

        cout << "\n--- Advanced Operations ---" << endl;
        benchmark_OpenCV_HistogramEqualize(imgsGray, false);
        benchmark_RPP_HOST_HistogramEqualize(imgsGray, false, handleHost);
        benchmark_RPP_HIP_HistogramEqualize(imgsGray, false, handleHip, stream);
        benchmark_RPP_HOST_HistogramEqualize_Batched(imgsGray, false, handleHostBatched);
        benchmark_RPP_HIP_HistogramEqualize_Batched(imgsGray, false, handleHipBatched, streamBatched);

        benchmark_OpenCV_LUT(imgsGray, false);
        benchmark_RPP_HOST_LUT(imgsGray, false, handleHost);
        benchmark_RPP_HIP_LUT(imgsGray, false, handleHip, stream);
        benchmark_RPP_HOST_LUT_Batched(imgsGray, false, handleHostBatched);
        benchmark_RPP_HIP_LUT_Batched(imgsGray, false, handleHipBatched, streamBatched);

        benchmark_OpenCV_Magnitude(imgsGray, false);
        benchmark_RPP_HOST_Magnitude(imgsGray, false, handleHost);
        benchmark_RPP_HIP_Magnitude(imgsGray, false, handleHip, stream);
        benchmark_RPP_HOST_Magnitude_Batched(imgsGray, false, handleHostBatched);
        benchmark_RPP_HIP_Magnitude_Batched(imgsGray, false, handleHipBatched, streamBatched);

        benchmark_OpenCV_Phase(imgsGray, false);
        benchmark_RPP_HOST_Phase(imgsGray, false, handleHost);
        benchmark_RPP_HIP_Phase(imgsGray, false, handleHip, stream);
        benchmark_RPP_HOST_Phase_Batched(imgsGray, false, handleHostBatched);
        benchmark_RPP_HIP_Phase_Batched(imgsGray, false, handleHipBatched, streamBatched);

        benchmark_OpenCV_Normalize(imgsGray, false);
        benchmark_RPP_HOST_Normalize(imgsGray, false, handleHost);
        benchmark_RPP_HIP_Normalize(imgsGray, false, handleHip, stream);
        benchmark_RPP_HOST_Normalize_Batched(imgsGray, false, handleHostBatched);
        benchmark_RPP_HIP_Normalize_Batched(imgsGray, false, handleHipBatched, streamBatched);

        benchmark_OpenCV_FusedMultiplyAddScalar(imgsGray, false, 1.2f, 10.0f);
        benchmark_RPP_HOST_FusedMultiplyAddScalar(imgsGray, false, 1.2f, 10.0f, handleHost);
        benchmark_RPP_HIP_FusedMultiplyAddScalar(imgsGray, false, 1.2f, 10.0f, handleHip, stream);
        benchmark_RPP_HOST_FusedMultiplyAddScalar_Batched(imgsGray, false, 1.2f, 10.0f, handleHostBatched);
        benchmark_RPP_HIP_FusedMultiplyAddScalar_Batched(imgsGray, false, 1.2f, 10.0f, handleHipBatched, streamBatched);

        benchmark_OpenCV_Remap(imgsGray, false);
        benchmark_RPP_HOST_Remap(imgsGray, false, handleHost);
        benchmark_RPP_HIP_Remap(imgsGray, false, handleHip, stream);
        benchmark_RPP_HOST_Remap_Batched(imgsGray, false, handleHostBatched);
        benchmark_RPP_HIP_Remap_Batched(imgsGray, false, handleHipBatched, streamBatched);
    }
    // ==================== RGB ====================
    if (!imgsRGB.empty()) {
        cout << "\n========== RGB IMAGES ==========" << endl;

        cout << "\n--- Color Augmentations ---" << endl;
        benchmark_OpenCV_Brightness(imgsRGB, true, alpha, beta);
        benchmark_RPP_HOST_Brightness(imgsRGB, true, alpha, beta, handleHost);
        benchmark_RPP_HIP_Brightness(imgsRGB, true, alpha, beta, handleHip, stream);
        benchmark_RPP_HOST_Brightness_Batched(imgsRGB, true, alpha, beta, handleHostBatched);
        benchmark_RPP_HIP_Brightness_Batched(imgsRGB, true, alpha, beta, handleHipBatched, streamBatched);

        benchmark_OpenCV_GammaCorrection(imgsRGB, true, gamma);
        benchmark_RPP_HOST_GammaCorrection(imgsRGB, true, gamma, handleHost);
        benchmark_RPP_HIP_GammaCorrection(imgsRGB, true, gamma, handleHip, stream);
        benchmark_RPP_HOST_GammaCorrection_Batched(imgsRGB, true, gamma, handleHostBatched);
        benchmark_RPP_HIP_GammaCorrection_Batched(imgsRGB, true, gamma, handleHipBatched, streamBatched);

        benchmark_OpenCV_Blend(imgsRGB, true, blendAlpha);
        benchmark_RPP_HOST_Blend(imgsRGB, true, blendAlpha, handleHost);
        benchmark_RPP_HIP_Blend(imgsRGB, true, blendAlpha, handleHip, stream);
        benchmark_RPP_HOST_Blend_Batched(imgsRGB, true, blendAlpha, handleHostBatched);
        benchmark_RPP_HIP_Blend_Batched(imgsRGB, true, blendAlpha, handleHipBatched, streamBatched);

        benchmark_OpenCV_Contrast(imgsRGB, true, contrastFactor, contrastCenter);
        benchmark_RPP_HOST_Contrast(imgsRGB, true, contrastFactor, contrastCenter, handleHost);
        benchmark_RPP_HIP_Contrast(imgsRGB, true, contrastFactor, contrastCenter, handleHip, stream);
        benchmark_RPP_HOST_Contrast_Batched(imgsRGB, true, contrastFactor, contrastCenter, handleHostBatched);
        benchmark_RPP_HIP_Contrast_Batched(imgsRGB, true, contrastFactor, contrastCenter, handleHipBatched, streamBatched);

        benchmark_OpenCV_Exposure(imgsRGB, true, exposureFactor);
        benchmark_RPP_HOST_Exposure(imgsRGB, true, exposureFactor, handleHost);
        benchmark_RPP_HIP_Exposure(imgsRGB, true, exposureFactor, handleHip, stream);
        benchmark_RPP_HOST_Exposure_Batched(imgsRGB, true, exposureFactor, handleHostBatched);
        benchmark_RPP_HIP_Exposure_Batched(imgsRGB, true, exposureFactor, handleHipBatched, streamBatched);

        benchmark_OpenCV_Hue(imgsRGB, hueDelta);
        benchmark_RPP_HOST_Hue(imgsRGB, hueDelta, handleHost);
        benchmark_RPP_HIP_Hue(imgsRGB, hueDelta, handleHip, stream);
        benchmark_RPP_HOST_Hue_Batched(imgsRGB, true, hueDelta, handleHostBatched);
        benchmark_RPP_HIP_Hue_Batched(imgsRGB, true, hueDelta, handleHipBatched, streamBatched);

        benchmark_OpenCV_Saturation(imgsRGB, satFactor);
        benchmark_RPP_HOST_Saturation(imgsRGB, satFactor, handleHost);
        benchmark_RPP_HIP_Saturation(imgsRGB, satFactor, handleHip, stream);
        benchmark_RPP_HOST_Saturation_Batched(imgsRGB, true, satFactor, handleHostBatched);
        benchmark_RPP_HIP_Saturation_Batched(imgsRGB, true, satFactor, handleHipBatched, streamBatched);

        benchmark_OpenCV_ColorToGreyscale(imgsRGB);
        benchmark_RPP_HOST_ColorToGreyscale(imgsRGB, handleHost);
        benchmark_RPP_HIP_ColorToGreyscale(imgsRGB, handleHip, stream);
        benchmark_RPP_HOST_ColorToGreyscale_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_ColorToGreyscale_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        // benchmark_OpenCV_ColorJitter(imgsRGB, 1.2f, 1.3f, 1.2f, 10.f);
        // benchmark_RPP_HOST_ColorJitter(imgsRGB, 1.2f, 1.3f, 1.2f, 10.f, handleHost);
        
        cout << "\n--- Filter Augmentations ---" << endl;
        benchmark_OpenCV_BoxFilter(imgsRGB, true, filterKernel);
        benchmark_RPP_HOST_BoxFilter(imgsRGB, true, filterKernel, handleHost);
        benchmark_RPP_HIP_BoxFilter(imgsRGB, true, filterKernel, handleHip, stream);
        benchmark_RPP_HOST_BoxFilter_Batched(imgsRGB, true, filterKernel, handleHostBatched);
        benchmark_RPP_HIP_BoxFilter_Batched(imgsRGB, true, filterKernel, handleHipBatched, streamBatched);

        benchmark_OpenCV_MedianFilter(imgsRGB, true, medianKernel);
        benchmark_RPP_HOST_MedianFilter(imgsRGB, true, medianKernel, handleHost);
        benchmark_RPP_HIP_MedianFilter(imgsRGB, true, medianKernel, handleHip, stream);
        benchmark_RPP_HOST_MedianFilter_Batched(imgsRGB, true, medianKernel, handleHostBatched);
        benchmark_RPP_HIP_MedianFilter_Batched(imgsRGB, true, medianKernel, handleHipBatched, streamBatched);

        benchmark_OpenCV_GaussianFilter(imgsRGB, true, filterKernel, gaussSigma);
        benchmark_RPP_HOST_GaussianFilter(imgsRGB, true, filterKernel, gaussSigma, handleHost);
        benchmark_RPP_HIP_GaussianFilter(imgsRGB, true, filterKernel, gaussSigma, handleHip, stream);
        benchmark_RPP_HOST_GaussianFilter_Batched(imgsRGB, true, filterKernel, gaussSigma, handleHostBatched);
        benchmark_RPP_HIP_GaussianFilter_Batched(imgsRGB, true, filterKernel, gaussSigma, handleHipBatched, streamBatched);

        // SobelFilter skipped for RGB (only works on grayscale - see grayscale section)
        benchmark_OpenCV_Emboss(imgsRGB, true, 3, 1.0f);
        benchmark_RPP_HOST_Emboss(imgsRGB, true, 3, 1.0f, handleHost);
        benchmark_RPP_HIP_Emboss(imgsRGB, true, 3, 1.0f, handleHip, stream);
        benchmark_RPP_HOST_Emboss_Batched(imgsRGB, true, 3, 1.0f, handleHostBatched);
        benchmark_RPP_HIP_Emboss_Batched(imgsRGB, true, 3, 1.0f, handleHipBatched, streamBatched);


        cout << "\n--- Geometric Augmentations ---" << endl;
        benchmark_OpenCV_Crop(imgsRGB, true, cropW, cropH);
        benchmark_RPP_HOST_Crop(imgsRGB, true, cropW, cropH, handleHost);
        benchmark_RPP_HIP_Crop(imgsRGB, true, cropW, cropH, handleHip, stream);
        benchmark_RPP_HOST_Crop_Batched(imgsRGB, true, cropW, cropH, handleHostBatched);
        benchmark_RPP_HIP_Crop_Batched(imgsRGB, true, cropW, cropH, handleHipBatched, streamBatched);

        benchmark_OpenCV_Resize(imgsRGB, true, resizeW, resizeH, INTER_NEAREST, "Nearest");
        benchmark_RPP_HOST_Resize(imgsRGB, true, resizeW, resizeH,
                             RpptInterpolationType::NEAREST_NEIGHBOR, "Nearest", handleHost);
        benchmark_RPP_HIP_Resize(imgsRGB, true, resizeW, resizeH,
                             RpptInterpolationType::NEAREST_NEIGHBOR, "Nearest", handleHip, stream);
        benchmark_RPP_HOST_Resize_Batched(imgsRGB, true, resizeW, resizeH,
                             RpptInterpolationType::NEAREST_NEIGHBOR, "Nearest", handleHostBatched);
        benchmark_RPP_HIP_Resize_Batched(imgsRGB, true, resizeW, resizeH,
                             RpptInterpolationType::NEAREST_NEIGHBOR, "Nearest", handleHipBatched, streamBatched);

        benchmark_OpenCV_Resize(imgsRGB, true, resizeW, resizeH, INTER_LINEAR, "Bilinear");
        benchmark_RPP_HOST_Resize(imgsRGB, true, resizeW, resizeH, RpptInterpolationType::BILINEAR,
                             "Bilinear", handleHost);
        benchmark_RPP_HIP_Resize(imgsRGB, true, resizeW, resizeH, RpptInterpolationType::BILINEAR,
                             "Bilinear", handleHip, stream);
        benchmark_RPP_HOST_Resize_Batched(imgsRGB, true, resizeW, resizeH, RpptInterpolationType::BILINEAR,
                             "Bilinear", handleHostBatched);
        benchmark_RPP_HIP_Resize_Batched(imgsRGB, true, resizeW, resizeH, RpptInterpolationType::BILINEAR,
                             "Bilinear", handleHipBatched, streamBatched);

        benchmark_OpenCV_Resize(imgsRGB, true, resizeW, resizeH, INTER_CUBIC, "Bicubic");
        benchmark_RPP_HOST_Resize(imgsRGB, true, resizeW, resizeH, RpptInterpolationType::BICUBIC,
                             "Bicubic", handleHost);
        benchmark_RPP_HIP_Resize(imgsRGB, true, resizeW, resizeH, RpptInterpolationType::BICUBIC,
                             "Bicubic", handleHip, stream);
        benchmark_RPP_HOST_Resize_Batched(imgsRGB, true, resizeW, resizeH, RpptInterpolationType::BICUBIC,
                             "Bicubic", handleHostBatched);
        benchmark_RPP_HIP_Resize_Batched(imgsRGB, true, resizeW, resizeH, RpptInterpolationType::BICUBIC,
                             "Bicubic", handleHipBatched, streamBatched);

        benchmark_OpenCV_Flip(imgsRGB, true, 1);
        benchmark_RPP_HOST_Flip(imgsRGB, true, 1, handleHost);
        benchmark_RPP_HIP_Flip(imgsRGB, true, 1, handleHip, stream);
        benchmark_RPP_HOST_Flip_Batched(imgsRGB, true, 1, handleHostBatched);
        benchmark_RPP_HIP_Flip_Batched(imgsRGB, true, 1, handleHipBatched, streamBatched);

        // benchmark_OpenCV_Flip(imgsRGB, true, 0);
        // benchmark_RPP_HOST_Flip(imgsRGB, true, 0, handleHost);
        // benchmark_RPP_HIP_Flip(imgsRGB, true, 0, handleHip, stream);
        // benchmark_RPP_HOST_Flip_Batched(imgsRGB, true, 0, handleHostBatched);
        // benchmark_RPP_HIP_Flip_Batched(imgsRGB, true, 0, handleHipBatched, streamBatched);

        // benchmark_OpenCV_Flip(imgsRGB, true, -1);
        // benchmark_RPP_HOST_Flip(imgsRGB, true, -1, handleHost);
        // benchmark_RPP_HIP_Flip(imgsRGB, true, -1, handleHip, stream);
        // benchmark_RPP_HOST_Flip_Batched(imgsRGB, true, -1, handleHostBatched);
        // benchmark_RPP_HIP_Flip_Batched(imgsRGB, true, -1, handleHipBatched, streamBatched);

        benchmark_OpenCV_Rotate(imgsRGB, true, angleDeg);
        benchmark_RPP_HOST_Rotate(imgsRGB, true, angleDeg, handleHost);
        benchmark_RPP_HIP_Rotate(imgsRGB, true, angleDeg, handleHip, stream);
        benchmark_RPP_HOST_Rotate_Batched(imgsRGB, true, angleDeg, handleHostBatched);
        benchmark_RPP_HIP_Rotate_Batched(imgsRGB, true, angleDeg, handleHipBatched, streamBatched);

        benchmark_OpenCV_WarpAffine(imgsRGB, true);
        benchmark_RPP_HOST_WarpAffine(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_WarpAffine(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_WarpAffine_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_WarpAffine_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        benchmark_OpenCV_WarpPerspective(imgsRGB, true);
        benchmark_RPP_HOST_WarpPerspective(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_WarpPerspective(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_WarpPerspective_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_WarpPerspective_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        benchmark_OpenCV_Fisheye(imgsRGB, true);
        benchmark_RPP_HOST_Fisheye(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_Fisheye(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_Fisheye_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_Fisheye_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        benchmark_OpenCV_LensCorrection(imgsRGB, true);
        benchmark_RPP_HOST_LensCorrection(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_LensCorrection(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_LensCorrection_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_LensCorrection_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        cout << "\n--- Morphological Operations ---" << endl;
        benchmark_OpenCV_Erode(imgsRGB, true, morphKernel);
        benchmark_RPP_HOST_Erode(imgsRGB, true, morphKernel, handleHost);
        benchmark_RPP_HIP_Erode(imgsRGB, true, morphKernel, handleHip, stream);
        benchmark_RPP_HOST_Erode_Batched(imgsRGB, true, morphKernel, handleHostBatched);
        benchmark_RPP_HIP_Erode_Batched(imgsRGB, true, morphKernel, handleHipBatched, streamBatched);

        benchmark_OpenCV_Dilate(imgsRGB, true, morphKernel);
        benchmark_RPP_HOST_Dilate(imgsRGB, true, morphKernel, handleHost);
        benchmark_RPP_HIP_Dilate(imgsRGB, true, morphKernel, handleHip, stream);
        benchmark_RPP_HOST_Dilate_Batched(imgsRGB, true, morphKernel, handleHostBatched);
        benchmark_RPP_HIP_Dilate_Batched(imgsRGB, true, morphKernel, handleHipBatched, streamBatched);

        cout << "\n--- Arithmetic Operations ---" << endl;
        benchmark_OpenCV_AddScalar(imgsRGB, true, addVal);
        benchmark_RPP_HOST_AddScalar(imgsRGB, true, addVal, handleHost);
        benchmark_RPP_HIP_AddScalar(imgsRGB, true, addVal, handleHip, stream);
        benchmark_RPP_HOST_AddScalar_Batched(imgsRGB, true, addVal, handleHostBatched);
        benchmark_RPP_HIP_AddScalar_Batched(imgsRGB, true, addVal, handleHipBatched, streamBatched);

        benchmark_OpenCV_SubtractScalar(imgsRGB, true, subVal);
        benchmark_RPP_HOST_SubtractScalar(imgsRGB, true, subVal, handleHost);
        benchmark_RPP_HIP_SubtractScalar(imgsRGB, true, subVal, handleHip, stream);
        benchmark_RPP_HOST_SubtractScalar_Batched(imgsRGB, true, subVal, handleHostBatched);
        benchmark_RPP_HIP_SubtractScalar_Batched(imgsRGB, true, subVal, handleHipBatched, streamBatched);

        benchmark_OpenCV_MultiplyScalar(imgsRGB, true, mulVal);
        benchmark_RPP_HOST_MultiplyScalar(imgsRGB, true, mulVal, handleHost);
        benchmark_RPP_HIP_MultiplyScalar(imgsRGB, true, mulVal, handleHip, stream);
        benchmark_RPP_HOST_MultiplyScalar_Batched(imgsRGB, true, mulVal, handleHostBatched);
        benchmark_RPP_HIP_MultiplyScalar_Batched(imgsRGB, true, mulVal, handleHipBatched, streamBatched);

        benchmark_OpenCV_Blend(imgsRGB, true, blendAlpha);
        benchmark_RPP_HOST_Blend(imgsRGB, true, blendAlpha, handleHost);
        benchmark_RPP_HIP_Blend(imgsRGB, true, blendAlpha, handleHip, stream);
        benchmark_RPP_HOST_Blend_Batched(imgsRGB, true, blendAlpha, handleHostBatched);
        benchmark_RPP_HIP_Blend_Batched(imgsRGB, true, blendAlpha, handleHipBatched, streamBatched);

        cout << "\n--- Bitwise Operations ---" << endl;
        benchmark_OpenCV_BitwiseAnd(imgsRGB, true);
        benchmark_RPP_HOST_BitwiseAnd(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_BitwiseAnd(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_BitwiseAnd_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_BitwiseAnd_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        benchmark_OpenCV_BitwiseOr(imgsRGB, true);
        benchmark_RPP_HOST_BitwiseOr(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_BitwiseOr(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_BitwiseOr_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_BitwiseOr_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        benchmark_OpenCV_BitwiseNot(imgsRGB, true);
        benchmark_RPP_HOST_BitwiseNot(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_BitwiseNot(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_BitwiseNot_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_BitwiseNot_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        benchmark_OpenCV_BitwiseXor(imgsRGB, true);
        benchmark_RPP_HOST_BitwiseXor(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_BitwiseXor(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_BitwiseXor_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_BitwiseXor_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        cout << "\n--- Statistical Operations ---" << endl;
        benchmark_OpenCV_TensorMin(imgsRGB, true);
        benchmark_RPP_HOST_TensorMin(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_TensorMin(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_TensorMin_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_TensorMin_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        benchmark_OpenCV_TensorMax(imgsRGB, true);
        benchmark_RPP_HOST_TensorMax(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_TensorMax(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_TensorMax_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_TensorMax_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        benchmark_OpenCV_TensorSum(imgsRGB, true);
        benchmark_RPP_HOST_TensorSum(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_TensorSum(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_TensorSum_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_TensorSum_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        benchmark_OpenCV_TensorMean(imgsRGB, true);
        benchmark_RPP_HOST_TensorMean(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_TensorMean(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_TensorMean_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_TensorMean_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        benchmark_OpenCV_TensorStddev(imgsRGB, true);
        benchmark_RPP_HOST_TensorStddev(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_TensorStddev(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_TensorStddev_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_TensorStddev_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        benchmark_OpenCV_Threshold(imgsRGB, true, threshVal);
        benchmark_RPP_HOST_Threshold(imgsRGB, true, threshVal, handleHost);
        benchmark_RPP_HIP_Threshold(imgsRGB, true, threshVal, handleHip, stream);
        benchmark_RPP_HOST_Threshold_Batched(imgsRGB, true, threshVal, handleHostBatched);
        benchmark_RPP_HIP_Threshold_Batched(imgsRGB, true, threshVal, handleHipBatched, streamBatched);

        cout << "\n--- Effects Augmentations ---" << endl;
        benchmark_OpenCV_GaussianNoise(imgsRGB, true, noiseMean, noiseStd);
        benchmark_RPP_HOST_GaussianNoise(imgsRGB, true, noiseMean, noiseStd, handleHost);
        benchmark_RPP_HIP_GaussianNoise(imgsRGB, true, noiseMean, noiseStd, handleHip, stream);
        benchmark_RPP_HOST_GaussianNoise_Batched(imgsRGB, true, noiseMean, noiseStd, handleHostBatched);
        benchmark_RPP_HIP_GaussianNoise_Batched(imgsRGB, true, noiseMean, noiseStd, handleHipBatched, streamBatched);

        benchmark_OpenCV_SaltAndPepperNoise(imgsRGB, true, noiseProb);
        benchmark_RPP_HOST_SaltAndPepperNoise(imgsRGB, true, noiseProb, handleHost);
        benchmark_RPP_HIP_SaltAndPepperNoise(imgsRGB, true, noiseProb, handleHip, stream);
        benchmark_RPP_HOST_SaltAndPepperNoise_Batched(imgsRGB, true, noiseProb, handleHostBatched);
        benchmark_RPP_HIP_SaltAndPepperNoise_Batched(imgsRGB, true, noiseProb, handleHipBatched, streamBatched);

        benchmark_OpenCV_NoiseShot(imgsRGB, true, 0.2f);
        benchmark_RPP_HOST_NoiseShot(imgsRGB, true, 0.2f, handleHost);
        benchmark_RPP_HIP_NoiseShot(imgsRGB, true, 0.2f, handleHip, stream);
        benchmark_RPP_HOST_NoiseShot_Batched(imgsRGB, true, 0.2f, handleHostBatched);
        benchmark_RPP_HIP_NoiseShot_Batched(imgsRGB, true, 0.2f, handleHipBatched, streamBatched);

        benchmark_OpenCV_JpegCompressionDistortion(imgsRGB, true, 50);
        benchmark_RPP_HOST_JpegCompressionDistortion(imgsRGB, true, 50, handleHost);
        benchmark_RPP_HIP_JpegCompressionDistortion(imgsRGB, true, 50, handleHip, stream);
        benchmark_RPP_HOST_JpegCompressionDistortion_Batched(imgsRGB, true, 50, handleHostBatched);
        benchmark_RPP_HIP_JpegCompressionDistortion_Batched(imgsRGB, true, 50, handleHipBatched, streamBatched);

        benchmark_OpenCV_Posterize(imgsRGB, true, 4);
        benchmark_RPP_HOST_Posterize(imgsRGB, true, 4, handleHost);
        benchmark_RPP_HIP_Posterize(imgsRGB, true, 4, handleHip, stream);
        benchmark_RPP_HOST_Posterize_Batched(imgsRGB, true, 4, handleHostBatched);
        benchmark_RPP_HIP_Posterize_Batched(imgsRGB, true, 4, handleHipBatched, streamBatched);

        benchmark_OpenCV_Solarize(imgsRGB, true, 128);
        benchmark_RPP_HOST_Solarize(imgsRGB, true, 128, handleHost);
        benchmark_RPP_HIP_Solarize(imgsRGB, true, 128, handleHip, stream);
        benchmark_RPP_HOST_Solarize_Batched(imgsRGB, true, 128, handleHostBatched);
        benchmark_RPP_HIP_Solarize_Batched(imgsRGB, true, 128, handleHipBatched, streamBatched);

        benchmark_OpenCV_ColorCast(imgsRGB, true, 20.0f, 10.0f, -15.0f);
        benchmark_RPP_HOST_ColorCast(imgsRGB, true, 20.0f, 10.0f, -15.0f, handleHost);
        benchmark_RPP_HIP_ColorCast(imgsRGB, true, 20.0f, 10.0f, -15.0f, handleHip, stream);
        benchmark_RPP_HOST_ColorCast_Batched(imgsRGB, true, 20.0f, 10.0f, -15.0f, handleHostBatched);
        benchmark_RPP_HIP_ColorCast_Batched(imgsRGB, true, 20.0f, 10.0f, -15.0f, handleHipBatched, streamBatched);

        benchmark_OpenCV_ColorTemperature(imgsRGB, true, 40);
        benchmark_RPP_HOST_ColorTemperature(imgsRGB, true, 40, handleHost);
        benchmark_RPP_HIP_ColorTemperature(imgsRGB, true, 40, handleHip, stream);
        benchmark_RPP_HOST_ColorTemperature_Batched(imgsRGB, true, 40, handleHostBatched);
        benchmark_RPP_HIP_ColorTemperature_Batched(imgsRGB, true, 40, handleHipBatched, streamBatched);

        benchmark_OpenCV_ColorTwist(imgsRGB, true);
        benchmark_RPP_HOST_ColorTwist(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_ColorTwist(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_ColorTwist_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_ColorTwist_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        benchmark_OpenCV_Vignette(imgsRGB, true, 0.5f);
        benchmark_RPP_HOST_Vignette(imgsRGB, true, 0.5f, handleHost);
        benchmark_RPP_HIP_Vignette(imgsRGB, true, 0.5f, handleHip, stream);
        benchmark_RPP_HOST_Vignette_Batched(imgsRGB, true, 0.5f, handleHostBatched);
        benchmark_RPP_HIP_Vignette_Batched(imgsRGB, true, 0.5f, handleHipBatched, streamBatched);

        benchmark_OpenCV_NonLinearBlend(imgsRGB, true, 50.0f);
        benchmark_RPP_HOST_NonLinearBlend(imgsRGB, true, 50.0f, handleHost);
        benchmark_RPP_HIP_NonLinearBlend(imgsRGB, true, 50.0f, handleHip, stream);
        benchmark_RPP_HOST_NonLinearBlend_Batched(imgsRGB, true, 50.0f, handleHostBatched);
        benchmark_RPP_HIP_NonLinearBlend_Batched(imgsRGB, true, 50.0f, handleHipBatched, streamBatched);

        cout << "\n--- Dropout Augmentations ---" << endl;
        benchmark_OpenCV_Erase(imgsRGB, true, 4);
        benchmark_RPP_HOST_Erase(imgsRGB, true, 4, handleHost);
        benchmark_RPP_HIP_Erase(imgsRGB, true, 4, handleHip, stream);
        benchmark_RPP_HOST_Erase_Batched(imgsRGB, true, 4, handleHostBatched);
        benchmark_RPP_HIP_Erase_Batched(imgsRGB, true, 4, handleHipBatched, streamBatched);

        benchmark_OpenCV_RandomErase(imgsRGB, true, 4);
        benchmark_RPP_HOST_RandomErase(imgsRGB, true, 4, handleHost);
        benchmark_RPP_HIP_RandomErase(imgsRGB, true, 4, handleHip, stream);
        benchmark_RPP_HOST_RandomErase_Batched(imgsRGB, true, 4, handleHostBatched);
        benchmark_RPP_HIP_RandomErase_Batched(imgsRGB, true, 4, handleHipBatched, streamBatched);

        benchmark_OpenCV_CoarseDropout(imgsRGB, true, 8);
        benchmark_RPP_HOST_CoarseDropout(imgsRGB, true, 8, handleHost);
        benchmark_RPP_HIP_CoarseDropout(imgsRGB, true, 8, handleHip, stream);
        benchmark_RPP_HOST_CoarseDropout_Batched(imgsRGB, true, 8, handleHostBatched);
        benchmark_RPP_HIP_CoarseDropout_Batched(imgsRGB, true, 8, handleHipBatched, streamBatched);

        benchmark_OpenCV_GridDropout(imgsRGB, true, 10, 10);
        benchmark_RPP_HOST_GridDropout(imgsRGB, true, 10, 10, handleHost);
        benchmark_RPP_HIP_GridDropout(imgsRGB, true, 10, 10, handleHip, stream);
        benchmark_RPP_HOST_GridDropout_Batched(imgsRGB, true, 10, 10, handleHostBatched);
        benchmark_RPP_HIP_GridDropout_Batched(imgsRGB, true, 10, 10, handleHipBatched, streamBatched);

        benchmark_OpenCV_Gridmask(imgsRGB, true, 96, 0.6f);
        benchmark_RPP_HOST_Gridmask(imgsRGB, true, 96, 0.6f, handleHost);
        benchmark_RPP_HIP_Gridmask(imgsRGB, true, 96, 0.6f, handleHip, stream);
        benchmark_RPP_HOST_Gridmask_Batched(imgsRGB, true, 96, 0.6f, handleHostBatched);
        benchmark_RPP_HIP_Gridmask_Batched(imgsRGB, true, 96, 0.6f, handleHipBatched, streamBatched);

        benchmark_OpenCV_ChannelDropout(imgsRGB, true, 0.4f);
        benchmark_RPP_HOST_ChannelDropout(imgsRGB, true, 0.4f, handleHost);
        benchmark_RPP_HIP_ChannelDropout(imgsRGB, true, 0.4f, handleHip, stream);
        benchmark_RPP_HOST_ChannelDropout_Batched(imgsRGB, true, 0.4f, handleHostBatched);
        benchmark_RPP_HIP_ChannelDropout_Batched(imgsRGB, true, 0.4f, handleHipBatched, streamBatched);

        benchmark_OpenCV_CutoutDropout(imgsRGB, true, 4);
        benchmark_RPP_HOST_CutoutDropout(imgsRGB, true, 4, handleHost);
        benchmark_RPP_HIP_CutoutDropout(imgsRGB, true, 4, handleHip, stream);
        benchmark_RPP_HOST_CutoutDropout_Batched(imgsRGB, true, 4, handleHostBatched);
        benchmark_RPP_HIP_CutoutDropout_Batched(imgsRGB, true, 4, handleHipBatched, streamBatched);

        cout << "\n--- Data Operations ---" << endl;
        benchmark_OpenCV_Copy(imgsRGB, true);
        benchmark_RPP_HOST_Copy(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_Copy(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_Copy_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_Copy_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        benchmark_OpenCV_Slice(imgsRGB, true);
        benchmark_RPP_HOST_Slice(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_Slice(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_Slice_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_Slice_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        benchmark_OpenCV_ChannelPermute(imgsRGB, true);
        benchmark_RPP_HOST_ChannelPermute(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_ChannelPermute(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_ChannelPermute_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_ChannelPermute_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        benchmark_OpenCV_Transpose(imgsRGB, true);
        benchmark_RPP_HOST_Transpose(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_Transpose(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_Transpose_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_Transpose_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        cout << "\n--- Advanced Operations ---" << endl;
        benchmark_OpenCV_LUT(imgsRGB, true);
        benchmark_RPP_HOST_LUT(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_LUT(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_LUT_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_LUT_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        benchmark_OpenCV_Magnitude(imgsRGB, true);
        benchmark_RPP_HOST_Magnitude(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_Magnitude(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_Magnitude_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_Magnitude_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        benchmark_OpenCV_Phase(imgsRGB, true);
        benchmark_RPP_HOST_Phase(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_Phase(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_Phase_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_Phase_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        benchmark_OpenCV_Normalize(imgsRGB, true);
        benchmark_RPP_HOST_Normalize(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_Normalize(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_Normalize_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_Normalize_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        benchmark_OpenCV_FusedMultiplyAddScalar(imgsRGB, true, 1.2f, 10.0f);
        benchmark_RPP_HOST_FusedMultiplyAddScalar(imgsRGB, true, 1.2f, 10.0f, handleHost);
        benchmark_RPP_HIP_FusedMultiplyAddScalar(imgsRGB, true, 1.2f, 10.0f, handleHip, stream);
        benchmark_RPP_HOST_FusedMultiplyAddScalar_Batched(imgsRGB, true, 1.2f, 10.0f, handleHostBatched);
        benchmark_RPP_HIP_FusedMultiplyAddScalar_Batched(imgsRGB, true, 1.2f, 10.0f, handleHipBatched, streamBatched);

        benchmark_OpenCV_Remap(imgsRGB, true);
        benchmark_RPP_HOST_Remap(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_Remap(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_Remap_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_Remap_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        cout << "\n--- Composite Operations ---" << endl;
        benchmark_OpenCV_CropAndPatch(imgsRGB, true);
        benchmark_RPP_HOST_CropAndPatch(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_CropAndPatch(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_CropAndPatch_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_CropAndPatch_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        benchmark_OpenCV_CropMirrorNormalize(imgsRGB, true);
        benchmark_RPP_HOST_CropMirrorNormalize(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_CropMirrorNormalize(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_CropMirrorNormalize_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_CropMirrorNormalize_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        benchmark_OpenCV_ResizeMirrorNormalize(imgsRGB, true);
        benchmark_RPP_HOST_ResizeMirrorNormalize(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_ResizeMirrorNormalize(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_ResizeMirrorNormalize_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_ResizeMirrorNormalize_Batched(imgsRGB, true, handleHipBatched, streamBatched);

        benchmark_OpenCV_ResizeCropMirror(imgsRGB, true);
        benchmark_RPP_HOST_ResizeCropMirror(imgsRGB, true, handleHost);
        benchmark_RPP_HIP_ResizeCropMirror(imgsRGB, true, handleHip, stream);
        benchmark_RPP_HOST_ResizeCropMirror_Batched(imgsRGB, true, handleHostBatched);
        benchmark_RPP_HIP_ResizeCropMirror_Batched(imgsRGB, true, handleHipBatched, streamBatched);
    }

    cout << "\n========================================" << endl;
    cout << "BENCHMARKING COMPLETE" << endl;
    cout << "OpenCV vs RPP HOST Backend Comparison" << endl;
    cout << "========================================\n" << endl;

    // Export results to Excel
    ostringstream excelFilenameStream;
    excelFilenameStream << "opencv_vs_rpp_benchmark_results_" << getGPUArchitectureName() << ".xlsx";
    string excelFilename = excelFilenameStream.str();

    cout << "Exporting results to: " << excelFilename << endl;
    bool exportSuccess = writeResultsToExcel(excelFilename, grayscaleResults, rgbResults, maxAvailableThreads);
    if (!exportSuccess) {
        cerr
            << "\nWarning: Benchmark completed successfully, but failed to export results to Excel."
            << endl;
        cerr << "         Benchmark data is still available in memory but not saved to disk."
             << endl;
    } else {
        cout << "Results successfully exported to: " << excelFilename << endl;
    }

    // Cleanup RPP handles
    rppDestroy(handleHost, RPP_HOST_BACKEND);
    rppDestroy(handleHip, RPP_HIP_BACKEND);
    (void)hipStreamDestroy(stream);

    // Cleanup Batched RPP handles
    rppDestroy(handleHostBatched, RPP_HOST_BACKEND);
    rppDestroy(handleHipBatched, RPP_HIP_BACKEND);
    (void)hipStreamDestroy(streamBatched);

    return exportSuccess ? 0 : 1;
}
