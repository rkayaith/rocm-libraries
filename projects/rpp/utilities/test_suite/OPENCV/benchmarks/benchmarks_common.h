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

#ifndef BENCHMARKS_COMMON_H
#define BENCHMARKS_COMMON_H

#include <dirent.h>
#include <glob.h>
#include <omp.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/utsname.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <unordered_set>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include <hip/hip_runtime.h>
#include <amd_smi/amdsmi.h>
#include "rpp.h"
#include "rpp_version.h"
#include "xlsxwriter.h"

using namespace std;
using namespace cv;
using namespace chrono;

// Device type for energy measurement
enum class DeviceType {
    CPU,    // For OpenCV, RPP HOST, RPP HOST BATCH
    GPU     // For RPP HIP, RPP HIP BATCH
};

// Performance monitoring class with energy measurement
class PerformanceMonitor {
private:
    // Time measurement
    high_resolution_clock::time_point startTime;
    high_resolution_clock::time_point endTime;
    double totalTimeMs;

    // Energy measurement
    DeviceType device;

    double totalEnergyJ;         // Total energy in Joules

    // CPU-specific cached paths
    string cpuEnergyPath;   // Cached path to CPU energy file
    uint64_t startEnergyUj;      // Energy in microjoules at start
    uint64_t endEnergyUj;        // Energy in microjoules at end
    uint64_t cpuMaxEnergyRangeUj; // Energy counter max ranges
    
    // GPU-specific AMD SMI and sysfs
    int gpuDeviceId;
    amdsmi_socket_handle socketHandle;
    amdsmi_processor_handle processorHandle;
    bool amdSmiInitialized;
    bool gpuEnergySupported;     // True if energy counters work, false if need power fallback
    uint64_t startEnergyGpuUj;   // GPU energy in microjoules at start
    uint64_t endEnergyGpuUj;     // GPU energy in microjoules at end
    float gpuEnergyResolution;   // GPU energy counter resolution in μJ
    uint64_t startPowerUw;       // GPU power in microwatts (fallback)
    uint64_t endPowerUw;         // GPU power in microwatts (fallback)
    string gpuPowerPath;         // sysfs path for power (final fallback)

    // Helper methods - implemented in benchmarks_utils.cpp
    void initializeCPU();
    void initializeGPU(int deviceId);
    void shutdownGPU();

    uint64_t readCpuEnergy();
    uint64_t readGpuEnergy();
    uint64_t readGpuPower();     // Fallback for GPUs without energy counters

    uint64_t calculateDelta(uint64_t start, uint64_t end, uint64_t maxRange);

public:
    PerformanceMonitor();
    ~PerformanceMonitor();

    // Updated methods
    void start(DeviceType type = DeviceType::CPU, int deviceId = 0);
    void stop();

    // Getters
    double getTotalTime() const { return totalTimeMs; }
    double getTotalEnergy() const { return totalEnergyJ; }

};

// Global performance monitor instance
extern PerformanceMonitor perfMonitor;

// Test image paths - can be overridden via command line
#define DEFAULT_GRAY_IMAGE_PATH "input_images_dataset/"
#define DEFAULT_RGB_IMAGE_PATH "input_images_dataset/"

// Global configuration variables (set at runtime)
extern int HOST_PERF_RUNS;
extern int HIP_PERF_RUNS;
extern int WARMUP_RUNS;
extern int HOST_TOTAL_RUNS;
extern int HIP_TOTAL_RUNS;
extern int NUM_THREADS;
extern string GRAY_IMAGE_PATH;
extern string RGB_IMAGE_PATH;

// Structure to store benchmark results
struct BenchmarkResult {
    string operationName;
    string parameters;
    string imageSize;
    string dtype;
    int batchSize;
    int hostPerfRuns;
    int hipPerfRuns;

    // Time measurements
    double opencvTime;
    double rppHostTime;
    double rppHipTime;
    double rppHostBatchTime;
    double rppHipBatchTime;

    // Energy measurements (in Joules)
    double opencvEnergy;
    double rppHostEnergy;
    double rppHipEnergy;
    double rppHostBatchEnergy;
    double rppHipBatchEnergy;

    BenchmarkResult(const string& name, const string& params, double cvTime, double rHostTime,
                    double rHipTime, const string& imgSize = "", const string& dataType = "",
                    int batch = 0, int hostRuns = 0, int hipRuns = 0, double rHostBatchTime = 0.0,
                    double rHipBatchTime = 0.0, double cvEnergy = 0.0, double rHostEnergy = 0.0,
                    double rHipEnergy = 0.0, double rHostBatchEnergy = 0.0, double rHipBatchEnergy = 0.0)
        : operationName(name),
          parameters(params),
          imageSize(imgSize),
          dtype(dataType),
          batchSize(batch),
          hostPerfRuns(hostRuns),
          hipPerfRuns(hipRuns),
          opencvTime(cvTime),
          rppHostTime(rHostTime),
          rppHipTime(rHipTime),
          rppHostBatchTime(rHostBatchTime),
          rppHipBatchTime(rHipBatchTime),
          opencvEnergy(cvEnergy),
          rppHostEnergy(rHostEnergy),
          rppHipEnergy(rHipEnergy),
          rppHostBatchEnergy(rHostBatchEnergy),
          rppHipBatchEnergy(rHipBatchEnergy) {
    }
};

// Global vectors to store results
extern vector<BenchmarkResult> grayscaleResults;
extern vector<BenchmarkResult> rgbResults;

// Global variables for image metadata
extern string grayImageSize;
extern string grayImageDtype;
extern string rgbImageSize;
extern string rgbImageDtype;
extern int grayBatchSize;
extern int rgbBatchSize;

// Utility functions
vector<Mat> loadBatchImages(const string& directory, int& batchSize, int& maxWidth, int& maxHeight,
                            bool isColor);
void printResult(const string& name, int batchSize, bool isColor, double totalMs,
                 double totalEnergy = 0.0, const string& params = "");
string getCPUInfo();
string getMemoryInfo();
string getOSInfo();
string getGPUInfo();
string getGPUArchitecture();
string getGPUArchitectureName();
string getRPPVersion();
string getROCmVersion();
string getCurrentDateTime();
string getDtypeString(int cvType);
bool writeResultsToExcel(const string& filename, const vector<BenchmarkResult>& grayResults,
                         const vector<BenchmarkResult>& colorResults, int maxAvailableThreads);

// RPP utility functions
RpptDesc createRppDescriptor(const Mat& img, RpptLayout layout = RpptLayout::NHWC);
RpptROI createFullImageROI(const Mat& img);
RpptGenericDesc toGenericDesc(const RpptDesc& desc);
RpptROI3D createFullImageROI3D(const Mat& img);

// Helper functions from rpp_test_suite_image.h
void set_descriptor_dims_and_strides(RpptDesc* descPtr, int noOfImages, int maxHeight,
                                    int maxWidth, int numChannels, int offsetInBytes,
                                    int additionalStride = 0);
void generate_channel_dropout_mask(Rpp8u* dropoutTensor, Rpp32f* dropoutProbability, int batchSize,
                                   int channels, int seed);
void init_cutout_dropout(int batchSize, int maxBoxesPerImage, Rpp32u* numOfBoxes,
                        RpptRoiLtrb* anchorBoxInfoTensor, RpptROI* roiTensorPtrSrc,
                        int channels, int BitDepthTestMode, int seed, int dropoutType,
                        void* colorBuffer = NULL);
void set_generic_descriptor(RpptGenericDescPtr descriptorPtr3D, int noOfImages, int maxX,
                           int maxY, int maxZ, int numChannels, int offsetInBytes,
                           int layoutType);
void init_remap(RpptDescPtr tableDescPtr, RpptDescPtr srcDescPtr, RpptROIPtr roiTensorPtrSrc,
               Rpp32f* rowRemapTable, Rpp32f* colRemapTable);

// Helper initialization functions for dropout operators
void init_grid_dropout_boxes(int batchCount, RpptRoiLtrb* anchorBoxInfoTensor,
                             RpptROI* roiTensorPtrSrc, Rpp32u gridH, Rpp32u gridW, Rpp32u& maxHoleW,
                             Rpp32u& maxHoleH, Rpp32f holeRatio, int seed);
void init_ricap_boxes(int maxWidth, int maxHeight, int batchSize, Rpp32u* permutationTensor,
                      RpptROI* roiPtrInputCropRegion);

// RPP Benchmark function declarations
void benchmark_RPP_HOST_Brightness(const vector<Mat>& imgs, bool isColor, float alpha, float beta,
                              rppHandle_t handle);
void benchmark_RPP_HOST_GammaCorrection(const vector<Mat>& imgs, bool isColor, float gamma,
                                   rppHandle_t handle);
void benchmark_RPP_HOST_Blend(const vector<Mat>& imgs, bool isColor, float alpha, rppHandle_t handle);
void benchmark_RPP_HOST_Contrast(const vector<Mat>& imgs, bool isColor, float contrastFactor,
                            float contrastCenter, rppHandle_t handle);
void benchmark_RPP_HOST_Exposure(const vector<Mat>& imgs, bool isColor, float exposureFactor,
                            rppHandle_t handle);
void benchmark_RPP_HOST_Hue(const vector<Mat>& imgs, float hueDelta, rppHandle_t handle);
void benchmark_RPP_HOST_Saturation(const vector<Mat>& imgs, float satFactor, rppHandle_t handle);
void benchmark_RPP_HOST_ColorToGreyscale(const vector<Mat>& imgs, rppHandle_t handle);
void benchmark_RPP_HOST_ColorJitter(const vector<Mat>& imgs, float brightness, float contrast,
                               float saturation, float hue, rppHandle_t handle);
void benchmark_RPP_HOST_BoxFilter(const vector<Mat>& imgs, bool isColor, int kernelSize,
                             rppHandle_t handle);
void benchmark_RPP_HOST_MedianFilter(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                rppHandle_t handle);
void benchmark_RPP_HOST_GaussianFilter(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                  double sigma, rppHandle_t handle);
void benchmark_RPP_HOST_SobelFilter(const vector<Mat>& imgs, bool isColor, int sobelType,
                               rppHandle_t handle);
void benchmark_RPP_HOST_Emboss(const vector<Mat>& imgs, bool isColor, int kernelSize, float strength,
                          rppHandle_t handle);
void benchmark_RPP_HOST_Crop(const vector<Mat>& imgs, bool isColor, int cropWidth, int cropHeight,
                        rppHandle_t handle);
void benchmark_RPP_HOST_Resize(const vector<Mat>& imgs, bool isColor, int dstW, int dstH,
                          RpptInterpolationType interpType, const string& interpName,
                          rppHandle_t handle);
void benchmark_RPP_HOST_Flip(const vector<Mat>& imgs, bool isColor, int flipCode, rppHandle_t handle);
void benchmark_RPP_HOST_Rotate(const vector<Mat>& imgs, bool isColor, float angleDeg,
                          rppHandle_t handle);
void benchmark_RPP_HOST_WarpAffine(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_Fisheye(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_LensCorrection(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_Erode(const vector<Mat>& imgs, bool isColor, int kernelSize, rppHandle_t handle);
void benchmark_RPP_HOST_Dilate(const vector<Mat>& imgs, bool isColor, int kernelSize,
                          rppHandle_t handle);
void benchmark_RPP_HOST_AddScalar(const vector<Mat>& imgs, bool isColor, float addVal,
                             rppHandle_t handle);
void benchmark_RPP_HOST_SubtractScalar(const vector<Mat>& imgs, bool isColor, float subVal,
                                  rppHandle_t handle);
void benchmark_RPP_HOST_MultiplyScalar(const vector<Mat>& imgs, bool isColor, float mulVal,
                                  rppHandle_t handle);
void benchmark_RPP_HOST_BitwiseAnd(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_BitwiseOr(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_BitwiseNot(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_Threshold(const vector<Mat>& imgs, bool isColor, double thresh,
                             rppHandle_t handle);
void benchmark_RPP_HOST_BitwiseXor(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_HistogramEqualize(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_LUT(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_Magnitude(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_Phase(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_WarpPerspective(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_Remap(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_Normalize(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_FusedMultiplyAddScalar(const vector<Mat>& imgs, bool isColor, Rpp32f mul,
                                          Rpp32f add, rppHandle_t handle);
void benchmark_RPP_HOST_Emboss(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_TensorMin(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_TensorMax(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_TensorSum(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_TensorMean(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_TensorStddev(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_GaussianNoise(const vector<Mat>& imgs, bool isColor, float mean, float stddev,
                                 rppHandle_t handle);
void benchmark_RPP_HOST_SaltAndPepperNoise(const vector<Mat>& imgs, bool isColor, float noiseProb,
                                      rppHandle_t handle);
void benchmark_RPP_HOST_Copy(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_Posterize(const vector<Mat>& imgs, bool isColor, Rpp32u bits,
                             rppHandle_t handle);
void benchmark_RPP_HOST_Solarize(const vector<Mat>& imgs, bool isColor, Rpp8u threshold,
                            rppHandle_t handle);
void benchmark_RPP_HOST_NoiseShot(const vector<Mat>& imgs, bool isColor, float shotNoiseFactor,
                             rppHandle_t handle);
void benchmark_RPP_HOST_Gridmask(const vector<Mat>& imgs, bool isColor, Rpp32u tileWidth,
                            Rpp32f gridRatio, rppHandle_t handle);
void benchmark_RPP_HOST_ColorCast(const vector<Mat>& imgs, bool isColor, Rpp32f rShift, Rpp32f gShift,
                             Rpp32f bShift, rppHandle_t handle);
void benchmark_RPP_HOST_ColorTemperature(const vector<Mat>& imgs, bool isColor, Rpp32s adjustmentValue,
                                    rppHandle_t handle);
void benchmark_RPP_HOST_Vignette(const vector<Mat>& imgs, bool isColor, Rpp32f vignetteIntensity,
                            rppHandle_t handle);
void benchmark_RPP_HOST_NonLinearBlend(const vector<Mat>& imgs, bool isColor, Rpp32f stdDev,
                                  rppHandle_t handle);
void benchmark_RPP_HOST_Erase(const vector<Mat>& imgs, bool isColor, Rpp32u numBoxes,
                         rppHandle_t handle);
void benchmark_RPP_HOST_CoarseDropout(const vector<Mat>& imgs, bool isColor, Rpp32u numDropouts,
                                 rppHandle_t handle);
void benchmark_RPP_HOST_GridDropout(const vector<Mat>& imgs, bool isColor, Rpp32u numGridsPerRow,
                               Rpp32u numGridsPerColumn, rppHandle_t handle);
void benchmark_RPP_HOST_RandomErase(const vector<Mat>& imgs, bool isColor, int numBoxes, rppHandle_t handle);
void benchmark_RPP_HOST_ColorTwist(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_CropAndPatch(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_CropMirrorNormalize(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_ResizeMirrorNormalize(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_ResizeCropMirror(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_RICAP(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_ChannelDropout(const vector<Mat>& imgs, bool isColor, float dropoutProb,
                                  rppHandle_t handle);
void benchmark_RPP_HOST_CutoutDropout(const vector<Mat>& imgs, bool isColor, Rpp32u numBoxes,
                                 rppHandle_t handle);
void benchmark_RPP_HOST_JpegCompressionDistortion(const vector<Mat>& imgs, bool isColor, Rpp32s quality,
                                             rppHandle_t handle);
void benchmark_RPP_HOST_ChannelPermute(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_Slice(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_Transpose(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);

// RPP HIP Benchmark function declarations
void benchmark_RPP_HIP_Brightness(const vector<Mat>& imgs, bool isColor, float alpha, float beta,
                             rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_GammaCorrection(const vector<Mat>& imgs, bool isColor, float gamma,
                                  rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Blend(const vector<Mat>& imgs, bool isColor, float alpha,
                        rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Contrast(const vector<Mat>& imgs, bool isColor, float contrastFactor,
                           float contrastCenter, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Exposure(const vector<Mat>& imgs, bool isColor, float exposureFactor,
                           rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Hue(const vector<Mat>& imgs, float hueDelta, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Saturation(const vector<Mat>& imgs, float satFactor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_ColorToGreyscale(const vector<Mat>& imgs, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_ColorJitter(const vector<Mat>& imgs, float brightness, float contrast,
                              float saturation, float hue, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_BoxFilter(const vector<Mat>& imgs, bool isColor, int kernelSize,
                            rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_MedianFilter(const vector<Mat>& imgs, bool isColor, int kernelSize,
                               rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_GaussianFilter(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                 double sigma, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_SobelFilter(const vector<Mat>& imgs, bool isColor, int sobelType,
                              rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Erode(const vector<Mat>& imgs, bool isColor, int kernelSize,
                        rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Dilate(const vector<Mat>& imgs, bool isColor, int kernelSize,
                         rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Emboss(const vector<Mat>& imgs, bool isColor, int kernelSize, float strength,
                         rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Crop(const vector<Mat>& imgs, bool isColor, int cropWidth, int cropHeight,
                       rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Resize(const vector<Mat>& imgs, bool isColor, int dstW, int dstH,
                         RpptInterpolationType interpType, const string& interpName,
                         rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Flip(const vector<Mat>& imgs, bool isColor, int flipCode, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Rotate(const vector<Mat>& imgs, bool isColor, float angleDeg,
                         rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_WarpAffine(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_WarpPerspective(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Fisheye(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_LensCorrection(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Erode(const vector<Mat>& imgs, bool isColor, int kernelSize, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Dilate(const vector<Mat>& imgs, bool isColor, int kernelSize,
                         rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_AddScalar(const vector<Mat>& imgs, bool isColor, float addVal,
                            rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_SubtractScalar(const vector<Mat>& imgs, bool isColor, float subVal,
                                 rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_MultiplyScalar(const vector<Mat>& imgs, bool isColor, float mulVal,
                                 rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_BitwiseAnd(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_BitwiseOr(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_BitwiseNot(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_BitwiseXor(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Threshold(const vector<Mat>& imgs, bool isColor, double thresh,
                            rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_HistogramEqualize(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_LUT(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Magnitude(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Phase(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Remap(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_TensorMin(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_TensorMax(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_TensorSum(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_TensorMean(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_TensorStddev(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_GaussianNoise(const vector<Mat>& imgs, bool isColor, float mean, float stdDev,
                                rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_SaltAndPepperNoise(const vector<Mat>& imgs, bool isColor, float prob,
                                     rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_NoiseShot(const vector<Mat>& imgs, bool isColor, float shotNoiseParam,
                            rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Posterize(const vector<Mat>& imgs, bool isColor, int bits,
                            rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Solarize(const vector<Mat>& imgs, bool isColor, int threshold,
                           rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_ColorCast(const vector<Mat>& imgs, bool isColor, float r, float g, float b,
                            rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_ColorTemperature(const vector<Mat>& imgs, bool isColor, int adjustment,
                                   rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_ColorTwist(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Vignette(const vector<Mat>& imgs, bool isColor, float strength,
                           rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_NonLinearBlend(const vector<Mat>& imgs, bool isColor, float strength,
                                 rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Erase(const vector<Mat>& imgs, bool isColor, int numBoxes,
                        rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_RandomErase(const vector<Mat>& imgs, bool isColor, int numBoxes, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_CoarseDropout(const vector<Mat>& imgs, bool isColor, Rpp32u maxBoxesPerImage,
                                rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_GridDropout(const vector<Mat>& imgs, bool isColor, Rpp32u numGridsPerRow, Rpp32u numGridsPerColumn,
                              rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Gridmask(const vector<Mat>& imgs, bool isColor, int tileWidth, float ratio,
                           rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_ChannelDropout(const vector<Mat>& imgs, bool isColor, float dropoutProb,
                                 rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_CutoutDropout(const vector<Mat>& imgs, bool isColor, Rpp32u numBoxes,
                                rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_JpegCompressionDistortion(const vector<Mat>& imgs, bool isColor, Rpp32s quality,
                                            rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Copy(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_ChannelPermute(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Slice(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Transpose(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Normalize(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_FusedMultiplyAddScalar(const vector<Mat>& imgs, bool isColor, float mulVal, float addVal,
                                         rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_CropAndPatch(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_CropMirrorNormalize(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_ResizeMirrorNormalize(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_ResizeCropMirror(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);

// OpenCV Benchmark function declarations
void benchmark_OpenCV_Brightness(const vector<Mat>& imgs, bool isColor, float alpha, float beta);
void benchmark_OpenCV_GammaCorrection(const vector<Mat>& imgs, bool isColor, float gamma);
void benchmark_OpenCV_Blend(const vector<Mat>& imgs, bool isColor, float alpha);
void benchmark_OpenCV_Contrast(const vector<Mat>& imgs, bool isColor, float contrastFactor,
                               float contrastCenter);
void benchmark_OpenCV_Exposure(const vector<Mat>& imgs, bool isColor, float stop);
void benchmark_OpenCV_Hue(const vector<Mat>& imgs, float hueDelta);
void benchmark_OpenCV_Saturation(const vector<Mat>& imgs, float satFactor);
void benchmark_OpenCV_ColorToGreyscale(const vector<Mat>& imgs);
void benchmark_OpenCV_ColorJitter(const vector<Mat>& imgs, float brightness, float contrast,
                                  float saturation, float hue);
void benchmark_OpenCV_BoxFilter(const vector<Mat>& imgs, bool isColor, int kernelSize);
void benchmark_OpenCV_MedianFilter(const vector<Mat>& imgs, bool isColor, int kernelSize);
void benchmark_OpenCV_GaussianFilter(const vector<Mat>& imgs, bool isColor, int kernelSize,
                                     double sigma);
void benchmark_OpenCV_SobelFilter(const vector<Mat>& imgs, bool isColor, int sobelType);
void benchmark_OpenCV_Emboss(const vector<Mat>& imgs, bool isColor, int kernelSize, float strength);
void benchmark_OpenCV_Crop(const vector<Mat>& imgs, bool isColor, int cropWidth, int cropHeight);
void benchmark_OpenCV_Resize(const vector<Mat>& imgs, bool isColor, int dstW, int dstH,
                             int interpType, const string& interpName);
void benchmark_OpenCV_Flip(const vector<Mat>& imgs, bool isColor, int flipCode);
void benchmark_OpenCV_Rotate(const vector<Mat>& imgs, bool isColor, float angleDeg);
void benchmark_OpenCV_WarpAffine(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_Fisheye(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_LensCorrection(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_Erode(const vector<Mat>& imgs, bool isColor, int kernelSize);
void benchmark_OpenCV_Dilate(const vector<Mat>& imgs, bool isColor, int kernelSize);
void benchmark_OpenCV_AddScalar(const vector<Mat>& imgs, bool isColor, float addVal);
void benchmark_OpenCV_SubtractScalar(const vector<Mat>& imgs, bool isColor, float subVal);
void benchmark_OpenCV_MultiplyScalar(const vector<Mat>& imgs, bool isColor, float mulVal);
void benchmark_OpenCV_BitwiseAnd(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_BitwiseOr(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_BitwiseNot(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_Threshold(const vector<Mat>& imgs, bool isColor, double thresh);
void benchmark_OpenCV_Normalize(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_FusedMultiplyAddScalar(const vector<Mat>& imgs, bool isColor, float mul,
                                             float add);
void benchmark_OpenCV_Transpose(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_BitwiseXor(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_HistogramEqualize(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_LUT(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_Magnitude(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_Phase(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_WarpPerspective(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_Remap(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_Emboss(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_TensorMin(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_TensorMax(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_TensorSum(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_TensorMean(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_TensorStddev(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_GaussianNoise(const vector<Mat>& imgs, bool isColor, float mean,
                                    float stddev);
void benchmark_OpenCV_SaltAndPepperNoise(const vector<Mat>& imgs, bool isColor, float noiseProb);
void benchmark_OpenCV_Copy(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_Posterize(const vector<Mat>& imgs, bool isColor, Rpp32u bits);
void benchmark_OpenCV_Solarize(const vector<Mat>& imgs, bool isColor, Rpp8u threshold);
void benchmark_OpenCV_NoiseShot(const vector<Mat>& imgs, bool isColor, float shotNoiseFactor);
void benchmark_OpenCV_Gridmask(const vector<Mat>& imgs, bool isColor, Rpp32u tileWidth,
                               Rpp32f gridRatio);
void benchmark_OpenCV_ColorCast(const vector<Mat>& imgs, bool isColor, Rpp32f rShift, Rpp32f gShift,
                                Rpp32f bShift);
void benchmark_OpenCV_ColorTemperature(const vector<Mat>& imgs, bool isColor,
                                       Rpp32s adjustmentValue);
void benchmark_OpenCV_Vignette(const vector<Mat>& imgs, bool isColor, Rpp32f vignetteIntensity);
void benchmark_OpenCV_NonLinearBlend(const vector<Mat>& imgs, bool isColor, Rpp32f stdDev);
void benchmark_OpenCV_Erase(const vector<Mat>& imgs, bool isColor, Rpp32u numBoxes);
void benchmark_OpenCV_CoarseDropout(const vector<Mat>& imgs, bool isColor, Rpp32u numDropouts);
void benchmark_OpenCV_GridDropout(const vector<Mat>& imgs, bool isColor, Rpp32u numGridsPerRow,
                                  Rpp32u numGridsPerColumn);
void benchmark_OpenCV_RandomErase(const vector<Mat>& imgs, bool isColor, int numBoxes);
void benchmark_OpenCV_ColorTwist(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_CropAndPatch(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_CropMirrorNormalize(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_ResizeMirrorNormalize(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_ResizeCropMirror(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_RICAP(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_ChannelDropout(const vector<Mat>& imgs, bool isColor, float dropoutProb);
void benchmark_OpenCV_CutoutDropout(const vector<Mat>& imgs, bool isColor, Rpp32u numBoxes);
void benchmark_OpenCV_JpegCompressionDistortion(const vector<Mat>& imgs, bool isColor,
                                                Rpp32s quality);
void benchmark_OpenCV_ChannelPermute(const vector<Mat>& imgs, bool isColor);
void benchmark_OpenCV_Slice(const vector<Mat>& imgs, bool isColor);

// ============================================================================
// BATCHED PROCESSING BENCHMARK FUNCTION DECLARATIONS
// ============================================================================
// HIP Batched Operations
void benchmark_RPP_HIP_Flip_Batched(const vector<Mat>& imgs, bool isColor, int flipCode, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Resize_Batched(const vector<Mat>& imgs, bool isColor, int dstW, int dstH, RpptInterpolationType interpType, const string& interpName, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Crop_Batched(const vector<Mat>& imgs, bool isColor, int cropW, int cropH, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Rotate_Batched(const vector<Mat>& imgs, bool isColor, float angleDeg, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_BoxFilter_Batched(const vector<Mat>& imgs, bool isColor, int kernelSize, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_GaussianFilter_Batched(const vector<Mat>& imgs, bool isColor, int kernelSize, float sigma, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_MedianFilter_Batched(const vector<Mat>& imgs, bool isColor, int kernelSize, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_SobelFilter_Batched(const vector<Mat>& imgs, bool isColor, int sobelType, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_HistogramEqualize_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Hue_Batched(const vector<Mat>& imgs, bool isColor, float hueFactor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Saturation_Batched(const vector<Mat>& imgs, bool isColor, float satFactor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_ColorToGreyscale_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Brightness_Batched(const vector<Mat>& imgs, bool isColor, float alpha, float beta, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Contrast_Batched(const vector<Mat>& imgs, bool isColor, float contrastFactor, float contrastCenter, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Emboss_Batched(const vector<Mat>& imgs, bool isColor, int kernelSize, float strength, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_AddScalar_Batched(const vector<Mat>& imgs, bool isColor, float addVal, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_SubtractScalar_Batched(const vector<Mat>& imgs, bool isColor, float subVal, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_MultiplyScalar_Batched(const vector<Mat>& imgs, bool isColor, float mulVal, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_GaussianNoise_Batched(const vector<Mat>& imgs, bool isColor, float mean, float stddev, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_SaltAndPepperNoise_Batched(const vector<Mat>& imgs, bool isColor, float noiseProb, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_NoiseShot_Batched(const vector<Mat>& imgs, bool isColor, float shotNoiseFactor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_ColorCast_Batched(const vector<Mat>& imgs, bool isColor, float rShift, float gShift, float bShift, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_ColorTemperature_Batched(const vector<Mat>& imgs, bool isColor, int adjustmentValue, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_ColorTwist_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Vignette_Batched(const vector<Mat>& imgs, bool isColor, float vignetteIntensity, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_NonLinearBlend_Batched(const vector<Mat>& imgs, bool isColor, float stdDev, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Posterize_Batched(const vector<Mat>& imgs, bool isColor, int levelBits, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Solarize_Batched(const vector<Mat>& imgs, bool isColor, int threshold, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Glitch_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_JpegCompressionDistortion_Batched(const vector<Mat>& imgs, bool isColor, Rpp32s quality, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Erase_Batched(const vector<Mat>& imgs, bool isColor, int numBoxes, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_RandomErase_Batched(const vector<Mat>& imgs, bool isColor, int numBoxes, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_CoarseDropout_Batched(const vector<Mat>& imgs, bool isColor, Rpp32u maxBoxesPerImage, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_GridDropout_Batched(const vector<Mat>& imgs, bool isColor, Rpp32u numGridsPerRow, Rpp32u numGridsPerColumn, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Gridmask_Batched(const vector<Mat>& imgs, bool isColor, int tileWidth, float ratio, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_ChannelDropout_Batched(const vector<Mat>& imgs, bool isColor, float dropoutProb, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_CutoutDropout_Batched(const vector<Mat>& imgs, bool isColor, Rpp32u numBoxes, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Copy_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Slice_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_ChannelPermute_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Transpose_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_TensorMin_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_TensorMax_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_TensorSum_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_TensorMean_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_TensorStddev_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Threshold_Batched(const vector<Mat>& imgs, bool isColor, float thresh, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_WarpAffine_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_WarpPerspective_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Fisheye_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_LensCorrection_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_GammaCorrection_Batched(const vector<Mat>& imgs, bool isColor, float gamma, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Exposure_Batched(const vector<Mat>& imgs, bool isColor, float exposureFactor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Blend_Batched(const vector<Mat>& imgs, bool isColor, float alpha, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Erode_Batched(const vector<Mat>& imgs, bool isColor, int kernelSize, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Dilate_Batched(const vector<Mat>& imgs, bool isColor, int kernelSize, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_BitwiseAnd_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_BitwiseOr_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_BitwiseNot_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_BitwiseXor_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);

// HOST Batched Operations
void benchmark_RPP_HOST_Flip_Batched(const vector<Mat>& imgs, bool isColor, int flipCode, rppHandle_t handle);
void benchmark_RPP_HOST_Resize_Batched(const vector<Mat>& imgs, bool isColor, int dstW, int dstH, RpptInterpolationType interpType, const string& interpName, rppHandle_t handle);
void benchmark_RPP_HOST_Crop_Batched(const vector<Mat>& imgs, bool isColor, int cropW, int cropH, rppHandle_t handle);
void benchmark_RPP_HOST_Rotate_Batched(const vector<Mat>& imgs, bool isColor, float angleDeg, rppHandle_t handle);
void benchmark_RPP_HOST_BoxFilter_Batched(const vector<Mat>& imgs, bool isColor, int kernelSize, rppHandle_t handle);
void benchmark_RPP_HOST_GaussianFilter_Batched(const vector<Mat>& imgs, bool isColor, int kernelSize, float sigma, rppHandle_t handle);
void benchmark_RPP_HOST_MedianFilter_Batched(const vector<Mat>& imgs, bool isColor, int kernelSize, rppHandle_t handle);
void benchmark_RPP_HOST_SobelFilter_Batched(const vector<Mat>& imgs, bool isColor, int sobelType, rppHandle_t handle);
void benchmark_RPP_HOST_HistogramEqualize_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_Hue_Batched(const vector<Mat>& imgs, bool isColor, float hueFactor, rppHandle_t handle);
void benchmark_RPP_HOST_Saturation_Batched(const vector<Mat>& imgs, bool isColor, float satFactor, rppHandle_t handle);
void benchmark_RPP_HOST_ColorToGreyscale_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_Brightness_Batched(const vector<Mat>& imgs, bool isColor, float alpha, float beta, rppHandle_t handle);
void benchmark_RPP_HOST_Contrast_Batched(const vector<Mat>& imgs, bool isColor, float contrastFactor, float contrastCenter, rppHandle_t handle);
void benchmark_RPP_HOST_Emboss_Batched(const vector<Mat>& imgs, bool isColor, int kernelSize, float strength, rppHandle_t handle);
void benchmark_RPP_HOST_AddScalar_Batched(const vector<Mat>& imgs, bool isColor, float addVal, rppHandle_t handle);
void benchmark_RPP_HOST_SubtractScalar_Batched(const vector<Mat>& imgs, bool isColor, float subVal, rppHandle_t handle);
void benchmark_RPP_HOST_MultiplyScalar_Batched(const vector<Mat>& imgs, bool isColor, float mulVal, rppHandle_t handle);
void benchmark_RPP_HOST_GaussianNoise_Batched(const vector<Mat>& imgs, bool isColor, float mean, float stddev, rppHandle_t handle);
void benchmark_RPP_HOST_SaltAndPepperNoise_Batched(const vector<Mat>& imgs, bool isColor, float noiseProb, rppHandle_t handle);
void benchmark_RPP_HOST_NoiseShot_Batched(const vector<Mat>& imgs, bool isColor, float shotNoiseFactor, rppHandle_t handle);
void benchmark_RPP_HOST_ColorCast_Batched(const vector<Mat>& imgs, bool isColor, float rShift, float gShift, float bShift, rppHandle_t handle);
void benchmark_RPP_HOST_ColorTemperature_Batched(const vector<Mat>& imgs, bool isColor, int adjustmentValue, rppHandle_t handle);
void benchmark_RPP_HOST_ColorTwist_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_Vignette_Batched(const vector<Mat>& imgs, bool isColor, float vignetteIntensity, rppHandle_t handle);
void benchmark_RPP_HOST_NonLinearBlend_Batched(const vector<Mat>& imgs, bool isColor, float stdDev, rppHandle_t handle);
void benchmark_RPP_HOST_Posterize_Batched(const vector<Mat>& imgs, bool isColor, int levelBits, rppHandle_t handle);
void benchmark_RPP_HOST_Solarize_Batched(const vector<Mat>& imgs, bool isColor, int threshold, rppHandle_t handle);
void benchmark_RPP_HOST_Glitch_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_JpegCompressionDistortion_Batched(const vector<Mat>& imgs, bool isColor, Rpp32s quality, rppHandle_t handle);
void benchmark_RPP_HOST_Erase_Batched(const vector<Mat>& imgs, bool isColor, int numBoxes, rppHandle_t handle);
void benchmark_RPP_HOST_RandomErase_Batched(const vector<Mat>& imgs, bool isColor, int numBoxes, rppHandle_t handle);
void benchmark_RPP_HOST_CoarseDropout_Batched(const vector<Mat>& imgs, bool isColor, Rpp32u maxBoxesPerImage, rppHandle_t handle);
void benchmark_RPP_HOST_GridDropout_Batched(const vector<Mat>& imgs, bool isColor, Rpp32u numGridsPerRow, Rpp32u numGridsPerColumn, rppHandle_t handle);
void benchmark_RPP_HOST_Gridmask_Batched(const vector<Mat>& imgs, bool isColor, int tileWidth, float ratio, rppHandle_t handle);
void benchmark_RPP_HOST_ChannelDropout_Batched(const vector<Mat>& imgs, bool isColor, float dropoutProb, rppHandle_t handle);
void benchmark_RPP_HOST_CutoutDropout_Batched(const vector<Mat>& imgs, bool isColor, int numBoxes, rppHandle_t handle);
void benchmark_RPP_HOST_Copy_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_Slice_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_ChannelPermute_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_Transpose_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_TensorMin_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_TensorMax_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_TensorSum_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_TensorMean_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_TensorStddev_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_Threshold_Batched(const vector<Mat>& imgs, bool isColor, float thresh, rppHandle_t handle);
void benchmark_RPP_HOST_WarpAffine_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_WarpPerspective_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_Fisheye_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_LensCorrection_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_GammaCorrection_Batched(const vector<Mat>& imgs, bool isColor, float gamma, rppHandle_t handle);
void benchmark_RPP_HOST_Exposure_Batched(const vector<Mat>& imgs, bool isColor, float exposureFactor, rppHandle_t handle);
void benchmark_RPP_HOST_Blend_Batched(const vector<Mat>& imgs, bool isColor, float alpha, rppHandle_t handle);
void benchmark_RPP_HOST_Erode_Batched(const vector<Mat>& imgs, bool isColor, int kernelSize, rppHandle_t handle);
void benchmark_RPP_HOST_Dilate_Batched(const vector<Mat>& imgs, bool isColor, int kernelSize, rppHandle_t handle);
void benchmark_RPP_HOST_BitwiseAnd_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_BitwiseOr_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_BitwiseNot_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_BitwiseXor_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);

// Advanced Operations - Batched
void benchmark_RPP_HOST_LUT_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_Magnitude_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_Phase_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_Normalize_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_FusedMultiplyAddScalar_Batched(const vector<Mat>& imgs, bool isColor, float mulVal, float addVal, rppHandle_t handle);
void benchmark_RPP_HOST_Remap_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HIP_LUT_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Magnitude_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Phase_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Normalize_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_FusedMultiplyAddScalar_Batched(const vector<Mat>& imgs, bool isColor, float mulVal, float addVal, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_Remap_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);

// Composite Operations - Batched
void benchmark_RPP_HOST_CropAndPatch_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_CropMirrorNormalize_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_ResizeMirrorNormalize_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HOST_ResizeCropMirror_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle);
void benchmark_RPP_HIP_CropAndPatch_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_CropMirrorNormalize_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_ResizeMirrorNormalize_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);
void benchmark_RPP_HIP_ResizeCropMirror_Batched(const vector<Mat>& imgs, bool isColor, rppHandle_t handle, hipStream_t stream);

#endif  // BENCHMARKS_COMMON_H
