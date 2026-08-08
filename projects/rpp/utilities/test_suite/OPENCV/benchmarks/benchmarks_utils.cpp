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

// Global configuration variables (defined here, declared extern in header)
int PERF_RUNS = 10;   // Default number of runs, can be overridden via command line
int WARMUP_RUNS = 50;       // Default number of warmup runs, can be overridden via command line
int TOTAL_RUNS = 0;         // Will be computed as WARMUP_RUNS + PERF_RUNS after initialization
int NUM_THREADS = 0;  // Will be set at runtime
string GRAY_IMAGE_PATH = DEFAULT_GRAY_IMAGE_PATH;
string RGB_IMAGE_PATH = DEFAULT_RGB_IMAGE_PATH;

// Global vectors to store results (defined here, declared extern in header)
vector<BenchmarkResult> grayscaleResults;
vector<BenchmarkResult> rgbResults;

// Global variables for image metadata
string grayImageSize;
string grayImageDtype;
string rgbImageSize;
string rgbImageDtype;
int grayBatchSize = 0;
int rgbBatchSize = 0;

vector<Mat> loadBatchImages(const string& directory, int& batchSize, int& maxWidth, int& maxHeight,
                            bool isColor) {
    vector<Mat> images;
    DIR* dir;
    struct dirent* entry;

    maxWidth = 0;
    maxHeight = 0;

    if ((dir = opendir(directory.c_str())) == NULL) {
        cerr << "Could not open directory: " << directory << endl;
        return images;
    }

    while ((entry = readdir(dir)) != NULL) {
        string filename = entry->d_name;
        if (filename == "." || filename == "..") continue;

        string ext = filename.substr(filename.find_last_of(".") + 1);
        transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext != "jpg" && ext != "jpeg" && ext != "png" && ext != "bmp" && ext != "tiff")
            continue;

        string filePath = directory + "/" + filename;
        Mat img = imread(filePath, isColor ? IMREAD_COLOR : IMREAD_GRAYSCALE);

        if (!img.empty()) {
            maxWidth = max(maxWidth, img.cols);
            maxHeight = max(maxHeight, img.rows);
            images.push_back(move(img));
        } else {
            cerr << "Warning: Could not read image " << filePath << endl;
        }
    }

    closedir(dir);
    batchSize = images.size();
    return images;
}

// Map to store benchmark times and parameters by operation name
struct BenchmarkData {
    double rppHostTime;
    double rppHipTime;
    double opencvTime;
    double rppHostBatchTime;
    double rppHipBatchTime;
    string parameters;
    bool rppHostCalled;
    bool rppHipCalled;
    bool opencvCalled;
    bool rppHostBatchCalled;
    bool rppHipBatchCalled;
    bool resultCreated;
    int resultIndex;  // Index in results vector, -1 if not created

    BenchmarkData()
        : rppHostTime(0), rppHipTime(0), opencvTime(0), rppHostBatchTime(0), rppHipBatchTime(0),
          parameters(""), rppHostCalled(false), rppHipCalled(false), opencvCalled(false),
          rppHostBatchCalled(false), rppHipBatchCalled(false), resultCreated(false), resultIndex(-1) {}
};

static map<string, BenchmarkData> benchmarkTimes;  // operationName -> data
static string currentOperation;
static bool currentIsColor;

void printResult(const string& name, int batchSize, bool isColor, double totalMs,
                 const string& params) {
    double avgTime = totalMs / PERF_RUNS;
    cout << name << " (Avg per run, WARMUP_RUNS=" << WARMUP_RUNS
         << ", PERF_RUNS=" << PERF_RUNS << ", " << batchSize << " images, "
         << (isColor ? "RGB" : "Grayscale");
    if (!params.empty()) cout << ", " << params;
    cout << "): " << avgTime << " ms" << endl;

    string opName = name;
    string prefix;

    // Check for prefixes in order (longest first to avoid mis-matching)
    const string RPP_HIP_BATCH_PREFIX = "RPP HIP BATCH ";
    const string RPP_HIP_PREFIX = "RPP HIP ";
    const string RPP_HOST_BATCH_PREFIX = "RPP HOST BATCH ";
    const string RPP_HOST_PREFIX = "RPP HOST ";
    const string OPENCV_PREFIX = "OpenCV ";

    if (opName.find(RPP_HIP_BATCH_PREFIX) == 0) {
        prefix = RPP_HIP_BATCH_PREFIX;
        opName = opName.substr(RPP_HIP_BATCH_PREFIX.length());
    } else if (opName.find(RPP_HIP_PREFIX) == 0) {
        prefix = RPP_HIP_PREFIX;
        opName = opName.substr(RPP_HIP_PREFIX.length());
    } else if (opName.find(RPP_HOST_BATCH_PREFIX) == 0) {
        prefix = RPP_HOST_BATCH_PREFIX;
        opName = opName.substr(RPP_HOST_BATCH_PREFIX.length());
    } else if (opName.find(RPP_HOST_PREFIX) == 0) {
        prefix = RPP_HOST_PREFIX;
        opName = opName.substr(RPP_HOST_PREFIX.length());
        currentOperation = opName;
        currentIsColor = isColor;
    } else if (opName.find(OPENCV_PREFIX) == 0) {
        prefix = OPENCV_PREFIX;
        opName = opName.substr(OPENCV_PREFIX.length());
    } else {
        return;
    }

    // Operations that use type parameter for unique naming (e.g., Resize_Bilinear, SobelFilter_0)
    static const unordered_set<string> opsWithTypeParameter = {"Resize", "Flip", "SobelFilter"};

    string displayName = opName;
    if (opsWithTypeParameter.count(opName) && !params.empty()) {
        size_t typePos = params.find("type=");
        if (typePos != string::npos) {
            size_t endPos = params.find(",", typePos);
            string typeValue = (endPos != string::npos)
                                   ? params.substr(typePos + 5, endPos - typePos - 5)
                                   : params.substr(typePos + 5);
            displayName = opName + "_" + typeValue;
        }
    }

    string key = displayName + "|" + (isColor ? "rgb" : "gray");
    auto& data = benchmarkTimes[key];

    if (prefix == RPP_HOST_PREFIX) {
        data.rppHostTime = avgTime;
        data.rppHostCalled = true;
        data.parameters = params;
    } else if (prefix == RPP_HOST_BATCH_PREFIX) {
        data.rppHostBatchTime = avgTime;
        data.rppHostBatchCalled = true;
        if (data.parameters.empty())
            data.parameters = params;
    } else if (prefix == RPP_HIP_PREFIX) {
        data.rppHipTime = avgTime;
        data.rppHipCalled = true;
        if (data.parameters.empty())
            data.parameters = params;
    } else if (prefix == RPP_HIP_BATCH_PREFIX) {
        data.rppHipBatchTime = avgTime;
        data.rppHipBatchCalled = true;
        if (data.parameters.empty())
            data.parameters = params;
    } else if (prefix == OPENCV_PREFIX) {
        data.opencvTime = avgTime;
        data.opencvCalled = true;
        if (data.parameters.empty())
            data.parameters = params;
    }

    // Create result when we have the base three benchmarks
    if (data.opencvCalled && data.rppHostCalled && data.rppHipCalled && !data.resultCreated) {
        if (isColor) {
            rgbResults.emplace_back(displayName, data.parameters, data.opencvTime,
                                    data.rppHostTime, data.rppHipTime, rgbImageSize,
                                    rgbImageDtype, rgbBatchSize, PERF_RUNS,
                                    data.rppHostBatchTime, data.rppHipBatchTime);
            data.resultIndex = rgbResults.size() - 1;
        } else {
            grayscaleResults.emplace_back(displayName, data.parameters, data.opencvTime,
                                          data.rppHostTime, data.rppHipTime, grayImageSize,
                                          grayImageDtype, grayBatchSize, PERF_RUNS,
                                          data.rppHostBatchTime, data.rppHipBatchTime);
            data.resultIndex = grayscaleResults.size() - 1;
        }
        data.resultCreated = true;
    }
    // Update existing result if batch times are added later
    else if (data.resultCreated && data.resultIndex >= 0) {
        if (prefix == RPP_HOST_BATCH_PREFIX || prefix == RPP_HIP_BATCH_PREFIX) {
            if (isColor && data.resultIndex < (int)rgbResults.size()) {
                auto& result = rgbResults[data.resultIndex];
                result.rppHostBatchTime = data.rppHostBatchTime;
                result.rppHipBatchTime = data.rppHipBatchTime;
            } else if (!isColor && data.resultIndex < (int)grayscaleResults.size()) {
                auto& result = grayscaleResults[data.resultIndex];
                result.rppHostBatchTime = data.rppHostBatchTime;
                result.rppHipBatchTime = data.rppHipBatchTime;
            }
        }
    }
}

// Helper to get CPU model information
string getCPUInfo() {
    ifstream cpuinfo("/proc/cpuinfo");
    string line;
    while (getline(cpuinfo, line)) {
        if (line.find("model name") != string::npos) {
            size_t pos = line.find(":");
            if (pos != string::npos) return line.substr(pos + 2);
        }
    }
    return "Unknown CPU";
}

// Helper to get memory information (in GB)
string getMemoryInfo() {
    ifstream meminfo("/proc/meminfo");
    string line;
    while (getline(meminfo, line)) {
        if (line.find("MemTotal") != string::npos) {
            istringstream iss(line);
            string label;
            long memKB;
            iss >> label >> memKB;
            double memGB = memKB / 1024.0 / 1024.0;
            ostringstream oss;
            oss.precision(2);
            oss << fixed << memGB << " GB";
            return oss.str();
        }
    }
    return "Unknown";
}

// Helper to get OS information
string getOSInfo() {
    struct utsname unameData;
    if (uname(&unameData) == 0) {
        ostringstream oss;
        oss << unameData.sysname << " " << unameData.release;
        return oss.str();
    }
    return "Unknown OS";
}

// Helper to get GPU architecture with family name
string getGPUArchitecture() {
    int deviceCount = 0;
    if (hipGetDeviceCount(&deviceCount) != hipSuccess) {
        return "GPU detection failed";
    }
    if (deviceCount > 0) {
        hipDeviceProp_t prop;
        if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
            return "Failed to get GPU properties";
        }
        string gcnArchName(prop.gcnArchName);
        if (gcnArchName.empty()) {
            return "Unknown";
        }

        ostringstream oss;

        // Determine architecture family and version
        if (gcnArchName.find("gfx12") == 0) {
            oss << "RDNA3 (" << gcnArchName << ")";
        } else if (gcnArchName.find("gfx11") == 0) {
            oss << "RDNA3 (" << gcnArchName << ")";
        } else if (gcnArchName.find("gfx103") == 0) {
            oss << "RDNA2 (" << gcnArchName << ")";
        } else if (gcnArchName.find("gfx101") == 0) {
            oss << "RDNA1 (" << gcnArchName << ")";
        } else if (gcnArchName == "gfx940" || gcnArchName == "gfx941" || gcnArchName == "gfx942") {
            oss << "CDNA3 (" << gcnArchName << ")";
        } else if (gcnArchName == "gfx90a") {
            oss << "CDNA2 (" << gcnArchName << ")";
        } else if (gcnArchName == "gfx908") {
            oss << "CDNA1 (" << gcnArchName << ")";
        } else if (gcnArchName.find("gfx9") == 0) {
            oss << "GCN5 (" << gcnArchName << ")";
        } else if (gcnArchName.find("gfx8") == 0) {
            oss << "GCN4 (" << gcnArchName << ")";
        } else if (gcnArchName.find("gfx7") == 0) {
            oss << "GCN3 (" << gcnArchName << ")";
        } else if (gcnArchName.find("gfx6") == 0) {
            oss << "GCN2 (" << gcnArchName << ")";
        } else {
            // Unknown architecture, just return the gcnArchName
            oss << gcnArchName;
        }

        return oss.str();
    }
    return "No GPU detected";
}

// Helper to get just the GPU architecture name (e.g., "gfx1201")
string getGPUArchitectureName() {
    int deviceCount = 0;
    if (hipGetDeviceCount(&deviceCount) != hipSuccess) {
        return "unknown";
    }
    if (deviceCount > 0) {
        hipDeviceProp_t prop;
        if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
            return "unknown";
        }
        string gcnArchName(prop.gcnArchName);
        if (!gcnArchName.empty()) {
            // Strip feature flags after colon for clean file naming
            size_t colonPos = gcnArchName.find(':');
            if (colonPos != string::npos) {
                return gcnArchName.substr(0, colonPos);
            }
            return gcnArchName;
        }
    }
    return "unknown";
}

// Helper to get GPU memory information (in GB)
string getGPUMemoryInfo() {
    int deviceCount = 0;
    if (hipGetDeviceCount(&deviceCount) != hipSuccess) {
        return "GPU detection failed";
    }
    if (deviceCount > 0) {
        hipDeviceProp_t prop;
        if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
            return "Failed to get GPU properties";
        }
        double memGB = prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0);
        ostringstream oss;
        oss.precision(2);
        oss << fixed << memGB << " GB";
        return oss.str();
    }
    return "No GPU detected";
}

// Helper to get GPU info
string getGPUInfo() {
    int deviceCount = 0;
    if (hipGetDeviceCount(&deviceCount) != hipSuccess) {
        return "GPU detection failed";
    }
    if (deviceCount > 0) {
        hipDeviceProp_t prop;
        if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
            return "Failed to get GPU properties";
        }
        ostringstream oss;
        oss << prop.name;

        // Determine compute units based on architecture
        // RDNA architectures (gfx10xx, gfx11xx, gfx12xx) report WGPs in multiProcessorCount
        // Each WGP contains 2 CUs, so we need to multiply by 2
        // GCN and CDNA architectures report CUs directly
        int computeUnits = prop.multiProcessorCount;
        string gcnArchName(prop.gcnArchName);

        // Check architecture using both gcnArchName string and major version number
        bool isRDNA = false;
        if (!gcnArchName.empty()) {
            // RDNA1: gfx10xx (major=10)
            // RDNA2: gfx10xx (major=10, specifically gfx1030+)
            // RDNA3: gfx11xx (major=11) and gfx12xx (major=12)
            isRDNA = (gcnArchName.find("gfx10") == 0 ||
                      gcnArchName.find("gfx11") == 0 ||
                      gcnArchName.find("gfx12") == 0);
        } else {
            // Fallback to major version if gcnArchName is not available
            isRDNA = (prop.major >= 10 && prop.major <= 12);
        }

        if (isRDNA) {
            computeUnits *= 2;  // Convert WGPs to CUs for RDNA architectures
        }

        oss << " (" << computeUnits << " CUs)";
        return oss.str();
    }
    return "No GPU detected";
}

// Helper to get RPP version
string getRPPVersion() {
    ostringstream oss;
    oss << RPP_VERSION_MAJOR << "." << RPP_VERSION_MINOR << "." << RPP_VERSION_PATCH;
    return oss.str();
}

// Helper to get ROCm version
string getROCmVersion() {
    const char* rocm_path = getenv("ROCM_PATH");
    if (!rocm_path) {
        rocm_path = "/opt/rocm";
    }

    string version_file = string(rocm_path) + "/.info/version";
    ifstream vfile(version_file);
    if (vfile.is_open()) {
        string version;
        getline(vfile, version);
        vfile.close();
        return version;
    }
    return "Unknown";
}

// Helper to get current date and time
string getCurrentDateTime() {
    auto now = chrono::system_clock::now();
    time_t now_time = chrono::system_clock::to_time_t(now);
    tm local_tm = *localtime(&now_time);

    ostringstream oss;
    oss << put_time(&local_tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}

// Helper to get data type string from OpenCV type
string getDtypeString(int cvType) {
    int depth = cvType & CV_MAT_DEPTH_MASK;

    switch (depth) {
        case CV_8U:
            return "U8";
        case CV_8S:
            return "S8";
        case CV_16U:
            return "U16";
        case CV_16S:
            return "S16";
        case CV_32S:
            return "S32";
        case CV_32F:
            return "F32";
        case CV_64F:
            return "F64";
        default:
            return "Unknown";
    }
}

// Helper to create RPP descriptor from OpenCV Mat
RpptDesc createRppDescriptor(const Mat& img, RpptLayout layout) {
    RpptDesc desc;
    desc.n = 1;
    desc.h = img.rows;
    desc.w = img.cols;
    desc.c = img.channels();
    desc.layout = layout;
    desc.dataType = (img.depth() == CV_32F) ? RpptDataType::F32 : RpptDataType::U8;
    desc.offsetInBytes = 0;
    desc.strides.nStride = desc.h * desc.w * desc.c;
    if (layout == RpptLayout::NHWC) {
        desc.strides.hStride = desc.w * desc.c;
        desc.strides.wStride = desc.c;
        desc.strides.cStride = 1;
    } else {  // NCHW
        desc.strides.cStride = desc.h * desc.w;
        desc.strides.hStride = desc.w;
        desc.strides.wStride = 1;
    }
    return desc;
}

// Helper to create full image ROI
RpptROI createFullImageROI(const Mat& img) {
    RpptROI roi;
    roi.xywhROI.xy.x = 0;
    roi.xywhROI.xy.y = 0;
    roi.xywhROI.roiWidth = img.cols;
    roi.xywhROI.roiHeight = img.rows;
    return roi;
}

// Helper to convert RpptDesc to RpptGenericDesc
RpptGenericDesc toGenericDesc(const RpptDesc& desc) {
    RpptGenericDesc genericDesc;
    genericDesc.numDims = 4;
    genericDesc.offsetInBytes = desc.offsetInBytes;
    genericDesc.dataType = desc.dataType;
    genericDesc.layout = desc.layout;

    // Set dims and strides based on layout
    if (desc.layout == RpptLayout::NHWC) {
        // NHWC: dims = [N, H, W, C]
        genericDesc.dims[0] = desc.n;
        genericDesc.dims[1] = desc.h;
        genericDesc.dims[2] = desc.w;
        genericDesc.dims[3] = desc.c;
        genericDesc.strides[0] = desc.strides.nStride;
        genericDesc.strides[1] = desc.strides.hStride;
        genericDesc.strides[2] = desc.strides.wStride;
        genericDesc.strides[3] = desc.strides.cStride;
    } else {  // NCHW
        // NCHW: dims = [N, C, H, W]
        genericDesc.dims[0] = desc.n;
        genericDesc.dims[1] = desc.c;
        genericDesc.dims[2] = desc.h;
        genericDesc.dims[3] = desc.w;
        genericDesc.strides[0] = desc.strides.nStride;
        genericDesc.strides[1] = desc.strides.cStride;
        genericDesc.strides[2] = desc.strides.hStride;
        genericDesc.strides[3] = desc.strides.wStride;
    }

    return genericDesc;
}

// Helper to create full image ROI3D
RpptROI3D createFullImageROI3D(const Mat& img) {
    RpptROI3D roi3d;
    roi3d.xyzwhdROI.xyz.x = 0;
    roi3d.xyzwhdROI.xyz.y = 0;
    roi3d.xyzwhdROI.xyz.z = 0;
    roi3d.xyzwhdROI.roiWidth = img.cols;
    roi3d.xyzwhdROI.roiHeight = img.rows;
    roi3d.xyzwhdROI.roiDepth = 1;
    return roi3d;
}

// Helper to set descriptor dimensions and strides
void set_descriptor_dims_and_strides(RpptDesc* descPtr, int noOfImages, int maxHeight,
                                    int maxWidth, int numChannels, int offsetInBytes,
                                    int additionalStride) {
    descPtr->numDims = 4;
    descPtr->offsetInBytes = offsetInBytes;
    descPtr->n = noOfImages;
    descPtr->h = maxHeight;
    descPtr->w = maxWidth;
    descPtr->c = numChannels;

    // BUGFIX: Padding causes stride/buffer mismatch for batched operations
    // The padding was causing segfaults in BICUBIC interpolation because:
    // 1. Buffer allocated with padded width
    // 2. Data copied with actual width
    // 3. RPP kernel accesses with padded stride but uninitialized padding
    // 4. BICUBIC's 4x4 neighborhood exposes this by reading beyond valid data
    // Optionally set w stride as a multiple of 8 for src/dst
    // descPtr->w = (descPtr->w / 8) * 8 + 8 + additionalStride;  // DISABLED - causes segfault
    // set strides
    if (descPtr->layout == RpptLayout::NHWC) {
        descPtr->strides.nStride = descPtr->c * descPtr->w * descPtr->h;
        descPtr->strides.hStride = descPtr->c * descPtr->w;
        descPtr->strides.wStride = descPtr->c;
        descPtr->strides.cStride = 1;
    } else if (descPtr->layout == RpptLayout::NCHW) {
        descPtr->strides.nStride = descPtr->c * descPtr->w * descPtr->h;
        descPtr->strides.cStride = descPtr->w * descPtr->h;
        descPtr->strides.hStride = descPtr->w;
        descPtr->strides.wStride = 1;
    }
}

// Helper to generate channel dropout mask
void generate_channel_dropout_mask(Rpp8u* dropoutTensor, Rpp32f* dropoutProbability, int batchSize,
                                   int channels, int seed) {
    omp_set_dynamic(0);

#pragma omp parallel for num_threads(omp_get_max_threads())
    for (int batchCount = 0; batchCount < batchSize; batchCount++) {
        std::mt19937 rng(seed + batchCount);
        std::bernoulli_distribution keepDist(1.0f - dropoutProbability[batchCount]);
        Rpp8u* maskPtrTemp = dropoutTensor + (batchCount * channels);
        bool atLeastOne = false;

        for (int channel = 0; channel < channels; channel++) {
            maskPtrTemp[channel] = keepDist(rng);
            atLeastOne |= maskPtrTemp[channel];
        }

        if (!atLeastOne) maskPtrTemp[rng() % channels] = 1;
    }
}

// Helper to initialize cutout dropout
void init_cutout_dropout(int batchSize, int maxBoxesPerImage, Rpp32u* numOfBoxes,
                        RpptRoiLtrb* anchorBoxInfoTensor, RpptROI* roiTensorPtrSrc,
                        int channels, int BitDepthTestMode, int seed, int dropoutType,
                        void* colorBuffer) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> pos_ratio(0.1f, 0.9f);
    std::uniform_real_distribution<float> wh_ratio_cutout(0.4f, 0.6f);

    Rpp8u* colors8u = reinterpret_cast<Rpp8u*>(colorBuffer);
    Rpp16f* colors16f = reinterpret_cast<Rpp16f*>(colorBuffer);
    Rpp32f* colors32f = reinterpret_cast<Rpp32f*>(colorBuffer);
    Rpp8s* colors8s = reinterpret_cast<Rpp8s*>(colorBuffer);

    for (int i = 0; i < batchSize; i++) {
        const auto& roi = roiTensorPtrSrc[i].xywhROI;
        const float roiW = static_cast<float>(roi.roiWidth);
        const float roiH = static_cast<float>(roi.roiHeight);
        const float roiX = static_cast<float>(roi.xy.x);
        const float roiY = static_cast<float>(roi.xy.y);

        float boxW, boxH;

        float squareSize = wh_ratio_cutout(rng) * std::min(roiW, roiH);
        boxW = boxH = std::max(1.0f, squareSize);
        const float x_start = std::max(0.0f, std::min(pos_ratio(rng) * (roiW - boxW), roiW - boxW));
        const float y_start = std::max(0.0f, std::min(pos_ratio(rng) * (roiH - boxH), roiH - boxH));

        RpptRoiLtrb& box = anchorBoxInfoTensor[i * maxBoxesPerImage];
        box.lt.x = static_cast<Rpp32u>(roiX + x_start);
        box.lt.y = static_cast<Rpp32u>(roiY + y_start);
        box.rb.x = static_cast<Rpp32u>(roiX + x_start + boxW - 1.0f);
        box.rb.y = static_cast<Rpp32u>(roiY + y_start + boxH - 1.0f);

        if (colorBuffer != nullptr) {
            int colorOffset = (i * maxBoxesPerImage) * channels;
            Rpp32f dropoutColor = 0.0f;
            for (int c = 0; c < channels; c++) {
                if (BitDepthTestMode == 0) // U8_TO_U8
                    colors8u[colorOffset + c] = (Rpp8u)dropoutColor;
                else if (BitDepthTestMode == 2) // F16_TO_F16
                    colors16f[colorOffset + c] = (Rpp16f)(dropoutColor / 255.0f);
                else if (BitDepthTestMode == 3) // F32_TO_F32
                    colors32f[colorOffset + c] = (Rpp32f)(dropoutColor);
                else if (BitDepthTestMode == 4) // I8_TO_I8
                    colors8s[colorOffset + c] = (Rpp8s)(dropoutColor - 128);
            }
        }
        numOfBoxes[i] = 1;
    }
}

// ==================== RPP COLOR AUGMENTATIONS ====================
bool writeResultsToExcel(const string& filename, const vector<BenchmarkResult>& grayResults,
                         const vector<BenchmarkResult>& colorResults, int maxAvailableThreads) {
    lxw_workbook* workbook = workbook_new(filename.c_str());
    if (!workbook) {
        cerr << "Error: Failed to create Excel workbook: " << filename << endl;
        cerr << "       Possible causes: insufficient permissions, invalid path, or disk full"
             << endl;
        return false;
    }

    // Create formats
    lxw_format* header_format = workbook_add_format(workbook);
    format_set_bold(header_format);
    format_set_bg_color(header_format, 0x4472C4);
    format_set_font_color(header_format, LXW_COLOR_WHITE);
    format_set_align(header_format, LXW_ALIGN_CENTER);

    lxw_format* info_label_format = workbook_add_format(workbook);
    format_set_bold(info_label_format);

    lxw_format* speedup_format = workbook_add_format(workbook);
    format_set_num_format(speedup_format, "0.00000");

    lxw_format* time_format = workbook_add_format(workbook);
    format_set_num_format(time_format, "0.00000");

    // Sheet 1: System Information
    lxw_worksheet* info_sheet = workbook_add_worksheet(workbook, "System Information");

    worksheet_set_column(info_sheet, 0, 0, 25, NULL);
    worksheet_set_column(info_sheet, 1, 1, 40, NULL);

    int row = 0;
    worksheet_write_string(info_sheet, row++, 0, "Parameter", header_format);
    worksheet_write_string(info_sheet, row - 1, 1, "Value", header_format);

    worksheet_write_string(info_sheet, row, 0, "Benchmark Date & Time", info_label_format);
    worksheet_write_string(info_sheet, row++, 1, getCurrentDateTime().c_str(), NULL);

    worksheet_write_string(info_sheet, row, 0, "RPP Version", info_label_format);
    worksheet_write_string(info_sheet, row++, 1, getRPPVersion().c_str(), NULL);

    worksheet_write_string(info_sheet, row, 0, "ROCm Version", info_label_format);
    worksheet_write_string(info_sheet, row++, 1, getROCmVersion().c_str(), NULL);

    worksheet_write_string(info_sheet, row, 0, "OpenCV Version", info_label_format);
    worksheet_write_string(info_sheet, row++, 1, CV_VERSION, NULL);

    worksheet_write_string(info_sheet, row, 0, "Operating System", info_label_format);
    worksheet_write_string(info_sheet, row++, 1, getOSInfo().c_str(), NULL);

    worksheet_write_string(info_sheet, row, 0, "CPU", info_label_format);
    worksheet_write_string(info_sheet, row++, 1, getCPUInfo().c_str(), NULL);

    worksheet_write_string(info_sheet, row, 0, "Maximum Available Threads", info_label_format);
    worksheet_write_number(info_sheet, row++, 1, omp_get_max_threads(), NULL);

    worksheet_write_string(info_sheet, row, 0, "System RAM", info_label_format);
    worksheet_write_string(info_sheet, row++, 1, getMemoryInfo().c_str(), NULL);

    worksheet_write_string(info_sheet, row, 0, "GPU", info_label_format);
    worksheet_write_string(info_sheet, row++, 1, getGPUInfo().c_str(), NULL);

    worksheet_write_string(info_sheet, row, 0, "GPU Architecture", info_label_format);
    worksheet_write_string(info_sheet, row++, 1, getGPUArchitecture().c_str(), NULL);

    worksheet_write_string(info_sheet, row, 0, "GPU Memory", info_label_format);
    worksheet_write_string(info_sheet, row++, 1, getGPUMemoryInfo().c_str(), NULL);

    // Sheet 2: Grayscale Results
    lxw_worksheet* gray_sheet = workbook_add_worksheet(workbook, "Grayscale Benchmarks");

    worksheet_set_column(gray_sheet, 0, 0, 30, NULL);
    worksheet_set_column(gray_sheet, 1, 1, 40, NULL);
    worksheet_set_column(gray_sheet, 2, 2, 15, NULL);
    worksheet_set_column(gray_sheet, 3, 3, 12, NULL);
    worksheet_set_column(gray_sheet, 4, 5, 12, NULL);
    worksheet_set_column(gray_sheet, 6, 10, 22, NULL);

    // Create column headers with thread information
    ostringstream opencvHeader, rppHostHeader, rppHostBatchHeader;
    opencvHeader << "OpenCV (avg ms, " << NUM_THREADS << " threads)";
    rppHostHeader << "RPP HOST (avg ms, " << NUM_THREADS << " threads)";
    rppHostBatchHeader << "RPP HOST BATCH (avg ms, " << maxAvailableThreads << " threads)";

    row = 0;
    worksheet_write_string(gray_sheet, row, 0, "Operation", header_format);
    worksheet_write_string(gray_sheet, row, 1, "Parameters", header_format);
    worksheet_write_string(gray_sheet, row, 2, "Image Size", header_format);
    worksheet_write_string(gray_sheet, row, 3, "DType", header_format);
    worksheet_write_string(gray_sheet, row, 4, "Batch Size", header_format);
    worksheet_write_string(gray_sheet, row, 5, "Runs", header_format);
    worksheet_write_string(gray_sheet, row, 6, opencvHeader.str().c_str(), header_format);
    worksheet_write_string(gray_sheet, row, 7, rppHostHeader.str().c_str(), header_format);
    worksheet_write_string(gray_sheet, row, 8, rppHostBatchHeader.str().c_str(), header_format);
    worksheet_write_string(gray_sheet, row, 9, "RPP HIP (avg ms)", header_format);
    worksheet_write_string(gray_sheet, row++, 10, "RPP HIP BATCH (avg ms)", header_format);

    for (const auto& result : grayResults) {
        worksheet_write_string(gray_sheet, row, 0, result.operationName.c_str(), NULL);
        worksheet_write_string(gray_sheet, row, 1, result.parameters.c_str(), NULL);
        worksheet_write_string(gray_sheet, row, 2, result.imageSize.c_str(), NULL);
        worksheet_write_string(gray_sheet, row, 3, result.dtype.c_str(), NULL);
        worksheet_write_number(gray_sheet, row, 4, result.batchSize, NULL);
        worksheet_write_number(gray_sheet, row, 5, result.numRuns, NULL);
        worksheet_write_number(gray_sheet, row, 6, result.opencvTime, time_format);
        worksheet_write_number(gray_sheet, row, 7, result.rppHostTime, time_format);
        worksheet_write_number(gray_sheet, row, 8, result.rppHostBatchTime, time_format);
        worksheet_write_number(gray_sheet, row, 9, result.rppHipTime, time_format);
        worksheet_write_number(gray_sheet, row, 10, result.rppHipBatchTime, time_format);
        row++;
    }

    // Sheet 3: RGB Results
    lxw_worksheet* rgb_sheet = workbook_add_worksheet(workbook, "RGB Benchmarks");

    worksheet_set_column(rgb_sheet, 0, 0, 30, NULL);
    worksheet_set_column(rgb_sheet, 1, 1, 40, NULL);
    worksheet_set_column(rgb_sheet, 2, 2, 15, NULL);
    worksheet_set_column(rgb_sheet, 3, 3, 12, NULL);
    worksheet_set_column(rgb_sheet, 4, 5, 12, NULL);
    worksheet_set_column(rgb_sheet, 6, 10, 22, NULL);

    row = 0;
    worksheet_write_string(rgb_sheet, row, 0, "Operation", header_format);
    worksheet_write_string(rgb_sheet, row, 1, "Parameters", header_format);
    worksheet_write_string(rgb_sheet, row, 2, "Image Size", header_format);
    worksheet_write_string(rgb_sheet, row, 3, "DType", header_format);
    worksheet_write_string(rgb_sheet, row, 4, "Batch Size", header_format);
    worksheet_write_string(rgb_sheet, row, 5, "Runs", header_format);
    worksheet_write_string(rgb_sheet, row, 6, opencvHeader.str().c_str(), header_format);
    worksheet_write_string(rgb_sheet, row, 7, rppHostHeader.str().c_str(), header_format);
    worksheet_write_string(rgb_sheet, row, 8, rppHostBatchHeader.str().c_str(), header_format);
    worksheet_write_string(rgb_sheet, row, 9, "RPP HIP (avg ms)", header_format);
    worksheet_write_string(rgb_sheet, row++, 10, "RPP HIP BATCH (avg ms)", header_format);

    for (const auto& result : colorResults) {
        worksheet_write_string(rgb_sheet, row, 0, result.operationName.c_str(), NULL);
        worksheet_write_string(rgb_sheet, row, 1, result.parameters.c_str(), NULL);
        worksheet_write_string(rgb_sheet, row, 2, result.imageSize.c_str(), NULL);
        worksheet_write_string(rgb_sheet, row, 3, result.dtype.c_str(), NULL);
        worksheet_write_number(rgb_sheet, row, 4, result.batchSize, NULL);
        worksheet_write_number(rgb_sheet, row, 5, result.numRuns, NULL);
        worksheet_write_number(rgb_sheet, row, 6, result.opencvTime, time_format);
        worksheet_write_number(rgb_sheet, row, 7, result.rppHostTime, time_format);
        worksheet_write_number(rgb_sheet, row, 8, result.rppHostBatchTime, time_format);
        worksheet_write_number(rgb_sheet, row, 9, result.rppHipTime, time_format);
        worksheet_write_number(rgb_sheet, row, 10, result.rppHipBatchTime, time_format);
        row++;
    }

    lxw_error error = workbook_close(workbook);
    if (error != LXW_NO_ERROR) {
        cerr << "Error: Failed to close Excel workbook: " << filename << " (Error code: " << error
             << ")" << endl;
        cerr << "       The file may be corrupted or incomplete" << endl;
        return false;
    }

    cout << "\nResults exported successfully to: " << filename << endl;
    return true;
}

// ==================== MAIN ====================

// Helper to initialize RICAP boxes for 4-way cutmix
void init_ricap_boxes(int maxWidth, int maxHeight, int batchSize, Rpp32u* permutationTensor,
                      RpptROI* roiPtrInputCropRegion) {
    // Simple RICAP: divide output into 4 quadrants
    int halfW = maxWidth / 2;
    int halfH = maxHeight / 2;

    roiPtrInputCropRegion[0].xywhROI.xy.x = 0;
    roiPtrInputCropRegion[0].xywhROI.xy.y = 0;
    roiPtrInputCropRegion[0].xywhROI.roiWidth = halfW;
    roiPtrInputCropRegion[0].xywhROI.roiHeight = halfH;

    roiPtrInputCropRegion[1].xywhROI.xy.x = halfW;
    roiPtrInputCropRegion[1].xywhROI.xy.y = 0;
    roiPtrInputCropRegion[1].xywhROI.roiWidth = halfW;
    roiPtrInputCropRegion[1].xywhROI.roiHeight = halfH;

    roiPtrInputCropRegion[2].xywhROI.xy.x = 0;
    roiPtrInputCropRegion[2].xywhROI.xy.y = halfH;
    roiPtrInputCropRegion[2].xywhROI.roiWidth = halfW;
    roiPtrInputCropRegion[2].xywhROI.roiHeight = halfH;

    roiPtrInputCropRegion[3].xywhROI.xy.x = halfW;
    roiPtrInputCropRegion[3].xywhROI.xy.y = halfH;
    roiPtrInputCropRegion[3].xywhROI.roiWidth = halfW;
    roiPtrInputCropRegion[3].xywhROI.roiHeight = halfH;

    // Permutation: which source image for each quadrant
    for (int i = 0; i < batchSize; i++) {
        permutationTensor[i * 4 + 0] = (i + 0) % batchSize;
        permutationTensor[i * 4 + 1] = (i + 1) % batchSize;
        permutationTensor[i * 4 + 2] = (i + 2) % batchSize;
        permutationTensor[i * 4 + 3] = (i + 3) % batchSize;
    }
}

// Helper to initialize grid dropout boxes
void init_grid_dropout_boxes(int batchCount, RpptRoiLtrb* anchorBoxInfoTensor,
                             RpptROI* roiTensorPtrSrc, Rpp32u gridH, Rpp32u gridW, Rpp32u& maxHoleW,
                             Rpp32u& maxHoleH, Rpp32f holeRatio, int seed) {
    std::mt19937 rng(seed);

    for (int i = 0; i < batchCount; i++) {
        Rpp32u roiW = roiTensorPtrSrc[i].xywhROI.roiWidth;
        Rpp32u roiH = roiTensorPtrSrc[i].xywhROI.roiHeight;
        Rpp32s x_base = roiTensorPtrSrc[i].xywhROI.xy.x;
        Rpp32s y_base = roiTensorPtrSrc[i].xywhROI.xy.y;

        Rpp32u cellW = std::max(1u, roiW / gridW);
        Rpp32u cellH = std::max(1u, roiH / gridH);
        Rpp32u holeW = std::max(1u, static_cast<Rpp32u>(cellW * holeRatio));
        Rpp32u holeH = std::max(1u, static_cast<Rpp32u>(cellH * holeRatio));
        if (holeW > maxHoleW) maxHoleW = holeW;
        if (holeH > maxHoleH) maxHoleH = holeH;

        std::uniform_int_distribution<int> distX(0, (cellW > holeW) ? cellW - holeW : 0);
        std::uniform_int_distribution<int> distY(0, (cellH > holeH) ? cellH - holeH : 0);

        int boxOffset = i * gridH * gridW;
        for (Rpp32u row = 0; row < gridH; ++row) {
            for (Rpp32u col = 0; col < gridW; ++col) {
                Rpp32s cellX = x_base + col * cellW;
                Rpp32s cellY = y_base + row * cellH;

                Rpp32s offsetX = 0, offsetY = 0;
                if (cellW > holeW && cellH > holeH) {
                    offsetX = distX(rng);
                    offsetY = distY(rng);
                }

                Rpp32s x1 = std::min(cellX + offsetX, x_base + (Rpp32s)roiW - 1);
                Rpp32s y1 = std::min(cellY + offsetY, y_base + (Rpp32s)roiH - 1);
                Rpp32s x2 = std::min(x1 + (Rpp32s)holeW - 1, x_base + (Rpp32s)roiW - 1);
                Rpp32s y2 = std::min(y1 + (Rpp32s)holeH - 1, y_base + (Rpp32s)roiH - 1);

                int boxIdx = boxOffset + (row * gridW + col);
                anchorBoxInfoTensor[boxIdx].lt.x = x1;
                anchorBoxInfoTensor[boxIdx].lt.y = y1;
                anchorBoxInfoTensor[boxIdx].rb.x = x2;
                anchorBoxInfoTensor[boxIdx].rb.y = y2;
            }
        }
    }
}

// sets generic descriptor dimensions and strides for 5D tensors
void set_generic_descriptor(RpptGenericDescPtr descriptorPtr3D, int noOfImages, int maxX,
                           int maxY, int maxZ, int numChannels, int offsetInBytes,
                           int layoutType) {
    descriptorPtr3D->numDims = 5;
    descriptorPtr3D->offsetInBytes = offsetInBytes;
    descriptorPtr3D->dataType = RpptDataType::F32;

    if (layoutType == 0) {
        descriptorPtr3D->layout = RpptLayout::NCDHW;
        descriptorPtr3D->dims[0] = noOfImages;
        descriptorPtr3D->dims[1] = numChannels;
        descriptorPtr3D->dims[2] = maxZ;
        descriptorPtr3D->dims[3] = maxY;
        descriptorPtr3D->dims[4] = maxX;
    } else if (layoutType == 1) {
        descriptorPtr3D->layout = RpptLayout::NDHWC;
        descriptorPtr3D->dims[0] = noOfImages;
        descriptorPtr3D->dims[1] = maxZ;
        descriptorPtr3D->dims[2] = maxY;
        descriptorPtr3D->dims[3] = maxX;
        descriptorPtr3D->dims[4] = numChannels;
    }

    descriptorPtr3D->strides[0] = descriptorPtr3D->dims[1] * descriptorPtr3D->dims[2] *
                                  descriptorPtr3D->dims[3] * descriptorPtr3D->dims[4];
    descriptorPtr3D->strides[1] =
        descriptorPtr3D->dims[2] * descriptorPtr3D->dims[3] * descriptorPtr3D->dims[4];
    descriptorPtr3D->strides[2] = descriptorPtr3D->dims[3] * descriptorPtr3D->dims[4];
    descriptorPtr3D->strides[3] = descriptorPtr3D->dims[4];
    descriptorPtr3D->strides[4] = 1;
}

// initialize remap tables for horizontal flip effect
void init_remap(RpptDescPtr tableDescPtr, RpptDescPtr srcDescPtr, RpptROIPtr roiTensorPtrSrc,
               Rpp32f* rowRemapTable, Rpp32f* colRemapTable) {
    tableDescPtr->c = 1;
    tableDescPtr->strides.nStride = srcDescPtr->h * srcDescPtr->w;
    tableDescPtr->strides.hStride = srcDescPtr->w;
    tableDescPtr->strides.wStride = tableDescPtr->strides.cStride = 1;
    Rpp32u batchSize = srcDescPtr->n;

    for (Rpp32u count = 0; count < batchSize; count++) {
        Rpp32f *rowRemapTableTemp, *colRemapTableTemp;
        rowRemapTableTemp = rowRemapTable + count * tableDescPtr->strides.nStride;
        colRemapTableTemp = colRemapTable + count * tableDescPtr->strides.nStride;
        Rpp32u halfWidth = roiTensorPtrSrc[count].xywhROI.roiWidth / 2;
        for (Rpp32u i = 0; i < roiTensorPtrSrc[count].xywhROI.roiHeight; i++) {
            Rpp32f *rowRemapTableTempRow, *colRemapTableTempRow;
            rowRemapTableTempRow = rowRemapTableTemp + i * tableDescPtr->strides.hStride;
            colRemapTableTempRow = colRemapTableTemp + i * tableDescPtr->strides.hStride;
            Rpp32u j = 0;
            for (; j < halfWidth; j++) {
                *rowRemapTableTempRow = i;
                *colRemapTableTempRow = halfWidth - j;

                rowRemapTableTempRow++;
                colRemapTableTempRow++;
            }
            for (; j < roiTensorPtrSrc[count].xywhROI.roiWidth; j++) {
                *rowRemapTableTempRow = i;
                *colRemapTableTempRow = j;

                rowRemapTableTempRow++;
                colRemapTableTempRow++;
            }
        }
    }
}
