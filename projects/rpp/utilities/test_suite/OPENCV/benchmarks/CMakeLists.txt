#[[
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
]]

cmake_minimum_required(VERSION 3.10)
project(opencv_benchmarking_script CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROCM Path
if(DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
elseif(ROCM_PATH)
    message("-- ${PROJECT_NAME}: INFO - ROCM_PATH Set -- ${ROCM_PATH}")
else()
    set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
endif()

# Set message options
if(NOT WIN32)
    string(ASCII 27 Esc)
    set(ColourReset "${Esc}[m")
    set(Red         "${Esc}[31m")
    set(Green       "${Esc}[32m")
    set(Yellow      "${Esc}[33m")
    set(White       "${Esc}[37m")
endif()

# Add CMake module paths
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake)
list(APPEND CMAKE_MODULE_PATH ${ROCM_PATH}/share/rpp/test/cmake)

# Find required packages
# First find OpenCV without components to detect version
find_package(OpenCV REQUIRED)

# OpenCV 5.x renamed calib3d to calib - verify correct components are available
if(OpenCV_VERSION VERSION_GREATER_EQUAL "5.0.0")
    # For OpenCV 5.x, check for 'calib' component
    find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib)
    message(STATUS "Using OpenCV ${OpenCV_VERSION} with calib component")
else()
    # For OpenCV 4.x, check for 'calib3d' component
    find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs calib3d)
    message(STATUS "Using OpenCV ${OpenCV_VERSION} with calib3d component")
endif()

find_package(OpenMP REQUIRED)

# Find libxlsxwriter (for Excel output)
# Support both system and local installations
if(DEFINED XLSXWRITER_ROOT)
    # Look for local installation first
    find_library(XLSXWRITER_LIBRARY
        NAMES xlsxwriter libxlsxwriter
        PATHS ${XLSXWRITER_ROOT}/lib
        NO_DEFAULT_PATH
    )
    find_path(XLSXWRITER_INCLUDE_DIR
        NAMES xlsxwriter.h
        PATHS ${XLSXWRITER_ROOT}/include
        NO_DEFAULT_PATH
    )
else()
    # Look for system installation
    find_library(XLSXWRITER_LIBRARY
    NAMES xlsxwriter libxlsxwriter
        PATHS /usr/lib /usr/local/lib ${ROCM_PATH}/lib $ENV{HOME}/.local/libxlsxwriter/lib
        PATH_SUFFIXES lib lib64
    )
    find_path(XLSXWRITER_INCLUDE_DIR
        NAMES xlsxwriter.h
        PATHS /usr/include /usr/local/include ${ROCM_PATH}/include $ENV{HOME}/.local/libxlsxwriter/include
    )
endif()


# Find RPP library
find_package(rpp 3.1.2 QUIET)
if(rpp_FOUND)
    message("-- ${White}${PROJECT_NAME}: rpp found with find_package(rpp QUIET)${ColourReset}")
    message("-- \t${White}rpp_INCLUDE_DIR -- ${rpp_INCLUDE_DIR}${ColourReset}")
    message("-- \t${White}rpp_LIBRARY -- ${rpp_LIBRARY}${ColourReset}")
    message("-- \t${White}rpp_BACKEND_TYPE -- ${rpp_BACKEND_TYPE}${ColourReset}")
else()
    message(FATAL_ERROR
        "-- ${Red}${PROJECT_NAME} requires RPP. Install RPP or set RPP_DIR to the RPP installation path${ColourReset}")
endif()

if(OpenCV_FOUND)
    message("-- ${Green}OpenCV found at ${OpenCV_INCLUDE_DIRS}${ColourReset}")
else()
    message(FATAL_ERROR "-- ${Red}${PROJECT_NAME} requires OpenCV${ColourReset}")
endif()

if(OpenMP_FOUND)
    message("-- ${Green}OpenMP found${ColourReset}")
else()
    message(FATAL_ERROR "-- ${Red}${PROJECT_NAME} requires OpenMP${ColourReset}")
endif()

if(XLSXWRITER_LIBRARY)
    message("-- ${Green}xlsxwriter found at ${XLSXWRITER_LIBRARY}${ColourReset}")
    if(XLSXWRITER_INCLUDE_DIR)
        message("-- ${Green}xlsxwriter headers at ${XLSXWRITER_INCLUDE_DIR}${ColourReset}")
    endif()
else()
    message(FATAL_ERROR
        "-- ${Red}${PROJECT_NAME} requires libxlsxwriter for Excel output.\n"
        "   Install with one of:\n"
        "   1. System-wide (requires sudo): sudo apt-get install libxlsxwriter-dev\n"
        "   2. Local (no sudo): Run ./run_benchmarking.sh which will build it locally\n"
        "   3. Manual local build: Set -DXLSXWRITER_ROOT=/path/to/libxlsxwriter${ColourReset}")
endif()

# Create executable from separated source files
add_executable(opencv_vs_rpp_host_hip_benchmarking
    benchmarks_utils.cpp
    rpp_host_benchmarks.cpp
    rpp_hip_benchmarks.cpp
    opencv_benchmarks.cpp
    opencv_vs_rpp_host_hip_benchmarking.cpp
)

# Include directories
set(BENCHMARK_INCLUDE_DIRS
        ${OpenCV_INCLUDE_DIRS}
        ${rpp_INCLUDE_DIR}
        ${rpp_INCLUDE_DIR}/rpp
        ${ROCM_PATH}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Add xlsxwriter include directory if found
if(XLSXWRITER_INCLUDE_DIR)
    list(APPEND BENCHMARK_INCLUDE_DIRS ${XLSXWRITER_INCLUDE_DIR})
endif()

target_include_directories(opencv_vs_rpp_host_hip_benchmarking
    PRIVATE
        ${BENCHMARK_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(opencv_vs_rpp_host_hip_benchmarking
    PRIVATE
        ${OpenCV_LIBS}
        OpenMP::OpenMP_CXX
        rpp::rpp
        ${XLSXWRITER_LIBRARY}
)

# Compiler options
target_compile_options(opencv_vs_rpp_host_hip_benchmarking PRIVATE -O3)

# Define HIP platform for AMD ROCm
target_compile_definitions(opencv_vs_rpp_host_hip_benchmarking PRIVATE __HIP_PLATFORM_AMD__ GPU_SUPPORT)

message("-- ${White}${PROJECT_NAME} configured successfully${ColourReset}")
message("-- \tOpenCV: ${OpenCV_VERSION}")
message("-- \tOpenMP: Found")
message("-- \tRPP: ${rpp_VERSION}")
message("-- \txlsxwriter: Found")
