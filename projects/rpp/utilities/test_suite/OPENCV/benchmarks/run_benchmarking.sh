#!/bin/bash

# MIT License
#
# Copyright (c) 2019 - 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Complete setup script for OpenCV benchmark WITHOUT SUDO ACCESS
# This script builds OpenCV 5.0.0 locally and doesn't require sudo

set -e

# Default values
NUM_THREADS=""
PERF_RUNS=""
WARMUP_RUNS=""
CLEAN_BUILD=1  # Default to fresh build
OPENCV_INSTALL_DIR="$HOME/.local/opencv-5.0.0"

# Parse command-line arguments
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --threads <N>          Number of threads to use (default: auto-detect)"
    echo "  -n, --num-runs <N>         Number of benchmark runs (default: 100)"
    echo "  -w, --warmup-runs <N>      Number of warmup runs (default: 50)"
    echo "  --opencv-dir <PATH>        Custom OpenCV installation directory"
    echo "                             (default: \$HOME/.local/opencv-5.0.0)"
    echo "  --no-clean                 Skip clean build (use existing build)"
    echo "  --skip-opencv              Skip OpenCV installation (use existing)"
    echo "  -h, --help                 Display this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Fresh build, install OpenCV locally"
    echo "  $0 -t 64                              # Build with 64 threads"
    echo "  $0 -t 32 -n 200 -w 100                # 32 threads, 200 runs, 100 warmup runs"
    echo "  $0 --opencv-dir ~/my-opencv           # Use custom OpenCV location"
    echo "  $0 --skip-opencv                      # Skip OpenCV build (already installed)"
    echo ""
    exit 0
}

SKIP_OPENCV=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--threads)
            NUM_THREADS="$2"
            shift 2
            ;;
        -n|--num-runs)
            PERF_RUNS="$2"
            shift 2
            ;;
        -w|--warmup-runs)
            WARMUP_RUNS="$2"
            shift 2
            ;;
        --opencv-dir)
            OPENCV_INSTALL_DIR="$2"
            shift 2
            ;;
        --no-clean)
            CLEAN_BUILD=0
            shift
            ;;
        --skip-opencv)
            SKIP_OPENCV=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Error: Unknown option: $1"
            usage
            ;;
    esac
done

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "OpenCV Benchmark - No-Sudo Setup"
echo "========================================"
echo ""
echo "Configuration:"
echo "  OpenCV Install: $OPENCV_INSTALL_DIR"
[ -n "$NUM_THREADS" ] && echo "  Threads: $NUM_THREADS"
[ -n "$PERF_RUNS" ] && echo "  Runs: $PERF_RUNS"
[ -n "$WARMUP_RUNS" ] && echo "  Warmup Runs: $WARMUP_RUNS"
echo ""

# Check for required system dependencies
echo "Checking system dependencies..."
echo "--------------------------------------"
MISSING_DEPS=()

# Check for cmake
if ! command -v cmake &> /dev/null; then
    MISSING_DEPS+=("cmake")
fi

# Check for g++ (build-essential)
if ! command -v g++ &> /dev/null; then
    MISSING_DEPS+=("build-essential")
fi

# Check for libgomp1
if ! ldconfig -p 2>/dev/null | grep -q libgomp.so; then
    MISSING_DEPS+=("libgomp1")
fi

# Check for libxlsxwriter-dev
if ! dpkg -l 2>/dev/null | grep -q "^ii.*libxlsxwriter-dev"; then
    if ! pkg-config --exists xlsxwriter 2>/dev/null; then
        MISSING_DEPS+=("libxlsxwriter-dev")
    fi
fi

# Check for python3-pip
if ! command -v pip3 &> /dev/null; then
    MISSING_DEPS+=("python3-pip")
fi

# Check for python3-venv
if ! python3 -m venv --help &> /dev/null; then
    MISSING_DEPS+=("python3-venv")
fi

# Report missing dependencies
if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo ""
    echo "⚠ WARNING: Missing required system dependencies:"
    for dep in "${MISSING_DEPS[@]}"; do
        echo "  - $dep"
    done
    echo ""
    echo "These packages are required to build and run the benchmark."
    echo "Please install them using one of these methods:"
    echo ""
    echo "1. If you have sudo access:"
    echo "   sudo apt-get update"
    echo "   sudo apt-get install -y ${MISSING_DEPS[*]}"
    echo ""
    echo "2. If you don't have sudo access:"
    echo "   Contact your system administrator to install: ${MISSING_DEPS[*]}"
    echo ""
    echo "Press Enter to continue anyway (may fail), or Ctrl+C to exit..."
    read -r
else
    echo "✓ All required system dependencies found"
fi
echo ""

# Function to check if OpenCV 5.0.0 is already installed
check_opencv_5() {
    local install_dir="$1"

    if [ -f "$install_dir/lib/libopencv_core.so.5.0.0" ] && \
       [ -f "$install_dir/lib/libopencv_imgproc.so.5.0.0" ] && \
       [ -f "$install_dir/lib/libopencv_imgcodecs.so.5.0.0" ] && \
       [ -f "$install_dir/lib/libopencv_calib.so.5.0.0" ]; then
        return 0  # Found
    else
        return 1  # Not found
    fi
}

# Function to build and install OpenCV 5.0.0 locally
install_opencv_5_local() {
    local version="5.0.0"
    local install_dir="$1"

    echo ""
    echo "Step 1: Building OpenCV $version locally..."
    echo "--------------------------------------"
    echo "Install location: $install_dir"
    echo ""

    # Create installation directory
    mkdir -p "$install_dir"

    # Create temporary build directory
    BUILD_DIR=$(mktemp -d /tmp/opencv-build.XXXXXX)
    cd "$BUILD_DIR"

    echo "Downloading OpenCV $version..."
    wget -q --show-progress -O opencv.zip \
        https://github.com/opencv/opencv/archive/refs/tags/${version}.zip

    echo "Extracting..."
    unzip -q opencv.zip
    cd opencv-${version}
    mkdir -p build && cd build

    echo "Configuring with CMake..."
    echo "(This may take a few minutes...)"

    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX="$install_dir" \
          -D BUILD_SHARED_LIBS=ON \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D BUILD_DOCS=OFF \
          -D BUILD_opencv_apps=OFF \
          -D BUILD_opencv_python2=OFF \
          -D BUILD_opencv_python3=OFF \
          -D BUILD_opencv_java=OFF \
          -D WITH_CUDA=OFF \
          -D WITH_OPENCL=OFF \
          -D WITH_IPP=OFF \
          -D WITH_TBB=OFF \
          -D WITH_EIGEN=OFF \
          -D WITH_V4L=OFF \
          -D WITH_GTK=OFF \
          -D WITH_QT=OFF \
          -D BUILD_opencv_core=ON \
          -D BUILD_opencv_imgproc=ON \
          -D BUILD_opencv_imgcodecs=ON \
          -D BUILD_opencv_calib=ON \
          -D BUILD_opencv_features2d=ON \
          -D BUILD_opencv_flann=ON \
          -D BUILD_opencv_photo=ON \
          -D BUILD_opencv_video=ON \
          -D BUILD_opencv_videoio=ON \
          -D BUILD_opencv_highgui=ON \
          .. 2>&1 | grep -E "OpenCV|Found|Building|Install"

    echo ""
    echo "Building OpenCV (using $(nproc) cores)..."
    echo "(This will take 10-20 minutes depending on your system...)"
    make -j$(nproc)

    echo ""
    echo "Installing to $install_dir..."
    make install

    # Clean up
    cd "$SCRIPT_DIR"
    rm -rf "$BUILD_DIR"

    echo ""
    echo "✓ OpenCV $version installed successfully to:"
    echo "  $install_dir"
    echo ""
}

# Check if we need to install OpenCV
if [ $SKIP_OPENCV -eq 0 ]; then
    if check_opencv_5 "$OPENCV_INSTALL_DIR"; then
        echo "✓ OpenCV 5.0.0 already installed at:"
        echo "  $OPENCV_INSTALL_DIR"
        echo ""
    else
        install_opencv_5_local "$OPENCV_INSTALL_DIR"
    fi
else
    echo "Skipping OpenCV installation (--skip-opencv specified)"
    if check_opencv_5 "$OPENCV_INSTALL_DIR"; then
        echo "✓ Using existing OpenCV at: $OPENCV_INSTALL_DIR"
    else
        echo "⚠ WARNING: OpenCV not found at $OPENCV_INSTALL_DIR"
        echo "  Build may fail. Remove --skip-opencv to install."
    fi
    echo ""
fi

# Set up Python environment and install dependencies
VENV_DIR="$SCRIPT_DIR/.venv"
USE_VENV=0

echo "Step 2: Setting up Python environment..."
echo "--------------------------------------"

# Check if virtual environment exists and is valid
if [ -f "$VENV_DIR/bin/activate" ] && [ -f "$VENV_DIR/bin/python3" ]; then
    echo "✓ Using existing virtual environment at $VENV_DIR"
    USE_VENV=1
elif [ -d "$VENV_DIR" ]; then
    # Directory exists but is incomplete - remove it
    echo "Removing incomplete virtual environment..."
    rm -rf "$VENV_DIR"
fi

# Try to create virtual environment if not already valid
if [ $USE_VENV -eq 0 ]; then
    if python3 -m venv "$VENV_DIR" 2>/dev/null; then
        echo "✓ Virtual environment created at $VENV_DIR"
        USE_VENV=1
    else
        echo "⚠ Could not create virtual environment (python3-venv not available)"
        echo "  Falling back to user installation (--user)"
        # Clean up failed attempt
        [ -d "$VENV_DIR" ] && rm -rf "$VENV_DIR"
        USE_VENV=0
    fi
fi

# Install Python dependencies (Pillow and numpy)
if [ $USE_VENV -eq 1 ]; then
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    # Check and install dependencies
    DEPS_TO_INSTALL=()
    python3 -c "import PIL" 2>/dev/null || DEPS_TO_INSTALL+=("Pillow")
    python3 -c "import numpy" 2>/dev/null || DEPS_TO_INSTALL+=("numpy")

    if [ ${#DEPS_TO_INSTALL[@]} -gt 0 ]; then
        echo "Installing Python packages in virtual environment: ${DEPS_TO_INSTALL[*]}"
        pip3 install --quiet "${DEPS_TO_INSTALL[@]}"
        echo "✓ Python packages installed"
    else
        echo "✓ Python packages already installed in virtual environment"
    fi
else
    # Fall back to --user installation
    DEPS_TO_INSTALL=()
    python3 -c "import PIL" 2>/dev/null || DEPS_TO_INSTALL+=("Pillow")
    python3 -c "import numpy" 2>/dev/null || DEPS_TO_INSTALL+=("numpy")

    if [ ${#DEPS_TO_INSTALL[@]} -gt 0 ]; then
        echo "Installing Python packages with --user flag: ${DEPS_TO_INSTALL[*]}"
        pip3 install --user --quiet "${DEPS_TO_INSTALL[@]}"
        echo "✓ Python packages installed"
    else
        echo "✓ Python packages already installed"
    fi
fi
echo ""

# Check if dataset exists
if [ ! -d "input_images_dataset" ] || [ -z "$(ls -A input_images_dataset 2>/dev/null)" ]; then
    echo "Step 3: Generating test dataset..."
    echo "--------------------------------------"
    # Use Python from virtual environment if available, otherwise system python3
    if [ $USE_VENV -eq 1 ]; then
        "$VENV_DIR/bin/python3" generate_test_dataset.py
    else
        python3 generate_test_dataset.py
    fi
    echo ""
else
    echo "✓ Dataset already exists ($(ls -1 input_images_dataset | wc -l) images)"
    echo ""
fi

# Deactivate virtual environment if it was activated
if [ $USE_VENV -eq 1 ]; then
    deactivate
fi

# Build benchmark with local OpenCV
echo "Step 4: Building benchmark..."
echo "--------------------------------------"

# Set up environment for CMake to find local OpenCV
export OpenCV_DIR="$OPENCV_INSTALL_DIR/lib/cmake/opencv5"
export PKG_CONFIG_PATH="$OPENCV_INSTALL_DIR/lib/pkgconfig:$PKG_CONFIG_PATH"
export LD_LIBRARY_PATH="$OPENCV_INSTALL_DIR/lib:$LD_LIBRARY_PATH"

if [ $CLEAN_BUILD -eq 1 ]; then
    echo "Performing fresh build..."
    echo "  OpenCV_DIR: $OpenCV_DIR"
    rm -rf build
    mkdir -p build
    cd build
    cmake -D OpenCV_DIR="$OpenCV_DIR" ..
    make -j$(nproc)
    cd ..
    echo "✓ Fresh build complete"
else
    echo "Using existing build (incremental)..."
    mkdir -p build
    cd build
    [ ! -f Makefile ] && cmake -D OpenCV_DIR="$OpenCV_DIR" ..
    make -j$(nproc)
    cd ..
    echo "✓ Incremental build complete"
fi
echo ""

# Create a wrapper script that sets LD_LIBRARY_PATH
WRAPPER_SCRIPT="build/run_benchmark_wrapper.sh"
cat > "$WRAPPER_SCRIPT" << EOF
#!/bin/bash
# Auto-generated wrapper to set library paths
export LD_LIBRARY_PATH="$OPENCV_INSTALL_DIR/lib:\$LD_LIBRARY_PATH"
exec "\$(dirname "\$0")/opencv_vs_rpp_host_hip_benchmarking" "\$@"
EOF
chmod +x "$WRAPPER_SCRIPT"

# Build command with optional arguments
cmd="$WRAPPER_SCRIPT"
cmd_args=""

[ -n "$NUM_THREADS" ] && cmd_args="$cmd_args --threads $NUM_THREADS"
[ -n "$PERF_RUNS" ] && cmd_args="$cmd_args --num-runs $PERF_RUNS"
[ -n "$WARMUP_RUNS" ] && cmd_args="$cmd_args --warmup-runs $WARMUP_RUNS"

runs_text="${PERF_RUNS:-100}"
warmup_text="${WARMUP_RUNS:-50}"
threads_text="${NUM_THREADS:-auto-detect}"

echo "========================================"
echo "Starting Benchmark"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Threads: $threads_text"
echo "  Runs: $runs_text"
echo "  Warmup Runs: $warmup_text"
echo "  OpenCV: $OPENCV_INSTALL_DIR"
echo ""
echo "This will take several minutes..."
echo ""

# Set library path for this session
export LD_LIBRARY_PATH="$OPENCV_INSTALL_DIR/lib:$LD_LIBRARY_PATH"

./build/opencv_vs_rpp_host_hip_benchmarking $cmd_args

echo ""
echo "========================================"
echo "Benchmark Complete!"
echo "========================================"
echo ""
echo "Note: To run the benchmark manually later, use:"
echo "  export LD_LIBRARY_PATH=$OPENCV_INSTALL_DIR/lib:\$LD_LIBRARY_PATH"
echo "  ./build/opencv_vs_rpp_host_hip_benchmarking"
echo ""
