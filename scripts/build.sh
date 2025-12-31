#!/bin/bash
# Fresnel build script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
BUILD_TYPE="${BUILD_TYPE:-Release}"

echo "=== Fresnel Build Script ==="
echo "Project root: $PROJECT_ROOT"
echo "Build type: $BUILD_TYPE"
echo ""

# Check for required tools
check_tool() {
    if ! command -v "$1" &> /dev/null; then
        echo "ERROR: $1 is required but not found"
        exit 1
    fi
}

check_tool cmake
check_tool ninja || check_tool make

# Check for GLSL compiler
if ! command -v glslangValidator &> /dev/null && ! command -v glslc &> /dev/null; then
    echo "WARNING: No GLSL compiler found (glslangValidator or glslc)"
    echo "         Install vulkan-sdk or shaderc for shader compilation"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
echo "Configuring with CMake..."
if command -v ninja &> /dev/null; then
    cmake -G Ninja \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        "$PROJECT_ROOT"
else
    cmake \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        "$PROJECT_ROOT"
fi

# Build
echo ""
echo "Building..."
cmake --build . --parallel

# Link compile_commands.json for IDE support
if [ -f "$BUILD_DIR/compile_commands.json" ]; then
    ln -sf "$BUILD_DIR/compile_commands.json" "$PROJECT_ROOT/compile_commands.json"
fi

echo ""
echo "Build complete! Binaries in: $BUILD_DIR"
echo ""
echo "To run tests:  $BUILD_DIR/test_vulkan_compute"
echo "To run main:   $BUILD_DIR/fresnel"
