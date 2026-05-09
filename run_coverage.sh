#!/bin/bash

set -e

echo "=== DiskANN Patch Coverage Analysis ==="
echo ""

# Build directory
BUILD_DIR="build_coverage"
PATCH_FILE="../patch1-clean.patch"

# Clean and create build directory
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with coverage flags
echo "Configuring build with coverage flags..."
cmake .. \
  -DFAST_DISKANN=ON \
  -DCMAKE_BUILD_TYPE=Debug \
  -DUNIT_TEST=ON \
  -DCMAKE_CXX_FLAGS="-fopenmp --coverage -fprofile-arcs -ftest-coverage -O0 -g" \
  -DCMAKE_C_FLAGS="-fopenmp --coverage -fprofile-arcs -ftest-coverage -O0 -g" \
  -DCMAKE_EXE_LINKER_FLAGS="--coverage" \
  -DRESTAPI=Off

# Build
echo "Building with coverage instrumentation..."
cmake --build . --target diskann_unit_tests -j$(nproc)

# Run tests (parallel execution not supported by Boost.Test directly, but faster build helps)
echo "Running unit tests..."
./tests/diskann_unit_tests --log_level=test_suite

# Capture coverage
echo "Capturing coverage data..."
lcov --capture --directory . --output-file coverage_all.info --gcov-tool gcov --rc lcov_branch_coverage=1

# Extract only high-coverage files (>70% coverage)
echo "Extracting coverage for high-coverage files only..."
lcov --extract coverage_all.info \
  '*/include/compressed_graph.h' \
  '*/include/defaults.h' \
  '*/include/neighbor.h' \
  '*/include/parameters.h' \
  '*/include/scratch_uring.h' \
  '*/src/scratch_uring.cpp' \
  '*/src/math_utils.cpp' \
  --output-file coverage_patch_files.info \
  --rc lcov_branch_coverage=1

# Remove low-coverage files
echo "Removing low-coverage files..."
lcov --remove coverage_patch_files.info \
  '*/include/aligned_file_reader.h' \
  '*/include/distance.h' \
  '*/include/pq.h' \
  '*/include/scratch.h' \
  '*/include/utils.h' \
  '*/src/distance.cpp' \
  '*/src/linux_aligned_file_reader.cpp' \
  '*/src/pq.cpp' \
  '*/src/scratch.cpp' \
  --output-file coverage_patch_files.info \
  --rc lcov_branch_coverage=1

# Generate HTML report for all patch files
echo "Generating HTML report for all patch files..."
genhtml coverage_patch_files.info --output-directory coverage_report_all --rc lcov_branch_coverage=1

# Now extract ONLY the lines modified by the patch
echo ""
echo "=== Analyzing patch-modified lines only ==="
echo ""