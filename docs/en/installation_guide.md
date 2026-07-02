# Installation guide

## Verified Environment

To use DiskANN smoothly and securely, ensure that your environment is one of the verified environments.

[**Table 1**] Verified environment of DiskANN<a id="verified-environment-of-diskann"></a>

| OS | CPU | Memory | Compiler |
| --- | --- | --- | --- |
| openEuler 24.03 LTS SP3 | Kunpeng 950 processor | 24 × 64 GB | GCC 12.3.1 |
| Debian 12 | Kunpeng 950 processor | 24 × 64 GB | LLVM 16.0.6 |

## Compilation and Installation

Obtain the open-source code of DiskANN from the GitHub repository. Install necessary dependencies and libraries. Then, obtain the patch optimized based on the Kunpeng platform from GitCode to recompile DiskANN. This enables Kunpeng-specific optimizations, significantly reducing computation latency while boosting computational efficiency.

1. Obtain the open-source code of DiskANN with the tag `0.7.0`. Assume that the code is stored in `/path/to/DiskANN`.

    ```bash
    git clone --branch 0.7.0 --single-branch https://github.com/microsoft/DiskANN.git
    ```

2. Obtain the Kunpeng-optimized patch file with the tag `v1.0.0`. Assume that the file is stored in `/path/to/diskann-patch`.

    ```bash
    git clone --branch v1.0.0 https://gitcode.com/boostkit/diskann.git diskann-patch
    ```

    >**Note:**
    >The following explains the patch files optimized for Kunpeng. Select a patch file as required.
    >- `0001-diskann_0.7.0-optimize-neq.patch`: non-equivalence optimization patch. It delivers optimal performance and ensures precision, but does not guarantee that the values or sequence of top K results are completely consistent with the original version.
    >- `0002-diskann_0.7.0-optimize-eqv.patch`: equivalence optimization patch. It ensures that the values and sequence of top K results are completely consistent with the original version.

3. Install LLVM 16.0.6.

    1. Download and decompress the LLVM 16.0.6 source package.

       ```bash
       wget -O llvm-project-16.0.6.src.tar.xz https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/llvm-project-16.0.6.src.tar.xz --no-check-certificate
       tar xf llvm-project-16.0.6.src.tar.xz
       ```

    2. Compile and install LLVM.

       ```bash
       cd llvm-project-16.0.6.src
       mkdir -p build && cd build
       cmake -G Ninja ../llvm \
         -DCMAKE_BUILD_TYPE=Release \
         -DCMAKE_INSTALL_PREFIX=/opt/llvm-16.0.6 \
         -DLLVM_TARGETS_TO_BUILD="AArch64" \
         -DLLVM_ENABLE_PROJECTS="clang;lld;clang-tools-extra;openmp" \
         -DLLVM_ENABLE_TERMINFO=ON \
         -DLLVM_ENABLE_ZLIB=ON
       ninja -j$(nproc)
       ```

    3. Configure the LLVM environment variables (temporarily effective).

       ```bash
       export CXX=/opt/llvm-16.0.6/bin/clang++
       export CC=/opt/llvm-16.0.6/bin/clang
       export PATH=/opt/llvm-16.0.6/bin:$PATH
       ```

4. DiskANN depends on the math library. Download the open-source OpenBLAS source code from the [GitHub repository](https://github.com/OpenMathLib/OpenBLAS) using the `v0.3.29` tag. Save the file to a path accessible to the compiler, such as `/path/to/OpenBLAS-0.3.29`.

    ```bash
    git clone --branch v0.3.29 --single-branch https://github.com/OpenMathLib/OpenBLAS.git
    ```

5. Compile the source code to obtain **libopenblas.so**.

    ```bash
    cd /path/to/OpenBLAS-0.3.29/OpenBLAS
    make
    make install
    ```

    >**Note:**
    >You can run the `make install PREFIX=/path/to/openblas/install` command to specify the installation path `/path/to/openblas/install`. The default installation path is `/opt/OpenBLAS`.

6. Apply the non-equivalence optimization patch (`0001-diskann_0.7.0-optimize-neq.patch`) or the equivalence optimization patch (`0002-diskann_0.7.0-optimize-eqv.patch`).

    ```bash
    cd /path/to/DiskANN
    patch -p1 < /path/to/diskann-patch/0001-diskann_0.7.0-optimize-neq.patch
    # patch -p1 < /path/to/diskann-patch/0002-diskann_0.7.0-optimize-eqv.patch
    ```

    After the non-equivalence optimization patch is applied, the complete directory structure of DiskANN is as follows:

    ```text
    DiskANN
    ├─ diskann/
    │   ├─ CMakeLists.txt                          // Top-level build configuration
    │   ├─ setup.py                                // Python package installation configuration
    │   ├─ unit_tester.sh                          // Unit test script
    │   │
    │   ├─ include/                                // Header file directory
    │   │   ├─ abstract_data_store.h               // Abstract data store interface
    │   │   ├─ abstract_graph_store.h              // Abstract graph store interface
    │   │   ├─ abstract_index.h                    // Abstract index interface
    │   │   ├─ aligned_file_reader.h               // Aligned file reader interface
    │   │   ├─ ann_exception.h                     // Exception handling
    │   │   ├─ cached_io.h                         // Cache I/O
    │   │   ├─ common_includes.h                   // Common header file
    │   │   ├─ compressed_graph.h                  // Compressed graph structure
    │   │   ├─ defaults.h                          // Default parameter configuration
    │   │   ├─ disk_utils.h                        // Disk utility functions
    │   │   ├─ distance.h                          // Distance calculation interface
    │   │   ├─ filter_utils.h                      // Filtering utilities
    │   │   ├─ index.h                             // Main in-memory index interface
    │   │   ├─ io_uring_aligned_file_reader.h      // io_uring aligned file reader
    │   │   ├─ linux_aligned_file_reader.h         // Linux aligned file reader
    │   │   ├─ log.h                               // Log tool
    │   │   ├─ logger.h                            // Logger
    │   │   ├─ math_utils.h                        // Math utilities
    │   │   ├─ neighbor.h                          // Neighbor structure definition
    │   │   ├─ parameters.h                        // Parameter configuration
    │   │   ├─ partition.h                         // Data partitioning
    │   │   ├─ pq.h                                // Product Quantization (PQ)
    │   │   ├─ pq_flash_index.h                    // PQ disk-resident index interface
    │   │   ├─ pq_flash_index_mg_uring.h           // PQ disk-resident index interface (io_uring version)
    │   │   ├─ scratch.h                           // Scratch buffer manager
    │   │   ├─ scratch_uring.h                     // io_uring scratch buffer
    │   │   ├─ timer.h                             // Timer
    │   │   ├─ types.h                             // Type definition
    │   │   ├─ utils.h                             // Common utilities
    │   │   ├─ v2/                                 // V2 interfaces
    │   │   └─ windows_aligned_file_reader.h       // Windows aligned file reader
    │   │
    │   ├─ src/                                    // Source file directory
    │   │   ├─ abstract_data_store.cpp             // Abstract data store implementation
    │   │   ├─ abstract_graph_store.cpp            // Abstract graph store implementation
    │   │   ├─ abstract_index.cpp                  // Abstract index implementation
    │   │   ├─ disk_utils.cpp                      // Disk utility implementation
    │   │   ├─ distance.cpp                        // Distance calculation implementation
    │   │   ├─ filter_utils.cpp                    // Filtering utility implementation
    │ │ ├─ index.cpp // Memory index implementation
    │   │   ├─ io_uring_aligned_file_reader.cpp    // io_uring aligned file reader implementation
    │   │   ├─ linux_aligned_file_reader.cpp       // Linux aligned file reader implementation
    │   │   ├─ logger.cpp                          // Logger implementation
    │   │   ├─ math_utils.cpp                      // Math utility implementation
    │   │   ├─ partition.cpp                       // Data partitioning implementation
    │   │   ├─ pq.cpp                              // Product quantization implementation
    │   │   ├─ pq_flash_index.cpp                  // PQ disk-resident index implementation
    │   │   ├─ pq_flash_index_mg_uring.cpp         // PQ disk-resident index (io_uring version) implementation
    │   │   ├─ scratch.cpp                         // Scratch buffer implementation
    │   │   ├─ scratch_uring.cpp                   // io_uring scratch buffer implementation
    │   │   ├─ utils.cpp                           // General utility implementation
    │   │   └─ windows_aligned_file_reader.cpp     // Windows aligned file reader implementation
    │   │
    │   ├─ perf_test/                              // Performance test directory
    │   └─ tests/                                  // Function test directory
    │
    └─ README.md
    ```

7. Install the compilation dependencies.

    ```bash
    apt install make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev liburing-dev
    ```

8. Compile the DiskANN source code to generate the binaries in the `bin` directory. Ensure that the Kunpeng optimization macro is enabled to achieve optimal performance.

    ```bash
    cd /path/to/DiskANN
    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release .. -DFAST_DISKANN=ON
    make -j
    ```

   - If you apply the non-equivalence optimization patch (`0001-diskann_0.7.0-optimize-neq.patch`), you can enable the following macro to boost performance:
     - `-DFAST_DISKANN=ON`: Delivers optimal performance and ensures high precision, but does not guarantee that the values or sequence of the Top-K results will be completely identical to the original implementation.
   - If you apply the equivalence optimization patch `0002-diskann_0.7.0-optimize-eqv.patch`, you can enable the following macro to boost performance:
     - `-DFAST_DISKANN=ON`: Guarantees that both the values and sequence of the Top-K results remain completely identical to the original implementation.
