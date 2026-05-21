# 安装指南

## 已验证环境

为保证您可以顺利安全地使用DiskANN，请确保所使用的环境信息在已验证环境范围内。

**表 1** DiskANN已验证环境<a id="DiskANN已验证环境"></a>

| 操作系统 | CPU | 内存 | 编译器 |
|---|---|---|---|
| openEuler 24.03 LTS SP3 | 鲲鹏950 7592C | 24×64GB | GCC 12.3.1 |
| Debian 12 | 鲲鹏950 7592C | 24×64GB | LLVM 16.0.6 |

## 编译安装

从GitHub获取DiskANN开源代码，安装必要的依赖工具、库，从GitCode获取基于鲲鹏平台优化后的Patch然后重新编译DiskANN，以便应用优化后特性，降低计算时延，提升计算效率。

1. 获取DiskANN开源代码，标签为**0.7.0**。假设代码存放于“/path/to/DiskANN“。

    ```bash
    git clone --branch 0.7.0 --single-branch https://github.com/microsoft/DiskANN.git
    ```

2. 获取基于鲲鹏优化的补丁文件，标签为**v1.0.0**。假设存放于“/path/to/diskann-patch“。

    ```bash
    git clone --branch v1.0.0 https://gitcode.com/boostkit/diskann.git diskann-patch
    ```

    >![](public_sys-resources/icon-note.gif) **说明：** 
    >鲲鹏优化补丁文件描述如下，请根据需要自行选择：
    >- 0001-diskann\_1.8.0-optimize-neq.patch：全量优化补丁，性能最优，保证精度，但不保证Top-K的值或顺序与原生完全一致。
    >- 0002-diskann\_1.8.0-optimize-eqv.patch：等价优化补丁，保证Top-K的值与顺序与原生保持完全一致。

3. 安装LLVM 16.0.6。

    1. 下载并解压LLVM 16.0.6源码。

       ```bash
       wget -O llvm-project-16.0.6.src.tar.xz https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/llvm-project-16.0.6.src.tar.xz --no-check-certificate
       tar xf llvm-project-16.0.6.src.tar.xz
       ```

    2. 编译并安装LLVM。

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

    3. 配置LLVM环境变量（临时生效）。

       ```bash
       export CXX=/opt/llvm-16.0.6/bin/clang++
       export CC=/opt/llvm-16.0.6/bin/clang
       export PATH=/opt/llvm-16.0.6/bin:$PATH
       ```

4. DiskANN依赖数学库，从[Github仓](https://github.com/OpenMathLib/OpenBLAS.git)下载开源OpenBLAS源代码，标签为**v0.3.29**。保存在编译机器可访问的路径中，假设位于“/path/to/OpenBLAS-0.3.29“。

    ```bash
    git clone --branch v0.3.29 --single-branch https://github.com/OpenMathLib/OpenBLAS.git
    ```

5. 编译源代码获取libopenblas.so。

    ```bash
    cd /path/to/OpenBLAS-0.3.29/OpenBLAS
    make
    make install
    ```

    >![](public_sys-resources/icon-note.gif) **说明：** 
    >您可通过**make install PREFIX=/path/to/openblas/install**设置“/path/to/openblas/install”以指定安装路径，默认安装路径为“/opt/OpenBLAS”。

6. 安装全量补丁文件0001-diskann\_0.7.0-optimize-neq.patch或等价优化补丁0002-diskann\_0.7.0-optimize-eqv.patch。

    ```bash
    cd /path/to/DiskANN
    patch -p1 < /path/to/diskann-patch/0001-diskann_0.7.0-optimize-neq.patch
    # patch -p1 < /path/to/diskann-patch/0002-diskann_0.7.0-optimize-eqv.patch
    ```

    合入全量优化补丁后DiskANN完整的目录结构如下所示：

    ```text
    DiskANN
    ├─ diskann/
    │   ├─ CMakeLists.txt                          // 顶层构建配置
    │   ├─ setup.py                                // Python包安装配置
    │   ├─ unit_tester.sh                          // 单元测试脚本
    │   │
    │   ├─ include/                                // 头文件目录
    │   │   ├─ abstract_data_store.h               // 抽象数据存储接口
    │   │   ├─ abstract_graph_store.h              // 抽象图存储接口
    │   │   ├─ abstract_index.h                    // 抽象索引接口
    │   │   ├─ aligned_file_reader.h               // 对齐文件读取器接口
    │   │   ├─ ann_exception.h                     // 异常处理
    │   │   ├─ cached_io.h                         // 缓存I/O
    │   │   ├─ common_includes.h                   // 通用头文件
    │   │   ├─ compressed_graph.h                  // 压缩图结构
    │   │   ├─ defaults.h                          // 默认参数配置
    │   │   ├─ disk_utils.h                        // 磁盘工具函数
    │   │   ├─ distance.h                          // 距离计算接口
    │   │   ├─ filter_utils.h                      // 过滤工具
    │   │   ├─ index.h                             // 内存索引主接口
    │   │   ├─ io_uring_aligned_file_reader.h      // io_uring文件读取器
    │   │   ├─ linux_aligned_file_reader.h         // Linux对齐文件读取器
    │   │   ├─ log.h                               // 日志工具
    │   │   ├─ logger.h                            // 日志记录器
    │   │   ├─ math_utils.h                        // 数学工具函数
    │   │   ├─ neighbor.h                          // 邻居结构定义
    │   │   ├─ parameters.h                        // 参数配置
    │   │   ├─ partition.h                         // 数据分区
    │   │   ├─ pq.h                                // 乘积量化(PQ)
    │   │   ├─ pq_flash_index.h                    // PQ磁盘索引接口
    │   │   ├─ pq_flash_index_mg_uring.h           // PQ磁盘索引(io_uring版本)
    │   │   ├─ scratch.h                           // 临时缓冲区管理
    │   │   ├─ scratch_uring.h                     // io_uring临时缓冲区
    │   │   ├─ timer.h                             // 计时器
    │   │   ├─ types.h                             // 类型定义
    │   │   ├─ utils.h                             // 通用工具函数
    │   │   ├─ v2/                                 // V2版本接口
    │   │   └─ windows_aligned_file_reader.h       // Windows对齐文件读取器
    │   │
    │   ├─ src/                                    // 源文件目录
    │   │   ├─ abstract_data_store.cpp             // 抽象数据存储实现
    │   │   ├─ abstract_graph_store.cpp            // 抽象图存储实现
    │   │   ├─ abstract_index.cpp                  // 抽象索引实现
    │   │   ├─ disk_utils.cpp                      // 磁盘工具实现
    │   │   ├─ distance.cpp                        // 距离计算实现
    │   │   ├─ filter_utils.cpp                    // 过滤工具实现
    │   │   ├─ index.cpp                           // 内存索引实现
    │   │   ├─ io_uring_aligned_file_reader.cpp    // io_uring文件读取器实现
    │   │   ├─ linux_aligned_file_reader.cpp       // Linux对齐文件读取器实现
    │   │   ├─ logger.cpp                          // 日志记录器实现
    │   │   ├─ math_utils.cpp                      // 数学工具实现
    │   │   ├─ partition.cpp                       // 数据分区实现
    │   │   ├─ pq.cpp                              // 乘积量化实现
    │   │   ├─ pq_flash_index.cpp                  // PQ磁盘索引实现
    │   │   ├─ pq_flash_index_mg_uring.cpp         // PQ磁盘索引(io_uring版本)实现
    │   │   ├─ scratch.cpp                         // 临时缓冲区实现
    │   │   ├─ scratch_uring.cpp                   // io_uring临时缓冲区实现
    │   │   ├─ utils.cpp                           // 通用工具实现
    │   │   └─ windows_aligned_file_reader.cpp     // Windows对齐文件读取器实现
    │   │
    │   ├─ perf_test/                              // 性能测试目录
    │   └─ tests/                                  // 功能测试目录
    │
    └─ README.md
    ```

7. 安装编译依赖。

    ```bash
    apt install make cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-all-dev liburing-dev
    ```

8. 编译DiskANN代码获取二进制文件，文件位于bin文件夹下。需启用鲲鹏优化宏以获得性能提升。

    ```bash
    cd /path/to/DiskANN
    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release .. -DFAST_DISKANN=ON
    make -j
    ```

   - 若您选择使用全量优化补丁0001-diskann\_0.7.0-optimize-neq.patch，可选择开启以下宏获得性能提升：
     - **-DFAST_DISKANN=ON**：性能最优，保证精度，但不保证Top-K的值或顺序与原生完全一致；
   - 若您选择使用等价优化补丁0002-diskann\_0.7.0-optimize-eqv.patch，可选择开启以下宏获得性能提升：
     - **-DFAST_DISKANN=ON**：保证Top-K的值与顺序与原生保持完全一致。
