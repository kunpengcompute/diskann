# 特性指南

## 技术架构

```text
原始DiskANN v0.7.0 (x86_64)
├── 0001-diskann_0.7.0-optimize-neq.patch  # 全量优化版本（改变搜索行为）
│   ├── ARM64 NEON向量化（IP/L2距离计算 + PQ 8bit查表）
│   ├── 数据布局优化（邻接表驻留内存，原始向量留SSD）
│   ├── 异步IO流水化（CPU计算与SSD读取并行重叠）
│   ├── 精排队列缩减（reorder_ratio × top-k，降低IO次数）
│   └── 内存缓存加速（cache_budget，高频节点/向量驻留内存）
└── 0002-diskann_0.7.0-optimize-eqv.patch  # 等价优化版本（不改变搜索行为，仅加速）
    ├── ARM64 NEON向量化（IP/L2距离计算 + PQ 8bit查表）
    └── 异步IO流水化（CPU计算与SSD读取并行重叠）
```

## 全量优化（0001-diskann_0.7.0-optimize-neq.patch）

基于开源DiskANN v0.7.0做侵入式修改，针对鲲鹏950平台（ARM64 NEON）进行全链路性能优化。

### 主要优化内容

- **SIMD向量化**：将x86 AVX2指令替换为ARM64 NEON实现，覆盖IP/L2距离计算与PQ 8bit查表两条核心计算路径。
- **数据布局优化**：将图索引（邻接表）从SSD搬入内存，原始向量保留在SSD。邻接表经无损压缩后与PQ量化数据共同驻留DRAM，使图检索的循环展开无需磁盘IO。
  - 构建阶段通过`generate_mem_file=true`生成独立的内存邻接表文件。
  - 搜索阶段通过`memory_graph_path`参数加载该文件。
- **异步IO流水化**：图搜索线程在执行节点扩展时，提前对有希望成为最近邻的候选点发起异步读取请求，使SSD数据加载与CPU图遍历计算相互重叠，提升检索吞吐与响应效率。
- **精排队列缩减**：开源DiskANN会对搜索列表L中所有候选节点加载精确向量并计算精确距离。优化后通过`reorder_ratio`参数控制精排队列长度为`reorder_ratio × top-k`，大幅减少冗余IO次数。
  - 用户可根据召回率需求调节`reorder_ratio`超参数，在精度与性能间取得平衡。

### 变更文件

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `include/compressed_graph.h` | 新增 | 压缩图数据结构，使用变长整数编码优化存储 |
| `include/io_uring_aligned_file_reader.h` | 新增 | 基于io_uring的高性能异步文件读取器（LinuxAlignedFileReaderV2） |
| `include/pq_flash_index_mg_uring.h` | 新增 | 使用io_uring的PQFlashIndexMGV2索引类 |
| `include/scratch_uring.h` | 新增 | io_uring查询临时缓冲区管理（ScratchStoreManagerV2） |
| `include/log.h` | 新增 | 日志工具类，提供统一的日志输出接口 |
| `src/io_uring_aligned_file_reader.cpp` | 新增 | LinuxAlignedFileReaderV2实现，封装io_uring系统调用 |
| `src/pq_flash_index_mg_uring.cpp` | 新增 | PQFlashIndexMGV2实现，支持增量构建和压缩图 |
| `src/scratch_uring.cpp` | 新增 | ScratchStoreManagerV2实现，管理查询临时内存池 |

### 数据布局优化详解

开源DiskANN将图索引（邻接表）与原始向量共同存储在SSD上，每次图遍历展开节点时都需要通过磁盘IO加载邻居信息。

#### 构建阶段

在索引构建时，通过设置`generate_mem_file=true`参数，额外生成一份独立的内存邻接表文件（`*_mem.index.vamana.comp`）。该文件仅包含图结构（节点ID与邻居列表），不含原始向量数据，经无损压缩后体积远小于完整索引文件，可在有限DRAM中与PQ量化数据共同驻留。

#### 搜索阶段

搜索时通过`memory_graph_path`参数加载内存邻接表。图检索的循环展开过程中，PQ粗距离计算与邻居遍历完全在内存中完成，仅在需要计算精确L2距离时才发起SSD读取请求。这将搜索阶段的IO模式从"每次展开都读盘"转变为"仅精排时读盘"，大幅减少磁盘访问次数。

### 异步IO流水化详解

在数据布局优化的基础上，搜索阶段仍需从SSD加载原始向量以计算精确距离。开源DiskANN采用同步IO方式，CPU在等待SSD返回数据期间处于空闲状态。

#### 流水化机制

优化后采用异步IO实现计算与IO的重叠流水：

1. 图搜索线程在PQ粗距离计算后，从候选队列中筛选出有希望成为最近邻的节点。
2. 对这些候选节点提前发起异步读取请求（`io_uring` / `libaio`），不阻塞当前线程。
3. CPU继续执行下一轮图遍历与PQ距离计算，与SSD数据传输并行进行。
4. 当需要精确距离时，检查异步IO是否完成，已就绪的数据直接使用，未就绪的等待完成。

#### 效果

CPU计算与SSD读取形成流水线，消除了同步IO的等待间隙，在IOPS受限场景下（如限速1000K 8k IOPS）尤为明显，可显著提升多线程检索吞吐。

### 精排队列缩减详解

开源DiskANN的检索流程中，Beam Search结束后会对搜索列表L中所有访问过的候选节点进行精确距离重排序。每个候选节点的精排都需要从SSD加载原始向量，IO开销与候选数量线性相关。

#### 缩减机制

实际检索中，搜索列表L的长度（通常200–1000）远大于最终需要的top-K结果（通常10），精排队列中绝大部分候选点是冗余的，不会进入最终结果。

优化后引入`reorder_ratio`超参数，将精排队列长度从L缩减为`reorder_ratio × K`：

- 仅对PQ粗距离排名前`reorder_ratio × K`的候选节点加载原始向量并计算精确L2距离。
- 其余候选节点直接丢弃，不产生IO开销。

### 内存缓存加速详解

在异步IO流水化的基础上，搜索阶段仍需从SSD加载原始向量。对于访问频率高的热点节点（如入口点附近的节点），每次查询都重复读取SSD是低效的。

#### 缓存机制

通过`cache_budget`参数指定缓存预算（GB），系统在加载索引后将高优先级的图节点和对应向量缓存到内存中：

1. 根据`graph_priority_file`（节点优先级文件）确定缓存顺序，优先缓存被访问频率最高的节点。
2. 搜索时先检查候选节点是否已缓存，命中则直接从内存读取，跳过SSD IO。
3. 未命中的节点仍走异步IO路径从SSD加载。

#### 使用方式

- 命令行参数`--cache_budget 2.0`指定2GB缓存预算。
- `--graph_priority_file`指定节点优先级文件路径（可通过`calculate_hops_from_entry`工具生成）。
- `cache_budget = 0`表示不缓存但使用cache代码路径，`cache_budget = -1`（默认）走非cache路径。

#### 效果

在内存充足的场景下，通过缓存高频节点可显著减少SSD访问次数，降低尾延迟，提升高并发场景下的搜索吞吐。

---

## 等价优化（0002-diskann_0.7.0-optimize-eqv.patch）

等价优化是全量优化的子集，仅包含**不改变搜索行为**的优化项。

### 包含的优化

| 优化项 | 说明 |
|--------|------|
| ARM64 NEON向量化 | IP/L2距离计算、PQ 8bit查表替换为NEON intrinsics |
| 异步IO流水化 | 精排阶段使用io_uring异步读取原始向量，CPU计算与IO重叠 |
