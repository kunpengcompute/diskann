# API参考

本文档详细说明鲲鹏优化相对于原始DiskANN开源代码的接口变动，包括对外接口build_disk_index与search_disk_index的参数新增，以及新增的search_disk_index_cache接口。

## build_disk_index

### 接口说明

构建索引，生成PQ表、IO友好的完整向量等文件。

### 接口定义

```cpp
template <typename T, typename LabelT>
int build_disk_index(
    const char *dataFilePath,
    const char *indexFilePath,
    const char *indexBuildParameters,
    diskann::Metric compareMetric,
    bool use_opq,
    const std::string &codebook_prefix,
    bool use_filters,
    const std::string &label_file,
    const std::string &universal_label,
    const uint32_t filter_threshold,
    const uint32_t Lf,
    const bool generate_mem_file
);
```

### 参数说明

| 参数名称 | 数据类型 | 描述 | 取值范围 |
|---|---|---|---|
| `dataFilePath` | `const char *` | 输入数据集文件，二进制形式。 | 非空，真实路径 |
| `indexFilePath` | `const char *` | 输出，索引路径前缀，如`/mnt/data/my_ann_index`。 | 非空，真实路径 |
| `indexBuildParameters` | `const char *` | 输入参数组合，包括图邻居数R、search节点候选队列长度L、search阶段内存预算B、构建内存预算M、线程数num_threads、PQ索引字节数disk_PQ、是否在数据文件中包含全精度数据reorder、构建索引的字节数PQ、PQ量化维度QD。 | 5~9个由空格隔开的参数组合 |
| `compareMetric` | `diskann::Metric` | 距离计算函数。 | `{l2, mips, cosine}`，推荐 `l2` |
| `use_opq` | `bool` | 是否使用OPQ算法进行PQ计算。 | `{true, false}` |
| `codebook_prefix` | `const std::string` | 预训练码本的路径前缀。 | 默认为空 |
| `use_filters` | `bool` | 是否进入filter分支，由 `universal_label` 决定。 | `{true, false}` |
| `label_file` | `const std::string` | 用于构建过滤索引的标签文件路径（txt格式）。文件中每行对应一个图节点，包含逗号分隔的过滤器标签。 | 默认为空 |
| `universal_label` | `const std::string` | 通用标签，仅在构建过滤索引时与标签文件结合使用。若某图节点拥有所有相关标签，可为其分配一个特殊的通用过滤器，而无需列出所有标签。通用标签应在标签文件中分配给节点，DiskANN 不会自动分配。 | 默认为空 |
| `filter_threshold` | `const uint32_t` | 每个节点最多拥有的标签数量，超过此值将拆分节点。 | 默认为0 |
| `Lf` | `const uint32_t` | 构建过滤点的复杂度，更高的值生成更好的图。 | 默认为0 |
| `generate_mem_file` | `const bool` | **新增参数** ，是否单独生成内存邻接表。 | `{true, false}` |

### 返回值

| 返回值 | 说明 |
|---|---|
| `0` | 正常返回 |
| `-1` | 失败 |

---

## search_disk_index

### 接口说明

并行批量query搜索接口，多线程并发执行。加载PQ表、内存邻接表等至内存，并在查询时按需进行IO。

### 接口定义

```cpp
template <typename T, typename LabelT = uint32_t>
int search_disk_index(
    diskann::Metric &metric,
    const std::string &index_path_prefix,
    std::string &memory_graph_path,
    const std::string &result_output_prefix,
    const std::string &query_file,
    std::string &gt_file,
    const uint32_t num_threads,
    const uint32_t recall_at,
    const uint32_t beamwidth,
    const uint32_t num_nodes_to_cache,
    const uint32_t search_io_limit,
    const float reorder_ratio,
    const std::vector<uint32_t> &Lvec,
    const float fail_if_recall_below,
    const std::vector<std::string> &query_filters,
    const bool use_reorder_data = false
);
```

### 参数说明

| 参数名称 | 数据类型 | 描述 | 取值范围 |
|---|---|---|---|
| `metric` | `diskann::Metric` | 距离计算函数。 | `{l2, mips, cosine}`，推荐 `l2` |
| `index_path_prefix` | `const std::string` | 索引路径前缀，如`/mnt/data/my_ann_index`。 | 非空，真实路径 |
| `memory_graph_path` | `std::string` | **新增参数** ，邻接表路径。 | 非空，真实路径 |
| `result_output_prefix` | `const std::string` | 结果输出路径前缀。 | 非空，真实路径 |
| `query_file` | `const std::string` | 批次查询文件路径。 | 非空，真实路径 |
| `gt_file` | `std::string` | 查询对应的正确结果文件路径，仅验证recall时使用。 | 无限制，可以为空 |
| `num_threads` | `const uint32_t` | 线程数。 | 大于1的整数，默认为系统参数 |
| `recall_at` | `const uint32_t` | 召回率（输出）。 | 无限制 |
| `beamwidth` | `const uint32_t` | 搜索时一次load索引的个数。 | 正整数，默认为2 |
| `num_nodes_to_cache` | `const uint32_t` | 缓存的节点数量，大于数据集10%会自动缩减到10%；Fast分支置为0。 | 自然数，默认为0 |
| `search_io_limit` | `const uint32_t` | 单个查询的最大IO数量。 | 正整数，默认为最大值 |
| `reorder_ratio` | `const float` | **新增参数**， 控制精确距离计算的次数，有效值为 `reorder_ratio × top-k`。 | 大于0，默认为2.0 |
| `Lvec` | `const std::vector<uint32_t>` | 搜索列表的长度，主要控制粗距离计算表的列表。 | 正整数列表 |
| `fail_if_recall_below` | `const float` | recall可接受的最小值，低于此值返回-1。 | 0~100，默认为 0 |
| `query_filters` | `const std::vector<std::string>` | 查询的筛选标签。1个标签对所有查询生效，多个标签与每个查询一一对应（当前不涉及）。 | 无限制，默认为空 |
| `use_reorder_data` | `const bool` | 索引中是否包含全精度数据，仅在SSD上与压缩数据结合使用。 | `{true, false}`，默认为 `false` |

### 返回值

| 返回值 | 说明 |
|---|---|
| `0` | 正常返回 |
| `-1` | 失败（含recall低于`fail_if_recall_below`的情况） |

## search_disk_index_cache

### 接口说明

基于内存缓存的搜索接口。将高优先级的图节点和向量缓存到内存中，减少磁盘IO次数，提升搜索性能。当命令行参数`--cache_budget >= 0`时进入此分支。

### 接口定义

```cpp
template <typename T>
int search_disk_index_cache(
    diskann::Metric &metric,
    const std::string &index_path_prefix,
    const std::string &result_output_prefix,
    const std::string &query_file,
    std::string &gt_file,
    const uint32_t num_threads,
    const uint32_t recall_at,
    const float reorder_ratio,
    const uint32_t beamwidth,
    const double cache_budget_gb,
    const std::string &graph_priority_file,
    const uint32_t search_io_limit,
    const std::vector<uint32_t> &Lvec,
    const float fail_if_recall_below,
    const uint32_t repeat_count,
    const bool use_reorder_data = false
);
```

### 参数说明

| 参数名称 | 数据类型 | 描述 | 取值范围 |
|---|---|---|---|
| `metric` | `diskann::Metric` | 距离计算函数。 | `{l2, mips, cosine}`，推荐`l2` |
| `index_path_prefix` | `const std::string` | 索引路径前缀。 | 非空，真实路径 |
| `result_output_prefix` | `const std::string` | 结果输出路径前缀。 | 非空，真实路径 |
| `query_file` | `const std::string` | 批次查询文件路径。 | 非空，真实路径 |
| `gt_file` | `std::string` | 正确结果文件路径，仅验证recall时使用。 | 无限制，可以为空 |
| `num_threads` | `const uint32_t` | 线程数。 | 大于1的整数，默认为系统参数 |
| `recall_at` | `const uint32_t` | 召回率top-K。 | 正整数 |
| `reorder_ratio` | `const float` | 控制精确距离计算的次数，有效值为`reorder_ratio × top-k`。 | 大于0，默认为2.0 |
| `beamwidth` | `const uint32_t` | 搜索时一次load索引的个数。 | 正整数，默认为1 |
| `cache_budget_gb` | `const double` | **新增参数**，缓存预算（GB）。`0`表示不缓存但使用 cache 代码路径，`>0`表示实际缓存预算。 | `>= 0` |
| `graph_priority_file` | `const std::string` | **新增参数**，图节点优先级文件路径，用于决定哪些节点优先缓存。 | 默认为 `"null"` |
| `search_io_limit` | `const uint32_t` | 单个查询的最大IO数量。 | 正整数，默认为最大值 |
| `Lvec` | `const std::vector<uint32_t>` | 搜索列表长度。 | 正整数列表 |
| `fail_if_recall_below` | `const float` | recall可接受的最小值，低于此值返回-1。 | 0~100，默认为0 |
| `repeat_count` | `const uint32_t` | **新增参数**，重复搜索次数，用于取平均性能。 | 正整数，默认为7 |
| `use_reorder_data` | `const bool` | 索引中是否包含全精度数据。 | `{true, false}`，默认为 `false` |

### 返回值

| 返回值 | 说明 |
|---|---|
| `0` | 正常返回 |
| `-1` | 失败（含recall低于`fail_if_recall_below`的情况） |
