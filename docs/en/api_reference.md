# API Reference

This document describes the API changes of the optimized DiskANN code compared with the original open-source code, including the new parameters of the external APIs `build_disk_index` and `search_disk_index`, and the new API `search_disk_index_cache`.

## build_disk_index

### Interface Description

Builds an index and generates files such as the PQ table and I/O-friendly complete vector.

### Interface Definition

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

### Parameters

| Parameter | Data Type | Description | Value Range |
| --- | --- | --- | --- |
| `dataFilePath` | `const char` | Input dataset file in binary format. | A non-empty, actual path. |
| `indexFilePath` | `const char` | Output, the index path prefix, for example, `/mnt/data/my_ann_index`. | A non-empty, actual path. |
| `indexBuildParameters` | `const char` | Input parameter combination, including the number of graph neighbors `R`, length `L` of the candidate node queue for search, memory budget `B` for the search phase, memory budget `M` for the build phase, thread count `num_threads`, PQ index bytes `disk_PQ`, `reorder` (indicating whether to include full-precision data in the data file), `PQ` indicating the number of PQ bytes used for building the index, and PQ quantization dimension `QD`. | A combination of 5 to 9 parameters separated by spaces. |
| `compareMetric` | `diskann::Metric` | Distance calculation function. | `{l2, mips, cosine}`. `l2` is recommended. |
| `use_opq` | `bool` | Indicates whether to use the OPQ algorithm for PQ calculation. | `{true, false}` |
| `codebook_prefix` | `const std::string` | Path prefix of the pre-trained codebook. | The default value is empty. |
| `use_filters` | `bool` | Indicates whether to enter the filter branch, which is determined by `universal_label`. | `{true, false}` |
| `label_file` | `const std::string` | Path to the label file (in TXT format) used for constructing the filtered index. Each line in the file corresponds to a graph node and contains filter labels separated by commas (,). | The default value is empty. |
| `universal_label` | `const std::string` | A universal label used in combination with the label file only during the construction of the filtered index. If a graph node possesses all relevant labels, a special universal filter label can be assigned to it instead of listing every individual label. This universal label must be assigned to nodes within the label file. DiskANN will not automatically assign it to any node. | The default value is empty. |
| `filter_threshold` | `const uint32_t` | Maximum number of labels that a single node can possess. If the number of labels exceeds this value, the node will be split into multiple cloned nodes. | The default value is <code>0</code>. |
| `Lf` | `const uint32_t` | Complexity of constructing the filtered index. A higher value generates a higher-quality graph. | The default value is <code>0</code>. |
| `generate_mem_file` | `const bool` | **New parameter**, which specifies whether to generate an independent memory adjacency list. | `{true, false}` |

### Return Value

| Return Value | Description |
| --- | --- |
| `0` | Normal execution |
| `-1` | Failure |

---

## search_disk_index

### Interface Description

Performs parallel batch query searches, executing multiple threads concurrently. It loads PQ tables and memory adjacency lists into memory, performing disk I/O operations as required during the query phase.

### Interface Definition

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

### Parameters

| Parameter | Data Type | Description | Value Range |
| --- | --- | --- | --- |
| `metric` | `diskann::Metric` | Distance calculation function. | `{l2, mips, cosine}`. `l2` is recommended. |
| `index_path_prefix` | `const std::string` | Index path prefix, for example, `/mnt/data/my_ann_index`. | A non-empty, actual path. |
| `memory_graph_path` | `std::string` | **New parameter**, which indicates the path to the adjacency list. | A non-empty, actual path. |
| `result_output_prefix` | `const std::string` | Prefix of the result output path. | A non-empty, actual path. |
| `query_file` | `const std::string` | Path to the batch query file. | A non-empty, actual path. |
| `gt_file` | `std::string` | Path to the corresponding ground-truth result file, used only for recall verification. | Unrestricted. It can be left empty. |
| `num_threads` | `const uint32_t` | Thread count. | An integer greater than 1. The default value is determined by system parameters. |
| `recall_at` | `const uint32_t` | Recall rate (output). | Unrestricted. |
| `beamwidth` | `const uint32_t` | Beam width (number of nodes loaded at a time) during search. | Positive integer. The default value is `2`. |
| `num_nodes_to_cache` | `const uint32_t` | Number of cached nodes. If the value exceeds 10% of the dataset size, it will be automatically capped at 10%. For the Fast branch, set this parameter to `0`. | Natural number. The default value is `0`. |
| `search_io_limit` | `const uint32_t` | Maximum number of I/Os in a single query. | Positive integer. The default value is the maximum value. |
| `reorder_ratio` | `const float` | **New parameter**, which controls the number of exact distance calculations, with a valid value of `reorder_ratio × top-k`. | Greater than 0. The default value is `2.0`. |
| `Lvec` | `const std::vector<uint32_t>` | Length of the search list, which mainly controls the list for coarse distance calculations. | List of positive integers. |
| `fail_if_recall_below` | `const float` | Minimum acceptable recall rate. If the query recall drops below this threshold, the function returns `-1`. | 0 to 100. The default value is `0`. |
| `query_filters` | `const std::vector<std::string>` | Filter labels used for querying. If a single label is provided, it applies to all queries. If multiple labels are provided, they must map one-to-one to each query (this multi-label mapping is not currently supported). | Unrestricted. It is left empty by default. |
| `use_reorder_data` | `const bool` | Specifies whether the index includes full-precision data. This is used in combination with compressed data exclusively for SSD-resident indexes. | `{true, false}`. The default value is `false`. |

### Return Value

| Return Value | Description |
| --- | --- |
| `0` | Normal execution |
| `-1` | Failure (including the case where recall is lower than `fail_if_recall_below`) |

## search_disk_index_cache

### Interface Description

A search interface based on memory caching. High-priority graph nodes and vectors are cached in memory to reduce the number of disk I/Os and improve search performance. This branch is entered when `--cache_budget >= 0`.

### Interface Definition

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

### Parameters

| Parameter | Data Type | Description | Value Range |
| --- | --- | --- | --- |
| `metric` | `diskann::Metric` | Distance calculation function. | `{l2, mips, cosine}`. `l2` is recommended. |
| `index_path_prefix` | `const std::string` | Index path prefix. | A non-empty, actual path. |
| `result_output_prefix` | `const std::string` | Prefix of the result output path. | A non-empty, actual path. |
| `query_file` | `const std::string` | Path to the batch query file. | A non-empty, actual path. |
| `gt_file` | `std::string` | Path to the ground-truth result file, used only for recall verification. | Unrestricted. It can be left empty. |
| `num_threads` | `const uint32_t` | Thread count. | An integer greater than 1. The default value is determined by system parameters. |
| `recall_at` | `const uint32_t` | Recall rate `top-K`. | Positive integer. |
| `reorder_ratio` | `const float` | Controls the number of exact distance calculations. The effective value is `reorder_ratio × top-k`. | Greater than 0. The default value is `2.0`. |
| `beamwidth` | `const uint32_t` | Beam width (number of nodes loaded at a time) during search. | Positive integer. The default value is `1`. |
| `cache_budget_gb` | `const double` | **New parameter**, which indicates the cache budget (GB). The value `0` disables caching but executes the cache code path, while `> 0` specifies the actual cache budget. | `>= 0` |
| `graph_priority_file` | `const std::string` | **New parameter**, which indicates the path to the graph node priority file for determining the caching priority of different nodes. | The default value is **''null''**. |
| `search_io_limit` | `const uint32_t` | Maximum number of I/Os in a single query. | Positive integer. The default value is the maximum value. |
| `Lvec` | `const std::vector<uint32_t>` | Length of the search list. | List of positive integers. |
| `fail_if_recall_below` | `const float` | Minimum acceptable recall rate. If the query recall drops below this threshold, the function returns `-1`. | 0 to 100. The default value is `0`. |
| `repeat_count` | `const uint32_t` | **New parameter**, which indicates the number of search repetitions, used to calculate the average performance. | Positive integer. The default value is `7`. |
| `use_reorder_data` | `const bool` | Indicates whether the index contains full-precision data. | `{true, false}`. The default value is `false`. |

### Return Value

| Return Value | Description |
| --- | --- |
| `0` | Normal execution |
| `-1` | Failure (including the case where recall is lower than `fail_if_recall_below`). |
