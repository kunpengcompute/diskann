# Feature Guide

## Technical Architecture

```text
Original DiskANN v0.7.0 (x86_64)
├── 0001-diskann_0.7.0-optimize-neq.patch  # Non-equivalence optimization version (modifies search behavior)
│   ├── AArch64 NEON vectorization (IP/L2 distance calculation + PQ 8-bit table lookup)
│   ├── Data layout optimization (adjacency list resides in memory, and original vectors remain on SSDs)
│   ├── Asynchronous I/O pipelining (overlaps CPU computing with SSD reads)
│   ├── Reranking queue reduction (reorder_ratio × top-k, reducing I/O frequency)
│   └── Memory cache acceleration (cache_budget, high-frequency nodes/vector in memory)
└── 0002-diskann_0.7.0-optimize-eqv.patch  # Equivalence optimization version (maintains search behavior, acceleration only)
    ├── AArch64 NEON vectorization (IP/L2 distance calculation + PQ 8-bit table lookup)
    └── Asynchronous I/O pipelining (overlaps CPU computing with SSD reads)
```

## Non-Equivalence Optimization (0001-diskann_0.7.0-optimize-neq.patch)

This patch introduces intrusive modifications based on the open-source DiskANN v0.7.0, providing full-link performance optimization tailored for the Kunpeng platform (AArch64 NEON).

### Main Optimizations

- **SIMD vectorization**: replaces x86 AVX2 instructions with AArch64 NEON implementations, covering two core computing paths: Inner Product (IP)/L2 distance calculation and Product Quantization (PQ) 8-bit table lookup.
- **Data layout optimization**: migrates the graph index (adjacency list) from SSD to memory while retaining the original vectors on the SSD. The adjacency list, after lossless compression, co-resides in DRAM with the PQ quantized data, eliminating disk I/O during the loop unrolling of graph searches.
  - In the build phase, the independent memory adjacency list file is generated using `generate_mem_file=true`.
  - In the search phase, the file is loaded using `memory_graph_path`.
- **Asynchronous I/O pipelining**: When executing node expansion, the graph search thread pre-initiates asynchronous read requests for candidate points that are highly likely to become the nearest neighbors. This allows SSD data loading to overlap with CPU graph traversal, improving search throughput and response efficiency.
- **Reranking queue reduction**: The open-source DiskANN loads exact vectors and calculates exact distances for all candidate nodes within the search list `L`. This optimization introduces the `reorder_ratio` parameter to constrain the reranking queue length to `reorder_ratio × top-k`, significantly reducing redundant disk I/O operations.
  - Users can fine-tune the `reorder_ratio` hyperparameter based on recall rate requirements to achieve an optimal balance between precision and performance.

### File Changes

| File | Change Type | Description |
| ------ | --------- | ------ |
| `include/compressed_graph.h` | New | A compressed graph data structure that uses variable-length integer encoding to optimize storage. |
| `include/io_uring_aligned_file_reader.h` | New | A high-performance asynchronous file reader based on `io_uring` (LinuxAlignedFileReaderV2). |
| `include/pq_flash_index_mg_uring.h` | New | PQFlashIndexMGV2 index class using `io_uring`. |
| `include/scratch_uring.h` | New | Manages temporary query buffers for `io_uring` (ScratchStoreManagerV2). |
| `include/log.h` | New | A logging utility class to provide a unified log output interface. |
| `src/io_uring_aligned_file_reader.cpp` | New | LinuxAlignedFileReaderV2 implementation, which encapsulates `io_uring` system calls. |
| `src/pq_flash_index_mg_uring.cpp` | New | PQFlashIndexMGV2 implementation, which supports incremental construction and compressed graphs. |
| `src/scratch_uring.cpp` | New | ScratchStoreManagerV2 implementation, which manages the query temporary memory pool. |

### Data Layout Optimization

The open-source DiskANN stores both the graph index (adjacency list) and original vectors on the SSD. Consequently, each node expansion during graph traversal triggers disk I/O to fetch neighbor information.

#### Build Phase

During index construction, by setting `generate_mem_file=true`, the system generates an additional, independent memory adjacency list file (`*_mem.index.vamana.comp`). This file contains solely the graph topology (node IDs and neighbor lists) without original vector data. After lossless compression, its size is drastically smaller than the complete index file, allowing it to co-reside with PQ quantized data within a limited DRAM footprint.

#### Search Phase

During queries, the memory adjacency list is loaded via `memory_graph_path`. Throughout the loop unrolling of the graph search, PQ coarse distance calculations and neighbor traversals are executed entirely in memory. SSD read requests are dispatched only when exact L2 distances are required. This shifts the search I/O pattern from "disk reads per node expansion" to "disk reads exclusively during reranking," drastically reducing disk accesses.

### Asynchronous I/O Pipelining

Even with data layout optimization, the search phase still requires loading original vectors from the SSD to compute exact distances. The open-source DiskANN relies on synchronous I/O, causing the CPU to sit idle while waiting for data retrieval from the SSD.

#### Pipelining Mechanism

The optimization implements asynchronous I/O to enable concurrent pipelining of computation and I/O tasks:

1. Following the PQ coarse distance calculation, the graph search thread filters out candidate nodes that are promising nearest-neighbor choices from the candidate queue.
2. Asynchronous read requests (`io_uring`/`libaio`) are dispatched in advance for these candidate nodes without blocking the current thread.
3. The CPU proceeds to the next round of graph traversal and PQ distance calculation, running in parallel with the SSD data transfer.
4. When an exact distance is needed, the system verifies whether the asynchronous I/O operation has completed. Ready data is consumed immediately, while incomplete operations are awaited.

#### Benefits

The pipelined execution of CPU computation and SSD reads eliminates the idle waiting gaps inherent in synchronous I/O. This improvement is especially pronounced in IOPS-constrained environments (e.g., rate-limited to 1,000K IOPS with 8 KB block size), significantly boosting multi-threaded search throughput.

### Reranking Queue Reduction

In the open-source DiskANN retrieval flow, a Beam Search is followed by an exact distance reranking of all visited candidate nodes within the search list `L`. The reranking of each candidate node requires fetching its original vector from the SSD, resulting in an I/O overhead that scales linearly with the number of candidates.

#### Reduction Mechanism

In production environments, the length of the search list `L` (typically 200 to 1000) is substantially larger than the final requested `top-K` results (typically 10). Consequently, the vast majority of candidates in the reranking queue are redundant and will be filtered out of the final results.

This optimization introduces the `reorder_ratio` hyperparameter to scale down the reranking queue length from `L` to `reorder_ratio × K`:

- Original vectors are loaded, and exact L2 distances are calculated, only for candidate nodes whose PQ coarse distances rank within the top `reorder_ratio × K`.
- The remaining candidate nodes are discarded instantly, incurring zero I/O overhead.

### Memory Cache Acceleration

Despite asynchronous I/O pipelining, fetching original vectors from the SSD during the search phase remains mandatory. For frequently accessed hotspot nodes (such as those close to the entry point), repeatedly reading them from the SSD for every query is highly inefficient.

#### Cache Mechanism

Using the `cache_budget` parameter, users can allocate a dedicated cache budget (in GB). Upon loading the index, the system caches high-priority graph nodes and their corresponding vectors into memory:

1. The caching sequence is prioritized based on a node priority file (`graph_priority_file`), ensuring that the most frequently accessed nodes are cached first.
2. During searches, the system first checks whether a candidate node is cached. Upon a cache hit, data is read directly from memory, bypassing SSD I/O.
3. Cache misses continue to follow the asynchronous I/O pathway to fetch data from the SSD.

#### Instructions

- Use the command-line argument `--cache_budget 2.0` to specify a 2 GB cache budget.
- Use `--graph_priority_file` to point to the node priority file path (which can be generated using the `calculate_hops_from_entry` tool).
- Setting `cache_budget = 0` executes the cache code path without actual caching, while `cache_budget = -1` (default) bypasses the cache path.

#### Benefits

In environments with sufficient memory, caching high-frequency nodes significantly reduces disk access frequency, cuts down tail latency, and enhances search throughput under highly concurrent workloads.

## Equivalence Optimization (0002-diskann_0.7.0-optimize-eqv.patch)

As a subset of the non-equivalence optimization, equivalence optimization includes only those optimization entries that do not alter the search behavior.

### Optimization Item

| Optimization Item | Description |
| -------- | ------ |
| AArch64 NEON vectorization | Replaces IP/L2 distance calculations and PQ 8-bit table lookups with NEON intrinsics. |
| Asynchronous I/O pipelining | Leverages `io_uring` during the reranking phase to read original vectors asynchronously, achieving overlap between CPU computation and I/O. |
