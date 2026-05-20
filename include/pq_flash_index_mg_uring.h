// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#pragma once
#define _IO_INFO_COUNT
#include "common_includes.h"
#include "io_uring_aligned_file_reader.h"
#include "linux_aligned_file_reader.h"
#include "concurrent_queue.h"
#include "neighbor.h"
#include "parameters.h"
#include "percentile_stats.h"
#include "pq.h"
#include "utils.h"
#include "windows_customizations.h"
#include "scratch_uring.h"
#include "tsl/robin_map.h"
#include "tsl/robin_set.h"
#include "compressed_graph.h"
#ifdef FAST_DISKANN
#include "data_cache.h"
#endif
#include <optional>

// Inherit from base class
#include "pq_flash_index.h"

#define FULL_PRECISION_REORDER_MULTIPLIER 3

namespace diskann
{

template <typename T, typename LabelT = uint32_t>
class PQFlashIndexMGV2 : public PQFlashIndex<T, LabelT> {
public:
    DISKANN_DLLEXPORT PQFlashIndexMGV2(std::shared_ptr<AlignedFileReaderV2> &fileReader,
                                       diskann::Metric metric = diskann::Metric::L2);
    DISKANN_DLLEXPORT ~PQFlashIndexMGV2();

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT int load(diskann::MemoryMappedFiles &files, uint32_t num_threads, const char *index_prefix,
                               const char *mem_graph_path = nullptr, const bool compressed_graph = false,
                               const float reorder_ratio = -1);
#else
    // load compressed data, and obtains the handle to the disk-resident index
    DISKANN_DLLEXPORT int load(uint32_t num_threads, const char *index_prefix, const char *mem_graph_path = nullptr,
                               const bool compressed_graph = false, const float reorder_ratio = -1);
#endif
    DISKANN_DLLEXPORT int load_graph_to_memory(std::string graph_path);
    DISKANN_DLLEXPORT void set_reorder_ratio(float reorder_ratio);

#ifdef EXEC_ENV_OLS
    DISKANN_DLLEXPORT int load_from_separate_paths(diskann::MemoryMappedFiles &files, uint32_t num_threads,
                                                   const char *index_filepath, const char *pivots_filepath,
                                                   const char *compressed_filepath);
#else
    DISKANN_DLLEXPORT int load_from_separate_paths(uint32_t num_threads, const char *index_filepath,
                                                   const char *pivots_filepath, const char *compressed_filepath);
#endif

    DISKANN_DLLEXPORT void cached_beam_search_v2(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const bool use_reorder_data = false,
                                                 const bool use_compressed_graph = false, QueryStats *stats = nullptr);
    DISKANN_DLLEXPORT void cached_beam_search_v2(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const bool use_filter, const LabelT &filter_label,
                                                 const bool use_reorder_data = false,
                                                 const bool use_compressed_graph = false, QueryStats *stats = nullptr);
    DISKANN_DLLEXPORT void cached_beam_search_v2(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *indices, float *distances, const uint64_t beam_width,
                                                 const uint32_t io_limit, const bool use_reorder_data = false,
                                                 const bool use_compressed_graph = false, QueryStats *stats = nullptr);
    DISKANN_DLLEXPORT void cached_beam_search_v2(const T *query, const uint64_t k_search, const uint64_t l_search,
                                                 uint64_t *res_ids, float *res_dists, const uint64_t beam_width,
                                                 const bool use_filter, const LabelT &filter_label,
                                                 const uint32_t io_limit, const bool use_reorder_data = false,
                                                 const bool use_compressed_graph = false, QueryStats *stats = nullptr);

    DISKANN_DLLEXPORT uint32_t range_search(const T *query1, const double range, const uint64_t min_l_search,
                                            const uint64_t max_l_search, std::vector<uint64_t> &indices,
                                            std::vector<float> &distances, const uint64_t min_beam_width,
                                            QueryStats *stats = nullptr);

#ifdef FAST_DISKANN
    DISKANN_DLLEXPORT uint64_t cache_graph_by_priority(const uint64_t cache_budget_bytes, const std::string graph_priority_file);
    DISKANN_DLLEXPORT uint64_t cache_vectors_by_priority(const uint64_t cache_budget_bytes, const std::string vector_priority_file);

    DISKANN_DLLEXPORT template <bool UseCompressedGraph = false, bool ReorderCompressed = false>
    void cached_beam_search(const T *query, const uint64_t k_search, const uint64_t l_search, uint64_t *res_ids,
                            float *res_dists, const uint64_t beam_width, const bool use_filter,
                            const LabelT &filter_label, const uint32_t io_limit, const bool use_reorder_data = false,
                            QueryStats *stats = nullptr);
#endif

    // Hide base class reader, replace with V2 version
    std::shared_ptr<AlignedFileReaderV2> reader_v2;

protected:
    // Need to override
    DISKANN_DLLEXPORT void use_medoids_data_as_centroids();
    DISKANN_DLLEXPORT void setup_thread_data(uint64_t nthreads, uint64_t visited_reserve = 4096);

    // Inherited from base class: set_universal_label()

private:

    using mem_graph = std::vector<std::vector<unsigned>>;
    mem_graph _graph;

    uint64_t _width = 0;
    uint64_t _ep = 0;

    alignas(128) std::optional<CompressedGraph> _compressed_graph;

#ifdef FAST_DISKANN
    float _reorder_ratio = 0.0f;
#else
    float _reorder_ratio = 2.0f;
#endif

#ifdef FAST_DISKANN
    DataCache<T> _data_cache;
#endif

    ConcurrentQueue<SSDThreadDataV2<T> *> _thread_data_v2;

};
} // namespace diskann