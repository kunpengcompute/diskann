// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include "pq_flash_index_mg_uring.h"

#include <mutex>
#include <sys/mman.h>

#include "common_includes.h"
#include "cosine_similarity.h"
#include "pq.h"
#include "pq_scratch.h"
#include "timer.h"

#define READ_U64(stream, val) stream.read((char *)&val, sizeof(uint64_t))
#define READ_U32(stream, val) stream.read((char *)&val, sizeof(uint32_t))


namespace diskann
{
static std::shared_ptr<AlignedFileReader> _dummy_reader;

template <typename T, typename LabelT>
PQFlashIndexMGV2<T, LabelT>::PQFlashIndexMGV2(std::shared_ptr<AlignedFileReaderV2> &fileReader, diskann::Metric m)
    : PQFlashIndex<T, LabelT>(_dummy_reader, m), reader_v2(fileReader)
{
    if (!_dummy_reader) {
        _dummy_reader = std::shared_ptr<AlignedFileReader>(new LinuxAlignedFileReader());
    }
    diskann::cout << "PQFlashIndexMGV2: io_uring version" << std::endl;
    diskann::Metric metric_to_invoke = m;
    if (m == diskann::Metric::COSINE || m == diskann::Metric::INNER_PRODUCT)
    {
        if (std::is_floating_point<T>::value)
        {
            diskann::cout << "Since data is floating point, we assume that it has been appropriately pre-processed "
                             "(normalization for cosine, and convert-to-l2 by adding extra dimension for MIPS). So we "
                             "shall invoke an l2 distance function."
                          << std::endl;
            metric_to_invoke = diskann::Metric::L2;
        }
        else
        {
            diskann::cerr << "WARNING: Cannot normalize integral data types."
                          << " This may result in erroneous results or poor recall."
                          << " Consider using L2 distance with integral data types." << std::endl;
        }
    }

    this->_dist_cmp.reset(diskann::get_distance_function<T>(metric_to_invoke));
    this->_dist_cmp_float.reset(diskann::get_distance_function<float>(metric_to_invoke));
}

template <typename T, typename LabelT> PQFlashIndexMGV2<T, LabelT>::~PQFlashIndexMGV2()
{
#ifndef EXEC_ENV_OLS
    if (this->data != nullptr)
    {
        delete[] this->data;
    }
#endif

    if (this->_centroid_data != nullptr)
        aligned_free(this->_centroid_data);
    // delete backing bufs for nhood and coord cache
    if (this->_nhood_cache_buf != nullptr)
    {
        delete[] this->_nhood_cache_buf;
        diskann::aligned_free(this->_coord_cache_buf);
    }

    // MGV2 uses _thread_data_v2 and reader_v2 for cleanup
    // After cleanup, set _load_flag = false to avoid duplicate cleanup in base class destructor
    if (this->_load_flag)
    {
        diskann::cout << "PQFlashIndexMGV2: Clearing scratch" << std::endl;
        ScratchStoreManagerV2<SSDThreadDataV2<T>> manager(this->_thread_data_v2);
        manager.destroy();
        this->reader_v2->deregister_all_threads();
        reader_v2->close();
        this->_load_flag = false;  // Avoid duplicate cleanup in base class destructor
    }
    if (this->_pts_to_label_offsets != nullptr)
    {
        delete[] this->_pts_to_label_offsets;
    }
    if (this->_pts_to_label_counts != nullptr)
    {
        delete[] this->_pts_to_label_counts;
    }
    if (this->_pts_to_labels != nullptr)
    {
        delete[] this->_pts_to_labels;
    }
}


template <typename T, typename LabelT>
void PQFlashIndexMGV2<T, LabelT>::setup_thread_data(uint64_t nthreads, uint64_t visited_reserve)
{
    diskann::cout << "Setting up thread-specific contexts for nthreads: " << nthreads << std::endl;
#pragma omp parallel for num_threads((int)nthreads)
    for (int64_t thread = 0; thread < (int64_t)nthreads; thread++)
    {
#pragma omp critical
        {
            // Create V2 thread data (without ctx for now, as it's not used in current code path)
            SSDThreadDataV2<T> *data_v2 = new SSDThreadDataV2<T>(this->_aligned_dim, visited_reserve);
            this->_thread_data_v2.push(data_v2);

            // Create regular thread data for compatibility with read_nodes
            // Only register with this->reader (the parent class reader), not reader_v2
            SSDThreadData<T> *data = new SSDThreadData<T>(this->_aligned_dim, visited_reserve);
            this->reader->register_thread();
            data->ctx = this->reader->get_ctx();
            this->_thread_data.push(data);
        }
    }

    // Fallback: if OpenMP didn't execute, do it manually
    if (this->_thread_data_v2.size() == 0)
    {
        for (uint64_t thread = 0; thread < nthreads; thread++)
        {
            // Create V2 thread data
            SSDThreadDataV2<T> *data_v2 = new SSDThreadDataV2<T>(this->_aligned_dim, visited_reserve);
            this->_thread_data_v2.push(data_v2);

            // Create regular thread data for compatibility with read_nodes
            SSDThreadData<T> *data = new SSDThreadData<T>(this->_aligned_dim, visited_reserve);
            this->reader->register_thread();
            data->ctx = this->reader->get_ctx();
            this->_thread_data.push(data);
        }
    }

    this->_load_flag = true;
}

template <typename T, typename LabelT> void PQFlashIndexMGV2<T, LabelT>::use_medoids_data_as_centroids()
{
    if (this->_centroid_data != nullptr)
        aligned_free(this->_centroid_data);
    alloc_aligned(((void **)&this->_centroid_data), this->_num_medoids * this->_aligned_dim * sizeof(float), 32);
    if (this->_centroid_data == nullptr)
    {
        throw diskann::ANNException("Failed to allocate aligned memory for _centroid_data", -1);
    }
    std::memset(this->_centroid_data, 0, this->_num_medoids * this->_aligned_dim * sizeof(float));

    // borrow ctx
    ScratchStoreManagerV2<SSDThreadDataV2<T>> manager(this->_thread_data_v2);
    auto data = manager.scratch_space();
    void *ctx = data->ctx;
    diskann::cout << "Loading centroid data from medoids vector data of " << this->_num_medoids << " medoid(s)" << std::endl;

    std::vector<uint32_t> nodes_to_read;
    std::vector<T *> medoid_bufs;
    std::vector<std::pair<uint32_t, uint32_t *>> nbr_bufs;

    for (uint64_t cur_m = 0; cur_m < this->_num_medoids; cur_m++)
    {
        nodes_to_read.push_back(this->_medoids[cur_m]);
        medoid_bufs.push_back(new T[this->_data_dim]);
        nbr_bufs.emplace_back(0, nullptr);
    }

    auto read_status = this->read_nodes(nodes_to_read, medoid_bufs, nbr_bufs);

    for (uint64_t cur_m = 0; cur_m < this->_num_medoids; cur_m++)
    {
        if (read_status[cur_m] == true)
        {
            if (!this->_use_disk_index_pq)
            {
                for (uint32_t i = 0; i < this->_data_dim; i++)
                    this->_centroid_data[cur_m * this->_aligned_dim + i] = medoid_bufs[cur_m][i];
            }
            else
            {
                this->_disk_pq_table.inflate_vector((uint8_t *)medoid_bufs[cur_m], (this->_centroid_data + cur_m * this->_aligned_dim));
            }
        }
        else
        {
            std::stringstream stream;
            stream << "Unable to read a medoid";
            diskann::cerr << stream.str() << std::endl;
            throw ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        delete[] medoid_bufs[cur_m];
    }
}


template <typename T, typename LabelT> void PQFlashIndexMGV2<T, LabelT>::set_reorder_ratio(const float reorder_ratio)
{
    _reorder_ratio = reorder_ratio;
}


template <typename T, typename LabelT> int PQFlashIndexMGV2<T, LabelT>::load_graph_to_memory(std::string graph_path)
{
    if (file_exists(graph_path))
    {
        std::ifstream in(graph_path, std::ios::binary);
        uint32_t _width32, _ep32;
        in.read((char *)&_width32, sizeof(uint32_t));
        in.read((char *)&_ep32, sizeof(uint32_t));
        this->_width = _width32;
        this->_ep = _ep32;
        uint64_t cc = 0;
        while (!in.eof())
        {
            unsigned k = 0;
            in.read((char *)&k, sizeof(unsigned));
            if (in.eof())
                break;
            cc += k;
            std::vector<unsigned> tmp(k);
            in.read((char *)tmp.data(), k * sizeof(unsigned));
            _graph.push_back(tmp);
        }
        cc /= _graph.size();
        diskann::cout << "Mem graph info: " << "width: " << this->_width << " ep: " << this->_ep << " nodes: " << _graph.size()
                      << " cc: " << cc << std::endl;
        return 0;
    }
    diskann::cout << "Error. Mem graph file " << graph_path << "doesn't exist." << std::endl;
    return -1;
}


template <typename T, typename LabelT>
int PQFlashIndexMGV2<T, LabelT>::load(uint32_t num_threads, const char *index_prefix, const char *mem_graph_path,
                                      const bool compressed_graph, const float reorder_ratio)
{

    std::string pq_table_bin = std::string(index_prefix) + "_pq_pivots.bin";
    std::string pq_compressed_vectors = std::string(index_prefix) + "_pq_compressed.bin";
    std::string _disk_index_file = std::string(index_prefix) + "_disk.index";

    if (mem_graph_path == nullptr)
    {
        diskann::cout << "[Graph] mem_graph_path is null, skip loading graph." << std::endl;
    }
    else
    {
        const std::string graph_path(mem_graph_path);

        if (!compressed_graph)
        {
            diskann::cout << "[Graph] Loading plain graph into memory from: " << graph_path << std::endl;
            load_graph_to_memory(graph_path);
        }
        else
        {
            const std::string map_path = graph_path + ".map";
            const bool has_map = file_exists(map_path);

            if (has_map)
            {
                diskann::cout << "[Graph] Compressed graph enabled. Found map file: " << map_path
                              << ". It will be used when returning k-NN results." << std::endl;
            }
            else
            {
                diskann::cout << "[Graph] Compressed graph enabled, but map file " << map_path
                              << " does not exist. Proceeding without reorder map." << std::endl;
            }
            diskann::cout << "graph_path is " << graph_path <<std::endl;
            _compressed_graph.emplace(graph_path, has_map);
        }
    }
    if (reorder_ratio > 0.0f)
    {
        set_reorder_ratio(reorder_ratio);
    }

    return load_from_separate_paths(num_threads, _disk_index_file.c_str(), pq_table_bin.c_str(),
                                    pq_compressed_vectors.c_str());
}


template <typename T, typename LabelT>
int PQFlashIndexMGV2<T, LabelT>::load_from_separate_paths(uint32_t num_threads, const char *index_filepath,
                                                          const char *pivots_filepath, const char *compressed_filepath)
{
    std::string pq_table_bin = pivots_filepath;
    std::string pq_compressed_vectors = compressed_filepath;
    std::string _disk_index_file = index_filepath;
    std::string medoids_file = std::string(_disk_index_file) + "_medoids.bin";
    std::string centroids_file = std::string(_disk_index_file) + "_centroids.bin";

    std::string labels_file = std ::string(_disk_index_file) + "_labels.txt";
    std::string labels_to_medoids = std ::string(_disk_index_file) + "_labels_to_medoids.txt";
    std::string dummy_map_file = std ::string(_disk_index_file) + "_dummy_map.txt";
    std::string labels_map_file = std ::string(_disk_index_file) + "_labels_map.txt";
    size_t num_pts_in_label_file = 0;

    size_t pq_file_dim, pq_file_num_centroids;
    get_bin_metadata(pq_table_bin, pq_file_num_centroids, pq_file_dim, METADATA_SIZE);

    this->_disk_index_file = _disk_index_file;

    if (pq_file_num_centroids != 256)
    {
        diskann::cout << "Error. Number of PQ centroids is not 256. Exiting." << std::endl;
        return -1;
    }

    this->_data_dim = pq_file_dim;
    // will change later if we use PQ on disk or if we are using
    // inner product without PQ
    this->_disk_bytes_per_point = this->_data_dim * sizeof(T);
    this->_aligned_dim = ROUND_UP(pq_file_dim, 8);

    size_t npts_u64, nchunks_u64;
    diskann::load_bin<uint8_t>(pq_compressed_vectors, this->data, npts_u64, nchunks_u64);

    this->_num_points = npts_u64;
    this->_n_chunks = nchunks_u64;
    this->_pq_table.load_pq_centroid_bin(pq_table_bin.c_str(), nchunks_u64);

    diskann::cout << "Loaded PQ centroids and in-memory compressed vectors. #points: " << this->_num_points
                  << " #dim: " << this->_data_dim << " #aligned_dim: " << this->_aligned_dim << " #chunks: " << this->_n_chunks
                  << std::endl;

    if (this->_n_chunks > MAX_PQ_CHUNKS)
    {
        std::stringstream stream;
        stream << "Error loading index. Ensure that max PQ bytes for in-memory "
                  "PQ data does not exceed "
               << MAX_PQ_CHUNKS << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

// read index metadata
    std::ifstream index_metadata(_disk_index_file, std::ios::binary);

    diskann::cout << "Reading index metadata from " << _disk_index_file << std::endl;
    uint32_t nr, nc; // metadata itself is stored as bin format (nr is number of
                     // metadata, nc should be 1)
    READ_U32(index_metadata, nr);
    READ_U32(index_metadata, nc);

    diskann::cout << "Disk index metadata: " << nr << " " << nc << std::endl;

    uint64_t disk_nnodes;
    uint64_t disk_ndims; // can be disk PQ dim if disk_PQ is set to true
    READ_U64(index_metadata, disk_nnodes);
    READ_U64(index_metadata, disk_ndims);

    diskann::cout << "Disk index metadata: " << disk_nnodes << " " << disk_ndims << std::endl;

    if (disk_nnodes != this->_num_points)
    {
        diskann::cout << "Mismatch in #points for compressed data file and disk "
                         "index file: "
                      << disk_nnodes << " vs " << this->_num_points << std::endl;
        return -1;
    }

    size_t medoid_id_on_file = 0;
    READ_U64(index_metadata, medoid_id_on_file);
    READ_U64(index_metadata, this->_max_node_len);
    READ_U64(index_metadata, this->_nnodes_per_sector);

    this->_max_degree = ((this->_max_node_len - this->_disk_bytes_per_point) / sizeof(uint32_t)) - 1;

    if (this->_max_degree > defaults::MAX_GRAPH_DEGREE)
    {
        std::stringstream stream;
        stream << "Error loading index. Ensure that max graph degree (R) does "
                  "not exceed "
               << defaults::MAX_GRAPH_DEGREE << std::endl;
        throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // setting up concept of frozen points in disk index for streaming-DiskANN
    READ_U64(index_metadata, this->_num_frozen_points);
    uint64_t file_frozen_id;
    READ_U64(index_metadata, file_frozen_id);
    if (this->_num_frozen_points == 1)
        this->_frozen_location = file_frozen_id;
    if (this->_num_frozen_points == 1)
    {
        diskann::cout << " Detected frozen point in index at location " << this->_frozen_location
                      << ". Will not output it at search time." << std::endl;
    }

    READ_U64(index_metadata, this->_reorder_data_exists);
    if (this->_reorder_data_exists)
    {
        if (this->_use_disk_index_pq == false)
        {
            std::stringstream stream;
            stream << "Reordering is designed for used with disk PQ "
                      "compression option";
            diskann::cerr << stream.str() << std::endl;
            throw ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        READ_U64(index_metadata, this->_reorder_data_start_sector);
        READ_U64(index_metadata, this->_ndims_reorder_vecs);
        READ_U64(index_metadata, this->_nvecs_per_sector);
    }

    diskann::cout << "Disk-Index File Meta-data: ";
    diskann::cout << "# nodes per sector: " << this->_nnodes_per_sector;
    diskann::cout << ", max node len (bytes): " << this->_max_node_len;
    diskann::cout << ", max node degree: " << this->_max_degree << std::endl;

    index_metadata.close();

    // open AlignedFileReaderV2 handle to index_file
    std::string index_fname(this->_disk_index_file);
    reader_v2->open(index_fname, false, false);
    // Also open the parent class reader for compatibility with read_nodes
    this->reader->open(index_fname);
    this->setup_thread_data(num_threads);
    this->_max_nthreads = num_threads;

    if (file_exists(medoids_file))
    {
        size_t tmp_dim = 0;
        diskann::load_bin<uint32_t>(medoids_file, this->_medoids, this->_num_medoids, tmp_dim);

        if (tmp_dim != 1)
        {
            std::stringstream stream;
            stream << "Error loading medoids file. Expected bin format of m times "
                      "1 vector of uint32_t."
                   << std::endl;
            throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        if (!file_exists(centroids_file))
        {
            diskann::cout << "Centroid data file not found. Using corresponding vectors "
                             "for the medoids "
                          << std::endl;
            use_medoids_data_as_centroids();
        }
        else
        {
            size_t num_centroids, aligned_tmp_dim;
            diskann::load_aligned_bin<float>(centroids_file, this->_centroid_data, num_centroids, tmp_dim, aligned_tmp_dim);
            if (aligned_tmp_dim != this->_aligned_dim || num_centroids != this->_num_medoids)
            {
                std::stringstream stream;
                stream << "Error loading centroids data file. Expected bin format "
                          "of "
                          "m times data_dim vector of float, where m is number of "
                          "medoids "
                          "in medoids file.";
                diskann::cerr << stream.str() << std::endl;
                throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
            }
        }
    }
    else
    {
        this->_num_medoids = 1;
        this->_medoids = new uint32_t[1];
        this->_medoids[0] = (uint32_t)(medoid_id_on_file);
        use_medoids_data_as_centroids();
    }

    std::string norm_file = std::string(this->_disk_index_file) + "_max_base_norm.bin";

    if (file_exists(norm_file) && this->metric == diskann::Metric::INNER_PRODUCT)
    {
        uint64_t dumr, dumc;
        float *norm_val;
        diskann::load_bin<float>(norm_file, norm_val, dumr, dumc);
        this->_max_base_norm = norm_val[0];
        diskann::cout << "Setting re-scaling factor of base vectors to " << this->_max_base_norm << std::endl;
        delete[] norm_val;
    }

    return 0;
}


template <typename T, typename LabelT>
void PQFlashIndexMGV2<T, LabelT>::cached_beam_search_v2(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                    uint64_t *indices, float *distances, const uint64_t beam_width,
                                                    const bool use_reorder_data, const bool use_compressed_graph,
                                                    QueryStats *stats)
{
    cached_beam_search_v2(query1, k_search, l_search, indices, distances, beam_width, std::numeric_limits<uint32_t>::max(),
                       use_reorder_data, use_compressed_graph, stats);
}

template <typename T, typename LabelT>
void PQFlashIndexMGV2<T, LabelT>::cached_beam_search_v2(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                    uint64_t *indices, float *distances, const uint64_t beam_width,
                                                    const bool use_filter, const LabelT &filter_label,
                                                    const bool use_reorder_data, const bool use_compressed_graph,
                                                    QueryStats *stats)
{
    cached_beam_search_v2(query1, k_search, l_search, indices, distances, beam_width, use_filter, filter_label,
                       std::numeric_limits<uint32_t>::max(), use_reorder_data, use_compressed_graph, stats);
}

template <typename T, typename LabelT>
void PQFlashIndexMGV2<T, LabelT>::cached_beam_search_v2(const T *query1, const uint64_t k_search, const uint64_t l_search,
                                                    uint64_t *indices, float *distances, const uint64_t beam_width,
                                                    const uint32_t io_limit, const bool use_reorder_data,
                                                    const bool use_compressed_graph, QueryStats *stats)
{
    LabelT dummy_filter = 0;
    cached_beam_search_v2(query1, k_search, l_search, indices, distances, beam_width, false, dummy_filter, io_limit,
                       use_reorder_data, use_compressed_graph, stats);
}

template <typename T, typename LabelT>
void PQFlashIndexMGV2<T, LabelT>::cached_beam_search_v2(const T *query1, const uint64_t k_search,
                                                        const uint64_t l_search, uint64_t *indices, float *distances,
                                                        const uint64_t beam_width, const bool use_filter,
                                                        const LabelT &filter_label, const uint32_t io_limit,
                                                        const bool use_reorder_data, const bool use_compressed_graph,
                                                        QueryStats *stats)
{
    // auto &cg = _compressed_graph.value();
    uint64_t num_sector_per_nodes = DIV_ROUND_UP(this->_max_node_len, defaults::SECTOR_LEN);
    if (beam_width > num_sector_per_nodes * defaults::MAX_N_SECTOR_READS)
        throw ANNException("Beamwidth can not be higher than defaults::MAX_N_SECTOR_READS", -1, __FUNCSIG__, __FILE__,
                           __LINE__);
    ScratchStoreManagerV2<SSDThreadDataV2<T>> manager(this->_thread_data_v2);
    auto data = manager.scratch_space();
    void *ctx = reader_v2->get_ctx();
    auto query_scratch = &(data->scratch);
    auto pq_query_scratch = query_scratch->pq_scratch();

    // reset query scratch
    query_scratch->reset();

    // copy query to thread specific aligned and allocated memory (for distance
    // calculations we need aligned data)
    float query_norm = 0;
    T *aligned_query_T = query_scratch->aligned_query_T();
    float *query_float = pq_query_scratch->aligned_query_float;
    float *query_rotated = pq_query_scratch->rotated_query;

    // normalization step. for cosine, we simply normalize the query
    // for mips, we normalize the first d-1 dims, and add a 0 for last dim, since an extra coordinate was used to
    // convert MIPS to L2 search
    // when data type is fp16, and metric is inner product or cosine, there will be a bug.
    if (this->metric == diskann::Metric::INNER_PRODUCT || this->metric == diskann::Metric::COSINE)
    {
        uint64_t inherent_dim = (this->metric == diskann::Metric::COSINE) ? this->_data_dim : (uint64_t)(this->_data_dim - 1);
        for (size_t i = 0; i < inherent_dim; i++)
        {
            aligned_query_T[i] = query1[i];
            query_norm += query1[i] * query1[i];
        }
        if (this->metric == diskann::Metric::INNER_PRODUCT)
            aligned_query_T[this->_data_dim - 1] = 0;

        query_norm = std::sqrt(query_norm);

        for (size_t i = 0; i < inherent_dim; i++)
        {
            aligned_query_T[i] = (T)(aligned_query_T[i] / query_norm);
        }
        pq_query_scratch->initialize(this->_data_dim, aligned_query_T);
    }
    else
    {
        for (size_t i = 0; i < this->_data_dim; i++)
        {
            aligned_query_T[i] = query1[i];
        }
        pq_query_scratch->initialize(this->_data_dim, aligned_query_T);
    }

    // pointers to buffers for data
    T *data_buf = query_scratch->coord_scratch;

#ifdef _X86
    _mm_prefetch((char *)data_buf, _MM_HINT_T1);
#else
#ifdef _ARM
    __builtin_prefetch((char *)data_buf, 0, 2);
#endif
#endif
    // sector scratch
    char *sector_scratch = query_scratch->sector_scratch;
    uint64_t &sector_scratch_idx = query_scratch->sector_idx;
    const uint64_t num_sectors_per_node =
        this->_nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(this->_max_node_len, defaults::SECTOR_LEN);
    // query <-> PQ chunk centers distances
    this->_pq_table.preprocess_query(query_rotated); // center the query and rotate if
                                               // we have a rotation matrix
    float *pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;
    this->_pq_table.populate_chunk_distances(query_rotated, pq_dists);

    // query <-> neighbor list
    float *dist_scratch = pq_query_scratch->aligned_dist_scratch;
    uint8_t *pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;

    // lambda to batch compute query<-> node distances in PQ space
    auto compute_dists = [this, pq_coord_scratch, pq_dists](const uint32_t *ids, const uint64_t n_ids,
                                                            float *dists_out) {
        diskann::aggregate_coords(ids, n_ids, this->data, this->_n_chunks, pq_coord_scratch);
        diskann::pq_dist_lookup(pq_coord_scratch, n_ids, this->_n_chunks, pq_dists, dists_out);
    };

    Timer query_timer, io_timer, cpu_timer;

    tsl::robin_set<uint64_t> &visited = query_scratch->visited;
    NeighborPriorityQueue &retset = query_scratch->retset;
    OriginNeighborPriorityQueue &retset_lb = query_scratch->retset_lb;
    retset.reserve(l_search);
    retset_lb.reserve(k_search);
    std::vector<SmallNeighbor> &full_retset = query_scratch->full_retset;
    std::vector<uint32_t> &edges_buffer = query_scratch->edges_buffer;
    std::vector<uint32_t> &decode_buffer = query_scratch->decode_buffer;

    uint32_t best_medoid = 0; // Calculate distance from query to multiple centroids, select the best one
    float best_dist = (std::numeric_limits<float>::max)();
    if (!use_filter)
    {
        for (uint64_t cur_m = 0; cur_m < this->_num_medoids; cur_m++)
        {
            float cur_expanded_dist =
                this->_dist_cmp_float->compare(query_float, this->_centroid_data + this->_aligned_dim * cur_m, (uint32_t)this->_aligned_dim);
            if (cur_expanded_dist < best_dist)
            {
                best_medoid = this->_medoids[cur_m];
                best_dist = cur_expanded_dist;
            }
        }
    }
    else
    {
        std::stringstream stream;
        stream << "Filtering is not supported in this version.";
        diskann::cerr << stream.str() << std::endl;
        throw ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    compute_dists(&best_medoid, 1, dist_scratch);
    retset.insert(Neighbor(best_medoid, dist_scratch[0]));
    visited.insert(best_medoid);

    uint32_t cmps = 0;
    uint32_t hops = 0;
    uint32_t num_ios = 0;
    uint32_t finished_ios = 0;
    // uint32_t pre_loaded_count = 0;

    uint32_t reorder_vector_num = std::min((size_t)(k_search * _reorder_ratio), l_search);

    // cleared every iteration
    std::vector<uint32_t> frontier;
    frontier.reserve(2 * beam_width);

    bool ready_to_load = false; // TODO
    retset.set_loaded_size(reorder_vector_num);

    full_retset.clear();
    sector_scratch_idx = 0;

    // io-uring
    const uint64_t size_per_io = num_sectors_per_node * defaults::SECTOR_LEN;
    // diskann::cout << "!!!!!" << std::endl;
    // use pq code to travel memory graph, obtain knn result
    cpu_timer.reset();

    std::queue<uint32_t> free_io_reqs;
    const uint32_t max_io_reqs =
        std::min(diskann::defaults::MAX_N_SECTOR_READS / num_sectors_per_node, 128UL); // at most 128 io reqs in flight
    for (uint32_t i = 0; i < max_io_reqs; i++)
    {
        free_io_reqs.push(i);
    }
    std::queue<uint32_t> to_loaded_vec_ids;

    while (retset.has_unexpanded_node())
    {
        if (hops > 5)
        {
            ready_to_load = true;
        }
        hops++;
        // find new beam
        uint32_t num_seen = 0;
        frontier.clear();
        while (retset.has_unexpanded_node() && num_seen < beam_width)
        {
            auto nbr = retset.closest_unexpanded();
            num_seen++;
            frontier.push_back(nbr.id);
        }

        // compute pq distance and update retset
        if (!frontier.empty())
        {
            if (ready_to_load)
            {
                if (retset.has_unloaded_node())
                {
                    auto to_load_nbr = retset.closest_unloaded();
                    auto to_load_nbr_id = to_load_nbr.id;
                    num_ios++;

                    // try to send a io request
                    if (!free_io_reqs.empty())
                    {
                        uint32_t free_io_req_index = free_io_reqs.front();
                        free_io_reqs.pop();
                        auto buf = sector_scratch + free_io_req_index * size_per_io;
                        IORequest &req = query_scratch->reqs[free_io_req_index];
                        req = IORequest(this->get_node_sector(((size_t)to_load_nbr_id)) * defaults::SECTOR_LEN, size_per_io,
                                        buf, to_load_nbr_id);

                        reader_v2->send_read_no_alloc(req, ctx);

                        if (stats != nullptr)
                        {
                            stats->n_4k++;
                            stats->n_ios++;
                            stats->n_ios_preload++;
                        }
                    }
                    else
                    {
                        to_loaded_vec_ids.push(to_load_nbr_id);
                    }
                }
            }
            uint32_t nnbrs_beam = 0;
            for (uint64_t i = 0; i < frontier.size(); i++)
            {
                auto id = frontier[i];
                if (use_compressed_graph == false)
                {
                    uint32_t nnbrs = _graph[id].size();
                    uint32_t *edges = _graph[id].data();
                    edges_buffer.clear();
                    for (uint64_t m = 0; m < nnbrs; ++m)
                    {
                        uint32_t id = edges[m];
                        if (visited.insert(id).second)
                        {
                            cmps++;
                            edges_buffer.push_back(id);
                        }
                    }
                }
                else
                {
                    edges_buffer.clear();
                    _compressed_graph->getNeighbors(id, decode_buffer);
                    for (uint64_t m = 0; m < decode_buffer.size(); ++m)
                    {
                        uint32_t id = decode_buffer[m];
                        if (visited.insert(id).second)
                        {
                            cmps++;
                            edges_buffer.push_back(id);
                        }
                    }
                }

                auto deduped_nn_num = edges_buffer.size();
                nnbrs_beam += deduped_nn_num;
                compute_dists(edges_buffer.data(), deduped_nn_num, dist_scratch);
                for (uint64_t m = 0; m < deduped_nn_num; ++m)
                {
                    uint32_t id = edges_buffer[m];
                    float dist = dist_scratch[m];
                    Neighbor nn(id, dist);
                    retset.insert(nn);
                }
            }
            if (stats != nullptr)
            {
                stats->n_cmps += (uint32_t)nnbrs_beam;
            }

            // try to maintain a k-sized min-heap with real distances
            if (free_io_reqs.size() < max_io_reqs)
            {
                bool one_more_time = true;
                // there are some io requests in flight
                while (one_more_time)
                {
                    IORequest *req = reader_v2->poll_ior(ctx);
                    if (req != nullptr)
                    {
                        auto req_id = req->req_id;
                        free_io_reqs.push(req_id);
                        char *node_disk_buf = this->offset_to_node((char *)req->buf, req->id);
                        T *node_fp_coords = this->offset_to_node_coords(node_disk_buf);
                        memcpy(data_buf, node_fp_coords, this->_disk_bytes_per_point);
                        float dist = this->_dist_cmp->compare(aligned_query_T, data_buf, (uint32_t)this->_aligned_dim);
                        retset_lb.insert(OriginNeighbor(req->id, dist));
                        finished_ios++;
                        if (free_io_reqs.size() == max_io_reqs)
                        {
                            one_more_time = false;
                        }
                    }
                    else
                    {
                        one_more_time = false;
                    }
                }
            }

            // try to send some reqs if there are new free slots
            while (!to_loaded_vec_ids.empty())
            {
                if (!free_io_reqs.empty())
                {
                    auto to_load_nbr_id = to_loaded_vec_ids.front();
                    to_loaded_vec_ids.pop();
                    auto free_io_req_index = free_io_reqs.front();
                    free_io_reqs.pop();
                    // send read req
                    auto buf = sector_scratch + free_io_req_index * size_per_io;
                    IORequest &req = query_scratch->reqs[free_io_req_index];
                    req = IORequest(this->get_node_sector(((size_t)to_load_nbr_id)) * defaults::SECTOR_LEN, size_per_io, buf,
                                    to_load_nbr_id);

                    reader_v2->send_read_no_alloc(req, ctx);

                    if (stats != nullptr)
                    {
                        stats->n_4k++;
                        stats->n_ios++;
                        stats->n_ios_preload++;
                    }
                }
                else
                {
                    break;
                }
            }
        }
        if (stats != nullptr)
        {
            stats->n_hops++;
        }
    }

    if (stats != nullptr)
    {
        stats->cpu_us += (float)cpu_timer.elapsed();
    }

    // // Iterate all retset entries, only the first reorder_vector_num points.
    // // If already loaded, it was prefetched asynchronously; otherwise, add to list for later loading.
    // // Note: this iterates reorder_vector_num entries, not the full L length.
    for (uint32_t i = 0; i < reorder_vector_num; i++)
    {
        auto id = retset[i].id;
        if (!retset[i].loaded)
        {
            to_loaded_vec_ids.push(id);
            num_ios++;
        }
        else
        {
            if (stats != nullptr)
            {
                stats->n_ios_preload_hits++;
            }
        }
    }


    io_timer.reset();
    while (finished_ios < num_ios)
    {
        // try to send some reqs if there are new free slots
        while (!to_loaded_vec_ids.empty())
        {
            if (!free_io_reqs.empty())
            {
                auto to_load_nbr_id = to_loaded_vec_ids.front();
                to_loaded_vec_ids.pop();
                auto free_io_req_index = free_io_reqs.front();
                free_io_reqs.pop();
                // send read req
                auto buf = sector_scratch + free_io_req_index * size_per_io;
                IORequest &req = query_scratch->reqs[free_io_req_index];
                req = IORequest(this->get_node_sector(((size_t)to_load_nbr_id)) * defaults::SECTOR_LEN, size_per_io, buf,
                                to_load_nbr_id);

                reader_v2->send_read_no_alloc(req, ctx);

                if (stats != nullptr)
                {
                    stats->n_4k++;
                    stats->n_ios++;
                    stats->n_ios_preload++;
                }
            }
            else
            {
                break;
            }
        }

        // try to maintain a k-sized min-heap with real distances
        if (free_io_reqs.size() < max_io_reqs)
        {
            bool one_more_time = true;
            // there are some io requests in flight
            while (one_more_time)
            {
                IORequest *req = reader_v2->poll_ior(ctx);
                if (req != nullptr)
                {
                    auto req_id = req->req_id;
                    free_io_reqs.push(req_id);
                    char *node_disk_buf = this->offset_to_node((char *)req->buf, req->id);
                    T *node_fp_coords = this->offset_to_node_coords(node_disk_buf);
                    memcpy(data_buf, node_fp_coords, this->_disk_bytes_per_point);
                    float dist = this->_dist_cmp->compare(aligned_query_T, data_buf, (uint32_t)this->_aligned_dim);
                    retset_lb.insert(OriginNeighbor(req->id, dist));
                    finished_ios++;
                    if (free_io_reqs.size() == max_io_reqs)
                    {
                        one_more_time = false;
                    }
                }
                else
                {
                    one_more_time = false;
                }
            }
        }
    }

    if (stats != nullptr)
    {
        stats->io_us += io_timer.elapsed();
    }

    // copy k_search values
    for (uint64_t i = 0; i < k_search; i++)
    {
        indices[i] = retset_lb[i].id;
        auto key = (uint32_t)indices[i];
        if (this->_dummy_pts.find(key) != this->_dummy_pts.end())
        {
            indices[i] = this->_dummy_to_real_map[key];
        }

        if (distances != nullptr)
        {
            distances[i] = retset_lb[i].distance;
            if (this->metric == diskann::Metric::INNER_PRODUCT)
            {
                // flip the sign to convert min to max
                distances[i] = (-distances[i]);
                // rescale to revert back to original norms (cancelling the
                // effect of base and query pre-processing)
                if (this->_max_base_norm != 0)
                    distances[i] *= (this->_max_base_norm * query_norm);
            }
        }
    }

#ifdef USE_BING_INFRA
    ctx.m_completeCount = 0;
#endif

    if (stats != nullptr)
    {
        stats->total_us += (float)query_timer.elapsed();
    }
}


// instantiations
template class PQFlashIndexMGV2<uint8_t>;
template class PQFlashIndexMGV2<int8_t>;
template class PQFlashIndexMGV2<float>;

template class PQFlashIndexMGV2<uint8_t, uint16_t>;
template class PQFlashIndexMGV2<int8_t, uint16_t>;
template class PQFlashIndexMGV2<float, uint16_t>;

} // namespace diskann
