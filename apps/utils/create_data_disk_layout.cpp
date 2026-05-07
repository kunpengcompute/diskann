// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include "utils.h"

#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#ifdef _X86
#include <immintrin.h>
#else
#include <arm_neon.h>
#endif

void create_header(std::vector<int8_t> &block_cache, uint32_t nr = 0, uint32_t nc = 0, uint64_t disk_nnodes = 0,
                   uint64_t disk_ndims = 0, uint64_t medoid_id_on_file = 0, uint64_t max_node_len = 0,
                   uint64_t nnodes_per_sector = 0, uint64_t num_frozen_points = 0, uint64_t file_frozen_id = 0,
                   uint64_t reorder_data_exists = 0, uint64_t reorder_data_start_sector = 0,
                   uint64_t ndims_reorder_vecs = 0, uint64_t nvecs_per_sector = 0)
{
    if (block_cache.size() != 4096)
    {
        throw std::invalid_argument("block_cache.size() must be 4096");
    }

    // print metadata
    std::cout << "Metadata: nr = " << nr << ", nc = " << nc << std::endl;
    std::cout << "Metadata: disk_nnodes = " << disk_nnodes << ", disk_ndims = " << disk_ndims << std::endl;
    std::cout << "Metadata: medoid_id_on_file = " << medoid_id_on_file << ", max_node_len = " << max_node_len
              << std::endl;
    std::cout << "Metadata: nnodes_per_sector = " << nnodes_per_sector << ", num_frozen_points = " << num_frozen_points
              << std::endl;
    std::cout << "Metadata: file_frozen_id = " << file_frozen_id << ", reorder_data_exists = " << reorder_data_exists
              << std::endl;
    std::cout << "Metadata: reorder_data_start_sector = " << reorder_data_start_sector
              << ", ndims_reorder_vecs = " << ndims_reorder_vecs << std::endl;
    std::cout << "Metadata: nvecs_per_sector = " << nvecs_per_sector << std::endl;
    std::fill(block_cache.begin(), block_cache.end(), 0);

    int8_t *block_ptr_cur = block_cache.data();

    std::memcpy(block_ptr_cur, &nr, sizeof(uint32_t));
    block_ptr_cur += sizeof(uint32_t);
    std::memcpy(block_ptr_cur, &nc, sizeof(uint32_t));
    block_ptr_cur += sizeof(uint32_t);

    std::memcpy(block_ptr_cur, &disk_nnodes, sizeof(uint64_t));
    block_ptr_cur += sizeof(uint64_t);
    std::memcpy(block_ptr_cur, &disk_ndims, sizeof(uint64_t));
    block_ptr_cur += sizeof(uint64_t);

    std::memcpy(block_ptr_cur, &medoid_id_on_file, sizeof(uint64_t));
    block_ptr_cur += sizeof(uint64_t);
    std::memcpy(block_ptr_cur, &max_node_len, sizeof(uint64_t));
    block_ptr_cur += sizeof(uint64_t);
    std::memcpy(block_ptr_cur, &nnodes_per_sector, sizeof(uint64_t));
    block_ptr_cur += sizeof(uint64_t);

    std::memcpy(block_ptr_cur, &num_frozen_points, sizeof(uint64_t));
    block_ptr_cur += sizeof(uint64_t);

    std::memcpy(block_ptr_cur, &file_frozen_id, sizeof(uint64_t));
    block_ptr_cur += sizeof(uint64_t);

    std::memcpy(block_ptr_cur, &reorder_data_exists, sizeof(uint64_t));
    block_ptr_cur += sizeof(uint64_t);
    std::memcpy(block_ptr_cur, &reorder_data_start_sector, sizeof(uint64_t));
    block_ptr_cur += sizeof(uint64_t);
    std::memcpy(block_ptr_cur, &ndims_reorder_vecs, sizeof(uint64_t));
    block_ptr_cur += sizeof(uint64_t);
    std::memcpy(block_ptr_cur, &nvecs_per_sector, sizeof(uint64_t));
    block_ptr_cur += sizeof(uint64_t);
}

template <typename T>
void create_data_disk_layout(const std::string &fbin_data_file_to_use, const std::string &aligned_data_file_to_save)
{
    size_t train_size = 0;
    size_t npts = 0, ndims = 0;
    uint32_t npts32 = 0, ndims32 = 0;
    // amount to read in one shot
    size_t read_blk_size = 64 * 1024 * 1024;
    size_t single_block_size = 4096; // 4kb
    size_t real_vectors_per_block = 0;

    std::string centroids_file = std::string(aligned_data_file_to_save) + "_centroids.bin";
    std::string medoids_file = std::string(aligned_data_file_to_save) + "_medoids.bin";
    if (!file_exists(fbin_data_file_to_use))
    {
        std::cout << "fbin_data_file_to_use doesn't exist." << std::endl;
        exit(-1);
    }

    if (file_exists(aligned_data_file_to_save))
    {
        std::remove(aligned_data_file_to_save.c_str());
    }
    // create cached reader + writer
    cached_ifstream base_reader(fbin_data_file_to_use.c_str(), read_blk_size);
    cached_ofstream layout_writer(aligned_data_file_to_save.c_str(), read_blk_size);

    // metadata: npts, ndims
    base_reader.read((char *)&npts32, sizeof(uint32_t));
    base_reader.read((char *)&ndims32, sizeof(uint32_t));
    npts = npts32;
    ndims = ndims32;

    std::cout << "Metadata: #pts = " << npts << ", #dims = " << ndims << "..." << std::endl;

    size_t aligned_dim = ROUND_UP(ndims, 8);

    size_t single_vector_size = sizeof(T) * ndims;
    size_t vectors_per_block = 0;

    std::vector<int8_t> block_cache(single_block_size, 0);

    if (single_vector_size > single_block_size)
    {
        vectors_per_block = 0;
        real_vectors_per_block = ROUND_UP(single_vector_size, 4096);
        block_cache.resize(real_vectors_per_block);
        diskann::cout << "Warning: single vector size: " << single_vector_size
                      << " is larger than block size: " << single_block_size
                      << " bytes. So, using block size: " << block_cache.size() << " bytes." << std::endl;
    }
    else
    {
        vectors_per_block = single_block_size / single_vector_size;
    }
    diskann::cout << "Vectors per block: " << vectors_per_block << std::endl;
    size_t node_len = single_vector_size;
    if (vectors_per_block == 0)
    {
        node_len = real_vectors_per_block * 4096;
    }

    create_header(block_cache, 14, 1, npts, ndims, 0, node_len, vectors_per_block);
    layout_writer.write((char *)block_cache.data(), single_block_size); // for save index header

    std::unique_ptr<T[]> cur_vector_T = std::make_unique<T[]>(aligned_dim);

    std::unique_ptr<float[]> centroid_float = std::make_unique<float[]>(aligned_dim);
    std::fill(centroid_float.get(), centroid_float.get() + aligned_dim, 0.0f);

    std::unique_ptr<double[]> centroid_double = std::make_unique<double[]>(aligned_dim);
    std::fill(centroid_double.get(), centroid_double.get() + aligned_dim, 0.0);

    size_t ep_id = 0;
    std::unique_ptr<float[]> ep_float = std::make_unique<float[]>(aligned_dim);
    std::fill(ep_float.get(), ep_float.get() + aligned_dim, 0.0f);
    for (size_t i = 0; i < npts; i++)
    {
        base_reader.read((char *)cur_vector_T.get(), ndims * sizeof(T));
        if (vectors_per_block == 0)
        {
            std::copy(cur_vector_T.get(), cur_vector_T.get() + ndims, block_cache.data());
            layout_writer.write((char *)block_cache.data(), block_cache.size());
            std::fill(block_cache.begin(), block_cache.end(), 0);
        }
        else
        {
            size_t vector_id = i % vectors_per_block;
            std::copy(cur_vector_T.get(), cur_vector_T.get() + ndims,
                      block_cache.data() + vector_id * single_vector_size);
            if (vector_id == vectors_per_block - 1 || i == npts - 1)
            {
                layout_writer.write((char *)block_cache.data(), single_block_size);
                std::fill(block_cache.begin(), block_cache.end(), 0);
            }
        }
        if (i == ep_id)
        {
            for (size_t j = 0; j < ndims; j++)
            {
                ep_float[j] = static_cast<float>(cur_vector_T[j]);
            }
        }
        for (size_t j = 0; j < ndims; j++)
        {
            centroid_double[j] += static_cast<double>(cur_vector_T[j]);
        }
    }

    float sum = 0.0f;
    float max_val = -std::numeric_limits<float>::infinity();
    float min_val = std::numeric_limits<float>::infinity();

    for (size_t j = 0; j < ndims; ++j)
    {
        centroid_float[j] = static_cast<float>(centroid_double[j] / npts);
        float val = centroid_float[j];
        sum += val;
        if (val > max_val)
            max_val = val;
        if (val < min_val)
            min_val = val;
    }

    if (ndims == 0)
    {
        throw std::invalid_argument("ndims cannot be zero");
    }
    float mean = sum / ndims;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Centroid stats:" << std::endl;
    std::cout << "  Min  = " << min_val << std::endl;
    std::cout << "  Max  = " << max_val << std::endl;
    std::cout << "  Mean = " << mean << std::endl;

    uint32_t centroid_nums = 1;
    std::ofstream centroid_writer(centroids_file, std::ios::binary);
    if (!centroid_writer.is_open())
    {
        throw std::runtime_error("Failed to open file for writing: " + centroids_file);
    }
    centroid_writer.write((char *)&centroid_nums, sizeof(uint32_t));
    centroid_writer.write((char *)&ndims32, sizeof(uint32_t));
    centroid_writer.write((char *)centroid_float.get(), sizeof(float) * ndims);
    centroid_writer.close();

    std::ofstream medoids_writer(medoids_file, std::ios::binary);
    if (!medoids_writer.is_open())
    {
        throw std::runtime_error("Failed to open file for writing: " + medoids_file);
    }
    medoids_writer.write((char *)&centroid_nums, sizeof(uint32_t));
    uint32_t medoid_dim = 1;
    medoids_writer.write((char *)&medoid_dim, sizeof(uint32_t));
    uint32_t medoid_id = 0;
    medoids_writer.write((char *)&medoid_id, sizeof(uint32_t));
    medoids_writer.close();
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_data_file> <output_data_file>" << std::endl;
        return 1;
    }

    // Input data file and output cache file
    std::string fbin_data_file_to_use = argv[1];
    std::string aligned_data_file_to_save = argv[2];

    if (fbin_data_file_to_use == aligned_data_file_to_save)
    {
        std::cout << "fbin_data_file_to_use is equal to aligned_data_file_to_save " << std::endl;
        exit(-1);
    }

    auto start = std::chrono::high_resolution_clock::now();
    create_data_disk_layout<float>(fbin_data_file_to_use, aligned_data_file_to_save);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "Data layout creation completed successfully!" << std::endl;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds." << std::endl;

    return 0;
}
