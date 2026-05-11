// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include "utils.h"

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#ifdef _X86
#include <immintrin.h>
#else
#include <arm_neon.h>
#endif

constexpr size_t HEADER_BLOCK_SIZE = 4096;

void create_header(std::vector<int8_t> &block_cache, uint64_t npts, uint64_t ndims, uint64_t aligned_dims,
                   uint64_t size_of_elements, uint64_t centroids_num)
{
    if (block_cache.size() != HEADER_BLOCK_SIZE)
    {
        std::cerr << "Error: block_cache size != HEADER_BLOCK_SIZE" << std::endl;
        exit(-1);
    }

    std::fill(block_cache.begin(), block_cache.end(), 0);
    int8_t *block_ptr_cur = block_cache.data();

    uint32_t magic = 0xDEADBEEF;
    uint32_t version = 1;
    std::memcpy(block_ptr_cur, &magic, sizeof(uint32_t));
    block_ptr_cur += sizeof(uint32_t);
    std::memcpy(block_ptr_cur, &version, sizeof(uint32_t));
    block_ptr_cur += sizeof(uint32_t);

    std::memcpy(block_ptr_cur, &npts, sizeof(uint64_t));
    block_ptr_cur += sizeof(uint64_t);
    std::memcpy(block_ptr_cur, &ndims, sizeof(uint64_t));
    block_ptr_cur += sizeof(uint64_t);
    std::memcpy(block_ptr_cur, &aligned_dims, sizeof(uint64_t));
    block_ptr_cur += sizeof(uint64_t);
    std::memcpy(block_ptr_cur, &size_of_elements, sizeof(uint64_t));
    block_ptr_cur += sizeof(uint64_t);
    std::memcpy(block_ptr_cur, &centroids_num, sizeof(uint64_t));

    std::cout << "Header Info:" << std::endl;
    std::cout << "  Magic: 0x" << std::hex << magic << std::dec << std::endl;
    std::cout << "  Version: " << version << std::endl;
    std::cout << "  npts: " << npts << ", ndims: " << ndims << ", aligned_dims: " << aligned_dims
              << ", element_size: " << size_of_elements << ", centroids: " << centroids_num << std::endl;
}

template <typename T>
void fbin_to_block(const std::string &fbin_data_file_to_use, const std::string &aligned_data_file_to_save)
{
    if (!file_exists(fbin_data_file_to_use))
    {
        std::cerr << "Error: Input file doesn't exist." << std::endl;
        exit(-1);
    }

    if (file_exists(aligned_data_file_to_save))
    {
        std::remove(aligned_data_file_to_save.c_str());
    }

    std::ifstream base_reader(fbin_data_file_to_use.c_str(), std::ios::binary);
    if (!base_reader.is_open())
    {
        throw std::runtime_error("Failed to open input file: " + fbin_data_file_to_use);
    }
    std::ofstream layout_writer(aligned_data_file_to_save.c_str(), std::ios::binary);
    if (!layout_writer.is_open())
    {
        throw std::runtime_error("Failed to open output file: " + aligned_data_file_to_save);
    }

    uint32_t npts32, ndims32;
    base_reader.read((char *)&npts32, sizeof(uint32_t));
    base_reader.read((char *)&ndims32, sizeof(uint32_t));
    size_t npts = npts32, ndims = ndims32;
    size_t aligned_dim = ROUND_UP(ndims, 16);

    std::cout << "Metadata: #pts = " << npts << ", #dims = " << ndims << ", aligned to " << aligned_dim << "..."
              << std::endl;

    // block sizes
    size_t centroid_block_size = ROUND_UP(sizeof(float) * aligned_dim, 4096);
    size_t vector_size = sizeof(T) * aligned_dim;
    size_t data_block_size = (vector_size > 4096) ? ROUND_UP(vector_size, 4096) : 4096;

    // === header block ===
    std::vector<int8_t> header_block(HEADER_BLOCK_SIZE, 0);
    create_header(header_block, npts, ndims, aligned_dim, sizeof(T), 1);
    layout_writer.write((char *)header_block.data(), HEADER_BLOCK_SIZE);

    // === centroid block ===
    std::unique_ptr<double[]> centroid_double = std::make_unique<double[]>(aligned_dim);
    std::fill(centroid_double.get(), centroid_double.get() + aligned_dim, 0.0);

    std::unique_ptr<T[]> cur_vector = std::make_unique<T[]>(aligned_dim);

    std::vector<int8_t> data_block(data_block_size, 0);
    std::vector<int8_t> centroid_block(centroid_block_size, 0);

    size_t vectors_per_4KB = 4096 / vector_size;
    size_t vec_idx = 0;

    for (size_t i = 0; i < npts; ++i)
    {
        base_reader.read((char *)cur_vector.get(), ndims * sizeof(T));
        std::fill(cur_vector.get() + ndims, cur_vector.get() + aligned_dim, 0);

        for (size_t j = 0; j < ndims; ++j)
            centroid_double[j] += static_cast<double>(cur_vector[j]);

        if (vectors_per_4KB == 0)
        {
            layout_writer.write((char *)cur_vector.get(), vector_size);
        }
        else
        {
            std::memcpy(data_block.data() + vec_idx * vector_size, cur_vector.get(), vector_size);
            vec_idx++;
            if (vec_idx == vectors_per_4KB || i == npts - 1)
            {
                layout_writer.write((char *)data_block.data(), data_block_size);
                vec_idx = 0;
                std::fill(data_block.begin(), data_block.end(), 0);
            }
        }

        if (i % 100000 == 0 && i > 0)
            std::cout << "  Processed " << i << " vectors..." << std::endl;
    }

    std::unique_ptr<float[]> centroid_float = std::make_unique<float[]>(aligned_dim);
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    float sum = 0.0f;

    if (npts == 0)
    {
        throw std::invalid_argument("npts cannot be zero");
    }

    for (size_t j = 0; j < ndims; ++j)
    {
        centroid_float[j] = static_cast<float>(centroid_double[j] / npts);
        sum += centroid_float[j];
        min_val = std::min(min_val, centroid_float[j]);
        max_val = std::max(max_val, centroid_float[j]);
    }

    if (ndims == 0)
    {
        throw std::invalid_argument("ndims cannot be zero");
    }
    float mean = sum / ndims;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Centroid stats:\n  Min = " << min_val << "\n  Max = " << max_val << "\n  Mean = " << mean
              << std::endl;

    std::memcpy(centroid_block.data(), centroid_float.get(), sizeof(float) * aligned_dim);
    layout_writer.seekp(HEADER_BLOCK_SIZE);
    layout_writer.write((char *)centroid_block.data(), centroid_block_size);
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_fbin_file> <output_block_file>" << std::endl;
        return 1;
    }

    std::string input_file = argv[1], output_file = argv[2];

    if (input_file == output_file)
    {
        std::cerr << "Error: input and output file cannot be the same." << std::endl;
        return 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    fbin_to_block<float>(input_file, output_file);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Data layout creation completed!" << std::endl;
    std::cout << "Elapsed time: " << std::chrono::duration<double>(end - start).count() << " seconds." << std::endl;

    return 0;
}
