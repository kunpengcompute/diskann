// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "utils.h"
#include "pq.h"
#include <vector>
#include <string>
#include <fstream>

namespace test_helpers {

// Create minimal test index files for integration testing
class MinimalIndexBuilder {
public:
    MinimalIndexBuilder(const std::string& prefix, uint32_t num_points, uint32_t dim)
        : prefix_(prefix), num_points_(num_points), dim_(dim) {
        aligned_dim_ = ROUND_UP(dim, 8);
    }

    // Generate random test data
    void generate_data() {
        data_.resize(num_points_ * dim_);
        for (size_t i = 0; i < data_.size(); ++i) {
            data_[i] = static_cast<float>((i % 100) / 10.0f);
        }
    }

    // Create PQ pivots file
    void create_pq_pivots(uint32_t num_pq_chunks = 2, uint32_t max_k_means_reps = 3) {
        std::string pivots_file = prefix_ + "_pq_pivots.bin";
        diskann::generate_pq_pivots(data_.data(), num_points_, dim_, 256,
                                    num_pq_chunks, max_k_means_reps, pivots_file.c_str(), false);
    }

    // Create PQ compressed vectors
    void create_pq_compressed(uint32_t num_pq_chunks = 2) {
        std::string data_file = prefix_ + "_data.bin";
        diskann::save_bin<float>(data_file.c_str(), data_.data(), num_points_, dim_);

        std::string pivots_file = prefix_ + "_pq_pivots.bin";
        std::string compressed_file = prefix_ + "_pq_compressed.bin";

        diskann::generate_pq_data_from_pivots<float>(
            data_file, 256, num_pq_chunks, pivots_file.c_str(), compressed_file.c_str()
        );

        std::remove(data_file.c_str());
    }

    // Create disk index metadata file
    void create_disk_index_metadata() {
        std::string index_file = prefix_ + "_disk.index";
        std::ofstream out(index_file, std::ios::binary);

        // Write header (nr, nc)
        uint32_t nr = 5;  // number of metadata fields
        uint32_t nc = 1;
        out.write((char*)&nr, sizeof(uint32_t));
        out.write((char*)&nc, sizeof(uint32_t));

        // Write metadata
        uint64_t disk_nnodes = num_points_;
        uint64_t disk_ndims = dim_;
        uint64_t medoid_id = 0;
        uint64_t max_node_len = 1024;
        uint64_t nnodes_per_sector = 1;

        out.write((char*)&disk_nnodes, sizeof(uint64_t));
        out.write((char*)&disk_ndims, sizeof(uint64_t));
        out.write((char*)&medoid_id, sizeof(uint64_t));
        out.write((char*)&max_node_len, sizeof(uint64_t));
        out.write((char*)&nnodes_per_sector, sizeof(uint64_t));

        // Write frozen points info
        uint64_t num_frozen = 0;
        uint64_t frozen_id = 0;
        out.write((char*)&num_frozen, sizeof(uint64_t));
        out.write((char*)&frozen_id, sizeof(uint64_t));

        // Write reorder data info
        uint64_t reorder_data_exists = 0;
        out.write((char*)&reorder_data_exists, sizeof(uint64_t));

        out.close();
    }

    // Create medoids file
    void create_medoids_file() {
        std::string medoids_file = prefix_ + "_disk.index_medoids.bin";
        std::vector<uint32_t> medoids = {0};  // Use first point as medoid
        diskann::save_bin<uint32_t>(medoids_file.c_str(), medoids.data(), 1, 1);
    }

    // Build complete minimal index
    void build_all(uint32_t num_pq_chunks = 2, uint32_t max_k_means_reps = 3) {
        generate_data();
        create_pq_pivots(num_pq_chunks, max_k_means_reps);
        create_pq_compressed(num_pq_chunks);
        create_disk_index_metadata();
        create_medoids_file();
    }

    // Clean up all generated files
    void cleanup() {
        std::remove((prefix_ + "_pq_pivots.bin").c_str());
        std::remove((prefix_ + "_pq_compressed.bin").c_str());
        std::remove((prefix_ + "_disk.index").c_str());
        std::remove((prefix_ + "_disk.index_medoids.bin").c_str());
        std::remove((prefix_ + "_data.bin").c_str());
    }

private:
    std::string prefix_;
    uint32_t num_points_;
    uint32_t dim_;
    uint32_t aligned_dim_;
    std::vector<float> data_;
};

} // namespace test_helpers
