// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>
#include "utils.h"
#include "pq.h"
#include "distance.h"
#include <fstream>
#include <vector>

BOOST_AUTO_TEST_SUITE(MoreCoverageTests)

// Test more utils functions
BOOST_AUTO_TEST_CASE(utils_functions) {
    // Test save_bin and load_bin with different types
    const char* test_file = "/tmp/utils_test.bin";

    // Test with uint32_t
    std::vector<uint32_t> data_u32 = {1, 2, 3, 4, 5};
    diskann::save_bin<uint32_t>(test_file, data_u32.data(), 5, 1);

    std::unique_ptr<uint32_t[]> loaded_u32;
    size_t npts, ndims;
    diskann::load_bin<uint32_t>(test_file, loaded_u32, npts, ndims);
    BOOST_CHECK_EQUAL(npts, 5);
    BOOST_CHECK_EQUAL(ndims, 1);
    BOOST_CHECK_EQUAL(loaded_u32[0], 1);
    BOOST_CHECK_EQUAL(loaded_u32[4], 5);

    // Test with int8_t
    std::vector<int8_t> data_i8 = {-1, 0, 1, 2, 3};
    diskann::save_bin<int8_t>(test_file, data_i8.data(), 5, 1);

    std::unique_ptr<int8_t[]> loaded_i8;
    diskann::load_bin<int8_t>(test_file, loaded_i8, npts, ndims);
    BOOST_CHECK_EQUAL(loaded_i8[0], -1);
    BOOST_CHECK_EQUAL(loaded_i8[2], 1);

    // Test with uint8_t
    std::vector<uint8_t> data_u8 = {10, 20, 30, 40, 50};
    diskann::save_bin<uint8_t>(test_file, data_u8.data(), 5, 1);

    std::unique_ptr<uint8_t[]> loaded_u8;
    diskann::load_bin<uint8_t>(test_file, loaded_u8, npts, ndims);
    BOOST_CHECK_EQUAL(loaded_u8[0], 10);
    BOOST_CHECK_EQUAL(loaded_u8[4], 50);

    std::remove(test_file);
}

// Test more PQ functions with different configurations
BOOST_AUTO_TEST_CASE(pq_various_configs) {
    const uint32_t num_points = 30;
    const uint32_t dim = 8;
    std::vector<float> data(num_points * dim);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i % 10);
    }

    // Test with 4 chunks
    std::string pivots_file = "/tmp/pq_test_4chunks.bin";

    // Delete old file first to avoid using corrupted cache
    std::remove(pivots_file.c_str());

    int ret = diskann::generate_pq_pivots(data.data(), num_points, dim, 256, 4, 1, pivots_file.c_str(), false);
    BOOST_CHECK_EQUAL(ret, 0);

    // Test generate_pq_data_from_pivots
    std::string data_file = "/tmp/pq_test_data.bin";
    diskann::save_bin<float>(data_file.c_str(), data.data(), num_points, dim);

    std::string compressed_file = "/tmp/pq_test_compressed.bin";
    ret = diskann::generate_pq_data_from_pivots<float>(data_file, 256, 4, pivots_file.c_str(), compressed_file.c_str());
    BOOST_CHECK_EQUAL(ret, 0);

    // Verify compressed file exists and has data
    std::ifstream check(compressed_file, std::ios::binary);
    BOOST_CHECK(check.good());
    check.close();

    // Cleanup
    std::remove(pivots_file.c_str());
    std::remove(data_file.c_str());
    std::remove(compressed_file.c_str());
}

// Test with 1 chunk (edge case)
BOOST_AUTO_TEST_CASE(pq_single_chunk) {
    const uint32_t num_points = 25;
    const uint32_t dim = 4;
    std::vector<float> data(num_points * dim, 1.5f);

    std::string pivots_file = "/tmp/pq_test_1chunk.bin";

    // Delete old file first to avoid using corrupted cache
    std::remove(pivots_file.c_str());

    int ret = diskann::generate_pq_pivots(data.data(), num_points, dim, 256, 1, 1, pivots_file.c_str(), false);
    BOOST_CHECK_EQUAL(ret, 0);

    std::remove(pivots_file.c_str());
}

// Test distance functions with various inputs
BOOST_AUTO_TEST_CASE(distance_various_inputs) {
    // Test L2 with different sizes
    auto dist_l2 = diskann::get_distance_function<float>(diskann::Metric::L2);
    BOOST_CHECK(dist_l2 != nullptr);

    // Test cosine
    auto dist_cosine = diskann::get_distance_function<float>(diskann::Metric::COSINE);
    BOOST_CHECK(dist_cosine != nullptr);

    // Test inner product
    auto dist_ip = diskann::get_distance_function<float>(diskann::Metric::INNER_PRODUCT);
    BOOST_CHECK(dist_ip != nullptr);
}

// Test with int8 distance
BOOST_AUTO_TEST_CASE(distance_int8) {
    auto dist_func = diskann::get_distance_function<int8_t>(diskann::Metric::L2);
    BOOST_CHECK(dist_func != nullptr);
}

// Test with uint8 distance
BOOST_AUTO_TEST_CASE(distance_uint8) {
    auto dist_func = diskann::get_distance_function<uint8_t>(diskann::Metric::L2);
    BOOST_CHECK(dist_func != nullptr);
}

BOOST_AUTO_TEST_SUITE_END()
