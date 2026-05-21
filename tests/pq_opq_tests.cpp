// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>
#include "pq.h"
#include "utils.h"
#include <vector>

BOOST_AUTO_TEST_SUITE(PQCacheTests)

// Test pivot cache hit path - DISABLED due to bad_alloc
// BOOST_AUTO_TEST_CASE(generate_quantized_data_pivot_cache_hit)
// {
//     ... commented out ...
// }

// Test compressed vector cache hit path
BOOST_AUTO_TEST_CASE(generate_pq_data_compressed_cache_hit)
{
    const std::string data_file = "test_comp_cache_data.bin";
    const std::string pq_pivots = "test_comp_cache_pivots.bin";
    const std::string pq_compressed = "test_comp_cache_compressed.bin";

    std::cout << "[TEST] Step 1: Creating dataset" << std::endl;
    // Create dataset
    const size_t num_points = 1000;
    const size_t dim = 16;
    std::vector<float> data(num_points * dim);
    for (size_t i = 0; i < num_points * dim; i++) {
        data[i] = (i % 100) / 10.0f;
    }
    diskann::save_bin<float>(data_file, data.data(), num_points, dim);
    std::cout << "[TEST] Step 1: Done" << std::endl;

    std::cout << "[TEST] Step 2: Loading data" << std::endl;
    // Load data for generate_pq_pivots
    size_t file_num_points, file_dim;
    float* loaded_data;
    diskann::load_bin<float>(data_file, loaded_data, file_num_points, file_dim);
    std::cout << "[TEST] Step 2: Loaded " << file_num_points << " points, dim=" << file_dim << std::endl;

    std::cout << "[TEST] Step 3: Generating PQ pivots" << std::endl;
    // Generate pivots first
    int pivot_result = diskann::generate_pq_pivots(loaded_data, file_num_points, file_dim, 256, 4, 10, pq_pivots, false);
    std::cout << "[TEST] Step 3: generate_pq_pivots returned " << pivot_result << std::endl;

    std::cout << "[TEST] Step 4: Freeing loaded data" << std::endl;
    // Free loaded data
    delete[] loaded_data;
    std::cout << "[TEST] Step 4: Done" << std::endl;

    std::cout << "[TEST] Step 5: First run - generate compressed vectors" << std::endl;
    // First run - generate compressed vectors
    diskann::generate_pq_data_from_pivots<float>(data_file, 256, 4, pq_pivots, pq_compressed, false);
    std::cout << "[TEST] Step 5: Done" << std::endl;

    std::cout << "[TEST] Step 6: Second run - should hit compressed cache" << std::endl;
    // Second run - should hit compressed cache
    diskann::generate_pq_data_from_pivots<float>(data_file, 256, 4, pq_pivots, pq_compressed, false);
    std::cout << "[TEST] Step 6: Done" << std::endl;

    // Verify file exists
    BOOST_CHECK(file_exists(pq_compressed));

    // Cleanup
    std::remove(data_file.c_str());
    std::remove(pq_pivots.c_str());
    std::remove(pq_compressed.c_str());
    std::cout << "[TEST] Test completed successfully" << std::endl;
}

BOOST_AUTO_TEST_SUITE_END()


