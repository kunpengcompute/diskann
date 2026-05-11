// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>
#include "utils.h"
#include "compressed_graph.h"
#include <vector>
#include <fstream>

BOOST_AUTO_TEST_SUITE(EndToEndTests)

// Test CompressedGraph API
BOOST_AUTO_TEST_CASE(compressed_graph_workflow) {
    // Create adjacency list
    std::vector<std::vector<uint32_t>> adjList = {
        {1, 2, 3},
        {0, 2},
        {0, 1, 3},
        {0, 2}
    };

    // Create CompressedGraph
    CompressedGraph cg(adjList, 64, 0);

    // Test basic operations
    BOOST_CHECK_EQUAL(cg.getPointsNum(), 4);

    std::vector<uint32_t> neighbors;
    cg.getNeighbors(0, neighbors);
    BOOST_CHECK_EQUAL(neighbors.size(), 3);
    BOOST_CHECK_EQUAL(neighbors[0], 1);
}

// Test data conversion utilities
BOOST_AUTO_TEST_CASE(data_conversion_workflow) {
    const char* test_file = "/tmp/e2e_test_data.bin";

    // Create test data
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    diskann::save_bin<float>(test_file, data.data(), 2, 3);

    // Verify file exists and has correct size
    std::ifstream in(test_file, std::ios::binary | std::ios::ate);
    size_t file_size = in.tellg();
    in.close();

    // Header (8 bytes) + data (6 floats * 4 bytes)
    BOOST_CHECK_EQUAL(file_size, 8 + 24);

    // Load and verify
    std::unique_ptr<float[]> loaded_data;
    size_t npts, ndims;
    diskann::load_bin<float>(test_file, loaded_data, npts, ndims);

    BOOST_CHECK_EQUAL(npts, 2);
    BOOST_CHECK_EQUAL(ndims, 3);
    BOOST_CHECK_CLOSE(loaded_data[0], 1.0f, 0.001f);
    BOOST_CHECK_CLOSE(loaded_data[5], 6.0f, 0.001f);

    std::remove(test_file);
}

BOOST_AUTO_TEST_SUITE_END()
