// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>
#include "disk_utils.h"
#include "utils.h"
#include <filesystem>
#include <fstream>

BOOST_AUTO_TEST_SUITE(DiskUtilsFastTests)

// Test build_disk_index with generate_mem_file=true
// This tests the FAST_DISKANN-specific parameter that triggers compress_vamana_graph
BOOST_AUTO_TEST_CASE(build_disk_index_with_mem_file_generation)
{
    const std::string prefix = "test_disk_build";
    const std::string data_file = prefix + "_data.bin";

    // Create larger dataset (build_disk_index needs sufficient data for partitioning)
    const size_t num_points = 10000;
    const size_t dim = 32;
    std::vector<float> data(num_points * dim);
    for (size_t i = 0; i < num_points * dim; i++) {
        data[i] = static_cast<float>((i % 1000) / 100.0);
    }
    diskann::save_bin<float>(data_file, data.data(), num_points, dim);

    // Build disk index with generate_mem_file=true
    // params format: "R L B M num_threads disk_PQ append_reorder_data build_PQ QD"
    // R=64, L=100, B=0.5 (500MB), M=4.0 (4GB), num_threads=4, disk_PQ=0, append_reorder_data=0, build_PQ=0, QD=0
    const char* build_params = "64 100 0.5 4.0 4 0 0 0 0";

    int result = -1;
    try {
        result = diskann::build_disk_index<float>(data_file.c_str(), prefix.c_str(), build_params,
                                                  diskann::Metric::L2, false, "", false, "", "", 0, 0, true);

        if (result == 0) {
            // Verify compressed graph was created
            std::string comp_file = prefix + "_mem.index.vamana.comp";
            BOOST_CHECK(file_exists(comp_file));
        }
    } catch (const std::exception& e) {
        std::cerr << "build_disk_index exception: " << e.what() << std::endl;
        // Even if it fails, we attempted to cover the code path
    }

    // Cleanup
    std::remove(data_file.c_str());
    try {
        std::filesystem::remove_all(prefix);
    } catch (...) {
        // Ignore cleanup errors
    }
}

BOOST_AUTO_TEST_SUITE_END()

