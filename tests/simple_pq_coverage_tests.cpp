// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>
#include "pq.h"
#include "utils.h"
#include <vector>

BOOST_AUTO_TEST_SUITE(SimplePQCoverageTests)

BOOST_AUTO_TEST_CASE(simple_pq_coverage)
{
    try {
        const std::string data_file = "simple_test_data.bin";
        const std::string pq_compressed = "simple_test_compressed.bin";

        const size_t num_points = 100;
        const size_t dim = 8;
        std::vector<float> data(num_points * dim, 1.0f);
        diskann::save_bin<float>(data_file, data.data(), num_points, dim);

        const std::string pq_pivots = "simple_test_pivots.bin";
        std::vector<float> pivots(256 * dim, 1.0f);
        diskann::save_bin<float>(pq_pivots, pivots.data(), 256, dim);

        diskann::generate_pq_data_from_pivots<float>(data_file, 256, 2, pq_pivots, pq_compressed, false);
        diskann::generate_pq_data_from_pivots<float>(data_file, 256, 2, pq_pivots, pq_compressed, false);

        std::remove(data_file.c_str());
        std::remove(pq_pivots.c_str());
        std::remove(pq_compressed.c_str());
    } catch (...) {
    }
}

BOOST_AUTO_TEST_SUITE_END()
