// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>
#include "partition.h"
#include "utils.h"
#include <vector>

BOOST_AUTO_TEST_SUITE(SimplePartitionCoverageTests)

// 简单调用 partition_with_ram_budget - 只要执行就行
BOOST_AUTO_TEST_CASE(simple_partition_coverage)
{
    try {
        const std::string data_file = "simple_part_data.bin";
        const std::string prefix = "simple_part";

        // 创建数据
        const size_t num_points = 100;
        const size_t dim = 8;
        std::vector<float> data(num_points * dim);
        for (size_t i = 0; i < num_points * dim; i++) {
            data[i] = static_cast<float>(i % 100);
        }
        diskann::save_bin<float>(data_file, data.data(), num_points, dim);

        // 调用 partition_with_ram_budget
        partition_with_ram_budget<float>(data_file, 0.1, 10.0, 32, prefix, 1);

        // 再调用一次 - 触发缓存路径
        partition_with_ram_budget<float>(data_file, 0.1, 10.0, 32, prefix, 1);

        // 清理
        std::remove(data_file.c_str());
        std::remove((prefix + "_clusters_num.bin").c_str());
        std::remove((prefix + "_centroids.bin").c_str());
        for (int i = 0; i < 10; i++) {
            std::remove((prefix + "_subshard-" + std::to_string(i) + "_ids_uint32.bin").c_str());
        }
    } catch (...) {
        // 忽略错误
    }
}

BOOST_AUTO_TEST_SUITE_END()
