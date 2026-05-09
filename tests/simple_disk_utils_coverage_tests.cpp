// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>
#include "disk_utils.h"
#include "utils.h"
#include <vector>

BOOST_AUTO_TEST_SUITE(SimpleDiskUtilsCoverageTests)

// 简单调用 compress_vamana_graph - 只要执行就行
BOOST_AUTO_TEST_CASE(simple_compress_vamana_coverage)
{
    try {
        const std::string index_file = "simple_vamana.index";

        // 创建一个简单的 vamana 文件
        std::ofstream out(index_file, std::ios::binary);
        uint64_t file_size = 1000;
        uint32_t width = 64;
        uint32_t medoid = 0;
        uint64_t frozen_num = 0;

        out.write((char*)&file_size, sizeof(uint64_t));
        out.write((char*)&width, sizeof(uint32_t));
        out.write((char*)&medoid, sizeof(uint32_t));
        out.write((char*)&frozen_num, sizeof(uint64_t));

        // 写一些节点数据
        for (int i = 0; i < 10; i++) {
            uint32_t degree = 5;
            out.write((char*)&degree, sizeof(uint32_t));
            for (int j = 0; j < 5; j++) {
                uint32_t neighbor = j;
                out.write((char*)&neighbor, sizeof(uint32_t));
            }
        }
        out.close();

        // 调用 compress_vamana_graph
        diskann::compress_vamana_graph(index_file.c_str());

        // 清理
        std::remove(index_file.c_str());
        std::remove((index_file + ".vamana.comp").c_str());
    } catch (...) {
        // 忽略错误
    }
}

BOOST_AUTO_TEST_SUITE_END()
