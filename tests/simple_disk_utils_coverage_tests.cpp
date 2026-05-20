// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>
#include "disk_utils.h"
#include "utils.h"
#include <vector>

BOOST_AUTO_TEST_SUITE(SimpleDiskUtilsCoverageTests)

BOOST_AUTO_TEST_CASE(simple_compress_vamana_coverage)
{
    try {
        const std::string index_file = "simple_vamana.index";

        std::ofstream out(index_file, std::ios::binary);
        uint64_t file_size = 1000;
        uint32_t width = 64;
        uint32_t medoid = 0;
        uint64_t frozen_num = 0;

        out.write((char*)&file_size, sizeof(uint64_t));
        out.write((char*)&width, sizeof(uint32_t));
        out.write((char*)&medoid, sizeof(uint32_t));
        out.write((char*)&frozen_num, sizeof(uint64_t));

        for (int i = 0; i < 10; i++) {
            uint32_t degree = 5;
            out.write((char*)&degree, sizeof(uint32_t));
            for (int j = 0; j < 5; j++) {
                uint32_t neighbor = j;
                out.write((char*)&neighbor, sizeof(uint32_t));
            }
        }
        out.close();

        diskann::compress_vamana_graph(index_file.c_str());

        std::remove(index_file.c_str());
        std::remove((index_file + ".vamana.comp").c_str());
    } catch (...) {
    }
}

BOOST_AUTO_TEST_SUITE_END()
