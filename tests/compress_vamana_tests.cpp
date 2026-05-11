// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstdio>
#include "compressed_graph.h"
#include "disk_utils.h"

BOOST_AUTO_TEST_SUITE(CompressVamanaTests)

namespace
{

std::string create_vamana_file(const std::string &path, uint32_t width, uint32_t medoid,
                               const std::vector<std::vector<uint32_t>> &graph)
{
    std::ofstream out(path, std::ios::binary);
    uint64_t file_size = 0;
    out.write((char *)&file_size, sizeof(uint64_t));
    out.write((char *)&width, sizeof(uint32_t));
    out.write((char *)&medoid, sizeof(uint32_t));
    uint64_t frozen_num = 0;
    out.write((char *)&frozen_num, sizeof(uint64_t));

    for (const auto &neighbors : graph)
    {
        uint32_t degree = static_cast<uint32_t>(neighbors.size());
        out.write((char *)&degree, sizeof(uint32_t));
        if (degree > 0)
        {
            out.write((char *)neighbors.data(), degree * sizeof(uint32_t));
        }
    }

    uint64_t total_size = static_cast<uint64_t>(out.tellp());
    out.seekp(0);
    out.write((char *)&total_size, sizeof(uint64_t));
    out.close();
    return path;
}

} // namespace

BOOST_AUTO_TEST_CASE(compress_vamana_graph_basic)
{
    std::string mem_index = "/tmp/test_compress_vamana_basic.index";
    std::string comp_file = mem_index + ".vamana.comp";

    std::vector<std::vector<uint32_t>> graph = {{1, 2, 3}, {0, 2}, {0, 1, 3}, {0, 2}};

    create_vamana_file(mem_index, 64, 0, graph);

    int ret = diskann::compress_vamana_graph(mem_index.c_str());
    BOOST_CHECK_EQUAL(ret, 0);

    CompressedGraph cg(comp_file);
    BOOST_CHECK_EQUAL(cg.getPointsNum(), 4u);
    BOOST_CHECK_EQUAL(cg.getWidth(), 64u);
    BOOST_CHECK_EQUAL(cg.getEp(), 0u);

    for (uint32_t i = 0; i < 4; i++)
    {
        std::vector<uint32_t> neighbors;
        cg.getNeighbors(i, neighbors);
        std::vector<uint32_t> expected = graph[i];
        std::sort(expected.begin(), expected.end());
        BOOST_CHECK_EQUAL(neighbors.size(), expected.size());
        for (size_t j = 0; j < neighbors.size(); j++)
        {
            BOOST_CHECK_EQUAL(neighbors[j], expected[j]);
        }
    }

    std::remove(mem_index.c_str());
    std::remove(comp_file.c_str());
}

BOOST_AUTO_TEST_CASE(compress_vamana_graph_single_node)
{
    std::string mem_index = "/tmp/test_compress_vamana_single.index";
    std::string comp_file = mem_index + ".vamana.comp";

    std::vector<std::vector<uint32_t>> graph = {{}};

    create_vamana_file(mem_index, 32, 0, graph);

    int ret = diskann::compress_vamana_graph(mem_index.c_str());
    BOOST_CHECK_EQUAL(ret, 0);

    CompressedGraph cg(comp_file);
    BOOST_CHECK_EQUAL(cg.getPointsNum(), 1u);
    BOOST_CHECK_EQUAL(cg.getWidth(), 32u);
    BOOST_CHECK_EQUAL(cg.getEp(), 0u);

    std::vector<uint32_t> neighbors;
    cg.getNeighbors(0, neighbors);
    BOOST_CHECK_EQUAL(neighbors.size(), 0u);

    std::remove(mem_index.c_str());
    std::remove(comp_file.c_str());
}

BOOST_AUTO_TEST_CASE(compress_vamana_graph_varying_degrees)
{
    std::string mem_index = "/tmp/test_compress_vamana_varying.index";
    std::string comp_file = mem_index + ".vamana.comp";

    std::vector<std::vector<uint32_t>> graph = {
        {1, 2, 3, 4, 5, 6, 7},
        {0},
        {0, 1},
        {},
        {0, 1, 2, 3},
        {4, 6},
        {0, 7},
        {0, 1, 2, 3, 4, 5, 6}};

    create_vamana_file(mem_index, 128, 2, graph);

    int ret = diskann::compress_vamana_graph(mem_index.c_str());
    BOOST_CHECK_EQUAL(ret, 0);

    CompressedGraph cg(comp_file);
    BOOST_CHECK_EQUAL(cg.getPointsNum(), 8u);
    BOOST_CHECK_EQUAL(cg.getWidth(), 128u);
    BOOST_CHECK_EQUAL(cg.getEp(), 2u);

    for (uint32_t i = 0; i < 8; i++)
    {
        std::vector<uint32_t> neighbors;
        cg.getNeighbors(i, neighbors);
        std::vector<uint32_t> expected = graph[i];
        std::sort(expected.begin(), expected.end());
        BOOST_CHECK_EQUAL(neighbors.size(), expected.size());
        for (size_t j = 0; j < neighbors.size(); j++)
        {
            BOOST_CHECK_EQUAL(neighbors[j], expected[j]);
        }
    }

    std::remove(mem_index.c_str());
    std::remove(comp_file.c_str());
}

BOOST_AUTO_TEST_SUITE_END()
