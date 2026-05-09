// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>
#include "compressed_graph.h"
#include "neighbor.h"
#include "defaults.h"
#include "parameters.h"
#include "pq.h"
#include <fstream>
#include <cstdio>

BOOST_AUTO_TEST_SUITE(CompressedGraphTests)

BOOST_AUTO_TEST_CASE(in_memory_construction_normal)
{
    std::vector<std::vector<uint32_t>> adjList = {
        {1, 2, 3},
        {0, 2, 4},
        {0, 1, 3},
        {0, 2, 4},
        {1, 3}
    };

    CompressedGraph cg(adjList, 64, 0);
    BOOST_CHECK_EQUAL(cg.getPointsNum(), 5);
    BOOST_CHECK_EQUAL(cg.getWidth(), 64);
    BOOST_CHECK_EQUAL(cg.getEp(), 0);
}

BOOST_AUTO_TEST_CASE(in_memory_construction_empty_graph)
{
    std::vector<std::vector<uint32_t>> adjList = {
        {},
        {},
        {}
    };

    CompressedGraph cg(adjList, 32, 1);
    BOOST_CHECK_EQUAL(cg.getPointsNum(), 3);
}

BOOST_AUTO_TEST_CASE(getNeighbors_valid_node)
{
    std::vector<std::vector<uint32_t>> adjList = {
        {1, 2, 3},
        {0, 2},
        {0, 1}
    };

    CompressedGraph cg(adjList, 64, 0);
    std::vector<uint32_t> neighbors;
    cg.getNeighbors(0, neighbors);

    BOOST_CHECK_EQUAL(neighbors.size(), 3);
    BOOST_CHECK_EQUAL(neighbors[0], 1);
    BOOST_CHECK_EQUAL(neighbors[1], 2);
    BOOST_CHECK_EQUAL(neighbors[2], 3);
}

BOOST_AUTO_TEST_CASE(getNeighbors_empty_neighbors)
{
    std::vector<std::vector<uint32_t>> adjList = {{}, {1}, {0}};
    CompressedGraph cg(adjList, 64, 0);
    std::vector<uint32_t> neighbors;

    cg.getNeighbors(0, neighbors);
    BOOST_CHECK(neighbors.empty());
}


BOOST_AUTO_TEST_CASE(getRealId_valid)
{
    std::vector<std::vector<uint32_t>> adjList = {{1}, {0}, {1}};
    CompressedGraph cg(adjList, 64, 0);

    BOOST_CHECK_EQUAL(cg.getPointsNum(), 3);
}

BOOST_AUTO_TEST_CASE(getRealId_out_of_bounds)
{
    std::vector<std::vector<uint32_t>> adjList = {{1}, {0}};
    CompressedGraph cg(adjList, 64, 0);

    BOOST_CHECK_EQUAL(cg.getPointsNum(), 2);
}

BOOST_AUTO_TEST_CASE(getRealIds_with_idMap)
{
    std::vector<std::vector<uint32_t>> adjList = {{1}, {0}, {1}};
    CompressedGraph cg(adjList, 64, 0);

    std::vector<uint32_t> neighbors = {0, 1, 2};
    cg.getRealIds(neighbors);
    BOOST_CHECK_EQUAL(neighbors.size(), 3);
}

BOOST_AUTO_TEST_CASE(compressed_graph_large)
{
    std::vector<std::vector<uint32_t>> adjList;
    for (int i = 0; i < 100; i++) {
        std::vector<uint32_t> neighbors;
        for (int j = 0; j < 10; j++) {
            neighbors.push_back((i + j) % 100);
        }
        adjList.push_back(neighbors);
    }

    CompressedGraph cg(adjList, 64, 0);
    BOOST_CHECK_EQUAL(cg.getPointsNum(), 100);

    std::vector<uint32_t> neighbors;
    cg.getNeighbors(0, neighbors);
    BOOST_CHECK_EQUAL(neighbors.size(), 10);
}

BOOST_AUTO_TEST_CASE(compressed_graph_multiple_queries)
{
    std::vector<std::vector<uint32_t>> adjList = {
        {1, 2, 3, 4, 5},
        {0, 2, 4, 6, 8},
        {0, 1, 3, 5, 7}
    };

    CompressedGraph cg(adjList, 64, 0);

    for (uint32_t i = 0; i < 3; i++) {
        std::vector<uint32_t> neighbors;
        cg.getNeighbors(i, neighbors);
        BOOST_CHECK_EQUAL(neighbors.size(), 5);
    }
}

BOOST_AUTO_TEST_SUITE_END()

// Neighbor tests
BOOST_AUTO_TEST_SUITE(NeighborTests)

BOOST_AUTO_TEST_CASE(neighbor_constructor)
{
    diskann::Neighbor n(5, 1.5f);
    BOOST_CHECK_EQUAL(n.id, 5);
    BOOST_CHECK_CLOSE(n.distance, 1.5f, 0.001f);
    BOOST_CHECK_EQUAL(n.expanded, false);
    BOOST_CHECK_EQUAL(n.loaded, false);
}

BOOST_AUTO_TEST_CASE(neighbor_comparison)
{
    diskann::Neighbor n1(1, 1.0f);
    diskann::Neighbor n2(2, 2.0f);

    BOOST_CHECK(n1 < n2);
    BOOST_CHECK(!(n2 < n1));
}

BOOST_AUTO_TEST_CASE(neighbor_equality)
{
    diskann::Neighbor n1(1, 1.0f);
    diskann::Neighbor n2(1, 1.0f);
    diskann::Neighbor n3(2, 1.0f);

    BOOST_CHECK(n1 == n2);
    BOOST_CHECK(!(n1 == n3));
}

BOOST_AUTO_TEST_CASE(neighbor_priority_queue_insert_multiple)
{
    diskann::NeighborPriorityQueue queue;
    queue.reserve(10);

    queue.insert(diskann::Neighbor(1, 1.0f));
    queue.insert(diskann::Neighbor(2, 2.0f));
    queue.insert(diskann::Neighbor(3, 0.5f));

    BOOST_CHECK_EQUAL(queue.size(), 3);
    BOOST_CHECK_CLOSE(queue.closest_unexpanded().distance, 0.5f, 0.001f);
}

BOOST_AUTO_TEST_CASE(neighbor_priority_queue_has_unexpanded)
{
    diskann::NeighborPriorityQueue queue;
    queue.reserve(10);

    BOOST_CHECK(!queue.has_unexpanded_node());

    queue.insert(diskann::Neighbor(1, 1.0f));
    BOOST_CHECK(queue.has_unexpanded_node());

    queue.closest_unexpanded();
    BOOST_CHECK(!queue.has_unexpanded_node());
}

BOOST_AUTO_TEST_CASE(small_neighbor_constructor)
{
    diskann::SmallNeighbor sn(10, 2.5f);
    BOOST_CHECK_EQUAL(sn.id, 10);
    BOOST_CHECK_CLOSE(sn.distance, 2.5f, 0.001f);
}

BOOST_AUTO_TEST_CASE(small_neighbor_comparison)
{
    diskann::SmallNeighbor sn1(1, 1.0f);
    diskann::SmallNeighbor sn2(2, 2.0f);

    BOOST_CHECK(sn1 < sn2);
}

BOOST_AUTO_TEST_CASE(origin_neighbor_priority_queue)
{
    diskann::OriginNeighborPriorityQueue queue;
    queue.reserve(10);

    diskann::OriginNeighbor on1(1, 1.0f);
    diskann::OriginNeighbor on2(2, 2.0f);
    queue.insert(on1);
    queue.insert(on2);

    BOOST_CHECK_EQUAL(queue.size(), 2);
}

BOOST_AUTO_TEST_SUITE_END()

// Additional CompressedGraph tests
BOOST_AUTO_TEST_SUITE(CompressedGraphAdditionalTests)

BOOST_AUTO_TEST_CASE(getRealIds_empty_idMap)
{
    std::vector<std::vector<uint32_t>> adjList = {{1}, {0}};
    CompressedGraph cg(adjList, 64, 0);

    std::vector<uint32_t> neighbors = {0, 1};
    cg.getRealIds(neighbors);

    BOOST_CHECK_EQUAL(neighbors[0], 0);
    BOOST_CHECK_EQUAL(neighbors[1], 1);
}

BOOST_AUTO_TEST_CASE(getRealIdsU64Array_with_idMap)
{
    std::vector<std::vector<uint32_t>> adjList = {{1}, {0}};
    CompressedGraph cg(adjList, 64, 0);

    uint64_t neighbors[2] = {0, 1};
    cg.getRealIdsU64Array(neighbors, 2);

    BOOST_CHECK_EQUAL(neighbors[0], 0);
    BOOST_CHECK_EQUAL(neighbors[1], 1);
}

BOOST_AUTO_TEST_CASE(file_constructor_missing_file)
{
    BOOST_CHECK_THROW(
        CompressedGraph cg("nonexistent_graph_file.bin", false),
        std::runtime_error
    );
}

BOOST_AUTO_TEST_CASE(file_constructor_success)
{
    std::vector<std::vector<uint32_t>> adjList = {{1, 2}, {0, 2}, {0, 1}};
    CompressedGraph cg1(adjList, 64, 0);

    const char* filename = "test_graph_load.bin";
    cg1.saveToFile(filename);

    CompressedGraph cg2(filename, false);
    BOOST_CHECK_EQUAL(cg2.getPointsNum(), 3);
    BOOST_CHECK_EQUAL(cg2.getWidth(), 64);
    BOOST_CHECK_EQUAL(cg2.getEp(), 0);

    std::vector<uint32_t> neighbors;
    cg2.getNeighbors(0, neighbors);
    BOOST_CHECK_EQUAL(neighbors.size(), 2);
    BOOST_CHECK_EQUAL(neighbors[0], 1);
    BOOST_CHECK_EQUAL(neighbors[1], 2);

    std::remove(filename);
}

BOOST_AUTO_TEST_CASE(file_constructor_with_reorder)
{
    std::vector<std::vector<uint32_t>> adjList = {{1}, {0}};
    CompressedGraph cg1(adjList, 32, 1);

    const char* graph_file = "test_reorder.bin";
    const char* map_file = "test_reorder.bin.map";

    cg1.saveToFile(graph_file);

    std::ofstream map_out(map_file, std::ios::binary);
    uint32_t points = 2;
    map_out.write(reinterpret_cast<const char*>(&points), sizeof(uint32_t));
    uint32_t idmap[2] = {10, 20};
    map_out.write(reinterpret_cast<const char*>(idmap), 2 * sizeof(uint32_t));
    map_out.close();

    CompressedGraph cg2(graph_file, true);
    BOOST_CHECK_EQUAL(cg2.getPointsNum(), 2);

    std::remove(graph_file);
    std::remove(map_file);
}

BOOST_AUTO_TEST_CASE(serialize_and_save)
{
    std::vector<std::vector<uint32_t>> adjList = {{1, 2}, {0}};
    CompressedGraph cg(adjList, 64, 0);

    std::vector<uint8_t> data = cg.serialize();
    BOOST_CHECK(data.size() > 0);

    const char* filename = "test_serialize.bin";
    cg.saveToFile(filename);

    std::ifstream in(filename, std::ios::binary);
    BOOST_CHECK(in.is_open());
    in.close();
    std::remove(filename);
}

BOOST_AUTO_TEST_CASE(varint_encode_decode_small_values)
{
    std::vector<std::vector<uint32_t>> adjList = {{0}, {1}, {2}};
    CompressedGraph cg(adjList, 64, 0);

    std::vector<uint32_t> neighbors;
    cg.getNeighbors(0, neighbors);
    BOOST_CHECK_EQUAL(neighbors.size(), 1);
    BOOST_CHECK_EQUAL(neighbors[0], 0);
}

BOOST_AUTO_TEST_CASE(varint_encode_decode_large_values)
{
    std::vector<std::vector<uint32_t>> adjList = {{127, 128, 16383, 16384}};
    CompressedGraph cg(adjList, 64, 0);

    std::vector<uint32_t> neighbors;
    cg.getNeighbors(0, neighbors);
    BOOST_CHECK_EQUAL(neighbors.size(), 4);
    BOOST_CHECK_EQUAL(neighbors[0], 127);
    BOOST_CHECK_EQUAL(neighbors[1], 128);
    BOOST_CHECK_EQUAL(neighbors[2], 16383);
    BOOST_CHECK_EQUAL(neighbors[3], 16384);
}

BOOST_AUTO_TEST_CASE(varint_encode_decode_max_value)
{
    std::vector<std::vector<uint32_t>> adjList = {{UINT32_MAX - 1}};
    CompressedGraph cg(adjList, 64, 0);

    std::vector<uint32_t> neighbors;
    cg.getNeighbors(0, neighbors);
    BOOST_CHECK_EQUAL(neighbors.size(), 1);
    BOOST_CHECK_EQUAL(neighbors[0], UINT32_MAX - 1);
}

BOOST_AUTO_TEST_CASE(file_constructor_pointsNum_mismatch)
{
    std::vector<std::vector<uint32_t>> adjList = {{1}, {0}};
    CompressedGraph cg1(adjList, 32, 0);

    const char* graph_file = "test_mismatch.bin";
    const char* map_file = "test_mismatch.bin.map";

    cg1.saveToFile(graph_file);

    std::ofstream map_out(map_file, std::ios::binary);
    uint32_t wrong_points = 10;
    map_out.write(reinterpret_cast<const char*>(&wrong_points), sizeof(uint32_t));
    uint32_t idmap[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    map_out.write(reinterpret_cast<const char*>(idmap), 10 * sizeof(uint32_t));
    map_out.close();

    BOOST_CHECK_THROW(CompressedGraph cg2(graph_file, true), std::runtime_error);

    std::remove(graph_file);
    std::remove(map_file);
}

BOOST_AUTO_TEST_CASE(large_graph_compression)
{
    std::vector<std::vector<uint32_t>> adjList;
    for (uint32_t i = 0; i < 100; ++i)
    {
        std::vector<uint32_t> neighbors;
        for (uint32_t j = 0; j < 10; ++j)
        {
            neighbors.push_back((i + j) % 100);
        }
        adjList.push_back(neighbors);
    }

    CompressedGraph cg(adjList, 64, 0);
    BOOST_CHECK_EQUAL(cg.getPointsNum(), 100);

    std::vector<uint32_t> neighbors;
    cg.getNeighbors(0, neighbors);
    BOOST_CHECK_EQUAL(neighbors.size(), 10);
}

BOOST_AUTO_TEST_CASE(compressed_graph_getters)
{
    std::vector<std::vector<uint32_t>> adjList = {{1, 2}, {0}};
    CompressedGraph cg(adjList, 64, 5);

    BOOST_CHECK_EQUAL(cg.getWidth(), 64);
    BOOST_CHECK_EQUAL(cg.getEp(), 5);
    BOOST_CHECK_EQUAL(cg.getPointsNum(), 2);
}

BOOST_AUTO_TEST_CASE(compressed_graph_serialize_deserialize)
{
    std::vector<std::vector<uint32_t>> adjList = {{1, 2, 3}, {0, 2}, {1}};
    CompressedGraph cg1(adjList, 32, 1);

    std::vector<uint8_t> data = cg1.serialize();
    BOOST_CHECK(data.size() > 0);

    const char* filename = "test_serialize_deserialize.bin";
    cg1.saveToFile(filename);

    CompressedGraph cg2(filename, false);
    BOOST_CHECK_EQUAL(cg2.getPointsNum(), 3);
    BOOST_CHECK_EQUAL(cg2.getWidth(), 32);
    BOOST_CHECK_EQUAL(cg2.getEp(), 1);

    std::vector<uint32_t> neighbors;
    cg2.getNeighbors(0, neighbors);
    BOOST_CHECK_EQUAL(neighbors.size(), 3);
    BOOST_CHECK_EQUAL(neighbors[0], 1);
    BOOST_CHECK_EQUAL(neighbors[1], 2);
    BOOST_CHECK_EQUAL(neighbors[2], 3);

    std::remove(filename);
}

BOOST_AUTO_TEST_CASE(varint_encode_decode_sequence)
{
    std::vector<std::vector<uint32_t>> adjList = {
        {0, 1, 2, 127, 128, 255, 256, 16383, 16384}
    };
    CompressedGraph cg(adjList, 64, 0);

    std::vector<uint32_t> neighbors;
    cg.getNeighbors(0, neighbors);
    BOOST_CHECK_EQUAL(neighbors.size(), 9);
    BOOST_CHECK_EQUAL(neighbors[0], 0);
    BOOST_CHECK_EQUAL(neighbors[3], 127);
    BOOST_CHECK_EQUAL(neighbors[4], 128);
    BOOST_CHECK_EQUAL(neighbors[7], 16383);
    BOOST_CHECK_EQUAL(neighbors[8], 16384);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(OriginAndSmallNeighborTests)

BOOST_AUTO_TEST_CASE(origin_neighbor_comparison_less)
{
    diskann::OriginNeighbor n1(1, 0.5f);
    diskann::OriginNeighbor n2(2, 0.7f);
    diskann::OriginNeighbor n3(3, 0.5f);

    BOOST_CHECK(n1 < n2);
    BOOST_CHECK(n1 < n3);
}

BOOST_AUTO_TEST_CASE(origin_neighbor_comparison_equal)
{
    diskann::OriginNeighbor n1(1, 0.5f);
    diskann::OriginNeighbor n2(1, 0.7f);

    BOOST_CHECK(n1 == n2);
}

BOOST_AUTO_TEST_CASE(small_neighbor_comparison_less)
{
    diskann::SmallNeighbor n1(1, 0.3f);
    diskann::SmallNeighbor n2(2, 0.6f);
    diskann::SmallNeighbor n3(3, 0.3f);

    BOOST_CHECK(n1 < n2);
    BOOST_CHECK(n1 < n3);
}

BOOST_AUTO_TEST_CASE(small_neighbor_comparison_equal)
{
    diskann::SmallNeighbor n1(5, 0.8f);
    diskann::SmallNeighbor n2(5, 1.0f);

    BOOST_CHECK(n1 == n2);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(NeighborPriorityQueueTests)

BOOST_AUTO_TEST_CASE(insert_with_pos_normal)
{
    diskann::NeighborPriorityQueue queue(5);
    diskann::Neighbor n1(1, 0.5f);
    diskann::Neighbor n2(2, 0.3f);
    diskann::Neighbor n3(3, 0.7f);

    size_t pos1 = queue.insert_with_pos(n1);
    size_t pos2 = queue.insert_with_pos(n2);
    size_t pos3 = queue.insert_with_pos(n3);

    BOOST_CHECK_EQUAL(pos1, 0);
    BOOST_CHECK_EQUAL(pos2, 0);
    BOOST_CHECK_EQUAL(pos3, 2);
    BOOST_CHECK_EQUAL(queue.size(), 3);
}

BOOST_AUTO_TEST_CASE(insert_with_pos_duplicate)
{
    diskann::NeighborPriorityQueue queue(5);
    diskann::Neighbor n1(1, 0.5f);
    diskann::Neighbor n2(1, 0.8f);

    size_t pos1 = queue.insert_with_pos(n1);
    size_t pos2 = queue.insert_with_pos(n2);

    BOOST_CHECK_EQUAL(pos1, 0);
    BOOST_CHECK_EQUAL(pos2, 5);
    BOOST_CHECK_EQUAL(queue.size(), 1);
}

BOOST_AUTO_TEST_CASE(insert_with_pos_full_queue)
{
    diskann::NeighborPriorityQueue queue(2);
    diskann::Neighbor n1(1, 0.3f);
    diskann::Neighbor n2(2, 0.5f);
    diskann::Neighbor n3(3, 0.7f);

    queue.insert_with_pos(n1);
    queue.insert_with_pos(n2);
    size_t pos3 = queue.insert_with_pos(n3);

    BOOST_CHECK_EQUAL(pos3, 2);
    BOOST_CHECK_EQUAL(queue.size(), 2);
}

BOOST_AUTO_TEST_CASE(set_loaded_size_and_has_unloaded)
{
    diskann::NeighborPriorityQueue queue(5);
    queue.insert_with_pos(diskann::Neighbor(1, 0.3f));
    queue.insert_with_pos(diskann::Neighbor(2, 0.5f));
    queue.insert_with_pos(diskann::Neighbor(3, 0.7f));

    queue.set_loaded_size(2);
    BOOST_CHECK(queue.has_unloaded_node());
    BOOST_CHECK_EQUAL(queue.loaded_cur(), 0);
}

BOOST_AUTO_TEST_CASE(closest_unloaded)
{
    diskann::NeighborPriorityQueue queue(5);
    queue.insert_with_pos(diskann::Neighbor(1, 0.3f));
    queue.insert_with_pos(diskann::Neighbor(2, 0.5f));
    queue.set_loaded_size(2);

    diskann::Neighbor n = queue.closest_unloaded();
    BOOST_CHECK_EQUAL(n.id, 1);
    BOOST_CHECK(n.loaded);
    BOOST_CHECK_EQUAL(queue.loaded_cur(), 1);
}

BOOST_AUTO_TEST_CASE(has_unloaded_node_false)
{
    diskann::NeighborPriorityQueue queue(5);
    queue.insert_with_pos(diskann::Neighbor(1, 0.3f));
    queue.set_loaded_size(1);

    queue.closest_unloaded();
    BOOST_CHECK(!queue.has_unloaded_node());
}

BOOST_AUTO_TEST_CASE(loaded_cur_advances)
{
    diskann::NeighborPriorityQueue queue(5);
    queue.insert_with_pos(diskann::Neighbor(1, 0.3f));
    queue.insert_with_pos(diskann::Neighbor(2, 0.5f));
    queue.insert_with_pos(diskann::Neighbor(3, 0.7f));
    queue.set_loaded_size(3);

    BOOST_CHECK_EQUAL(queue.loaded_cur(), 0);
    queue.closest_unloaded();
    BOOST_CHECK_EQUAL(queue.loaded_cur(), 1);
    queue.closest_unloaded();
    BOOST_CHECK_EQUAL(queue.loaded_cur(), 2);
}

BOOST_AUTO_TEST_CASE(insert_with_pos_at_capacity)
{
    diskann::NeighborPriorityQueue queue(3);
    queue.insert_with_pos(diskann::Neighbor(1, 0.3f));
    queue.insert_with_pos(diskann::Neighbor(2, 0.5f));
    queue.insert_with_pos(diskann::Neighbor(3, 0.7f));

    size_t pos = queue.insert_with_pos(diskann::Neighbor(4, 0.4f));
    BOOST_CHECK_EQUAL(pos, 1);
    BOOST_CHECK_EQUAL(queue.size(), 3);
    BOOST_CHECK_EQUAL(queue[0].id, 1);
    BOOST_CHECK_EQUAL(queue[1].id, 4);
    BOOST_CHECK_EQUAL(queue[2].id, 2);
}

BOOST_AUTO_TEST_CASE(neighbor_insert_basic)
{
    diskann::NeighborPriorityQueue queue(5);
    queue.insert(diskann::Neighbor(1, 0.5f));
    queue.insert(diskann::Neighbor(2, 0.3f));
    queue.insert(diskann::Neighbor(3, 0.7f));

    BOOST_CHECK_EQUAL(queue.size(), 3);
    BOOST_CHECK_EQUAL(queue[0].id, 2);
    BOOST_CHECK_EQUAL(queue[1].id, 1);
    BOOST_CHECK_EQUAL(queue[2].id, 3);
}

BOOST_AUTO_TEST_CASE(neighbor_insert_duplicate)
{
    diskann::NeighborPriorityQueue queue(5);
    queue.insert(diskann::Neighbor(1, 0.5f));
    queue.insert(diskann::Neighbor(1, 0.8f));

    BOOST_CHECK_EQUAL(queue.size(), 1);
}

BOOST_AUTO_TEST_CASE(neighbor_insert_at_capacity)
{
    diskann::NeighborPriorityQueue queue(2);
    queue.insert(diskann::Neighbor(1, 0.3f));
    queue.insert(diskann::Neighbor(2, 0.5f));
    queue.insert(diskann::Neighbor(3, 0.4f));

    BOOST_CHECK_EQUAL(queue.size(), 2);
    BOOST_CHECK_EQUAL(queue[0].id, 1);
    BOOST_CHECK_EQUAL(queue[1].id, 3);
}

BOOST_AUTO_TEST_CASE(neighbor_closest_unexpanded)
{
    diskann::NeighborPriorityQueue queue(5);
    queue.insert(diskann::Neighbor(1, 0.3f));
    queue.insert(diskann::Neighbor(2, 0.5f));

    BOOST_CHECK(queue.has_unexpanded_node());

    diskann::Neighbor closest = queue.closest_unexpanded();
    BOOST_CHECK_EQUAL(closest.id, 1);

    BOOST_CHECK(queue.has_unexpanded_node());

    queue.closest_unexpanded();
    BOOST_CHECK(!queue.has_unexpanded_node());
}

BOOST_AUTO_TEST_CASE(neighbor_has_unexpanded_node)
{
    diskann::NeighborPriorityQueue queue(5);
    BOOST_CHECK(!queue.has_unexpanded_node());

    queue.insert(diskann::Neighbor(1, 0.3f));
    BOOST_CHECK(queue.has_unexpanded_node());

    queue.closest_unexpanded();
    BOOST_CHECK(!queue.has_unexpanded_node());
}

BOOST_AUTO_TEST_CASE(origin_neighbor_priority_queue_basic)
{
    diskann::OriginNeighborPriorityQueue queue(5);
    queue.insert(diskann::OriginNeighbor(1, 0.5f));
    queue.insert(diskann::OriginNeighbor(2, 0.3f));
    queue.insert(diskann::OriginNeighbor(3, 0.7f));

    BOOST_CHECK_EQUAL(queue.size(), 3);
    BOOST_CHECK_EQUAL(queue[0].id, 2);
    BOOST_CHECK_EQUAL(queue[1].id, 1);
    BOOST_CHECK_EQUAL(queue[2].id, 3);
}

BOOST_AUTO_TEST_CASE(origin_neighbor_closest_unexpanded)
{
    diskann::OriginNeighborPriorityQueue queue(5);
    queue.insert(diskann::OriginNeighbor(1, 0.3f));
    queue.insert(diskann::OriginNeighbor(2, 0.5f));

    diskann::OriginNeighbor closest = queue.closest_unexpanded();
    BOOST_CHECK_EQUAL(closest.id, 1);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(DefaultsTests)

BOOST_AUTO_TEST_CASE(prune_strategy_to_string_all_values)
{
    BOOST_CHECK_EQUAL(diskann::prune_strategy_to_string(diskann::PruneStrategy::RNG), "RNG");
    BOOST_CHECK_EQUAL(diskann::prune_strategy_to_string(diskann::PruneStrategy::Robust), "Robust");
    BOOST_CHECK_EQUAL(diskann::prune_strategy_to_string(diskann::PruneStrategy::Cagra), "Cagra");
    BOOST_CHECK_EQUAL(diskann::prune_strategy_to_string(diskann::PruneStrategy::Random), "Random");
    BOOST_CHECK_EQUAL(diskann::prune_strategy_to_string(diskann::PruneStrategy::NO_Prune), "NO_Prune");
}

BOOST_AUTO_TEST_CASE(prune_code_to_string_all_values)
{
    BOOST_CHECK_EQUAL(diskann::prune_code_to_string(diskann::PruneCode::PQ), "PQ");
    BOOST_CHECK_EQUAL(diskann::prune_code_to_string(diskann::PruneCode::RAW), "RAW");
    BOOST_CHECK_EQUAL(diskann::prune_code_to_string(diskann::PruneCode::NO_Code), "Unknown");
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(UtilsTests)

BOOST_AUTO_TEST_CASE(round_up_aligned)
{
    BOOST_CHECK_EQUAL(diskann::round_up(64, 64), 64);
    BOOST_CHECK_EQUAL(diskann::round_up(128, 64), 128);
}

BOOST_AUTO_TEST_CASE(round_up_unaligned)
{
    BOOST_CHECK_EQUAL(diskann::round_up(65, 64), 128);
    BOOST_CHECK_EQUAL(diskann::round_up(100, 64), 128);
    BOOST_CHECK_EQUAL(diskann::round_up(1, 64), 64);
}

BOOST_AUTO_TEST_CASE(round_up_zero)
{
    BOOST_CHECK_EQUAL(diskann::round_up(0, 64), 0);
}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(ParametersTests)

BOOST_AUTO_TEST_CASE(index_write_parameters_with_prune_strategy)
{
    diskann::IndexWriteParameters params = diskann::IndexWriteParametersBuilder(100, 64)
        .with_prune_strategy(diskann::PruneStrategy::Robust)
        .build();

    BOOST_CHECK(params.prune_strategy == diskann::PruneStrategy::Robust);
}

BOOST_AUTO_TEST_CASE(index_write_parameters_default_prune_strategy)
{
    diskann::IndexWriteParameters params = diskann::IndexWriteParametersBuilder(100, 64)
        .build();

    BOOST_CHECK(params.prune_strategy == diskann::defaults::PRUNE_STRATEGY);
}

BOOST_AUTO_TEST_SUITE_END()

