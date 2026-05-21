// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <queue>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <sys/resource.h>

int main(int argc, char *argv[])
{
    auto start_time = std::chrono::high_resolution_clock::now();
    struct rusage start_rusage;
    getrusage(RUSAGE_SELF, &start_rusage);
    const long start_maxrss = start_rusage.ru_maxrss;

    if (argc != 3)
    {
        std::cout << "Usage: calculate_hops_from_entry <vamana_graph_file> <output_bin_file>" << std::endl;
        std::cout << "Output binary file format:" << std::endl;
        std::cout << "  Header (24 bytes): file_size, num_nodes, entry_point, reserved" << std::endl;
        std::cout << "  Per node (12 bytes): node_id, neighbors_num, storage_bytes, hops" << std::endl;
        return -1;
    }

    const std::string vamana_graph_file_path = argv[1];
    const std::string output_bin_path = argv[2];

    // Open graph file
    std::ifstream graph_reader(vamana_graph_file_path, std::ios::binary);
    if (!graph_reader)
    {
        std::cerr << "Failed to open graph file: " << vamana_graph_file_path << std::endl;
        return 1;
    }

    // Read header (24 bytes)
    constexpr std::streamsize header_size = 24;
    char header_buffer[24];
    graph_reader.read(header_buffer, header_size);

    if (!graph_reader)
    {
        std::cerr << "Failed to read header from graph file" << std::endl;
        return 1;
    }

    // Parse header
    uint64_t file_size;
    uint32_t width;
    uint32_t entry_point;
    uint64_t frozen;
    std::memcpy(&file_size, header_buffer, sizeof(file_size));
    std::memcpy(&width, header_buffer + 8, sizeof(width));
    std::memcpy(&entry_point, header_buffer + 12, sizeof(entry_point));
    std::memcpy(&frozen, header_buffer + 16, sizeof(frozen));

    std::cout << "Graph file info:" << std::endl;
    std::cout << "  File size: " << file_size << " bytes" << std::endl;
    std::cout << "  Width: " << width << std::endl;
    std::cout << "  Entry point: " << entry_point << std::endl;
    std::cout << "  Frozen points: " << frozen << std::endl;

    // Load graph adjacency list
    graph_reader.seekg(header_size, std::ios::beg);

    std::vector<std::vector<uint32_t>> graph;
    std::vector<uint32_t> neighbors_num;
    uint64_t total_edges = 0;

    while (true)
    {
        uint32_t k;
        graph_reader.read(reinterpret_cast<char *>(&k), sizeof(uint32_t));
        if (graph_reader.eof())
            break;
        if (!graph_reader)
        {
            std::cerr << "Failed while reading node degree" << std::endl;
            return 1;
        }

        total_edges += k;
        neighbors_num.push_back(k);

        std::vector<uint32_t> neighbors(k);
        graph_reader.read(reinterpret_cast<char *>(neighbors.data()), static_cast<std::streamsize>(k) * sizeof(uint32_t));

        if (!graph_reader)
        {
            std::cerr << "Failed while reading neighbor list" << std::endl;
            return 1;
        }

        graph.push_back(std::move(neighbors));
    }

    const size_t num_nodes = graph.size();
    if (num_nodes == 0)
    {
        std::cerr << "Error: graph is empty" << std::endl;
        return 1;
    }

    const double avg_degree = static_cast<double>(total_edges) / static_cast<double>(num_nodes);

    std::cout << "Graph loaded:" << std::endl;
    std::cout << "  Number of nodes: " << num_nodes << std::endl;
    std::cout << "  Total edges: " << total_edges << std::endl;
    std::cout << "  Average degree: " << avg_degree << std::endl;

    // Validate entry point
    if (entry_point >= num_nodes)
    {
        std::cerr << "Error: Entry point (" << entry_point << ") is out of range [0, " << num_nodes << ")" << std::endl;
        return 1;
    }

    // BFS to find shortest hops from entry point to all nodes
    // Use uint32_t instead of int64_t to save memory; UINT32_MAX means unreachable
    std::vector<uint32_t> hops(num_nodes, UINT32_MAX);
    std::queue<uint32_t> bfs_queue;

    hops[entry_point] = 0;
    bfs_queue.push(entry_point);

    size_t visited_count = 1;
    uint32_t max_hops = 0;

    while (!bfs_queue.empty())
    {
        const uint32_t current = bfs_queue.front();
        bfs_queue.pop();

        const uint32_t current_hops = hops[current];
        if (current_hops > max_hops)
            max_hops = current_hops;

        for (const uint32_t neighbor : graph[current])
        {
            if (neighbor >= num_nodes)
            {
                std::cerr << "Warning: Invalid neighbor " << neighbor << " from node " << current << std::endl;
                continue;
            }

            if (hops[neighbor] == UINT32_MAX)
            {
                hops[neighbor] = current_hops + 1;
                bfs_queue.push(neighbor);
                visited_count++;
            }
        }
    }

    std::cout << "\nBFS completed:" << std::endl;
    std::cout << "  Reachable nodes: " << visited_count << " / " << num_nodes << std::endl;
    std::cout << "  Max hops from entry: " << max_hops << std::endl;

    // Calculate hop distribution
    std::vector<size_t> hop_distribution(static_cast<size_t>(max_hops) + 1, 0);
    for (const uint32_t h : hops)
    {
        if (h != UINT32_MAX)
            hop_distribution[h]++;
    }

    std::cout << "\nHop distribution:" << std::endl;
    for (size_t h = 0; h < hop_distribution.size(); ++h)
    {
        std::cout << "  Hop " << h << ": " << hop_distribution[h] << " nodes" << std::endl;
    }

    // Sort nodes by hops (ascending, reachable first, unreachable last)
    std::vector<std::pair<uint32_t, uint32_t>> hops_node_pairs(num_nodes);
    for (size_t i = 0; i < num_nodes; ++i)
        hops_node_pairs[i] = {hops[i], static_cast<uint32_t>(i)};

    std::sort(hops_node_pairs.begin(), hops_node_pairs.end(),
              [](const auto &a, const auto &b) {
                  if (a.first == UINT32_MAX && b.first == UINT32_MAX) return false;
                  if (a.first == UINT32_MAX) return false;
                  if (b.first == UINT32_MAX) return true;
                  return a.first < b.first;
              });

    // Write binary output file
    std::ofstream bin_output(output_bin_path, std::ios::binary);
    if (!bin_output)
    {
        std::cerr << "Failed to open output file: " << output_bin_path << std::endl;
        return 1;
    }

    // Header (24 bytes)
    const uint64_t file_size_bin = 24 + num_nodes * 12;
    const uint64_t num_nodes_bin = static_cast<uint64_t>(num_nodes);
    const uint32_t reserved = 0;

    bin_output.write(reinterpret_cast<const char *>(&file_size_bin), 8);
    bin_output.write(reinterpret_cast<const char *>(&num_nodes_bin), 8);
    bin_output.write(reinterpret_cast<const char *>(&entry_point), 4);
    bin_output.write(reinterpret_cast<const char *>(&reserved), 4);

    // Write each node (12 bytes): node_id, neighbors_num, storage_bytes, hops
    for (const auto &p : hops_node_pairs)
    {
        const uint32_t node_id = p.second;
        const uint16_t n_num = static_cast<uint16_t>(std::min(neighbors_num[node_id], (uint32_t)UINT16_MAX));
        const uint16_t storage_bytes = static_cast<uint16_t>(n_num * 4);
        const uint32_t hops_val = p.first;

        bin_output.write(reinterpret_cast<const char *>(&node_id), 4);
        bin_output.write(reinterpret_cast<const char *>(&n_num), 2);
        bin_output.write(reinterpret_cast<const char *>(&storage_bytes), 2);
        bin_output.write(reinterpret_cast<const char *>(&hops_val), 4);
    }

    if (!bin_output)
    {
        std::cerr << "Failed while writing binary output: " << output_bin_path << std::endl;
        return 1;
    }

    std::cout << "\nOutput saved to: " << output_bin_path << std::endl;
    std::cout << "  File size: " << file_size_bin << " bytes" << std::endl;
    std::cout << "  Nodes: " << num_nodes_bin << ", entry_point: " << entry_point << std::endl;

    // Report execution time and memory usage
    auto end_time = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    struct rusage end_rusage;
    getrusage(RUSAGE_SELF, &end_rusage);
    const long maxrss_kb = end_rusage.ru_maxrss - start_maxrss;
    const double maxrss_mb = static_cast<double>(maxrss_kb) / 1024.0;

    std::cout << "\nPerformance statistics:" << std::endl;
    std::cout << "  Execution time: " << duration.count() << " ms" << std::endl;
    std::cout << "  Max memory usage: " << maxrss_mb << " MB" << std::endl;

    return 0;
}
