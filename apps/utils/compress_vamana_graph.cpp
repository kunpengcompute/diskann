// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include "compressed_graph.h"

#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cout << "usage: compress_graph <vamana_graph_file_path> <compressed_graph_file_path>" << std::endl;
        return -1;
    }

    std::string vamana_graph_file_path = argv[1];
    std::string compressed_graph_file_path = argv[2];

    std::ifstream readr(vamana_graph_file_path, std::ios::binary);
    if (!readr)
    {
        std::cerr << "Failed to open: " << vamana_graph_file_path << std::endl;
        return 1;
    }

    // header 24B
    const std::streamsize buffer_size = 24; // 24B
    std::vector<char> buffer(buffer_size);

    readr.read(buffer.data(), buffer_size);

    uint32_t *header_ptr = reinterpret_cast<uint32_t *>(buffer.data());
    size_t *header_ptr_size = reinterpret_cast<size_t *>(buffer.data());

    const size_t file_size = header_ptr_size[0];     // offset 0: file size (8 bytes)
    const uint32_t _width32 = header_ptr[2];         // offset 8: graph width (4 bytes)
    const uint32_t _ep32 = header_ptr[3];            // offset 12: entry point (4 bytes)
    const uint32_t frozen = header_ptr_size[2];      // offset 16: frozen points (8 bytes)

    std::vector<std::vector<uint32_t>> _graph;
    std::ifstream in(vamana_graph_file_path, std::ios::binary);
    if (!in.is_open())
    {
        std::cerr << "Error: Unable to open file " << vamana_graph_file_path << std::endl;
        return -1;
    }
    in.seekg(24, std::ios::cur);

    uint64_t cc = 0;
    while (!in.eof())
    {
        unsigned k = 0;
        in.read((char *)&k, sizeof(unsigned));
        if (in.eof())
            break;
        cc += k;
        std::vector<unsigned> tmp(k);
        in.read((char *)tmp.data(), k * sizeof(unsigned));
        _graph.push_back(tmp);
    }
    cc /= _graph.size();
    std::cout << "Mem graph info: "
              << "width: " << _width32 << " ep: " << _ep32 << " nodes: " << _graph.size() << " cc: " << cc << std::endl;

    CompressedGraph CG = CompressedGraph(_graph, _width32, _ep32);
    CG.saveToFile(compressed_graph_file_path);
    return 0;
}