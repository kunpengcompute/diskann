// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>
#define CHECK 1

void load_nsg(const char *filename, std::vector<std::vector<uint32_t>> &graph)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        throw std::runtime_error(std::string("Failed to open file: ") + filename);
    }
    in.seekg(24, std::ios::cur);
    unsigned cc = 0;
    while (!in.eof())
    {
        unsigned k = 0;
        in.read((char *)&k, sizeof(unsigned));
        if (in.eof())
            break;
        cc += k;
        std::vector<unsigned> tmp(k);
        in.read((char *)tmp.data(), k * sizeof(unsigned));
        graph.push_back(tmp);
    }
}

void save_nsg(const char *filename, std::vector<std::vector<uint32_t>> &graph, unsigned width, unsigned ep_)
{
    std::ofstream out(filename, std::ios::binary | std::ios::out);
    if (!out.is_open())
    {
        throw std::runtime_error(std::string("Failed to open file for writing: ") + filename);
    }

    out.write((char *)&width, sizeof(unsigned));
    out.write((char *)&ep_, sizeof(unsigned));
    for (unsigned i = 0; i < graph.size(); i++)
    {
        unsigned GK = (unsigned)graph[i].size();
        out.write((char *)&GK, sizeof(unsigned));
        out.write((char *)graph[i].data(), GK * sizeof(unsigned));
    }
    out.close();
}

int main(int argc, char **argv)
{

    if (argc != 2)
    {
        std::cout << argv[0] << " input_mem_index" << std::endl;
        exit(-1);
    }

    std::ifstream readr(argv[1], std::ios::binary);
    if (!readr)
    {
        std::cerr << "Failed to open: " << argv[1] << std::endl;
        return 1;
    }

    // header 24B
    const std::streamsize buffer_size = 24; // 24B
    std::vector<char> buffer(buffer_size);

    readr.read(buffer.data(), buffer_size);

    uint32_t *header_ptr = reinterpret_cast<uint32_t *>(buffer.data());
    size_t *header_ptr_size = reinterpret_cast<size_t *>(buffer.data());

    const size_t file_size = header_ptr_size[0];     // offset 0: file size (8 bytes)
    const uint32_t r = header_ptr[2];                // offset 8: graph degree (4 bytes)
    const uint32_t ep = header_ptr[3];               // offset 12: entry point (4 bytes)
    const uint32_t frozen = header_ptr_size[2];      // offset 16: frozen points (8 bytes)

    std::cout << "file_size = " << file_size << std::endl;
    std::cout << "r = " << r << std::endl;
    std::cout << "ep = " << ep << std::endl;
    std::cout << "frozen = " << frozen << std::endl;
    std::vector<std::vector<uint32_t>> graph;
    load_nsg(argv[1], graph);
    const size_t points_num = graph.size();
    const size_t check_point = points_num * 0.1f;
    std::cout << "points_num = " << points_num << std::endl;

    if (check_point == 0)
    {
        std::cerr << "Error: check_point is zero (points_num too small)" << std::endl;
        return -1;
    }

    size_t point_id = 0;
    for (size_t i = 0; i < points_num; i++)
    {
        if (CHECK)
        {
            if (point_id % check_point == 0 || point_id == (points_num - 1))
            {
                std::cout << "point_id: " << point_id << " edge num: " << graph[point_id].size() << std::endl;
                std::cout << "edges : ";
                for (int i = 0; i < graph[point_id].size(); i++)
                {
                    std::cout << graph[point_id][i] << ' ';
                }
                std::cout << std::endl;
                std::cout << std::endl;
            }
        }
        point_id++;
    }

    return 0;
}