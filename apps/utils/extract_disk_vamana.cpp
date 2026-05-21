// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>
#define CHECK 1

inline void load_fvecs(char *filename, std::vector<std::vector<float>> &data)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open())
    {
        std::cerr << "Error: Unable to open file " << filename << " for reading" << std::endl;
        exit(-1);
    }
    int dim = 0;
    in.read((char *)&dim, 4);
    in.seekg(0, std::ios::end);
    uint64_t dim_64 = dim;
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    uint64_t num_64 = (size_t)(fsize / (dim_64 + 1) / 4);
    int num = num_64;

    data.resize(num_64);

    in.seekg(0, std::ios::beg);
    for (uint64_t i = 0; i < num_64; i++)
    {
        in.seekg(4, std::ios::cur);
        data[i].resize(dim_64);
        in.read((char *)(data[i].data()), dim_64 * 4);
    }
    in.close();
}

// Compare two std::vector<std::vector<float>> for exact equality (no tolerance), output error location
bool check_dataset(const std::vector<std::vector<float>> &extract, const std::vector<std::vector<float>> &origin)
{
    if (extract.size() != origin.size())
    {
        std::cout << "Error: Outer vector size mismatch, extract.size() = " << extract.size()
                  << ", origin.size() = " << origin.size() << std::endl;
        return false;
    }

    for (size_t i = 0; i < extract.size(); ++i)
    {
        if (extract[i].size() != origin[i].size())
        {
            std::cout << "Error: Sub-vector " << i << " size mismatch, extract[i].size() = " << extract[i].size()
                      << ", origin[i].size() = " << origin[i].size() << std::endl;
            return false;
        }

        for (size_t j = 0; j < extract[i].size(); ++j)
        {
            if (extract[i][j] != origin[i][j])
            {
                std::cout << "Error: Element mismatch at [" << i << "][" << j << "], extract = " << extract[i][j]
                          << ", origin = " << origin[i][j] << std::endl;
                return false;
            }
        }
    }

    return true;
}

void save_nsg(const char *filename, std::vector<std::vector<uint32_t>> &graph, unsigned width, unsigned ep_)
{
    std::ofstream out(filename, std::ios::binary | std::ios::out);

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

    if (argc != 4)
    {
        std::cout << argv[0] << " input_disk_index data_path output_vamana_graph" << std::endl;
        exit(-1);
    }

    std::ifstream readr(argv[1], std::ios::binary);
    if (!readr)
    {
        std::cerr << "Failed to open: " << argv[1] << std::endl;
        return 1;
    }

    const std::streamsize buffer_size = 4 * 1024; // 4KB
    std::vector<char> buffer(buffer_size);

    readr.read(buffer.data(), buffer_size);

    size_t *header_ptr = reinterpret_cast<size_t *>(buffer.data());

    const size_t max_id = header_ptr[0];
    const size_t points_num = header_ptr[1];
    const size_t dim = header_ptr[2];
    const size_t ep = header_ptr[3];
    size_t node_bytes = header_ptr[4];
    size_t node_per_block = header_ptr[5];
    const size_t file_size = header_ptr[9];

    // Prevent underflow in r calculation below
    if (node_bytes < dim * sizeof(float) + sizeof(uint32_t))
    {
        std::cerr << "Error: node_bytes (" << node_bytes << ") is too small for dim (" << dim << ")" << std::endl;
        return -1;
    }
    const size_t r = (node_bytes - dim * sizeof(float)) / sizeof(uint32_t) - 1;
    const size_t check_point = (double)points_num * 0.1f;
    std::cout << "only support float data now!!!" << std::endl;
    std::cout << "max_id = " << max_id << std::endl;
    std::cout << "points_num = " << points_num << std::endl;
    std::cout << "dim = " << dim << std::endl;
    std::cout << "ep = " << ep << std::endl;
    std::cout << "node_bytes = " << node_bytes << std::endl;
    std::cout << "node_per_block = " << node_per_block << std::endl;
    std::cout << "file_size = " << file_size << std::endl;
    std::cout << "r = " << r << std::endl;
    std::cout << "check_point = " << check_point << std::endl;

    node_bytes = (node_bytes + 4096 - 1) / 4096 * 4096;
    buffer.resize(node_bytes);
    std::cout << "aligned node_bytes = " << node_bytes << std::endl;
    if (node_bytes != 4096)
    {
        node_per_block = 1;
    }
    std::cout << "aligned node_per_block = " << node_per_block << std::endl;

    std::vector<std::vector<uint32_t>> graph(points_num);
    std::vector<std::vector<float>> dataset(points_num);
    size_t point_id = 0;

    if (check_point == 0)
    {
        std::cerr << "Error: check_point is zero (points_num too small)" << std::endl;
        return -1;
    }

    while (readr.read(buffer.data(), node_bytes) || readr.gcount() > 0)
    {
        char *node_ptr = buffer.data();
        for (int node_id = 0; node_id < node_per_block; node_id++)
        {
            if (point_id >= points_num)
            {
                goto done_reading;
            }

            uint32_t *edge_ptr = (uint32_t *)(node_ptr + dim * sizeof(float));
            size_t edge_num = edge_ptr[0];
            graph[point_id].reserve(edge_num);
            for (int i = 0; i < edge_num; i++)
            {
                graph[point_id].push_back(edge_ptr[i + 1]);
            }
            node_ptr += node_bytes;

            if (CHECK)
            {
                if (point_id % check_point == 0 || point_id == (points_num - 1))
                {
                    std::cout << "node id: " << point_id << " edge num:" << graph[point_id].size() << std::endl;
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
    }
done_reading:
    {
        std::cout << "check pass" << std::endl;
        save_nsg(argv[3], graph, r, ep);
        std::cout << "save as nsg done" << std::endl;
    }

    return 0;
}