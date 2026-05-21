// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>
#define CHECK 1

int main(int argc, char **argv)
{

    if (argc != 2)
    {
        std::cout << argv[0] << " input_disk_index" << std::endl;
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
    const size_t node_bytes = header_ptr[4];
    const size_t node_per_block = header_ptr[5];
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

    return 0;
}