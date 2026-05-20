// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <string>
#include "disk_utils.h"

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cout << "Usage: calculate_in_degree <vamana_graph_file> <output_bin_file>" << std::endl;
        std::cout << "Output binary file format:" << std::endl;
        std::cout << "  Header (24 bytes): file_size, num_nodes, max_in_degree, reserved" << std::endl;
        std::cout << "  Per node (12 bytes): node_id, neighbors_num, storage_bytes, in_degree" << std::endl;
        return -1;
    }

    return diskann::generate_priority_indegree(std::string(argv[1]), std::string(argv[2]));
}
