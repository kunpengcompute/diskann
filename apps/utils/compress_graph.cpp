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
    std::cout << "usage: compress_graph <nsg_graph_file_path> <compressed_graph_file_path>" << std::endl;
    return -1;
  }

  std::string nsg_graph_file_path = argv[1];
  std::string compressed_graph_file_path = argv[2];

  std::vector<std::vector<uint32_t>> _graph;
  std::ifstream in(nsg_graph_file_path, std::ios::binary);
  if (!in.is_open())
  {
    std::cerr << "Error: Unable to open file " << nsg_graph_file_path << std::endl;
    return -1;
  }
  uint32_t _width32, _ep32;
  in.read((char *)&_width32, sizeof(uint32_t));
  in.read((char *)&_ep32, sizeof(uint32_t));

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
  if (_graph.size() == 0)
  {
    std::cerr << "Error: graph is empty" << std::endl;
    return -1;
  }
  cc /= _graph.size();
  std::cout << "Mem graph info: "
            << "width: " << _width32 << " ep: " << _ep32 << " nodes: " << _graph.size() << " cc: " << cc << std::endl;

  CompressedGraph CG = CompressedGraph(_graph, _width32, _ep32);
  CG.saveToFile(compressed_graph_file_path);
  return 0;
}