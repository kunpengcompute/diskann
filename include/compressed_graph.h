// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <vector>
#include <cstdint>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cstring>
#include <stdexcept>
#include "logger.h"

class CompressedGraph
{
    typedef std::vector<std::vector<uint32_t>> Graph;

  private:
    alignas(128) std::vector<uint8_t> compressedData;
    alignas(128) std::vector<uint64_t> nodeOffsets;
    alignas(128) std::vector<uint32_t> idMap;
    alignas(128) uint32_t pointsNum;
    alignas(128) uint32_t width;
    alignas(128) uint32_t ep;

    void GraphInfo(){
        diskann::cout << "CompressedGraph pointsNum " << pointsNum << std::endl;
        diskann::cout << "CompressedGraph width " << width << std::endl;
        diskann::cout << "CompressedGraph ep " << ep << std::endl;
        diskann::cout << "CompressedGraph compressedData size " << compressedData.size() << std::endl;
        diskann::cout << "CompressedGraph nodeOffsets size " << nodeOffsets.size() << std::endl;
        diskann::cout << "CompressedGraph idMap size " << idMap.size() << std::endl;
    }

    // Varint encoding
    void encodeVarint(uint32_t value, std::vector<uint8_t> &output)
    {
        while (value >= 0x80)
        {
            output.push_back(static_cast<uint8_t>(value) | 0x80);
            value >>= 7;
        }
        output.push_back(static_cast<uint8_t>(value));
    }

    // Varint decoding
    uint32_t decodeVarint(const uint8_t *&data) const
    {
        uint32_t result = 0;
        int shift = 0;
        while (true)
        {
            uint8_t byte = *data++;
            result |= (byte & 0x7F) << shift;
            if ((byte & 0x80) == 0)
                break;
            shift += 7;
            if (shift >= 35)
                throw std::runtime_error("Malformed varint");
        }
        return result;
    }

    static inline __attribute__((always_inline)) uint32_t decodeVarintUnrolled(const uint8_t *&p)
    {
        uint32_t x;
        uint32_t b;

        b = *p++;
        x = (b & 0x7F);
        if (!(b & 0x80))
            return x;
        b = *p++;
        x |= (b & 0x7F) << 7;
        if (!(b & 0x80))
            return x;
        b = *p++;
        x |= (b & 0x7F) << 14;
        if (!(b & 0x80))
            return x;
        b = *p++;
        x |= (b & 0x7F) << 21;
        if (!(b & 0x80))
            return x;
        b = *p++;
        x |= (b & 0x7F) << 28;
        return x;
    }

  public:
    // online build compressed graph from adjacency list
    CompressedGraph(const Graph &adjList, const uint32_t width, const uint32_t ep)
        : pointsNum(adjList.size()), width(width), ep(ep)
    {
        nodeOffsets.reserve(pointsNum + 1);
        nodeOffsets.push_back(0); // First node offset is 0

        for (const auto &neighbors : adjList)
        {
            size_t startPos = compressedData.size();

            if (!neighbors.empty())
            {
                std::vector<uint32_t> sortedNeighbors(neighbors);
                std::sort(sortedNeighbors.begin(), sortedNeighbors.end());

                uint32_t prev = 0;
                for (uint32_t node : sortedNeighbors)
                {
                    uint32_t delta = node - prev;
                    encodeVarint(delta, compressedData);
                    prev = node;
                }
            }
            nodeOffsets.push_back(compressedData.size());
        }
    }

    // load compressed graph from binary file
    CompressedGraph(const std::string &graphPath, bool useReorder = false)
    {
        std::string mapPath = graphPath + ".map";
        if (useReorder)
        {
            std::ifstream in(mapPath, std::ios::binary);
            if (!in.is_open())
            {
                throw std::runtime_error("Error opening file: " + mapPath);
            }
            in.read((char *)(&pointsNum), sizeof(uint32_t));
            idMap.resize(pointsNum);
            in.read((char *)(idMap.data()), pointsNum * sizeof(uint32_t));
        }

        std::ifstream in(graphPath, std::ios::binary);
        if (!in.is_open())
        {
            throw std::runtime_error("Error opening file: " + graphPath);
        }
        uint32_t pointsNumGraph = 0;
        in.read((char *)(&pointsNumGraph), sizeof(uint32_t));
        in.read((char *)(&width), sizeof(uint32_t));
        in.read((char *)(&ep), sizeof(uint32_t));
        if (!useReorder)
        {
            pointsNum = pointsNumGraph;
        }

        if (pointsNumGraph != pointsNum)
        {
            throw std::runtime_error("pointsNumGraph != pointsNum, data mismatch");
        }

        uint64_t offsetsByteSize = 0;
        in.read((char *)(&offsetsByteSize), sizeof(uint64_t));
        if (offsetsByteSize != (sizeof(uint64_t) * (pointsNum + 1)))
        {
            const size_t expected = sizeof(uint64_t) * (pointsNum + 1);
            throw std::runtime_error("offsetsByteSize mismatch: actual=" + std::to_string(offsetsByteSize) +
                                     ", expected=" + std::to_string(expected));
        }
        nodeOffsets.resize(pointsNum + 1);
        in.read((char *)(nodeOffsets.data()), offsetsByteSize);

        uint64_t compressedDataByteSize = 0;
        in.read((char *)(&compressedDataByteSize), sizeof(uint64_t));
        compressedData.resize(compressedDataByteSize);
        in.read((char *)(compressedData.data()), compressedDataByteSize);
        in.close();
    }

    std::vector<uint8_t> serialize() const
    {
        std::vector<uint8_t> result;

        // reserve space for pointsNum(4) + width(4) + ep(4) + offsetsize(8) + offsets + datasize(8) + data
        size_t totalSize =
            sizeof(uint32_t) * 3 + sizeof(uint64_t) * 2 + nodeOffsets.size() * sizeof(uint64_t) + compressedData.size();
        result.reserve(totalSize);

        result.insert(result.end(), reinterpret_cast<const uint8_t *>(&pointsNum),
                      reinterpret_cast<const uint8_t *>(&pointsNum) + sizeof(uint32_t));
        result.insert(result.end(), reinterpret_cast<const uint8_t *>(&width),
                      reinterpret_cast<const uint8_t *>(&width) + sizeof(uint32_t));
        result.insert(result.end(), reinterpret_cast<const uint8_t *>(&ep),
                      reinterpret_cast<const uint8_t *>(&ep) + sizeof(uint32_t));

        uint64_t offsetsByteSize = static_cast<uint64_t>(nodeOffsets.size() * sizeof(uint64_t));
        result.insert(result.end(), reinterpret_cast<const uint8_t *>(&offsetsByteSize),
                      reinterpret_cast<const uint8_t *>(&offsetsByteSize) + sizeof(uint64_t));
        result.insert(result.end(), reinterpret_cast<const uint8_t *>(nodeOffsets.data()),
                      reinterpret_cast<const uint8_t *>(nodeOffsets.data() + nodeOffsets.size()));

        uint64_t dataSize = static_cast<uint64_t>(compressedData.size());
        result.insert(result.end(), reinterpret_cast<const uint8_t *>(&dataSize),
                      reinterpret_cast<const uint8_t *>(&dataSize) + sizeof(uint64_t));
        result.insert(result.end(), compressedData.begin(), compressedData.end());

        return result;
    }

    void getNeighbors(uint32_t nodeId, std::vector<uint32_t> &neighbors) const
    {
        neighbors.clear();

        const uint8_t *start = compressedData.data() + nodeOffsets[nodeId];
        const uint8_t *end = compressedData.data() + nodeOffsets[nodeId + 1];

        uint32_t prev = 0;
        while (start < end)
        {
            uint32_t delta = decodeVarintUnrolled(start);
            prev += delta;
            neighbors.push_back(prev);
        }

        return;
    }

    void getRealIds(std::vector<uint32_t> &neighbors) const
    {
        if (idMap.size() == 0)
            return;
        for (size_t i = 0; i < neighbors.size(); i++)
        {
            if (neighbors[i] >= idMap.size())
                throw std::out_of_range("neighbor ID out of range in idMap");
            neighbors[i] = idMap[neighbors[i]];
        }
    }

    void getRealIdsU64Array(uint64_t *neighbors, uint32_t size) const
    {
        if (idMap.size() == 0)
            return;
        for (uint32_t i = 0; i < size; i++)
        {
            if (neighbors[i] >= idMap.size())
                throw std::out_of_range("neighbor ID out of range in idMap");
            neighbors[i] = idMap[neighbors[i]];
        }
    }

    uint32_t getRealId(uint32_t originalId) const
    {
        if (originalId >= idMap.size())
            throw std::out_of_range("originalId out of range in idMap");
        return idMap[originalId];
    }

    uint32_t getWidth() const
    {
        return width;
    }

    uint32_t getEp() const
    {
        return ep;
    }

    uint32_t getPointsNum() const
    {
        return pointsNum;
    }

    void saveToFile(const std::string &filename) const
    {
        std::ofstream out(filename, std::ios::binary);
        if (!out)
            throw std::runtime_error("Cannot open file for writing");

        auto data = serialize();
        out.write(reinterpret_cast<const char *>(data.data()), data.size());
    }
};