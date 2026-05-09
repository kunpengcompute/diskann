// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <stdint.h>

namespace diskann
{

#ifdef FAST_DISKANN
enum class PruneCode : int
{
    PQ = 0,
    RAW = 1,
    NO_Code = 2,
};

enum class PruneStrategy : int
{
    RNG = 0,
    Robust = 1,
    Cagra = 2,
    Random = 3,
    NO_Prune = 4
};

inline std::string prune_strategy_to_string(PruneStrategy strategy)
{
    switch (strategy)
    {
    case PruneStrategy::RNG:
        return "RNG";
    case PruneStrategy::Robust:
        return "Robust";
    case PruneStrategy::Cagra:
        return "Cagra";
    case PruneStrategy::Random:
        return "Random";
    case PruneStrategy::NO_Prune:
        return "NO_Prune";
    default:
        return "Unknown";
    }
}

inline std::string prune_code_to_string(PruneCode code)
{
    switch (code)
    {
    case PruneCode::PQ:
        return "PQ";
    case PruneCode::RAW:
        return "RAW";
    default:
        return "Unknown";
    }
}
#endif

namespace defaults
{
const float ALPHA = 1.2f;
const uint32_t NUM_THREADS = 0;
const uint32_t MAX_OCCLUSION_SIZE = 750;
const bool HAS_LABELS = false;
const uint32_t FILTER_LIST_SIZE = 0;
const uint32_t NUM_FROZEN_POINTS_STATIC = 0;
const uint32_t NUM_FROZEN_POINTS_DYNAMIC = 1;

// In-mem index related limits
const float GRAPH_SLACK_FACTOR = 1.3;

// SSD Index related limits
const uint64_t MAX_GRAPH_DEGREE = 512;
const uint64_t SECTOR_LEN = 4096;
#ifdef FAST_DISKANN
const uint64_t MAX_N_SECTOR_READS = 512; // max 512 async 4k-io reads
#else
const uint64_t MAX_N_SECTOR_READS = 128;
#endif

// following constants should always be specified, but are useful as a
// sensible default at cli / python boundaries
const uint32_t MAX_DEGREE = 64;
const uint32_t BUILD_LIST_SIZE = 100;
const uint32_t SATURATE_GRAPH = false;
const uint32_t SEARCH_LIST_SIZE = 100;
#ifdef FAST_DISKANN
const PruneStrategy PRUNE_STRATEGY = PruneStrategy::RNG;
const PruneCode PRUNE_CODE = PruneCode::PQ;
const uint32_t MAX_KNN_NUM = 1000;
#endif
} // namespace defaults
} // namespace diskann
