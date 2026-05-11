// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#ifdef _WINDOWS
#include <numeric>
#endif
#include <string>
#include <vector>

#include "distance.h"
#include "parameters.h"

namespace diskann
{
struct QueryStats
{
    float total_us = 0; // total time to process query in micros
    float io_us = 0;    // total time spent in IO
    float cpu_us = 0;   // total time spent in CPU

    unsigned n_4k = 0;         // # of 4kB reads
    unsigned n_8k = 0;         // # of 8kB reads
    unsigned n_12k = 0;        // # of 12kB reads
    unsigned n_ios = 0;        // total # of IOs issued
    unsigned read_size = 0;    // total # of bytes read
    unsigned n_cmps_saved = 0; // # cmps saved
    unsigned n_cmps = 0;       // # cmps
    unsigned n_cache_hits = 0; // # cache_hits
    unsigned n_hops = 0;       // # search hops
#ifdef FAST_DISKANN
    unsigned n_ios_preload = 0;
    unsigned n_ios_preload_hits = 0;
    unsigned pruned_ios = 0;             // # of IOs pruned by ML model
    unsigned n_overflow_queries = 0;     // # of queries that overflowed the IO limit


    float preprocess_us = 0;             // total time before entering greedy search
    float cache_quant_compute_us = 0;    // total time spent on pq compute on cached nodes
    float cache_acc_compute_us = 0;      // total time spent on accurate distance compute on cached nodes
    float disk_req_prepare_us = 0;       // total time of preparing for read requests
    float disk_req_submit_us = 0;        // total time of submitting the read requests
    float disk_req_wait_us = 0;          // total time of waiting for read requests to be ready
    float disk_req_quant_compute_us = 0; // total time spent on pq compute on read nodes
    float disk_req_acc_compute_us = 0;   // total time spent on accurate distance compute on read nodes
    float postprocess_us = 0;            // postprocess after greedy search
#endif
};

template <typename T>
inline T get_percentile_stats(QueryStats *stats, uint64_t len, float percentile,
                              const std::function<T(const QueryStats &)> &member_fn)
{
    std::vector<T> vals(len);
    for (uint64_t i = 0; i < len; i++)
    {
        vals[i] = member_fn(stats[i]);
    }

    std::sort(vals.begin(), vals.end(), [](const T &left, const T &right) { return left < right; });

    auto retval = vals[(uint64_t)(percentile * len)];
    vals.clear();
    return retval;
}

template <typename T>
inline double get_mean_stats(QueryStats *stats, uint64_t len, const std::function<T(const QueryStats &)> &member_fn)
{
    double avg = 0;
    for (uint64_t i = 0; i < len; i++)
    {
        avg += (double)member_fn(stats[i]);
    }
    return avg / len;
}
} // namespace diskann
