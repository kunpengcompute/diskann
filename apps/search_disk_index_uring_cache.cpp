// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"
#include <boost/program_options.hpp>

#include "disk_utils.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition.h"
#include "pq_flash_index_mg_uring.h"
#include "timer.h"
#include "percentile_stats.h"
#include "program_options_utils.hpp"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "io_uring_aligned_file_reader.h"

#define WARMUP false
uint32_t repeat = 7;
namespace po = boost::program_options;

int getRecallThreshold() {
    const char* env_recall = std::getenv("RECALL");
    if (env_recall != nullptr) {
        try {
            return std::stoi(env_recall);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Invalid RECALL env var '" << env_recall
                      << "', using default 99" << std::endl;
        }
    }
    return 99;
}

std::string generate_csv_filename(const std::string &result_output_prefix, uint32_t num_threads, float reorder_ratio,
                                  uint32_t beamwidth, double cache_budget_gb)
{
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);
    char timestamp[20];
    std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", &tm);
    std::string filename = result_output_prefix;
    std::ostringstream oss;
    oss << filename << "/MGAIO_" << timestamp << "_T" << num_threads << "_Rerank" << reorder_ratio << "_W" << beamwidth
        << "_Cache" << cache_budget_gb << ".csv";
    return oss.str();
}

void print_stats(std::string category, std::vector<float> percentiles, std::vector<float> results)
{
    diskann::cout << std::setw(20) << category << ": " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++)
    {
        diskann::cout << std::setw(8) << percentiles[s] << "%";
    }
    diskann::cout << std::endl;
    diskann::cout << std::setw(22) << " " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++)
    {
        diskann::cout << std::setw(9) << results[s];
    }
    diskann::cout << std::endl;
}

template <typename T, typename LabelT = uint32_t>
int search_disk_index(diskann::Metric &metric, const std::string &index_path_prefix, std::string &memory_graph_path,
                      const std::string &result_output_prefix, const std::string &query_file, std::string &gt_file,
                      const uint32_t num_threads, const uint32_t recall_at, const float reorder_ratio,
                      const uint32_t beamwidth, const double cache_budget_gb, const std::string &graph_priority_file,
                      const uint32_t search_io_limit, const std::vector<uint32_t> &Lvec,
                      const float fail_if_recall_below, const std::vector<std::string> &query_filters,
                      const bool use_reorder_data = false)
{
    {
        double peak_mem_gb = diskann::get_max_rss_gb();
        diskann::cout << "Peak memory usage when enter search_disk_index: " << peak_mem_gb << " GB" << std::endl;
    }
    diskann::cout << "Search parameters: #threads: " << num_threads << ", ";
    if (beamwidth <= 0)
        diskann::cout << "beamwidth to be optimized for each L value" << std::flush;
    else
        diskann::cout << " beamwidth: " << beamwidth << std::flush;
    if (search_io_limit == std::numeric_limits<uint32_t>::max())
        diskann::cout << "." << std::endl;
    else
        diskann::cout << ", io_limit: " << search_io_limit << "." << std::endl;

    std::string warmup_query_file = index_path_prefix + "_sample_data.bin";

    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    bool filtered_search = false;
    if (!query_filters.empty())
    {
        diskann::cerr << "Filtering is not supported in this version." << std::endl;
        return -1;
    }

    bool calc_recall_flag = false;
    if (gt_file != std::string("null") && gt_file != std::string("NULL") && file_exists(gt_file))
    {
        diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
        if (gt_num != query_num)
        {
            diskann::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
        }
        calc_recall_flag = true;
    }

    std::shared_ptr<AlignedFileReaderV2> reader = nullptr;
    reader.reset(new LinuxAlignedFileReaderV2());
    std::unique_ptr<diskann::PQFlashIndexMGV2<T, LabelT>> _pFlashIndex(
        new diskann::PQFlashIndexMGV2<T, LabelT>(reader, metric));

    int res = _pFlashIndex->load(num_threads, index_path_prefix.c_str(), nullptr, false, reorder_ratio);

    if (res != 0)
    {
        return res;
    }

    double peak_mem_after_load = diskann::get_max_rss_gb();
    diskann::cout << "Peak memory usage after loading index: " << peak_mem_after_load << " GB" << std::endl;

#ifdef FAST_DISKANN
    std::string effective_priority_file = graph_priority_file;
    std::string mem_index_file = index_path_prefix + "_mem.index";
    if (!file_exists(mem_index_file))
    {
        diskann::cerr << "Warning: memory index file " << mem_index_file << " not found." << std::endl;
    }
    else
    {
        std::string comp_file = mem_index_file + ".vamana.comp";
        if (!file_exists(comp_file))
        {
            diskann::cout << "Compressed graph not found, generating: " << comp_file << std::endl;
            diskann::compress_vamana_graph(mem_index_file.c_str());
        }
        if (cache_budget_gb > 0 && (graph_priority_file == "null" || graph_priority_file == "NULL"))
        {
            std::string indegree_file = mem_index_file + ".priority.indegree";
            if (!file_exists(indegree_file))
            {
                diskann::cout << "Priority file not found, generating: " << indegree_file << std::endl;
                diskann::generate_priority_indegree(mem_index_file, indegree_file);
            }
            effective_priority_file = indegree_file;
        }
    }
#else
    const std::string &effective_priority_file = graph_priority_file;
#endif

    diskann::cout << "Using " << effective_priority_file << " as graph priority file." << std::endl;
    diskann::cout << "Caching " << cache_budget_gb << " GB of nodes around medoid(s)" << std::endl;

    if (cache_budget_gb > 0)
    {
        const uint64_t cache_budget_bytes = cache_budget_gb * 1024 * 1024 * 1024;
        const uint64_t graph_cache_budget_bytes =
            _pFlashIndex->cache_graph_by_priority(cache_budget_bytes, effective_priority_file);
        if (graph_cache_budget_bytes < cache_budget_bytes)
        {
            const uint64_t vector_cache_budget_bytes = cache_budget_bytes - graph_cache_budget_bytes;
            _pFlashIndex->cache_vectors_by_priority(vector_cache_budget_bytes, effective_priority_file);
        }
    }

    omp_set_num_threads(num_threads);

    uint64_t warmup_L = 20;
    uint64_t warmup_num = 0, warmup_dim = 0, warmup_aligned_dim = 0;
    T *warmup = nullptr;
    {
        double peak_mem_gb = diskann::get_max_rss_gb();
        diskann::cout << "Peak memory usage before cached_beam_search: " << peak_mem_gb << " GB" << std::endl;
    }
    diskann::cout << "Search start!" << std::endl;

    diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    diskann::cout.precision(2);

    std::string recall_string = "Recall@" + std::to_string(recall_at);
    diskann::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth" << std::setw(16) << "QPS" << std::setw(16)
                  << "Mean Latency" << std::setw(16) << "99.9 Latency" << std::setw(16) << "Mean IOs" << std::setw(16)
                  << "CPU (s)" << std::setw(16) << "IO (%)" << std::setw(16) << "Mean Pl IOs" << std::setw(16)
                  << "Preload Hit(%)" << std::setw(16);
    if (calc_recall_flag)
    {
        diskann::cout << std::setw(16) << recall_string << std::endl;
    }
    else
        diskann::cout << std::endl;
    diskann::cout << "==============================================================="
                     "======================================================="
                  << std::endl;

    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());

    uint32_t optimized_beamwidth = 1;

    double best_recall = 0.0;

    std::string csv_filename =
        generate_csv_filename(result_output_prefix, num_threads, reorder_ratio, beamwidth, cache_budget_gb);
    std::ofstream log_file(csv_filename, std::ios::app);
    if (!log_file.is_open())
    {
        std::cerr << "Error: Unable to open log file!" << std::endl;
        return 0;
    }

    log_file << "L,Beamwidth,T,RR,QPS,Latency(us),99.9 Latency(us),CPU(us),IO(us),IO(%),Cmps,Hops,IOs,Pl-IOs,Pl-IO "
                "Hits,Pl-IOs Hits(%),Graph Hit(%),Vector Hit(%)";
    log_file << ",Recall,Recall@";
    log_file << "\n";

    bool bi_cut_search_start = false;
    bool bi_cut_search_finish = false;
    uint32_t test_id = 0;
    uint32_t L = 0;
    uint32_t right_L = 0;
    uint32_t left_L = 0;
    uint32_t middle_L = 0;
    const int RECALL_THRESHOLD = getRecallThreshold();
    while (test_id < Lvec.size() && !bi_cut_search_finish)
    {
        if (bi_cut_search_start)
        {
            L = middle_L;
        }
        else {
            L = Lvec[test_id];
        }

        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        optimized_beamwidth = beamwidth;

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);

        auto stats = new diskann::QueryStats[query_num];

        std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
        double qps = 0;
        for (uint32_t iter = 0; iter < repeat; iter++)
        {
            auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
            for (int64_t i = 0; i < (int64_t)query_num; i++)
            {
                if (!filtered_search)
                {
                    _pFlashIndex->cached_beam_search(
                        query + (i * query_aligned_dim), recall_at, L, query_result_ids_64.data() + (i * recall_at),
                        query_result_dists[test_id].data() + (i * recall_at), optimized_beamwidth, false, 0,
                        std::numeric_limits<uint32_t>::max(), use_reorder_data, stats + i);
                }
                else
                {
                    diskann::cerr << "Filtering is not supported in this version." << std::endl;
                }
            }
            auto e = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = e - s;
            qps += (1.0 * query_num) / (1.0 * diff.count());
        }
        qps /= repeat;
        for (uint32_t i = 0; i < query_num; i++)
        {
            (stats + i)->total_us /= repeat;
            (stats + i)->io_us /= repeat;
            (stats + i)->cpu_us /= repeat;
            (stats + i)->n_4k /= repeat;
            (stats + i)->n_8k /= repeat;
            (stats + i)->n_12k /= repeat;
            (stats + i)->n_ios /= repeat;
            (stats + i)->n_cmps_saved /= repeat;
            (stats + i)->n_cmps /= repeat;
            (stats + i)->n_cache_hits /= repeat;
            (stats + i)->n_hops /= repeat;
            (stats + i)->n_ios_preload /= repeat;
            (stats + i)->n_ios_preload_hits /= repeat;
            (stats + i)->n_graph_hits /= repeat;
            (stats + i)->n_graph_reads /= repeat;
            (stats + i)->n_vector_hits /= repeat;
            (stats + i)->n_vector_reads /= repeat;
        }
        diskann::convert_types<uint64_t, uint32_t>(query_result_ids_64.data(), query_result_ids[test_id].data(),
                                                   query_num, recall_at);

        auto mean_latency = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto latency_999 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.999, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto mean_ios = diskann::get_mean_stats<uint32_t>(stats, query_num,
                                                          [](const diskann::QueryStats &stats) { return stats.n_ios; });

        auto mean_cpuus = diskann::get_mean_stats<float>(stats, query_num,
                                                         [](const diskann::QueryStats &stats) { return stats.cpu_us; });

        auto mean_ious = diskann::get_mean_stats<float>(stats, query_num,
                                                        [](const diskann::QueryStats &stats) { return stats.io_us; });

        auto mean_hops = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_hops; });

        auto mean_n_cmps = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_cmps; });

        auto mean_n_ios_preload = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_ios_preload; });

        auto mean_n_ios_preload_hit = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_ios_preload_hits; });

        auto io_preload_ratio = (double)mean_n_ios_preload_hit / (double)mean_n_ios_preload * 100.0;

        auto io_ratio = (double)mean_ious / (double)mean_latency * 100.0;

        auto mean_n_graph_hits = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_graph_hits; });

        auto mean_n_vector_hits = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_vector_hits; });

        auto mean_n_graph_reads = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_graph_reads; });

        auto mean_n_vector_reads = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_vector_reads; });

        double graph_hit_ratio = (double)mean_n_graph_hits / (double)mean_n_graph_reads * 100.0;
        double vector_hit_ratio = (double)mean_n_vector_hits / (double)mean_n_vector_reads * 100.0;

        double recall = 0;
        if (calc_recall_flag)
        {
            recall = diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                               query_result_ids[test_id].data(), recall_at, recall_at);
            best_recall = std::max(recall, best_recall);
        }
        if (!bi_cut_search_start)
        {
            if (recall > RECALL_THRESHOLD)
            {
                bi_cut_search_start = true;
                right_L = Lvec[test_id];
                left_L = test_id > 0 ? Lvec[test_id-1]: recall_at;
                middle_L = (left_L + right_L)/2;
            }
            test_id++;
        }
        else if(recall >= RECALL_THRESHOLD)
        {
            right_L = middle_L;
            middle_L = (left_L + right_L)/2;
            if (middle_L == left_L)
            {
                bi_cut_search_finish = true;
            }
        }
        else if (recall < RECALL_THRESHOLD)
        {
            left_L = middle_L;
            middle_L = (left_L + right_L)/2;
            if (middle_L == left_L)
            {
                middle_L = right_L;
            }
        }

        diskann::cout << std::setw(6) << L << std::setw(12) << optimized_beamwidth << std::setw(16) << qps
                      << std::setw(16) << mean_latency << std::setw(16) << latency_999 << std::setw(16) << mean_ios
                      << std::setw(16) << mean_cpuus << std::setw(16) << io_ratio << std::setw(16) << mean_n_ios_preload
                      << std::setw(16) << io_preload_ratio << std::setw(16);

        if (calc_recall_flag)
        {
            diskann::cout << std::setw(16) << recall << std::endl;
        }
        else
            diskann::cout << std::endl;
        log_file << L << "," << optimized_beamwidth << "," << num_threads << "," << reorder_ratio << "," << qps << ","
                 << mean_latency << "," << latency_999 << "," << mean_cpuus << "," << mean_ious << "," << io_ratio
                 << "," << mean_n_cmps << "," << mean_hops << "," << mean_ios << "," << mean_n_ios_preload << ","
                 << mean_n_ios_preload_hit << "," << io_preload_ratio << "," << graph_hit_ratio << ","
                 << vector_hit_ratio;
        log_file << "," << std::to_string(recall) << "," << recall_at;
        log_file << "\n";

        delete[] stats;
    }
    log_file.close();

    diskann::cout << "Done searching. Now saving results " << std::endl;

    diskann::aligned_free(query);
    if (warmup != nullptr)
        diskann::aligned_free(warmup);

    double peak_mem_gb = diskann::get_max_rss_gb();
    diskann::cout << "Peak memory usage: " << peak_mem_gb << " GB" << std::endl;

    return best_recall >= fail_if_recall_below ? 0 : -1;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, index_path_prefix, memory_graph_path, result_path_prefix, query_file, gt_file,
        filter_label, label_type, query_filters_file, graph_priority_file;
    uint32_t num_threads, K, W, search_io_limit;
    double cache_budget_gb = 0.0;
    float reorder_ratio = 0.0f;
    std::vector<uint32_t> Lvec;
    bool use_reorder_data = false;
    float fail_if_recall_below = 0.0f;

    po::options_description desc{program_options_utils::make_program_description(
        "search_disk_index_uring_cache", "Searches using disk index with io_uring and graph/vector cache")};
    try
    {
        desc.add_options()("help,h", "Print information on arguments");

        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("result_path", po::value<std::string>(&result_path_prefix)->required(),
                                       program_options_utils::RESULT_PATH_DESCRIPTION);
        required_configs.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                                       program_options_utils::QUERY_FILE_DESCRIPTION);
        required_configs.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                                       program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION);
        required_configs.add_options()("search_list,L",
                                       po::value<std::vector<uint32_t>>(&Lvec)->multitoken()->required(),
                                       program_options_utils::SEARCH_LIST_DESCRIPTION);

        po::options_description optional_configs("Optional");
        optional_configs.add_options()("gt_file", po::value<std::string>(&gt_file)->default_value(std::string("null")),
                                       program_options_utils::GROUND_TRUTH_FILE_DESCRIPTION);
        optional_configs.add_options()("reorder_ratio,rr", po::value<float>(&reorder_ratio)->default_value(0.0f),
                                       "Reorder ratio [0,1]. 0=full reorder(no early stop), 1=most aggressive early stop");
        optional_configs.add_options()("beamwidth,W", po::value<uint32_t>(&W)->default_value(1),
                                       program_options_utils::BEAMWIDTH);
        optional_configs.add_options()(
            "search_io_limit",
            po::value<uint32_t>(&search_io_limit)->default_value(std::numeric_limits<uint32_t>::max()),
            "Max #IOs for search.  Default value: uint32::max()");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("cache_budget", po::value<double>(&cache_budget_gb)->default_value(0.0),
                                       "Cache budget(GB) for graph and vectors. Default value: 0.0");
        optional_configs.add_options()("graph_priority_file",
                                       po::value<std::string>(&graph_priority_file)->default_value(std::string("null")),
                                       "Graph priority file. Default value: null");
        optional_configs.add_options()("repeat", po::value<uint32_t>(&repeat)->default_value(7),
                                       "Repeat times. Default value: 7");

        desc.add(required_configs).add(optional_configs);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    diskann::Metric metric;
    if (dist_fn == std::string("mips"))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else
    {
        std::cout << "Unsupported distance function. Currently only L2/ Inner "
                     "Product/Cosine are supported."
                  << std::endl;
        return -1;
    }

    if ((data_type != std::string("float")) && (metric == diskann::Metric::INNER_PRODUCT))
    {
        std::cout << "Currently support only floating point data for Inner Product." << std::endl;
        return -1;
    }

    std::vector<std::string> query_filters;

    try
    {
        if (data_type == std::string("float"))
            return search_disk_index<float>(metric, index_path_prefix, memory_graph_path, result_path_prefix,
                                            query_file, gt_file, num_threads, K, reorder_ratio, W, cache_budget_gb,
                                            graph_priority_file, search_io_limit, Lvec, fail_if_recall_below,
                                            query_filters, use_reorder_data);
        else if (data_type == std::string("int8"))
            return search_disk_index<int8_t>(metric, index_path_prefix, memory_graph_path, result_path_prefix,
                                             query_file, gt_file, num_threads, K, reorder_ratio, W, cache_budget_gb,
                                             graph_priority_file, search_io_limit, Lvec, fail_if_recall_below,
                                             query_filters, use_reorder_data);
        else if (data_type == std::string("uint8"))
            return search_disk_index<uint8_t>(metric, index_path_prefix, memory_graph_path, result_path_prefix,
                                              query_file, gt_file, num_threads, K, reorder_ratio, W, cache_budget_gb,
                                              graph_priority_file, search_io_limit, Lvec, fail_if_recall_below,
                                              query_filters, use_reorder_data);
        else
        {
            std::cerr << "Unsupported data type. Use float or int8 or uint8" << std::endl;
            return -1;
        }
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index search failed." << std::endl;
        return -1;
    }
}
