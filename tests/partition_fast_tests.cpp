// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstdio>
#include <string>
#include "partition.h"
#include "utils.h"

BOOST_AUTO_TEST_SUITE(PartitionFastTests)

namespace
{

void create_bin_data_file(const std::string &path, uint32_t npts, uint32_t ndims, const float *data)
{
    std::ofstream out(path, std::ios::binary);
    out.write((char *)&npts, sizeof(uint32_t));
    out.write((char *)&ndims, sizeof(uint32_t));
    out.write((char *)data, npts * ndims * sizeof(float));
    out.close();
}

void create_idmap_file(const std::string &path, const std::vector<uint32_t> &ids)
{
    std::ofstream out(path, std::ios::binary);
    int32_t npts = static_cast<int32_t>(ids.size());
    int32_t ncols = 1;
    out.write((char *)&npts, sizeof(int32_t));
    out.write((char *)&ncols, sizeof(int32_t));
    out.write((char *)ids.data(), ids.size() * sizeof(uint32_t));
    out.close();
}

} // namespace

BOOST_AUTO_TEST_CASE(partition_skip_when_existing_valid)
{
    const std::string prefix = "/tmp/test_partition_skip";
    const std::string data_file = prefix + "_data.bin";
    const std::string clusters_file = prefix + "_clusters_num.bin";

    const uint32_t npts = 20;
    const uint32_t ndims = 4;
    const size_t k_base = 2;
    const int num_parts = 2;

    std::vector<float> data(npts * ndims);
    for (size_t i = 0; i < data.size(); i++)
        data[i] = static_cast<float>(i) * 0.1f;
    create_bin_data_file(data_file, npts, ndims, data.data());

    float num_parts_f = static_cast<float>(num_parts);
    diskann::save_bin<float>(clusters_file, &num_parts_f, (size_t)1, (size_t)1);

    for (int i = 0; i < num_parts; i++)
    {
        std::string idmap_file = prefix + "_subshard-" + std::to_string(i) + "_ids_uint32.bin";
        size_t ids_per_shard = npts * k_base / num_parts;
        std::vector<uint32_t> ids(ids_per_shard);
        for (size_t j = 0; j < ids_per_shard; j++)
            ids[j] = static_cast<uint32_t>(j);
        create_idmap_file(idmap_file, ids);
    }

    int result = partition_with_ram_budget<float>(data_file, 0.1, 4.0, 64, prefix, k_base);

    BOOST_CHECK_LT(result, 0);

    std::remove(data_file.c_str());
    std::remove(clusters_file.c_str());
    for (int i = 0; i < num_parts; i++)
    {
        std::string idmap_file = prefix + "_subshard-" + std::to_string(i) + "_ids_uint32.bin";
        std::remove(idmap_file.c_str());
    }
}

BOOST_AUTO_TEST_CASE(gen_random_slice_file_size_mismatch)
{
    const std::string data_file = "/tmp/test_gen_random_slice_mismatch.bin";

    uint32_t npts = 10;
    uint32_t ndims = 4;
    std::vector<float> data(npts * ndims, 1.0f);

    {
        std::ofstream out(data_file, std::ios::binary);
        out.write((char *)&npts, sizeof(uint32_t));
        out.write((char *)&ndims, sizeof(uint32_t));
        out.write((char *)data.data(), (npts * ndims - 2) * sizeof(float));
        out.close();
    }

    float *sampled_data = nullptr;
    size_t slice_size = 0;
    size_t out_ndims = 0;

    BOOST_CHECK_THROW(gen_random_slice<float>(data_file, 0.5, sampled_data, slice_size, out_ndims),
                      std::runtime_error);

    std::remove(data_file.c_str());
}

BOOST_AUTO_TEST_CASE(gen_random_slice_basic)
{
    const std::string data_file = "/tmp/test_gen_random_slice_basic.bin";

    const uint32_t npts = 100;
    const uint32_t ndims = 4;
    std::vector<float> data(npts * ndims);
    for (size_t i = 0; i < data.size(); i++)
        data[i] = static_cast<float>(i) * 0.01f;
    create_bin_data_file(data_file, npts, ndims, data.data());

    float *sampled_data = nullptr;
    size_t slice_size = 0;
    size_t out_ndims = 0;

    gen_random_slice<float>(data_file, 1.0, sampled_data, slice_size, out_ndims);

    BOOST_CHECK_EQUAL(out_ndims, ndims);
    BOOST_CHECK_EQUAL(slice_size, npts);
    BOOST_CHECK(sampled_data != nullptr);

    if (sampled_data != nullptr)
        delete[] sampled_data;

    sampled_data = nullptr;
    slice_size = 0;
    out_ndims = 0;
    gen_random_slice<float>(data_file, 0.5, sampled_data, slice_size, out_ndims);

    BOOST_CHECK_EQUAL(out_ndims, ndims);
    BOOST_CHECK_GT(slice_size, 0u);
    BOOST_CHECK_LE(slice_size, npts);
    BOOST_CHECK(sampled_data != nullptr);

    if (sampled_data != nullptr)
        delete[] sampled_data;

    std::remove(data_file.c_str());
}

BOOST_AUTO_TEST_SUITE_END()
