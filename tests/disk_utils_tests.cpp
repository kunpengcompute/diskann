// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>
#include "disk_utils.h"
#include <fstream>
#include <cstdio>

BOOST_AUTO_TEST_SUITE(DiskUtilsTests)

BOOST_AUTO_TEST_CASE(get_memory_budget_from_double)
{
    double budget = diskann::get_memory_budget(10.0);
    BOOST_CHECK(budget > 0);
    double expected_bytes = 10.0 * 1024 * 1024 * 1024;
    BOOST_CHECK(budget <= expected_bytes);
}

BOOST_AUTO_TEST_CASE(get_memory_budget_from_string)
{
    double budget1 = diskann::get_memory_budget("10");
    double expected1 = 10.0 * 1024 * 1024 * 1024;
    BOOST_CHECK(budget1 > 0);
    BOOST_CHECK(budget1 <= expected1);

    double budget2 = diskann::get_memory_budget("5.5");
    double expected2 = 5.5 * 1024 * 1024 * 1024;
    BOOST_CHECK(budget2 > 0);
    BOOST_CHECK(budget2 <= expected2);
}

BOOST_AUTO_TEST_CASE(read_idmap_basic)
{
    const char* filename = "test_idmap.bin";

    std::ofstream out(filename, std::ios::binary);
    uint32_t num_ids = 5;
    uint32_t dim = 1;
    out.write(reinterpret_cast<const char*>(&num_ids), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(uint32_t));
    uint32_t ids[5] = {10, 20, 30, 40, 50};
    out.write(reinterpret_cast<const char*>(ids), 5 * sizeof(uint32_t));
    out.close();

    std::vector<uint32_t> ivecs;
    diskann::read_idmap(filename, ivecs);

    BOOST_CHECK_EQUAL(ivecs.size(), 5);
    BOOST_CHECK_EQUAL(ivecs[0], 10);
    BOOST_CHECK_EQUAL(ivecs[1], 20);
    BOOST_CHECK_EQUAL(ivecs[4], 50);

    std::remove(filename);
}

BOOST_AUTO_TEST_CASE(load_warmup_file_format)
{
    const char* filename = "test_warmup.bin";
    uint64_t warmup_num = 10;
    uint64_t warmup_dim = 8;
    uint64_t warmup_aligned_dim = 8;

    std::vector<float> data(warmup_num * warmup_aligned_dim);
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = static_cast<float>(i);
    }

    std::ofstream out(filename, std::ios::binary);
    uint32_t num = static_cast<uint32_t>(warmup_num);
    uint32_t dim = static_cast<uint32_t>(warmup_dim);
    out.write(reinterpret_cast<const char*>(&num), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(data.data()),
              data.size() * sizeof(float));
    out.close();

    uint64_t loaded_num = 0;
    float* loaded = diskann::load_warmup<float>(
        filename, loaded_num, warmup_dim, warmup_aligned_dim
    );

    BOOST_CHECK_EQUAL(loaded_num, warmup_num);
    BOOST_CHECK(loaded != nullptr);

    for (size_t i = 0; i < data.size(); ++i)
    {
        BOOST_CHECK_EQUAL(loaded[i], data[i]);
    }

    diskann::aligned_free(loaded);
    std::remove(filename);
}

BOOST_AUTO_TEST_CASE(load_warmup_int8)
{
    const char* filename = "test_warmup_int8.bin";
    uint64_t warmup_num = 5;
    uint64_t warmup_dim = 8;
    uint64_t warmup_aligned_dim = 8;

    std::vector<int8_t> data(warmup_num * warmup_aligned_dim);
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = static_cast<int8_t>(i % 128);
    }

    std::ofstream out(filename, std::ios::binary);
    uint32_t num = static_cast<uint32_t>(warmup_num);
    uint32_t dim = static_cast<uint32_t>(warmup_dim);
    out.write(reinterpret_cast<const char*>(&num), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(data.data()),
              data.size() * sizeof(int8_t));
    out.close();

    uint64_t loaded_num = 0;
    int8_t* loaded = diskann::load_warmup<int8_t>(
        filename, loaded_num, warmup_dim, warmup_aligned_dim
    );

    BOOST_CHECK_EQUAL(loaded_num, warmup_num);
    BOOST_CHECK(loaded != nullptr);

    diskann::aligned_free(loaded);
    std::remove(filename);
}

BOOST_AUTO_TEST_CASE(load_warmup_uint8)
{
    const char* filename = "test_warmup_uint8.bin";
    uint64_t warmup_num = 5;
    uint64_t warmup_dim = 8;
    uint64_t warmup_aligned_dim = 8;

    std::vector<uint8_t> data(warmup_num * warmup_aligned_dim);
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = static_cast<uint8_t>(i);
    }

    std::ofstream out(filename, std::ios::binary);
    uint32_t num = static_cast<uint32_t>(warmup_num);
    uint32_t dim = static_cast<uint32_t>(warmup_dim);
    out.write(reinterpret_cast<const char*>(&num), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&dim), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(data.data()),
              data.size() * sizeof(uint8_t));
    out.close();

    uint64_t loaded_num = 0;
    uint8_t* loaded = diskann::load_warmup<uint8_t>(
        filename, loaded_num, warmup_dim, warmup_aligned_dim
    );

    BOOST_CHECK_EQUAL(loaded_num, warmup_num);
    BOOST_CHECK(loaded != nullptr);

    diskann::aligned_free(loaded);
    std::remove(filename);
}

BOOST_AUTO_TEST_SUITE_END()
