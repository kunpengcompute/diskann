// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>
#include "utils.h"
#include <fstream>
#include <cstdio>

BOOST_AUTO_TEST_SUITE(UtilsTests)

BOOST_AUTO_TEST_CASE(save_and_load_bin_float)
{
    const char* filename = "test_utils_float.bin";
    const size_t npts = 10;
    const size_t ndims = 4;

    std::vector<float> data(npts * ndims);
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = static_cast<float>(i);
    }

    diskann::save_bin<float>(filename, data.data(), npts, ndims);

    size_t loaded_npts, loaded_ndims;
    float* loaded_data = nullptr;

    diskann::load_bin<float>(filename, loaded_data, loaded_npts, loaded_ndims);

    BOOST_CHECK_EQUAL(loaded_npts, npts);
    BOOST_CHECK_EQUAL(loaded_ndims, ndims);
    BOOST_CHECK(loaded_data != nullptr);

    for (size_t i = 0; i < npts * ndims; ++i)
    {
        BOOST_CHECK_EQUAL(loaded_data[i], data[i]);
    }

    delete[] loaded_data;
    std::remove(filename);
}

BOOST_AUTO_TEST_CASE(save_and_load_bin_uint8)
{
    const char* filename = "test_utils_uint8.bin";
    const size_t npts = 5;
    const size_t ndims = 8;

    std::vector<uint8_t> data(npts * ndims);
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = static_cast<uint8_t>(i % 256);
    }

    diskann::save_bin<uint8_t>(filename, data.data(), npts, ndims);

    size_t loaded_npts, loaded_ndims;
    uint8_t* loaded_data = nullptr;

    diskann::load_bin<uint8_t>(filename, loaded_data, loaded_npts, loaded_ndims);

    BOOST_CHECK_EQUAL(loaded_npts, npts);
    BOOST_CHECK_EQUAL(loaded_ndims, ndims);
    BOOST_CHECK(loaded_data != nullptr);

    for (size_t i = 0; i < npts * ndims; ++i)
    {
        BOOST_CHECK_EQUAL(loaded_data[i], data[i]);
    }

    delete[] loaded_data;
    std::remove(filename);
}

BOOST_AUTO_TEST_CASE(save_and_load_bin_int8)
{
    const char* filename = "test_utils_int8.bin";
    const size_t npts = 5;
    const size_t ndims = 8;

    std::vector<int8_t> data(npts * ndims);
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = static_cast<int8_t>((i % 128) - 64);
    }

    diskann::save_bin<int8_t>(filename, data.data(), npts, ndims);

    size_t loaded_npts, loaded_ndims;
    int8_t* loaded_data = nullptr;

    diskann::load_bin<int8_t>(filename, loaded_data, loaded_npts, loaded_ndims);

    BOOST_CHECK_EQUAL(loaded_npts, npts);
    BOOST_CHECK_EQUAL(loaded_ndims, ndims);
    BOOST_CHECK(loaded_data != nullptr);

    for (size_t i = 0; i < npts * ndims; ++i)
    {
        BOOST_CHECK_EQUAL(loaded_data[i], data[i]);
    }

    delete[] loaded_data;
    std::remove(filename);
}

BOOST_AUTO_TEST_CASE(load_aligned_bin_float)
{
    const char* filename = "test_aligned_float.bin";
    const size_t npts = 10;
    const size_t ndims = 8;
    const size_t aligned_dim = 8;

    std::vector<float> data(npts * ndims);
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = static_cast<float>(i);
    }

    diskann::save_bin<float>(filename, data.data(), npts, ndims);

    size_t loaded_npts, loaded_ndims, loaded_aligned_dim;
    float* loaded_data = nullptr;

    diskann::load_aligned_bin<float>(filename, loaded_data, loaded_npts,
                                     loaded_ndims, loaded_aligned_dim);

    BOOST_CHECK_EQUAL(loaded_npts, npts);
    BOOST_CHECK_EQUAL(loaded_ndims, ndims);
    BOOST_CHECK_EQUAL(loaded_aligned_dim, aligned_dim);
    BOOST_CHECK(loaded_data != nullptr);

    diskann::aligned_free(loaded_data);
    std::remove(filename);
}

BOOST_AUTO_TEST_CASE(copy_aligned_data_from_file)
{
    const char* filename = "test_copy_aligned.bin";
    size_t npts = 5;
    size_t ndims = 8;

    std::vector<float> data(npts * ndims);
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = static_cast<float>(i);
    }

    diskann::save_bin<float>(filename, data.data(), npts, ndims);

    float* dest_data = nullptr;
    diskann::alloc_aligned((void**)&dest_data, npts * ndims * sizeof(float), 8 * sizeof(float));

    diskann::copy_aligned_data_from_file<float>(filename, dest_data, npts, ndims, ndims);

    for (size_t i = 0; i < npts * ndims; ++i)
    {
        BOOST_CHECK_EQUAL(dest_data[i], data[i]);
    }

    diskann::aligned_free(dest_data);
    std::remove(filename);
}

BOOST_AUTO_TEST_SUITE_END()
