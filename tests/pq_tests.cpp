// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>
#include "pq.h"
#include <vector>
#include <cstring>

BOOST_AUTO_TEST_SUITE(PQTests)

BOOST_AUTO_TEST_CASE(round_up_various_values)
{
    BOOST_CHECK_EQUAL(diskann::round_up(0, 64), 0);
    BOOST_CHECK_EQUAL(diskann::round_up(1, 64), 64);
    BOOST_CHECK_EQUAL(diskann::round_up(63, 64), 64);
    BOOST_CHECK_EQUAL(diskann::round_up(64, 64), 64);
    BOOST_CHECK_EQUAL(diskann::round_up(65, 64), 128);
    BOOST_CHECK_EQUAL(diskann::round_up(127, 64), 128);
    BOOST_CHECK_EQUAL(diskann::round_up(128, 64), 128);
    BOOST_CHECK_EQUAL(diskann::round_up(1000, 256), 1024);
}

BOOST_AUTO_TEST_CASE(aggregate_coords_basic)
{
    const size_t ndims = 4;
    const size_t n_ids = 3;

    uint8_t all_coords[12] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };

    std::vector<uint32_t> ids = {0, 2};
    uint8_t out[8];

    diskann::aggregate_coords(ids, all_coords, ndims, out);

    BOOST_CHECK_EQUAL(out[0], 1);
    BOOST_CHECK_EQUAL(out[1], 2);
    BOOST_CHECK_EQUAL(out[2], 3);
    BOOST_CHECK_EQUAL(out[3], 4);
    BOOST_CHECK_EQUAL(out[4], 9);
    BOOST_CHECK_EQUAL(out[5], 10);
    BOOST_CHECK_EQUAL(out[6], 11);
    BOOST_CHECK_EQUAL(out[7], 12);
}

BOOST_AUTO_TEST_CASE(aggregate_coords_single_id)
{
    const size_t ndims = 3;
    uint8_t all_coords[9] = {
        10, 20, 30,
        40, 50, 60,
        70, 80, 90
    };

    std::vector<uint32_t> ids = {1};
    uint8_t out[3];

    diskann::aggregate_coords(ids, all_coords, ndims, out);

    BOOST_CHECK_EQUAL(out[0], 40);
    BOOST_CHECK_EQUAL(out[1], 50);
    BOOST_CHECK_EQUAL(out[2], 60);
}

BOOST_AUTO_TEST_CASE(pq_dist_lookup_basic)
{
    const size_t n_pts = 2;
    const size_t pq_nchunks = 2;

    uint8_t pq_ids[4] = {
        0, 1,  // point 0: chunk0=0, chunk1=1
        2, 3   // point 1: chunk0=2, chunk1=3
    };

    float pq_dists[512];
    for (int i = 0; i < 512; ++i) pq_dists[i] = 0.0f;

    pq_dists[0] = 1.0f;
    pq_dists[256 + 1] = 2.0f;
    pq_dists[2] = 3.0f;
    pq_dists[256 + 3] = 4.0f;

    std::vector<float> dists_out;
    diskann::pq_dist_lookup(pq_ids, n_pts, pq_nchunks, pq_dists, dists_out);

    BOOST_CHECK_EQUAL(dists_out.size(), 2);
    BOOST_CHECK_CLOSE(dists_out[0], 1.0f + 2.0f, 0.001f);
    BOOST_CHECK_CLOSE(dists_out[1], 3.0f + 4.0f, 0.001f);
}

BOOST_AUTO_TEST_CASE(pq_dist_lookup_single_chunk)
{
    const size_t n_pts = 3;
    const size_t pq_nchunks = 1;

    uint8_t pq_ids[3] = {0, 1, 2};
    float pq_dists[3] = {5.0f, 10.0f, 15.0f};

    std::vector<float> dists_out;
    diskann::pq_dist_lookup(pq_ids, n_pts, pq_nchunks, pq_dists, dists_out);

    BOOST_CHECK_EQUAL(dists_out.size(), 3);
    BOOST_CHECK_CLOSE(dists_out[0], 5.0f, 0.001f);
    BOOST_CHECK_CLOSE(dists_out[1], 10.0f, 0.001f);
    BOOST_CHECK_CLOSE(dists_out[2], 15.0f, 0.001f);
}

BOOST_AUTO_TEST_CASE(generate_pq_pivots_small_dataset)
{
    const size_t num_train = 20;
    const uint32_t dim = 4;
    const uint32_t num_centers = 4;
    const uint32_t num_pq_chunks = 2;

    std::vector<float> train_data(num_train * dim);
    for (size_t i = 0; i < num_train * dim; ++i)
    {
        train_data[i] = static_cast<float>(i % 10);
    }

    const char* pq_pivots_path = "test_pq_pivots.bin";

    int result = diskann::generate_pq_pivots(
        train_data.data(), num_train, dim, num_centers,
        num_pq_chunks, 10, pq_pivots_path, false
    );

    BOOST_CHECK_EQUAL(result, 0);

    std::ifstream in(pq_pivots_path, std::ios::binary);
    BOOST_CHECK(in.is_open());
    in.close();

    std::remove(pq_pivots_path);
}

BOOST_AUTO_TEST_CASE(FixedChunkPQTable_get_num_chunks)
{
    diskann::FixedChunkPQTable table;
    uint32_t chunks = table.get_num_chunks();
    BOOST_CHECK_EQUAL(chunks, 0);
}

BOOST_AUTO_TEST_CASE(aggregate_coords_array_version)
{
    const size_t ndims = 3;
    const size_t n_ids = 2;
    uint8_t all_coords[9] = {
        10, 20, 30,
        40, 50, 60,
        70, 80, 90
    };

    uint32_t ids[2] = {0, 2};
    uint8_t out[6];

    diskann::aggregate_coords(ids, n_ids, all_coords, ndims, out);

    BOOST_CHECK_EQUAL(out[0], 10);
    BOOST_CHECK_EQUAL(out[1], 20);
    BOOST_CHECK_EQUAL(out[2], 30);
    BOOST_CHECK_EQUAL(out[3], 70);
    BOOST_CHECK_EQUAL(out[4], 80);
    BOOST_CHECK_EQUAL(out[5], 90);
}

BOOST_AUTO_TEST_CASE(pq_dist_lookup_array_version)
{
    const size_t n_pts = 2;
    const size_t pq_nchunks = 1;

    uint8_t pq_ids[2] = {0, 1};
    float pq_dists[256];
    for (int i = 0; i < 256; ++i) pq_dists[i] = 0.0f;
    pq_dists[0] = 5.0f;
    pq_dists[1] = 10.0f;

    float dists_out[2];
    diskann::pq_dist_lookup(pq_ids, n_pts, pq_nchunks, pq_dists, dists_out);

    BOOST_CHECK_CLOSE(dists_out[0], 5.0f, 0.001f);
    BOOST_CHECK_CLOSE(dists_out[1], 10.0f, 0.001f);
}

BOOST_AUTO_TEST_CASE(FixedChunkPQTable_preprocess_query)
{
    diskann::FixedChunkPQTable table;
    float query[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    table.preprocess_query(query);
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(FixedChunkPQTable_load_and_query)
{
    const size_t num_train = 300;
    const uint32_t dim = 8;
    const uint32_t num_centers = 256;
    const uint32_t num_pq_chunks = 2;

    std::vector<float> train_data(num_train * dim);
    for (size_t i = 0; i < num_train * dim; ++i)
    {
        train_data[i] = static_cast<float>((i % 100) / 10.0f);
    }

    const char* pq_pivots_path = "test_pq_table.bin";

    int result = diskann::generate_pq_pivots(
        train_data.data(), num_train, dim, num_centers,
        num_pq_chunks, 10, pq_pivots_path, false
    );

    BOOST_CHECK_EQUAL(result, 0);

    diskann::FixedChunkPQTable table;
    table.load_pq_centroid_bin(pq_pivots_path, num_pq_chunks);

    BOOST_CHECK_EQUAL(table.get_num_chunks(), num_pq_chunks);

    float query[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    table.preprocess_query(query);

    float dist_vec[256 * num_pq_chunks];
    table.populate_chunk_distances(query, dist_vec);

    BOOST_CHECK(true);

    std::remove(pq_pivots_path);
}

BOOST_AUTO_TEST_CASE(FixedChunkPQTable_l2_distance)
{
    const size_t num_train = 300;
    const uint32_t dim = 8;
    const uint32_t num_centers = 256;
    const uint32_t num_pq_chunks = 2;

    std::vector<float> train_data(num_train * dim);
    for (size_t i = 0; i < num_train * dim; ++i)
    {
        train_data[i] = static_cast<float>((i % 100) / 10.0f);
    }

    const char* pq_pivots_path = "test_pq_l2.bin";
    diskann::generate_pq_pivots(train_data.data(), num_train, dim, num_centers,
                                num_pq_chunks, 10, pq_pivots_path, false);

    diskann::FixedChunkPQTable table;
    table.load_pq_centroid_bin(pq_pivots_path, num_pq_chunks);

    float query[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    table.preprocess_query(query);

    uint8_t base_vec[2] = {0, 0};
    float dist = table.l2_distance(query, base_vec);
    BOOST_CHECK(dist >= 0);

    std::remove(pq_pivots_path);
}

BOOST_AUTO_TEST_CASE(FixedChunkPQTable_inner_product)
{
    const size_t num_train = 300;
    const uint32_t dim = 8;
    const uint32_t num_centers = 256;
    const uint32_t num_pq_chunks = 2;

    std::vector<float> train_data(num_train * dim);
    for (size_t i = 0; i < num_train * dim; ++i)
    {
        train_data[i] = static_cast<float>((i % 100) / 10.0f);
    }

    const char* pq_pivots_path = "test_pq_ip.bin";
    diskann::generate_pq_pivots(train_data.data(), num_train, dim, num_centers,
                                num_pq_chunks, 10, pq_pivots_path, false);

    diskann::FixedChunkPQTable table;
    table.load_pq_centroid_bin(pq_pivots_path, num_pq_chunks);

    float query[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    table.preprocess_query(query);

    uint8_t base_vec[2] = {0, 0};
    float ip = table.inner_product(query, base_vec);
    BOOST_CHECK(true);

    std::remove(pq_pivots_path);
}

BOOST_AUTO_TEST_CASE(FixedChunkPQTable_inflate_vector)
{
    const size_t num_train = 300;
    const uint32_t dim = 8;
    const uint32_t num_centers = 256;
    const uint32_t num_pq_chunks = 2;

    std::vector<float> train_data(num_train * dim);
    for (size_t i = 0; i < num_train * dim; ++i)
    {
        train_data[i] = static_cast<float>((i % 100) / 10.0f);
    }

    const char* pq_pivots_path = "test_pq_inflate.bin";
    diskann::generate_pq_pivots(train_data.data(), num_train, dim, num_centers,
                                num_pq_chunks, 10, pq_pivots_path, false);

    diskann::FixedChunkPQTable table;
    table.load_pq_centroid_bin(pq_pivots_path, num_pq_chunks);

    uint8_t base_vec[2] = {0, 1};
    float out_vec[8];
    table.inflate_vector(base_vec, out_vec);

    BOOST_CHECK(true);

    std::remove(pq_pivots_path);
}

BOOST_AUTO_TEST_CASE(FixedChunkPQTable_populate_chunk_inner_products)
{
    const size_t num_train = 300;
    const uint32_t dim = 8;
    const uint32_t num_centers = 256;
    const uint32_t num_pq_chunks = 2;

    std::vector<float> train_data(num_train * dim);
    for (size_t i = 0; i < num_train * dim; ++i)
    {
        train_data[i] = static_cast<float>((i % 100) / 10.0f);
    }

    const char* pq_pivots_path = "test_pq_chunk_ip.bin";
    diskann::generate_pq_pivots(train_data.data(), num_train, dim, num_centers,
                                num_pq_chunks, 10, pq_pivots_path, false);

    diskann::FixedChunkPQTable table;
    table.load_pq_centroid_bin(pq_pivots_path, num_pq_chunks);

    float query[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float dist_vec[256 * num_pq_chunks];
    table.populate_chunk_inner_products(query, dist_vec);

    BOOST_CHECK(true);

    std::remove(pq_pivots_path);
}

BOOST_AUTO_TEST_CASE(FixedChunkPQTable_compute_query_residual_norm)
{
    const size_t num_train = 300;
    const uint32_t dim = 8;
    const uint32_t num_centers = 256;
    const uint32_t num_pq_chunks = 2;

    std::vector<float> train_data(num_train * dim);
    for (size_t i = 0; i < num_train * dim; ++i)
    {
        train_data[i] = static_cast<float>((i % 100) / 10.0f);
    }

    const char* pq_pivots_path = "test_pq_residual.bin";
    diskann::generate_pq_pivots(train_data.data(), num_train, dim, num_centers,
                                num_pq_chunks, 10, pq_pivots_path, false);

    diskann::FixedChunkPQTable table;
    table.load_pq_centroid_bin(pq_pivots_path, num_pq_chunks);

    float query[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    table.preprocess_query(query);

    float dist_vec[256 * num_pq_chunks];
    table.populate_chunk_distances(query, dist_vec);

    float residual_norm = table.compute_query_residual_norm(query, dist_vec);
    BOOST_CHECK(residual_norm >= 0);

    std::remove(pq_pivots_path);
}

BOOST_AUTO_TEST_CASE(generate_pq_data_from_pivots_float)
{
    // Create test data file
    const char* data_file = "test_data_for_pq.bin";
    const size_t num_points = 100;
    const uint32_t dim = 8;

    std::vector<float> data(num_points * dim);
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = static_cast<float>(i % 10);
    }
    diskann::save_bin<float>(data_file, data.data(), num_points, dim);

    // Generate pivots
    const char* pivots_file = "test_pivots_for_data.bin";
    const uint32_t num_centers = 256;
    const uint32_t num_pq_chunks = 2;
    diskann::generate_pq_pivots(data.data(), num_points, dim, num_centers,
                                num_pq_chunks, 10, pivots_file, false);

    // Generate PQ data
    const char* pq_compressed_file = "test_pq_compressed.bin";
    int result = diskann::generate_pq_data_from_pivots<float>(
        data_file, num_centers, num_pq_chunks, pivots_file, pq_compressed_file
    );

    BOOST_CHECK_EQUAL(result, 0);

    std::remove(data_file);
    std::remove(pivots_file);
    std::remove(pq_compressed_file);
}

BOOST_AUTO_TEST_SUITE_END()
