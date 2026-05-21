// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>
#include "pq.h"
#include <vector>
#include <cstring>
#include <cmath>

BOOST_AUTO_TEST_SUITE(PQNeonTests)

namespace
{

float compute_expected_dist(const uint8_t *pq_ids, size_t pt_idx, size_t pq_nchunks, const float *pq_dists)
{
    float dist = 0.0f;
    for (size_t chunk = 0; chunk < pq_nchunks; chunk++)
    {
        uint8_t center_id = pq_ids[pq_nchunks * pt_idx + chunk];
        dist += pq_dists[256 * chunk + center_id];
    }
    return dist;
}

} // namespace

BOOST_AUTO_TEST_CASE(pq_dist_lookup_8_points)
{
    const size_t n_pts = 8;
    const size_t pq_nchunks = 2;

    std::vector<uint8_t> pq_ids(n_pts * pq_nchunks);
    for (size_t i = 0; i < pq_ids.size(); i++)
    {
        pq_ids[i] = static_cast<uint8_t>(i % 256);
    }

    std::vector<float> pq_dists(256 * pq_nchunks);
    for (size_t i = 0; i < pq_dists.size(); i++)
    {
        pq_dists[i] = static_cast<float>(i) * 0.1f;
    }

    std::vector<float> dists_out(n_pts, 0.0f);
    diskann::pq_dist_lookup(pq_ids.data(), n_pts, pq_nchunks, pq_dists.data(), dists_out.data());

    for (size_t i = 0; i < n_pts; i++)
    {
        float expected = compute_expected_dist(pq_ids.data(), i, pq_nchunks, pq_dists.data());
        BOOST_CHECK_CLOSE(dists_out[i], expected, 0.001f);
    }
}

BOOST_AUTO_TEST_CASE(pq_dist_lookup_9_points)
{
    const size_t n_pts = 9;
    const size_t pq_nchunks = 2;

    std::vector<uint8_t> pq_ids(n_pts * pq_nchunks);
    for (size_t i = 0; i < pq_ids.size(); i++)
    {
        pq_ids[i] = static_cast<uint8_t>((i * 7) % 256);
    }

    std::vector<float> pq_dists(256 * pq_nchunks);
    for (size_t i = 0; i < pq_dists.size(); i++)
    {
        pq_dists[i] = static_cast<float>(i) * 0.05f;
    }

    std::vector<float> dists_out(n_pts, 0.0f);
    diskann::pq_dist_lookup(pq_ids.data(), n_pts, pq_nchunks, pq_dists.data(), dists_out.data());

    for (size_t i = 0; i < n_pts; i++)
    {
        float expected = compute_expected_dist(pq_ids.data(), i, pq_nchunks, pq_dists.data());
        BOOST_CHECK_CLOSE(dists_out[i], expected, 0.001f);
    }
}

BOOST_AUTO_TEST_CASE(pq_dist_lookup_16_points)
{
    const size_t n_pts = 16;
    const size_t pq_nchunks = 3;

    std::vector<uint8_t> pq_ids(n_pts * pq_nchunks);
    for (size_t i = 0; i < pq_ids.size(); i++)
    {
        pq_ids[i] = static_cast<uint8_t>((i * 13) % 256);
    }

    std::vector<float> pq_dists(256 * pq_nchunks);
    for (size_t i = 0; i < pq_dists.size(); i++)
    {
        pq_dists[i] = static_cast<float>(i % 50) * 0.2f;
    }

    std::vector<float> dists_out(n_pts, 0.0f);
    diskann::pq_dist_lookup(pq_ids.data(), n_pts, pq_nchunks, pq_dists.data(), dists_out.data());

    for (size_t i = 0; i < n_pts; i++)
    {
        float expected = compute_expected_dist(pq_ids.data(), i, pq_nchunks, pq_dists.data());
        BOOST_CHECK_CLOSE(dists_out[i], expected, 0.001f);
    }
}

BOOST_AUTO_TEST_CASE(pq_dist_lookup_7_points)
{
    const size_t n_pts = 7;
    const size_t pq_nchunks = 2;

    std::vector<uint8_t> pq_ids(n_pts * pq_nchunks);
    for (size_t i = 0; i < pq_ids.size(); i++)
    {
        pq_ids[i] = static_cast<uint8_t>((i * 3 + 5) % 256);
    }

    std::vector<float> pq_dists(256 * pq_nchunks);
    for (size_t i = 0; i < pq_dists.size(); i++)
    {
        pq_dists[i] = static_cast<float>(i) * 0.01f;
    }

    std::vector<float> dists_out(n_pts, 0.0f);
    diskann::pq_dist_lookup(pq_ids.data(), n_pts, pq_nchunks, pq_dists.data(), dists_out.data());

    for (size_t i = 0; i < n_pts; i++)
    {
        float expected = compute_expected_dist(pq_ids.data(), i, pq_nchunks, pq_dists.data());
        BOOST_CHECK_CLOSE(dists_out[i], expected, 0.001f);
    }
}

BOOST_AUTO_TEST_CASE(pq_dist_lookup_multi_chunks_large)
{
    const size_t n_pts = 10;
    const size_t pq_nchunks = 8;

    std::vector<uint8_t> pq_ids(n_pts * pq_nchunks);
    for (size_t i = 0; i < pq_ids.size(); i++)
    {
        pq_ids[i] = static_cast<uint8_t>((i * 11 + 3) % 256);
    }

    std::vector<float> pq_dists(256 * pq_nchunks);
    for (size_t i = 0; i < pq_dists.size(); i++)
    {
        pq_dists[i] = static_cast<float>(i % 100) * 0.03f;
    }

    std::vector<float> dists_out(n_pts, 0.0f);
    diskann::pq_dist_lookup(pq_ids.data(), n_pts, pq_nchunks, pq_dists.data(), dists_out.data());

    for (size_t i = 0; i < n_pts; i++)
    {
        float expected = compute_expected_dist(pq_ids.data(), i, pq_nchunks, pq_dists.data());
        BOOST_CHECK_CLOSE(dists_out[i], expected, 0.001f);
    }
}

BOOST_AUTO_TEST_CASE(pq_dist_lookup_vector_overload_8_points)
{
    const size_t n_pts = 8;
    const size_t pq_nchunks = 2;

    std::vector<uint8_t> pq_ids(n_pts * pq_nchunks);
    for (size_t i = 0; i < pq_ids.size(); i++)
    {
        pq_ids[i] = static_cast<uint8_t>(i % 256);
    }

    std::vector<float> pq_dists(256 * pq_nchunks);
    for (size_t i = 0; i < pq_dists.size(); i++)
    {
        pq_dists[i] = static_cast<float>(i) * 0.1f;
    }

    std::vector<float> dists_out;
    diskann::pq_dist_lookup(pq_ids.data(), n_pts, pq_nchunks, pq_dists.data(), dists_out);

    BOOST_CHECK_EQUAL(dists_out.size(), n_pts);
    for (size_t i = 0; i < n_pts; i++)
    {
        float expected = compute_expected_dist(pq_ids.data(), i, pq_nchunks, pq_dists.data());
        BOOST_CHECK_CLOSE(dists_out[i], expected, 0.001f);
    }
}

BOOST_AUTO_TEST_SUITE_END()
