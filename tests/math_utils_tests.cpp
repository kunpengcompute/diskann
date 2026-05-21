// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>
#include "math_utils.h"
#include <vector>
#include <cmath>

BOOST_AUTO_TEST_SUITE(MathUtilsTests)

BOOST_AUTO_TEST_CASE(calc_distance_basic) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> b = {2.0f, 3.0f, 4.0f, 5.0f};

    float dist = math_utils::calc_distance(a.data(), b.data(), 4);
    float expected = 4.0f;  // squared distance: (1+1+1+1)
    BOOST_CHECK_CLOSE(dist, expected, 0.01f);
}

BOOST_AUTO_TEST_CASE(calc_distance_zero) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {1.0f, 2.0f, 3.0f};

    float dist = math_utils::calc_distance(a.data(), b.data(), 3);
    BOOST_CHECK_SMALL(dist, 0.001f);
}

BOOST_AUTO_TEST_CASE(compute_vecs_l2sq_basic) {
    const size_t num_points = 3;
    const size_t dim = 4;

    std::vector<float> data = {
        1.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 3.0f, 4.0f, 5.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };

    std::vector<float> l2sq(num_points);
    math_utils::compute_vecs_l2sq(l2sq.data(), data.data(), num_points, dim);

    BOOST_CHECK_CLOSE(l2sq[0], 30.0f, 0.01f);  // 1+4+9+16
    BOOST_CHECK_CLOSE(l2sq[1], 54.0f, 0.01f);  // 4+9+16+25
    BOOST_CHECK_SMALL(l2sq[2], 0.001f);
}

BOOST_AUTO_TEST_CASE(compute_closest_centers_basic) {
    const size_t num_points = 2;
    const size_t dim = 4;
    const size_t num_centers = 3;

    std::vector<float> data = {
        1.0f, 1.0f, 1.0f, 1.0f,
        5.0f, 5.0f, 5.0f, 5.0f
    };

    std::vector<float> centers = {
        0.0f, 0.0f, 0.0f, 0.0f,
        2.0f, 2.0f, 2.0f, 2.0f,
        6.0f, 6.0f, 6.0f, 6.0f
    };

    std::vector<uint32_t> closest(num_points);
    math_utils::compute_closest_centers(data.data(), num_points, dim,
                                        centers.data(), num_centers, 1,
                                        closest.data(), nullptr, nullptr);

    BOOST_CHECK(closest[0] < num_centers);
    BOOST_CHECK(closest[1] < num_centers);
}

BOOST_AUTO_TEST_SUITE_END()
