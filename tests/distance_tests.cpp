// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>
#include "distance.h"
#include <vector>

BOOST_AUTO_TEST_SUITE(DistanceTests)

BOOST_AUTO_TEST_CASE(get_distance_function_l2_float) {
    auto dist_func = diskann::get_distance_function<float>(diskann::Metric::L2);
    BOOST_CHECK(dist_func != nullptr);

    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};

    float dist = dist_func->compare(a.data(), b.data(), 3);
    BOOST_CHECK(dist >= 0);  // distance should be non-negative
}

BOOST_AUTO_TEST_CASE(get_distance_function_cosine_float) {
    auto dist_func = diskann::get_distance_function<float>(diskann::Metric::COSINE);
    BOOST_CHECK(dist_func != nullptr);
}

BOOST_AUTO_TEST_CASE(get_distance_function_inner_product) {
    auto dist_func = diskann::get_distance_function<float>(diskann::Metric::INNER_PRODUCT);
    BOOST_CHECK(dist_func != nullptr);
}

BOOST_AUTO_TEST_CASE(get_distance_function_uint8) {
    auto dist_func = diskann::get_distance_function<uint8_t>(diskann::Metric::L2);
    BOOST_CHECK(dist_func != nullptr);
}

BOOST_AUTO_TEST_CASE(get_distance_function_int8) {
    auto dist_func = diskann::get_distance_function<int8_t>(diskann::Metric::L2);
    BOOST_CHECK(dist_func != nullptr);
}

BOOST_AUTO_TEST_SUITE_END()
