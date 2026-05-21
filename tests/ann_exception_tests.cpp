// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>
#include "ann_exception.h"
#include <string>

BOOST_AUTO_TEST_SUITE(ANNExceptionTests)

BOOST_AUTO_TEST_CASE(constructor_with_message) {
    std::string msg = "Test error message";
    diskann::ANNException ex(msg, -1);

    std::string what_str = ex.what();
    BOOST_CHECK(what_str.find(msg) != std::string::npos);
}

BOOST_AUTO_TEST_CASE(constructor_with_code) {
    diskann::ANNException ex("Error", -42);
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(constructor_with_all_params) {
    diskann::ANNException ex("Error", -1, "function", "file.cpp", 123);

    std::string what_str = ex.what();
    BOOST_CHECK(what_str.find("Error") != std::string::npos);
}

BOOST_AUTO_TEST_CASE(throw_and_catch) {
    try {
        throw diskann::ANNException("Test exception", -1);
        BOOST_FAIL("Should have thrown");
    } catch (const diskann::ANNException& e) {
        BOOST_CHECK(true);
    }
}

BOOST_AUTO_TEST_SUITE_END()
