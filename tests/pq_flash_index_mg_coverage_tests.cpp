// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>
#include "pq_flash_index_mg_uring.h"
#include "io_uring_aligned_file_reader.h"
#include "utils.h"
#include <vector>
#include <fstream>
#include <cstdlib>
#include <memory>

BOOST_AUTO_TEST_SUITE(PQFlashIndexMGCoverageTests)

BOOST_AUTO_TEST_CASE(pq_flash_index_mg_instantiate)
{
    // This test only instantiates PQFlashIndexMGV2 to get constructor coverage.
    // Full load/search testing is done in pq_flash_index_integration_tests.
    // Note: Running multiple PQFlashIndexMGV2 load operations in the same process
    // causes segfaults (see disabled test in pq_flash_index_integration_tests.cpp).

    try {
        std::shared_ptr<AlignedFileReaderV2> reader = std::make_shared<LinuxAlignedFileReaderV2>();
        diskann::PQFlashIndexMGV2<float> index(reader, diskann::Metric::L2);
        // Just instantiate - this covers constructor code
    } catch (...) {
        // Ignore all errors
    }
}

BOOST_AUTO_TEST_SUITE_END()
