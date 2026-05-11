// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>
#include "scratch_uring.h"
#include "concurrent_queue.h"
#include <vector>

BOOST_AUTO_TEST_SUITE(ScratchUringTests)

// InMemQueryScratchV2 tests
BOOST_AUTO_TEST_CASE(inmem_scratch_constructor) {
    diskann::InMemQueryScratchV2<float> scratch(100, 50, 64, 750, 128, 128, 8, false);
    BOOST_CHECK_EQUAL(scratch.get_L(), 100);
    BOOST_CHECK_EQUAL(scratch.get_R(), 64);
    BOOST_CHECK_EQUAL(scratch.get_maxc(), 750);
}

BOOST_AUTO_TEST_CASE(inmem_scratch_zero_params) {
    BOOST_CHECK_THROW(
        diskann::InMemQueryScratchV2<float>(0, 50, 64, 750, 128, 128, 8, false),
        diskann::ANNException
    );
}

BOOST_AUTO_TEST_CASE(inmem_scratch_resize) {
    diskann::InMemQueryScratchV2<float> scratch(50, 50, 64, 750, 128, 128, 8, false);
    BOOST_CHECK_EQUAL(scratch.get_L(), 50);
    scratch.resize_for_new_L(150);
    BOOST_CHECK_EQUAL(scratch.get_L(), 150);
}

BOOST_AUTO_TEST_CASE(inmem_scratch_clear) {
    diskann::InMemQueryScratchV2<float> scratch(100, 50, 64, 750, 128, 128, 8, false);
    scratch.pool().push_back(diskann::Neighbor(1, 0.5f));
    scratch.id_scratch().push_back(42);
    scratch.clear();
    BOOST_CHECK(scratch.pool().empty());
    BOOST_CHECK(scratch.id_scratch().empty());
}

BOOST_AUTO_TEST_CASE(inmem_scratch_with_pq) {
    diskann::InMemQueryScratchV2<float> scratch(100, 50, 64, 750, 128, 128, 8, true);
    BOOST_CHECK(scratch.pq_scratch() != nullptr);
}

BOOST_AUTO_TEST_CASE(inmem_scratch_getters) {
    diskann::InMemQueryScratchV2<float> scratch(100, 50, 64, 750, 128, 128, 8, false);
    BOOST_CHECK(scratch.aligned_query() != nullptr);
    BOOST_CHECK_EQUAL(scratch.pool().size(), 0);
    BOOST_CHECK_EQUAL(scratch.best_l_nodes().size(), 0);
    BOOST_CHECK_EQUAL(scratch.occlude_factor().size(), 0);
    BOOST_CHECK_EQUAL(scratch.id_scratch().size(), 0);
    BOOST_CHECK_EQUAL(scratch.dist_scratch().size(), 0);
}

// SSDQueryScratchV2 tests
BOOST_AUTO_TEST_CASE(ssd_scratch_constructor) {
    diskann::SSDQueryScratchV2<float> scratch(128, 4096);
    BOOST_CHECK(scratch.coord_scratch != nullptr);
    BOOST_CHECK(scratch.sector_scratch != nullptr);
    BOOST_CHECK_EQUAL(scratch.sector_idx, 0);
}

BOOST_AUTO_TEST_CASE(ssd_scratch_reset) {
    diskann::SSDQueryScratchV2<float> scratch(128, 4096);
    scratch.sector_idx = 5;
    scratch.visited.insert(10);
    scratch.edges_buffer.push_back(42);
    scratch.reset();
    BOOST_CHECK_EQUAL(scratch.sector_idx, 0);
    BOOST_CHECK(scratch.visited.empty());
    BOOST_CHECK(scratch.edges_buffer.empty());
}

// SSDThreadDataV2 tests
BOOST_AUTO_TEST_CASE(ssd_thread_data_constructor) {
    diskann::SSDThreadDataV2<float> thread_data(128, 4096);
    BOOST_CHECK(thread_data.scratch.coord_scratch != nullptr);
}

BOOST_AUTO_TEST_CASE(ssd_thread_data_clear) {
    diskann::SSDThreadDataV2<float> thread_data(128, 4096);
    thread_data.scratch.sector_idx = 10;
    thread_data.clear();
    BOOST_CHECK_EQUAL(thread_data.scratch.sector_idx, 0);
}

// ScratchStoreManagerV2 tests
BOOST_AUTO_TEST_CASE(scratch_manager_basic) {
    diskann::ConcurrentQueue<diskann::SSDThreadDataV2<float>*> queue;
    auto* data = new diskann::SSDThreadDataV2<float>(128, 4096);
    queue.push(data);

    {
        diskann::ScratchStoreManagerV2<diskann::SSDThreadDataV2<float>> manager(queue);
        auto* scratch = manager.scratch_space();
        BOOST_CHECK(scratch != nullptr);
        scratch->scratch.sector_idx = 5;
    }

    // After manager destructor, data should be back in queue
    auto* returned = queue.pop();
    BOOST_CHECK(returned != nullptr);
    BOOST_CHECK_EQUAL(returned->scratch.sector_idx, 0);  // cleared
    delete returned;
}

BOOST_AUTO_TEST_CASE(scratch_manager_destroy) {
    diskann::ConcurrentQueue<diskann::SSDThreadDataV2<float>*> queue;
    auto* data1 = new diskann::SSDThreadDataV2<float>(128, 4096);
    auto* data2 = new diskann::SSDThreadDataV2<float>(128, 4096);
    queue.push(data1);
    queue.push(data2);

    diskann::ScratchStoreManagerV2<diskann::SSDThreadDataV2<float>> manager(queue);
    manager.destroy();
    BOOST_CHECK(queue.empty());
}

BOOST_AUTO_TEST_SUITE_END()

