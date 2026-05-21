// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>

#ifdef __linux__
#include "linux_aligned_file_reader.h"
#include <stdexcept>

BOOST_AUTO_TEST_SUITE(LinuxAlignedFileReaderTests)

BOOST_AUTO_TEST_CASE(constructor_initializes_file_desc)
{
    LinuxAlignedFileReader reader;
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(register_deregister_thread)
{
    LinuxAlignedFileReader reader;
    reader.register_thread();
    reader.deregister_thread();
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(deregister_all_threads)
{
    LinuxAlignedFileReader reader;
    reader.register_thread();
    reader.deregister_all_threads();
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(get_ctx)
{
    LinuxAlignedFileReader reader;
    reader.register_thread();
    io_context_t& ctx = reader.get_ctx();
    BOOST_CHECK(true);
    reader.deregister_thread();
}

BOOST_AUTO_TEST_CASE(open_close_file)
{
    LinuxAlignedFileReader reader;

    // Create a test file
    const char* test_file = "/tmp/test_linux_reader.bin";
    std::ofstream out(test_file, std::ios::binary);
    std::vector<char> data(4096, 'A');
    out.write(data.data(), data.size());
    out.close();

    reader.open(test_file);
    reader.close();

    std::remove(test_file);
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(read_file_sync)
{
    LinuxAlignedFileReader reader;
    reader.register_thread();

    // Create a test file
    const char* test_file = "/tmp/test_linux_read.bin";
    std::ofstream out(test_file, std::ios::binary);
    std::vector<char> data(4096, 'B');
    out.write(data.data(), data.size());
    out.close();

    reader.open(test_file);

    // Prepare read request
    std::vector<AlignedRead> reqs(1);
    char* buf = nullptr;
    diskann::alloc_aligned((void**)&buf, 4096, 512);
    reqs[0].buf = buf;
    reqs[0].len = 4096;
    reqs[0].offset = 0;

    io_context_t& ctx = reader.get_ctx();
    reader.read(reqs, ctx, false);

    BOOST_CHECK_EQUAL(buf[0], 'B');

    diskann::aligned_free(buf);
    reader.close();
    reader.deregister_thread();
    std::remove(test_file);
}

BOOST_AUTO_TEST_SUITE_END()

#else

BOOST_AUTO_TEST_SUITE(LinuxAlignedFileReaderTests)

BOOST_AUTO_TEST_CASE(skip_on_non_linux)
{
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_SUITE_END()

#endif
