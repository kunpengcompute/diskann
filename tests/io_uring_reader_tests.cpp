// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include <boost/test/unit_test.hpp>
#include "io_uring_aligned_file_reader.h"
#include <stdexcept>

BOOST_AUTO_TEST_SUITE(IoUringAlignedFileReaderTests)

BOOST_AUTO_TEST_CASE(read_without_open)
{
    LinuxAlignedFileReaderV2 reader;
    std::vector<IORequest> reqs;
    void* ctx = nullptr;

    BOOST_CHECK_THROW(reader.read(reqs, ctx, false), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(write_without_open)
{
    LinuxAlignedFileReaderV2 reader;
    std::vector<IORequest> reqs;
    void* ctx = nullptr;

    BOOST_CHECK_THROW(reader.write(reqs, ctx, false), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(read_fd_without_open)
{
    LinuxAlignedFileReaderV2 reader;
    std::vector<IORequest> reqs;
    void* ctx = nullptr;

    BOOST_CHECK_THROW(reader.read_fd(0, reqs, ctx), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(write_fd_without_open)
{
    LinuxAlignedFileReaderV2 reader;
    std::vector<IORequest> reqs;
    void* ctx = nullptr;

    BOOST_CHECK_THROW(reader.write_fd(0, reqs, ctx), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(open_nonexistent_file)
{
    LinuxAlignedFileReaderV2 reader;
    BOOST_CHECK_THROW(reader.open("/nonexistent/path/file.bin", false, false), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(constructor_destructor) {
    LinuxAlignedFileReaderV2 reader;
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(register_deregister_thread) {
    LinuxAlignedFileReaderV2 reader;
    reader.register_thread();
    reader.deregister_thread();
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(deregister_all_threads) {
    LinuxAlignedFileReaderV2 reader;
    reader.register_thread();
    reader.deregister_all_threads();
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(get_ctx) {
    LinuxAlignedFileReaderV2 reader;
    void* ctx = reader.get_ctx(0);
    BOOST_CHECK(ctx != nullptr);
}

BOOST_AUTO_TEST_CASE(open_close_file) {
    LinuxAlignedFileReaderV2 reader;

    // Create a test file
    const char* test_file = "/tmp/test_uring_reader.bin";
    std::ofstream out(test_file, std::ios::binary);
    std::vector<char> data(4096, 'C');
    out.write(data.data(), data.size());
    out.close();

    reader.open(test_file, false, false);
    reader.close();

    std::remove(test_file);
    BOOST_CHECK(true);
}

BOOST_AUTO_TEST_CASE(read_file) {
    LinuxAlignedFileReaderV2 reader;

    // Create a test file
    const char* test_file = "/tmp/test_uring_read.bin";
    std::ofstream out(test_file, std::ios::binary);
    std::vector<char> data(4096, 'D');
    out.write(data.data(), data.size());
    out.close();

    reader.open(test_file, false, false);

    // Prepare read request
    std::vector<IORequest> reqs(1);
    char* buf = nullptr;
    diskann::alloc_aligned((void**)&buf, 4096, 512);
    reqs[0].buf = buf;
    reqs[0].len = 4096;
    reqs[0].offset = 0;

    void* ctx = reader.get_ctx(0);
    reader.read(reqs, ctx, false);

    BOOST_CHECK_EQUAL(buf[0], 'D');

    diskann::aligned_free(buf);
    reader.close();
    std::remove(test_file);
}

BOOST_AUTO_TEST_SUITE_END()
