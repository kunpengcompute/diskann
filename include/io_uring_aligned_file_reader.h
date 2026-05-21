// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#pragma once

#define MAX_IO_DEPTH 128

#include <fcntl.h>
#include <unistd.h>

#include <malloc.h>
#include <cstdio>
#include <utils.h>
#include "liburing/io_uring.h"

#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

struct alignas(128) IORequest
{
    uint64_t offset;   // where to read from (page)
    uint64_t len;      // how much to read
    void *buf;         // where to read into
    uint32_t id;       // vector id
    uint32_t finished; // for async IO
    uint32_t req_id;   // IORequest id
    uint8_t padding[92];

    IORequest() : offset(0), len(0), buf(nullptr), finished(0)
    {
    }

    IORequest(uint64_t offset, uint64_t len, void *buf, uint32_t id)
        : offset(offset), len(len), buf(buf), id(id), finished(0)
    {
        assert(IS_512_ALIGNED(offset));
        assert(IS_512_ALIGNED(len));
        assert(IS_512_ALIGNED(buf));
    }

    bool isfinished()
    {
        return finished == 1;
    }
};

static_assert(sizeof(IORequest) == 128, "IORequest must be 128 bytes");
static_assert(alignof(IORequest) == 128, "IORequest must be 128-byte aligned");

class AlignedFileReaderV2
{
  public:
    // returns the thread-specific io ring.
    // If not constructed, it will register the thread (using the flag) and return the context.
    // For io_uring reader, the flag is used to set up the ring (e.g., IORING_SETUP_SQPOLL).
    // For PipeSearch, we use IO_RING_SETUP_POLL to enable polling.
    // For all the other algorithms, we use 0 to disable polling.
    virtual void *get_ctx(int flag = 0) = 0;

    virtual ~AlignedFileReaderV2() {};

    // Open & close ops
    // Blocking calls
    virtual void open(const std::string &fname, bool enable_writes, bool enable_create) = 0;
    virtual void close() = 0;

    // process batch of aligned requests in parallel
    // NOTE :: blocking call
    virtual void read(std::vector<IORequest> &read_reqs, void *ctx, bool async = false) = 0;
    virtual void write(std::vector<IORequest> &write_reqs, void *ctx, bool async = false) = 0;
    virtual void read_fd(int fd, std::vector<IORequest> &read_reqs, void *ctx) = 0;
    virtual void write_fd(int fd, std::vector<IORequest> &write_reqs, void *ctx) = 0;

    virtual void read_alloc(std::vector<IORequest> &read_reqs, void *ctx,
                            std::vector<uint64_t> *page_ref = nullptr) = 0;

    virtual void send_io(IORequest &reqs, void *ctx, bool write) = 0;
    virtual void send_io(std::vector<IORequest> &reqs, void *ctx, bool write) = 0;
    // Note that this is not used in update, so no page_ref is adopted (returns n_ios).
    virtual int send_read_no_alloc(IORequest &req, void *ctx) = 0;
    virtual int send_read_no_alloc(std::vector<IORequest> &reqs, void *ctx) = 0;
    virtual int poll(void *ctx) = 0;
    virtual IORequest *poll_ior(void *ctx) = 0;
    virtual void poll_all(void *ctx) = 0;
    virtual void poll_wait(void *ctx) = 0;
    virtual IORequest *poll_wait_one(void *ctx) = 0;

    // register thread-id for a context
    virtual void register_thread(int flag = 0) = 0;
    // de-register thread-id for a context
    virtual void deregister_thread() = 0;
    virtual void deregister_all_threads() = 0;
};

class LinuxAlignedFileReaderV2 : public AlignedFileReaderV2
{
  private:
    uint64_t file_sz;
    FileHandle file_desc;
    void *bad_ctx = nullptr;

  public:
    LinuxAlignedFileReaderV2();
    ~LinuxAlignedFileReaderV2();

    void *get_ctx(int flag = 0);

    // Open & close ops
    // Blocking calls
    void open(const std::string &fname, bool enable_writes, bool enable_create);
    void close();

    // process batch of aligned requests in parallel
    // NOTE :: blocking call
    void read(std::vector<IORequest> &read_reqs, void *ctx, bool async = false);
    void write(std::vector<IORequest> &write_reqs, void *ctx, bool async = false);
    void read_fd(int fd, std::vector<IORequest> &read_reqs, void *ctx);
    void write_fd(int fd, std::vector<IORequest> &write_reqs, void *ctx);

    // read and update cache.
    void read_alloc(std::vector<IORequest> &read_reqs, void *ctx, std::vector<uint64_t> *page_ref = nullptr);
    // read but not update cache.
    int send_read_no_alloc(IORequest &req, void *ctx);
    int send_read_no_alloc(std::vector<IORequest> &reqs, void *ctx);

    void send_io(IORequest &reqs, void *ctx, bool write);
    void send_io(std::vector<IORequest> &reqs, void *ctx, bool write);
    int poll(void *ctx);
    IORequest *poll_ior(void *ctx);
    void poll_all(void *ctx);
    void poll_wait(void *ctx);
    IORequest *poll_wait_one(void *ctx);

    // register thread-id for a context
    void register_thread(int flag = 0);

    // de-register thread-id for a context
    void deregister_thread();

    void deregister_all_threads();
};
