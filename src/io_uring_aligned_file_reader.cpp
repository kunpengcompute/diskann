// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#include "io_uring_aligned_file_reader.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>

#include "liburing.h"
#include "log.h"

#define MAX_EVENTS 256

namespace
{
constexpr uint64_t kNoUserData = 0;
void execute_io(void *context, int fd, std::vector<IORequest> &reqs, uint64_t n_retries = 0, bool write = false)
{
    io_uring *ring = (io_uring *)context;
    constexpr uint64_t max_retries = 5;
    uint64_t attempts = 0;
    while (true)
    {
        for (uint64_t j = 0; j < reqs.size(); j++)
        {
            auto sqe = io_uring_get_sqe(ring);
            if (sqe == nullptr)
            {
                LOG(ERROR) << "Failed to get SQE";
                return;
            }
            sqe->user_data = kNoUserData;
            if (write)
            {
                io_uring_prep_write(sqe, fd, reqs[j].buf, reqs[j].len, reqs[j].offset);
            }
            else
            {
                io_uring_prep_read(sqe, fd, reqs[j].buf, reqs[j].len, reqs[j].offset);
            }
        }
        io_uring_submit(ring);

        io_uring_cqe *cqe = nullptr;
        bool fail = false;
        for (uint64_t j = 0; j < reqs.size(); j++)
        {
            int ret = 0;
            do
            {
                ret = io_uring_wait_cqe(ring, &cqe);
            } while (ret == -EINTR);

            if (ret < 0 || cqe->res < 0)
            {
                fail = true;
                LOG(ERROR) << "Failed " << strerror(-ret) << " " << ring << " " << j << " " << reqs[j].buf << " "
                           << reqs[j].len << " " << reqs[j].offset;
                io_uring_cqe_seen(ring, cqe);
                continue;
            }
            io_uring_cqe_seen(ring, cqe);
        }
        if (!fail)
        {
            break;
        }
        if (++attempts > max_retries)
        {
            LOG(ERROR) << "execute_io: exceeded max retries (" << max_retries << "), aborting";
            break;
        }
    }
}
} // namespace

LinuxAlignedFileReaderV2::LinuxAlignedFileReaderV2()
{
    this->file_desc = -1;
}

LinuxAlignedFileReaderV2::~LinuxAlignedFileReaderV2()
{
    int64_t ret;
    // check to make sure file_desc is closed
    ret = ::fcntl(this->file_desc, F_GETFD);
    if (ret == -1)
    {
        if (errno != EBADF)
        {
            diskann::cerr << "close() not called" << std::endl;
            // close file desc
            ret = ::close(this->file_desc);
            // error checks
            if (ret == -1)
            {
                diskann::cerr << "close() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno)
                              << std::endl;
            }
        }
    }
}

namespace ioctx
{
static thread_local io_uring *ring = nullptr;
};

void *LinuxAlignedFileReaderV2::get_ctx(int flag)
{
    if (unlikely(ioctx::ring == nullptr))
    {
        register_thread(flag);
    }
    return ioctx::ring;
}

void LinuxAlignedFileReaderV2::register_thread(int flag)
{
    if (ioctx::ring == nullptr)
    {
        ioctx::ring = new io_uring();
        int ret = io_uring_queue_init(MAX_EVENTS, ioctx::ring, flag);
        if (ret < 0)
        {
            LOG(ERROR) << "io_uring_queue_init failed: " << strerror(-ret);
            delete ioctx::ring;
            ioctx::ring = nullptr;
        }
    }
}

void LinuxAlignedFileReaderV2::deregister_thread()
{
    io_uring_queue_exit(ioctx::ring);
    delete ioctx::ring;
    ioctx::ring = nullptr;
}

void LinuxAlignedFileReaderV2::deregister_all_threads()
{
    return;
}

void LinuxAlignedFileReaderV2::open(const std::string &fname, bool enable_writes = false, bool enable_create = false)
{
    int flags = O_DIRECT | O_LARGEFILE | O_RDONLY;
    this->file_desc = ::open(fname.c_str(), flags, 0644);
    if (this->file_desc == -1)
    {
        throw std::runtime_error("Failed to open file: " + fname);
    }
}

void LinuxAlignedFileReaderV2::close()
{
    ::fcntl(this->file_desc, F_GETFD);
    ::close(this->file_desc);
}

void LinuxAlignedFileReaderV2::read(std::vector<IORequest> &read_reqs, void *ctx, bool async)
{
    if (this->file_desc == -1)
    {
        throw std::runtime_error("File not opened for read");
    }
    execute_io(ctx, this->file_desc, read_reqs);
    if (async == true)
    {
        diskann::cerr << "async only supported in Windows for now." << std::endl;
    }
}

void LinuxAlignedFileReaderV2::write(std::vector<IORequest> &write_reqs, void *ctx, bool async)
{
    if (this->file_desc == -1)
    {
        throw std::runtime_error("File not opened for write");
    }
    execute_io(ctx, this->file_desc, write_reqs, 0, true);
    if (async == true)
    {
        diskann::cerr << "async only supported in Windows for now." << std::endl;
    }
}

void LinuxAlignedFileReaderV2::read_fd(int fd, std::vector<IORequest> &read_reqs, void *ctx)
{
    if (this->file_desc == -1)
    {
        throw std::runtime_error("File not opened for read_fd");
    }
    execute_io(ctx, fd, read_reqs);
}

void LinuxAlignedFileReaderV2::write_fd(int fd, std::vector<IORequest> &write_reqs, void *ctx)
{
    if (this->file_desc == -1)
    {
        throw std::runtime_error("File not opened for write_fd");
    }
    execute_io(ctx, fd, write_reqs, 0, true);
}

void LinuxAlignedFileReaderV2::send_io(IORequest &req, void *ctx, bool write)
{
    io_uring *ring = (io_uring *)ctx;
    auto sqe = io_uring_get_sqe(ring);
    req.finished = 0;
    if (write)
    {
        io_uring_prep_write(sqe, this->file_desc, req.buf, req.len, req.offset);
    }
    else
    {
        io_uring_prep_read(sqe, this->file_desc, req.buf, req.len, req.offset);
    }
    sqe->user_data = (uint64_t)&req;
    int ret = io_uring_submit(ring);
    if (ret < 0)
    {
        diskann::cerr << "io_uring_submit failed with error: " << strerror(-ret) << std::endl;
    }
}

void LinuxAlignedFileReaderV2::send_io(std::vector<IORequest> &reqs, void *ctx, bool write)
{
    io_uring *ring = (io_uring *)ctx;
    for (uint64_t j = 0; j < reqs.size(); j++)
    {
        auto sqe = io_uring_get_sqe(ring);
        reqs[j].finished = 0;
        sqe->user_data = (uint64_t)&reqs[j];
        if (write)
        {
            io_uring_prep_write(sqe, this->file_desc, reqs[j].buf, reqs[j].len, reqs[j].offset);
        }
        else
        {
            io_uring_prep_read(sqe, this->file_desc, reqs[j].buf, reqs[j].len, reqs[j].offset);
        }
    }
    int ret = io_uring_submit(ring);
    if (ret < 0)
    {
        diskann::cerr << "io_uring_submit failed with error: " << strerror(-ret) << std::endl;
    }
}

int LinuxAlignedFileReaderV2::poll(void *ctx)
{
    io_uring *ring = (io_uring *)ctx;
    io_uring_cqe *cqe = nullptr;
    int ret = io_uring_peek_cqe(ring, &cqe);
    if (ret < 0)
    {
        return ret; // not finished yet.
    }
    if (cqe->res < 0)
    {
        LOG(ERROR) << "Failed " << strerror(-cqe->res);
    }
    IORequest *req = (IORequest *)cqe->user_data;
    if (req != nullptr)
    {
        req->finished = 1;
    }
    io_uring_cqe_seen(ring, cqe);
    return 0;
}

IORequest* LinuxAlignedFileReaderV2::poll_ior(void *ctx)
{
    io_uring *ring = (io_uring *)ctx;
    io_uring_cqe *cqe = nullptr;
    int ret = io_uring_peek_cqe(ring, &cqe);
    if (ret < 0)
    {
        return nullptr; // not finished yet.
    }
    if (cqe->res < 0)
    {
        LOG(ERROR) << "Failed " << strerror(-cqe->res);
    }
    IORequest *req = (IORequest *)cqe->user_data;
    if (req != nullptr)
    {
        req->finished = 1;
    }
    io_uring_cqe_seen(ring, cqe);
    return req;
}

void LinuxAlignedFileReaderV2::poll_all(void *ctx)
{
    io_uring *ring = (io_uring *)ctx;
    static __thread io_uring_cqe *cqes[MAX_EVENTS];
    int ret = io_uring_peek_batch_cqe(ring, cqes, MAX_EVENTS);
    if (ret < 0)
    {
        return; // not finished yet.
    }
    for (int i = 0; i < ret; i++)
    {
        if (cqes[i]->res < 0)
        {
            LOG(ERROR) << "Failed " << strerror(-cqes[i]->res);
        }
        IORequest *req = (IORequest *)cqes[i]->user_data;
        if (req != nullptr)
        {
            req->finished = 1;
        }else{
            LOG(ERROR) << "req == nullptr\n";
        }
        io_uring_cqe_seen(ring, cqes[i]);
    }
}

void LinuxAlignedFileReaderV2::poll_wait(void *ctx)
{
    io_uring *ring = (io_uring *)ctx;
    io_uring_cqe *cqe = nullptr;
    int ret = 0;
    do
    {
        ret = io_uring_wait_cqe(ring, &cqe);
    } while (ret == -EINTR);
    if (ret < 0 || cqe->res < 0)
    {
        LOG(ERROR) << "Failed " << strerror(-cqe->res);
    }
    IORequest *req = (IORequest *)cqe->user_data;
    if (req != nullptr)
    {
        req->finished = 1;
    }
    io_uring_cqe_seen(ring, cqe);
}

IORequest* LinuxAlignedFileReaderV2::poll_wait_one(void *ctx)
{
    io_uring *ring = (io_uring *)ctx;
    io_uring_cqe *cqe = nullptr;
    int ret = 0;
    do
    {
        ret = io_uring_wait_cqe(ring, &cqe);
    } while (ret == -EINTR);
    if (ret < 0 || cqe->res < 0)
    {
        LOG(ERROR) << "Failed " << strerror(-cqe->res);
    }
    IORequest *req = (IORequest *)cqe->user_data;
    if (req != nullptr)
    {
        req->finished = 1;
    }
    io_uring_cqe_seen(ring, cqe);
    return req;
}

int LinuxAlignedFileReaderV2::send_read_no_alloc(IORequest &req, void *ring)
{
    send_io(req, ring, false);
    return 1;
}

int LinuxAlignedFileReaderV2::send_read_no_alloc(std::vector<IORequest> &reqs, void *ring)
{
    send_io(reqs, ring, false);
    return reqs.size();
}

void LinuxAlignedFileReaderV2::read_alloc(std::vector<IORequest> &read_reqs, void *ctx, std::vector<uint64_t> *page_ref)
{

    read(read_reqs, ctx);
}
