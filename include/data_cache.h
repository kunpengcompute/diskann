// Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tsl/robin_map.h"

namespace diskann
{

template <typename T> class DataCache
{
  public:
    DataCache() = default;
    ~DataCache()
    {
        clear_all();
    }

    DataCache(const DataCache &) = delete;
    DataCache &operator=(const DataCache &) = delete;
    DataCache(DataCache &&) = delete;
    DataCache &operator=(DataCache &&) = delete;

    // Vector cache

    void reserve_vector_cache(uint64_t memory_budget_bytes, uint64_t dim);

    void add_vector(uint32_t vector_id, const T *data);

    inline bool is_vector_cached(uint32_t vector_id) const
    {
        return _vector_id_to_index.find(vector_id) != _vector_id_to_index.end();
    }

    const T *get_cached_vector(uint32_t vector_id) const;

    const T *get_cached_vector_by_index(uint32_t index) const
    {
        if (index >= _cached_vector_count)
        {
            return nullptr;
        }
        return _cached_vector_buf + static_cast<uint64_t>(index) * _vector_dim_aligned;
    }

    int32_t get_vector_index(uint32_t vector_id) const
    {
        auto it = _vector_id_to_index.find(vector_id);
        if (it == _vector_id_to_index.end())
        {
            return -1;
        }
        return static_cast<int32_t>(it->second);
    }

    inline uint32_t get_cached_vector_count() const
    {
        return _cached_vector_count;
    }

    inline uint64_t get_vector_dim() const
    {
        return _vector_dim_original;
    }

    inline uint64_t get_vector_dim_aligned() const
    {
        return _vector_dim_aligned;
    }

    inline uint64_t get_vector_cache_allocated_bytes() const
    {
        return _vector_cache_allocated_bytes;
    }

    inline uint64_t get_vector_cache_used_bytes() const
    {
        return static_cast<uint64_t>(_cached_vector_count) * _vector_dim_aligned * sizeof(T);
    }

    inline uint64_t get_vector_cache_budget_bytes() const
    {
        return _vector_memory_budget_bytes;
    }

    void clear_vector_cache();

    template <typename Callback> void for_each_vector(Callback callback) const
    {
        for (const auto &[vector_id, cache_index] : _vector_id_to_index)
        {
            const T *vec = _cached_vector_buf + static_cast<uint64_t>(cache_index) * _vector_dim_aligned;
            callback(vector_id, vec);
        }
    }

    // Graph cache

    void reserve_graph_cache(uint64_t memory_budget_bytes);

    void reserve_graph_node_map(uint32_t expected_node_count);

    void add_graph_node(uint32_t node_id, const uint32_t *neighbors, uint32_t degree);

    inline bool is_graph_node_cached(uint32_t node_id) const
    {
        return _node_id_to_offset.find(node_id) != _node_id_to_offset.end();
    }

    std::pair<uint32_t, const uint32_t *> get_cached_graph_adj(uint32_t node_id) const;

    inline uint32_t get_cached_graph_node_count() const
    {
        return _cached_graph_node_count;
    }

    inline uint64_t get_graph_cache_budget_bytes() const
    {
        return _graph_memory_budget_bytes;
    }

    inline uint64_t get_graph_cache_used_bytes() const
    {
        return _cached_graph_buf_used_bytes;
    }

    inline uint64_t get_graph_cache_allocated_bytes() const
    {
        return _cached_graph_buf_size_bytes;
    }

    void clear_graph_cache();

    template <typename Callback> void for_each_graph_node(Callback callback) const
    {
        for (const auto &[node_id, offset_bytes] : _node_id_to_offset)
        {
            const uint32_t *base_ptr = reinterpret_cast<const uint32_t *>(_cached_graph_buf + offset_bytes);
            const uint32_t degree = base_ptr[0];
            const uint32_t *adj_ptr = base_ptr + 1;
            callback(node_id, degree, adj_ptr);
        }
    }

    // Combined operations

    void clear_all()
    {
        clear_vector_cache();
        clear_graph_cache();
    }

    std::string get_cache_stats() const;

  private:
    static uint8_t *alloc_aligned_128(uint64_t size_bytes)
    {
        if (size_bytes == 0)
        {
            return nullptr;
        }

        void *ptr = nullptr;
        if (posix_memalign(&ptr, 128, static_cast<size_t>(size_bytes)) != 0)
        {
            throw std::bad_alloc();
        }
        return static_cast<uint8_t *>(ptr);
    }

    static void free_aligned(uint8_t *ptr)
    {
        if (ptr != nullptr)
        {
            std::free(ptr);
        }
    }

    static uint64_t checked_mul_u64(uint64_t a, uint64_t b)
    {
        if (a == 0 || b == 0)
        {
            return 0;
        }
        if (a > std::numeric_limits<uint64_t>::max() / b)
        {
            throw std::bad_alloc();
        }
        return a * b;
    }

    static uint64_t align_up_to_4(uint64_t dim)
    {
        return (dim + 3) & ~static_cast<uint64_t>(3);
    }

  private:
    T *_cached_vector_buf = nullptr;
    uint64_t _vector_dim_original = 0;
    uint64_t _vector_dim_aligned = 0;
    uint32_t _cached_vector_count = 0;
    uint32_t _cached_vector_capacity = 0;
    uint64_t _vector_cache_allocated_bytes = 0;
    uint64_t _vector_memory_budget_bytes = 0;

    tsl::robin_map<uint32_t, uint32_t> _vector_id_to_index;

    uint8_t *_cached_graph_buf = nullptr;

    uint32_t _cached_graph_node_count = 0;
    uint64_t _cached_graph_buf_size_bytes = 0;
    uint64_t _cached_graph_buf_used_bytes = 0;
    uint64_t _graph_memory_budget_bytes = 0;

    tsl::robin_map<uint32_t, uint64_t> _node_id_to_offset;
};

// Vector cache implementation

template <typename T> void DataCache<T>::reserve_vector_cache(uint64_t memory_budget_bytes, uint64_t dim)
{
    clear_vector_cache();

    _vector_memory_budget_bytes = memory_budget_bytes;
    _vector_dim_original = dim;
    _vector_dim_aligned = align_up_to_4(dim);

    if (memory_budget_bytes == 0 || dim == 0)
    {
        return;
    }

    const uint64_t bytes_per_vector = _vector_dim_aligned * sizeof(T);
    const uint32_t capacity = static_cast<uint32_t>(memory_budget_bytes / bytes_per_vector);

    if (capacity == 0)
    {
        std::cout << "Vector cache budget too small, cannot fit any vector." << std::endl;
        return;
    }

    _cached_vector_capacity = capacity;
    const uint64_t total_bytes = checked_mul_u64(static_cast<uint64_t>(capacity), bytes_per_vector);
    _vector_cache_allocated_bytes = total_bytes;

    _cached_vector_buf = reinterpret_cast<T *>(alloc_aligned_128(total_bytes));
    _vector_id_to_index.reserve(capacity);
}

template <typename T> void DataCache<T>::add_vector(uint32_t vector_id, const T *data)
{
    if (_cached_vector_buf == nullptr || _vector_dim_aligned == 0)
    {
        throw std::runtime_error("Vector cache buffer is not allocated.");
    }

    if (_vector_id_to_index.find(vector_id) != _vector_id_to_index.end())
    {
        throw std::runtime_error("Vector " + std::to_string(vector_id) + " already exists.");
    }

    if (_cached_vector_count >= _cached_vector_capacity)
    {
        throw std::runtime_error("Vector cache is full, cannot add more vectors.");
    }

    if (data == nullptr)
    {
        throw std::runtime_error("Vector data pointer is null.");
    }

    const uint32_t cache_index = _cached_vector_count;
    T *dst = _cached_vector_buf + static_cast<uint64_t>(cache_index) * _vector_dim_aligned;

    std::memcpy(dst, data, static_cast<size_t>(_vector_dim_original) * sizeof(T));

    if (_vector_dim_aligned > _vector_dim_original)
    {
        std::memset(dst + _vector_dim_original, 0,
                    static_cast<size_t>(_vector_dim_aligned - _vector_dim_original) * sizeof(T));
    }

    _vector_id_to_index.emplace(vector_id, cache_index);
    ++_cached_vector_count;
}

template <typename T> const T *DataCache<T>::get_cached_vector(uint32_t vector_id) const
{
    auto it = _vector_id_to_index.find(vector_id);
    if (it == _vector_id_to_index.end())
    {
        return nullptr;
    }
    return _cached_vector_buf + static_cast<uint64_t>(it->second) * _vector_dim_aligned;
}

template <typename T> void DataCache<T>::clear_vector_cache()
{
    free_aligned(reinterpret_cast<uint8_t *>(_cached_vector_buf));
    _cached_vector_buf = nullptr;

    _vector_dim_original = 0;
    _vector_dim_aligned = 0;
    _cached_vector_count = 0;
    _vector_cache_allocated_bytes = 0;
    _vector_memory_budget_bytes = 0;
    _vector_id_to_index.clear();
    _cached_vector_capacity = 0;
}

// Graph cache implementation

template <typename T> void DataCache<T>::reserve_graph_cache(uint64_t memory_budget_bytes)
{
    clear_graph_cache();

    _graph_memory_budget_bytes = memory_budget_bytes;
    _cached_graph_buf_size_bytes = memory_budget_bytes;

    if (memory_budget_bytes == 0)
    {
        std::cout << "Graph cache memory budget is 0, skip allocation." << std::endl;
        return;
    }

    _cached_graph_buf = alloc_aligned_128(memory_budget_bytes);
    _cached_graph_buf_used_bytes = 0;
    _cached_graph_node_count = 0;
}

template <typename T> void DataCache<T>::reserve_graph_node_map(uint32_t expected_node_count)
{
    _node_id_to_offset.reserve(expected_node_count);
}

template <typename T> void DataCache<T>::add_graph_node(uint32_t node_id, const uint32_t *neighbors, uint32_t degree)
{
    if (_cached_graph_buf == nullptr)
    {
        throw std::runtime_error("Graph cache buffer is not allocated.");
    }

    if (_node_id_to_offset.find(node_id) != _node_id_to_offset.end())
    {
        throw std::runtime_error("Graph node " + std::to_string(node_id) + " already exists.");
    }

    if (degree > 0 && neighbors == nullptr)
    {
        throw std::runtime_error("Graph node degree is > 0 but neighbors pointer is null.");
    }

    const uint64_t record_bytes =
        checked_mul_u64(static_cast<uint64_t>(1 + degree), static_cast<uint64_t>(sizeof(uint32_t)));

    if (_cached_graph_buf_size_bytes - _cached_graph_buf_used_bytes < record_bytes)
    {
        throw std::runtime_error("Graph cache has insufficient space for node " + std::to_string(node_id) +
                                 " (need " + std::to_string(record_bytes) + " bytes, have " +
                                 std::to_string(_cached_graph_buf_size_bytes - _cached_graph_buf_used_bytes) + ").");
    }

    const uint64_t offset_bytes = _cached_graph_buf_used_bytes;
    uint8_t *record_ptr = _cached_graph_buf + offset_bytes;

    uint32_t *degree_ptr = reinterpret_cast<uint32_t *>(record_ptr);
    *degree_ptr = degree;

    if (degree > 0)
    {
        std::memcpy(record_ptr + sizeof(uint32_t), neighbors, static_cast<size_t>(degree) * sizeof(uint32_t));
    }

    _node_id_to_offset.emplace(node_id, offset_bytes);

    _cached_graph_buf_used_bytes += record_bytes;
    ++_cached_graph_node_count;
}

template <typename T> std::pair<uint32_t, const uint32_t *> DataCache<T>::get_cached_graph_adj(uint32_t node_id) const
{
    auto it = _node_id_to_offset.find(node_id);
    if (it == _node_id_to_offset.end() || _cached_graph_buf == nullptr)
    {
        return {0, nullptr};
    }

    const uint64_t offset_bytes = it->second;
    const uint32_t *base_ptr = reinterpret_cast<const uint32_t *>(_cached_graph_buf + offset_bytes);
    return {base_ptr[0], base_ptr + 1};
}

template <typename T> void DataCache<T>::clear_graph_cache()
{
    free_aligned(_cached_graph_buf);
    _cached_graph_buf = nullptr;

    _cached_graph_node_count = 0;
    _cached_graph_buf_size_bytes = 0;
    _cached_graph_buf_used_bytes = 0;
    _graph_memory_budget_bytes = 0;
    _node_id_to_offset.clear();
}

// Stats

template <typename T> std::string DataCache<T>::get_cache_stats() const
{
    std::string stats;
    stats += "Vector Cache:\n";
    stats += "  Cached vectors: " + std::to_string(_cached_vector_count) + "\n";
    stats += "  Dimension (original): " + std::to_string(_vector_dim_original) + "\n";
    stats += "  Dimension (aligned): " + std::to_string(_vector_dim_aligned) + "\n";
    stats += "  Budget bytes: " + std::to_string(_vector_memory_budget_bytes) + "\n";
    stats += "  Allocated bytes: " + std::to_string(_vector_cache_allocated_bytes) + "\n";
    stats += "  Used bytes: " + std::to_string(get_vector_cache_used_bytes()) + "\n";

    stats += "Graph Cache:\n";
    stats += "  Cached nodes: " + std::to_string(_cached_graph_node_count) + "\n";
    stats += "  Budget bytes: " + std::to_string(get_graph_cache_budget_bytes()) + "\n";
    stats += "  Allocated bytes: " + std::to_string(get_graph_cache_allocated_bytes()) + "\n";
    stats += "  Used bytes: " + std::to_string(get_graph_cache_used_bytes()) + "\n";

    return stats;
}

} // namespace diskann
