// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstddef>
#include <mutex>
#include <vector>
#include "utils.h"

namespace diskann
{

struct Neighbor
{
    unsigned id;
    float distance;
    bool expanded;
#ifdef FAST_DISKANN
    bool loaded;
#endif

    Neighbor() = default;

#ifdef FAST_DISKANN
    Neighbor(unsigned id, float distance) : id{id}, distance{distance}, expanded(false), loaded(false)
#else
    Neighbor(unsigned id, float distance) : id{id}, distance{distance}, expanded(false)
#endif
    {
    }

    inline bool operator<(const Neighbor &other) const
    {
        return distance < other.distance || (distance == other.distance && id < other.id);
    }

    inline bool operator==(const Neighbor &other) const
    {
        return (id == other.id);
    }
};

#ifdef FAST_DISKANN
struct OriginNeighbor
{
    unsigned id;
    float distance;
    bool expanded;

    OriginNeighbor() = default;

    OriginNeighbor(unsigned id, float distance) : id{id}, distance{distance}, expanded(false)
    {
    }

    inline bool operator<(const OriginNeighbor &other) const
    {
        return distance < other.distance || (distance == other.distance && id < other.id);
    }

    inline bool operator==(const OriginNeighbor &other) const
    {
        return (id == other.id);
    }
};

struct SmallNeighbor
{
    unsigned id;
    float distance;

    SmallNeighbor() = default;

    SmallNeighbor(unsigned id, float distance) : id{id}, distance{distance}
    {
    }

    inline bool operator<(const SmallNeighbor &other) const
    {
        return distance < other.distance || (distance == other.distance && id < other.id);
    }

    inline bool operator==(const SmallNeighbor &other) const
    {
        return (id == other.id);
    }
};
#endif

// Invariant: after every `insert` and `closest_unexpanded()`, `_cur` points to
//            the first Neighbor which is unexpanded.
class NeighborPriorityQueue
{
  public:
#ifdef FAST_DISKANN
    NeighborPriorityQueue() : _size(0), _capacity(0), _cur(0), _loaded_cur(0), _loaded_size(0)
#else
    NeighborPriorityQueue() : _size(0), _capacity(0), _cur(0)
#endif
    {
    }

#ifdef FAST_DISKANN
    explicit NeighborPriorityQueue(size_t capacity)
        : _size(0), _capacity(capacity), _cur(0), _loaded_cur(0), _loaded_size(0), _data(capacity + 1)
#else
    explicit NeighborPriorityQueue(size_t capacity) : _size(0), _capacity(capacity), _cur(0), _data(capacity + 1)
#endif
    {
    }

    // Inserts the item ordered into the set up to the sets capacity.
    // The item will be dropped if it is the same id as an exiting
    // set item or it has a greated distance than the final
    // item in the set. The set cursor that is used to pop() the
    // next item will be set to the lowest index of an uncheck item
    void insert(const Neighbor &nbr)
    {
        if (_size == _capacity && _data[_size - 1] < nbr)
        {
            return;
        }

        size_t lo = 0, hi = _size;
        while (lo < hi)
        {
            size_t mid = (lo + hi) >> 1;
            if (nbr < _data[mid])
            {
                hi = mid;
                // Make sure the same id isn't inserted into the set
            }
            else if (_data[mid].id == nbr.id)
            {
                return;
            }
            else
            {
                lo = mid + 1;
            }
        }

        if (lo < _capacity)
        {
            std::memmove(&_data[lo + 1], &_data[lo], (_size - lo) * sizeof(Neighbor));
        }
        _data[lo] = {nbr.id, nbr.distance};
        if (_size < _capacity)
        {
            _size++;
        }
        if (lo < _cur)
        {
            _cur = lo;
        }
#ifdef FAST_DISKANN
        if (lo < _loaded_cur)
        {
            _loaded_cur = lo;
        }
#endif
    }

#ifdef FAST_DISKANN
    // Returns the actual insertion position.
    // If insertion fails (duplicate ID or full and worse than tail), returns _capacity.
    size_t insert_with_pos(const Neighbor &nbr)
    {
        // Case 1: list is full and the worst element is better than new one -> do nothing
        if (_size == _capacity && _data[_size - 1] < nbr)
            return _capacity;

        size_t lo = 0, hi = _size;
        while (lo < hi)
        {
            size_t mid = (lo + hi) >> 1;
            if (nbr < _data[mid])
            {
                hi = mid;
            }
            else if (_data[mid].id == nbr.id)
            {
                return _capacity; // duplicate, no insertion
            }
            else
            {
                lo = mid + 1;
            }
        }

        // Shift elements to make room for the new one
        if (lo < _capacity)
        {
            std::memmove(&_data[lo + 1], &_data[lo], (_size - lo) * sizeof(Neighbor));
        }

        _data[lo] = {nbr.id, nbr.distance};
        if (_size < _capacity)
            _size++;

        if (lo < _cur)
            _cur = lo;
        if (lo < _loaded_cur)
            _loaded_cur = lo;

        return lo; // return the position where the element was inserted
    }

    void set_loaded_size(size_t loaded_size)
    {
        _loaded_size = loaded_size;
    }
#endif

    Neighbor closest_unexpanded()
    {
        _data[_cur].expanded = true;
        size_t pre = _cur;
        while (_cur < _size && _data[_cur].expanded)
        {
            _cur++;
        }
        return _data[pre];
    }

#ifdef FAST_DISKANN
    Neighbor closest_unloaded()
    {
        _data[_loaded_cur].loaded = true;
        size_t pre = _loaded_cur;
        while (_loaded_cur < _size && _loaded_cur < _loaded_size && _data[_loaded_cur].loaded)
        {
            _loaded_cur++;
        }
        return _data[pre];
    }
#endif

    bool has_unexpanded_node() const
    {
        return _cur < _size;
    }

#ifdef FAST_DISKANN
    bool has_unloaded_node() const
    {
        return _loaded_cur < _loaded_size && _loaded_cur < _size;
    }
#endif

    size_t size() const
    {
        return _size;
    }

    size_t capacity() const
    {
        return _capacity;
    }

#ifdef FAST_DISKANN
    size_t loaded_cur() const
    {
        return _loaded_cur;
    }
#endif

    void reserve(size_t capacity)
    {
        if (capacity + 1 > _data.size())
        {
            _data.resize(capacity + 1);
        }
        _capacity = capacity;
    }

    Neighbor &operator[](size_t i)
    {
        return _data[i];
    }

    Neighbor operator[](size_t i) const
    {
        return _data[i];
    }

    void clear()
    {
        _size = 0;
        _cur = 0;
#ifdef FAST_DISKANN
        _loaded_cur = 0;
        _loaded_size = 0;
#endif
    }

  private:
#ifdef FAST_DISKANN
    size_t _size, _capacity, _cur, _loaded_cur, _loaded_size;
#else
    size_t _size, _capacity, _cur;
#endif
    std::vector<Neighbor> _data;
};

#ifdef FAST_DISKANN
class OriginNeighborPriorityQueue
{
  public:
    OriginNeighborPriorityQueue() : _size(0), _capacity(0), _cur(0)
    {
    }

    explicit OriginNeighborPriorityQueue(size_t capacity) : _size(0), _capacity(capacity), _cur(0), _data(capacity + 1)
    {
    }

    // Inserts the item ordered into the set up to the sets capacity.
    // The item will be dropped if it is the same id as an exiting
    // set item or it has a greated distance than the final
    // item in the set. The set cursor that is used to pop() the
    // next item will be set to the lowest index of an uncheck item
    void insert(const OriginNeighbor &nbr)
    {
        if (_size == _capacity && _data[_size - 1] < nbr)
        {
            return;
        }

        size_t lo = 0, hi = _size;
        while (lo < hi)
        {
            size_t mid = (lo + hi) >> 1;
            if (nbr < _data[mid])
            {
                hi = mid;
                // Make sure the same id isn't inserted into the set
            }
            else if (_data[mid].id == nbr.id)
            {
                return;
            }
            else
            {
                lo = mid + 1;
            }
        }

        if (lo < _capacity)
        {
            std::memmove(&_data[lo + 1], &_data[lo], (_size - lo) * sizeof(OriginNeighbor));
        }
        _data[lo] = {nbr.id, nbr.distance};
        if (_size < _capacity)
        {
            _size++;
        }
        if (lo < _cur)
        {
            _cur = lo;
        }
    }

    OriginNeighbor closest_unexpanded()
    {
        _data[_cur].expanded = true;
        size_t pre = _cur;
        while (_cur < _size && _data[_cur].expanded)
        {
            _cur++;
        }
        return _data[pre];
    }

    bool has_unexpanded_node() const
    {
        return _cur < _size;
    }

    size_t size() const
    {
        return _size;
    }

    size_t capacity() const
    {
        return _capacity;
    }

    void reserve(size_t capacity)
    {
        if (capacity + 1 > _data.size())
        {
            _data.resize(capacity + 1);
        }
        _capacity = capacity;
    }

    OriginNeighbor &operator[](size_t i)
    {
        return _data[i];
    }

    OriginNeighbor operator[](size_t i) const
    {
        return _data[i];
    }

    void clear()
    {
        _size = 0;
        _cur = 0;
    }

  private:
    size_t _size, _capacity, _cur;
    std::vector<OriginNeighbor> _data;
};
#endif

} // namespace diskann
