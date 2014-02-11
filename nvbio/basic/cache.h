/*
 * nvbio
 * Copyright (C) 2012-2014, NVIDIA Corporation
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * version 2 as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#pragma once

#include <nvbio/basic/types.h>
#include <map>
#include <vector>
#include <stack>

namespace nvbio {

///
/// LRU cache.
/// The template parameter CacheManager should supply the following interface:
///
/// bool acquire(const uint32 item);
///     try to acquire/load an element, returning false in case of failure:
///     the latter will trigger a cache release cycle.
///
/// void release(const uint32 item);
///     release an element, freeing any of the relative resources.
///
/// bool low_watermark() const
///     return true when the cache usage is below the low-watermark
///
struct cache_overflow {};

template <typename CacheManager>
struct LRU
{
    typedef CacheManager cache_manager_type;

    /// cache constructor
    ///
    LRU(CacheManager& manager);

    /// pin a given element, marking it as non-releasable
    ///
    void pin(const uint32 item);

    /// unpin a given element, marking it as releasable
    ///
    void unpin(const uint32 item);

private:
    struct List
    {
        List() {}
        List(const uint32 item, const uint32 next, const uint32 prev) :
            m_item( item ), m_next( next ), m_prev( prev ), m_pinned(true) {}

        uint32 m_item;
        uint32 m_next;
        uint32 m_prev;
        bool   m_pinned;
    };

    void touch(const uint32 list_idx, List& list);
    void release_cycle(const uint32 item);

    uint32 m_first;
    uint32 m_last;

    CacheManager*           m_manager;
    std::map<uint32,uint32> m_cache_map;
    std::vector<List>       m_cache_list;
    std::stack<uint32>      m_cache_pool;
};

} // namespace nvbio

#include <nvbio/basic/cache_inl.h>
