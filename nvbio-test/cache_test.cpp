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

// packedstream_test.cpp
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <nvbio/basic/types.h>
#include <nvbio/basic/cache.h>

using namespace nvbio;

struct CacheManager
{
    // constructor
    CacheManager() : m_size(0) {}

    // acquire element i
    bool acquire(const uint32 i)
    {
        if (m_size >= 100u)
            return false;

        ++m_size;
        return true;
    }

    // release element i
    void release(const uint32 i)
    {
        --m_size;
    }

    // is cache usage below the low-watermark?
    bool low_watermark() const
    {
        return (m_size < 75u);
    }

    uint32 m_size;
};

int cache_test()
{
    printf("cache test... started\n");
    CacheManager manager;

    LRU<CacheManager> cache( manager );
    for (uint32 i = 0; i < 1000; ++i)
    {
        cache.pin(i);
        cache.unpin(i);
    }

    printf("  test overflow... started\n");
    bool overflow = false;

    try
    {
        for (uint32 i = 0; i < 200; ++i)
            cache.pin(i);
    }
    catch (cache_overflow)
    {
        overflow = true;
    }
    if (overflow == false)
        printf("  error: overflow was expected, but did not occurr!\n");
    else
        printf("  test overflow... done\n");
    printf("cache test... done\n");
    return 0u;
}