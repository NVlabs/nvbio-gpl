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

#include <nvbio/basic/cached_iterator.h>
#include <nvbio/basic/packedstream.h>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

// helper class to convert {uint32,uint4} to uint32 streams
template <typename Iterator, typename value_type>
struct StreamAdapterBase {};

// helper class to convert uint32 to uint32 streams
template <typename Iterator>
struct StreamAdapterBase<Iterator,uint32>
{
    typedef const_cached_iterator<Iterator> type;

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    static type adapt(const Iterator it) { return type( it ); }
};

// helper class to convert uint4 to uint32 streams
template <typename Iterator>
struct StreamAdapterBase<Iterator,uint4>
{
    typedef const_cached_iterator<Iterator>                                         BaseCachedIterator;
    typedef const_cached_iterator< uint4_as_uint32_iterator<BaseCachedIterator> >   type;

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    static type adapt(const Iterator it) { return type( uint4_as_uint32_iterator<BaseCachedIterator>( BaseCachedIterator(it) ) ); }
};

// helper class to convert {uint32,uint4} to uint32 streams
template <typename Iterator>
struct StreamAdapter
{
    typedef typename std::iterator_traits<Iterator>::value_type                     value_type;
    typedef StreamAdapterBase<Iterator,value_type>                                  adapter_type;
    typedef typename adapter_type::type                                             type;

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    static type adapt(const Iterator it) { return adapter_type::adapt( it ); }
};

} // namespace cuda
} // namespace bowtie2

} // namespace nvbio
