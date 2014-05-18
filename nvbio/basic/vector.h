/*
 * nvbio
 * Copyright (C) 2011-2014, NVIDIA Corporation
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

/*! \file vector.h
 *   \brief Define host / device vectors
 */

#pragma once

#include <nvbio/basic/types.h>
#include <nvbio/basic/iterator.h>
#include <nvbio/basic/thrust_view.h>
#include <nvbio/basic/vector_view.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace nvbio {

/// a dynamic host/device vector class
///
template <typename system_tag, typename T>
struct vector {};

/// a dynamic host vector class
///
template <typename T>
struct vector<host_tag,T> : public thrust::host_vector<T>
{
    typedef host_tag                            system_tag;

    typedef thrust::host_vector<T>              base_type;
    typedef typename base_type::const_iterator  const_iterator;
    typedef typename base_type::iterator        iterator;
    typedef typename base_type::value_type      value_type;

    typedef nvbio::vector_view<T*,uint64>              plain_view_type;
    typedef nvbio::vector_view<const T*,uint64>  const_plain_view_type;

    /// constructor
    ///
    vector<host_tag,T>(const size_t size = 0, const T val = T()) : base_type( size, val ) {}
    vector<host_tag,T>(const thrust::host_vector<T>&   v) : base_type( v ) {}
    vector<host_tag,T>(const thrust::device_vector<T>& v) : base_type( v ) {}

    vector<host_tag,T>& operator= (const thrust::host_vector<T>& v)   { this->base_type::operator=(v); return *this; }
    vector<host_tag,T>& operator= (const thrust::device_vector<T>& v) { this->base_type::operator=(v); return *this; }

    /// conversion to plain_view_type
    ///
    operator plain_view_type() { return plain_view_type( base_type::size(), nvbio::raw_pointer( *this ) ); }

    /// conversion to const_plain_view_type
    ///
    operator const_plain_view_type() const { return const_plain_view_type( base_type::size(), nvbio::raw_pointer( *this ) ); }
};

/// a dynamic device vector class
///
template <typename T>
struct vector<device_tag,T> : public thrust::device_vector<T>
{
    typedef device_tag                          system_tag;

    typedef thrust::device_vector<T>            base_type;
    typedef typename base_type::const_iterator  const_iterator;
    typedef typename base_type::iterator        iterator;
    typedef typename base_type::value_type      value_type;

    typedef nvbio::vector_view<T*,uint64>              plain_view_type;
    typedef nvbio::vector_view<const T*,uint64>  const_plain_view_type;

    /// constructor
    ///
    vector<device_tag,T>(const size_t size = 0, const T val = T()) : base_type( size, val ) {}
    vector<device_tag,T>(const thrust::host_vector<T>&   v) : base_type( v ) {}
    vector<device_tag,T>(const thrust::device_vector<T>& v) : base_type( v ) {}

    vector<device_tag,T>& operator= (const thrust::host_vector<T>& v)   { this->base_type::operator=(v); return *this; }
    vector<device_tag,T>& operator= (const thrust::device_vector<T>& v) { this->base_type::operator=(v); return *this; }

    /// conversion to plain_view_type
    ///
    operator plain_view_type() { return plain_view_type( base_type::size(), nvbio::raw_pointer( *this ) ); }

    /// conversion to const_plain_view_type
    ///
    operator const_plain_view_type() const { return const_plain_view_type( base_type::size(), nvbio::raw_pointer( *this ) ); }
};

/// a utility meta-type to wrap naked device pointers as thrust::device_ptr
///
template <typename T>   struct device_iterator_type             { typedef T type; };
template <typename T>   struct device_iterator_type<T*>         { typedef thrust::device_ptr<T> type; };
template <typename T>   struct device_iterator_type<const T*>   { typedef thrust::device_ptr<const T> type; };

/// a convenience function to wrap naked device pointers as thrust::device_ptr
///
template <typename T>
typename device_iterator_type<T>::type device_iterator(const T it)
{
    // wrap the plain iterator
    return typename device_iterator_type<T>::type( it );
}

} // namespace nvbio
