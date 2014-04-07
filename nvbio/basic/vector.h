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

    /// constructor
    ///
    vector<host_tag,T>(const size_t size = 0, const T val = T()) : base_type( size, val ) {}
    vector<host_tag,T>(const thrust::host_vector<T>&   v) : base_type( v ) {}
    vector<host_tag,T>(const thrust::device_vector<T>& v) : base_type( v ) {}

    vector<host_tag,T>& operator= (const thrust::host_vector<T>& v)   { this->base_type::operator=(v); return *this; }
    vector<host_tag,T>& operator= (const thrust::device_vector<T>& v) { this->base_type::operator=(v); return *this; }
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

    /// constructor
    ///
    vector<device_tag,T>(const size_t size = 0, const T val = T()) : base_type( size, val ) {}
    vector<device_tag,T>(const thrust::host_vector<T>&   v) : base_type( v ) {}
    vector<device_tag,T>(const thrust::device_vector<T>& v) : base_type( v ) {}

    vector<device_tag,T>& operator= (const thrust::host_vector<T>& v)   { this->base_type::operator=(v); return *this; }
    vector<device_tag,T>& operator= (const thrust::device_vector<T>& v) { this->base_type::operator=(v); return *this; }
};

} // namespace nvbio
