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

/*! \file utils.h
 *   \brief Define CUDA utilities.
 */

#pragma once

#include <nvbio/basic/types.h>
#include <nvbio/basic/vector_view.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace nvbio {

template <typename T> struct device_view_subtype< thrust::device_vector<T> > { typedef vector_view<T*,uint64> type; };
template <typename T> struct plain_view_subtype< thrust::host_vector<T> >   { typedef vector_view<T*,uint64> type; };
template <typename T> struct plain_view_subtype< thrust::device_vector<T> > { typedef vector_view<T*,uint64> type; };

/// return the plain view of a device vector
///
template <typename T>
vector_view<T*,uint64> device_view(thrust::device_vector<T>& vec) { return vector_view<T*,uint64>( vec.size(), vec.size() ? thrust::raw_pointer_cast( &vec.front() ) : NULL ); }

/// return the plain view of a device vector
///
template <typename T>
vector_view<const T*,uint64> device_view(const thrust::device_vector<T>& vec) { return vector_view<const T*,uint64>( vec.size(), vec.size() ? thrust::raw_pointer_cast( &vec.front() ) : NULL ); }

/// return the plain view of a device vector
///
template <typename T>
vector_view<T*,uint64> plain_view(thrust::device_vector<T>& vec) { return vector_view<T*,uint64>( vec.size(), vec.size() ? thrust::raw_pointer_cast( &vec.front() ) : NULL ); }

/// return the plain view of a device vector
///
template <typename T>
vector_view<const T*,uint64> plain_view(const thrust::device_vector<T>& vec) { return vector_view<const T*,uint64>( vec.size(), vec.size() ? thrust::raw_pointer_cast( &vec.front() ) : NULL ); }

/// return the plain view of a device vector
///
template <typename T>
vector_view<T*,uint64> plain_view(thrust::host_vector<T>& vec) { return vector_view<T*,uint64>( vec.size(), vec.size() ? thrust::raw_pointer_cast( &vec.front() ) : NULL ); }

/// return the plain view of a device vector
///
template <typename T>
vector_view<const T*,uint64> plain_view(const thrust::host_vector<T>& vec) { return vector_view<const T*,uint64>( vec.size(), vec.size() ? thrust::raw_pointer_cast( &vec.front() ) : NULL ); }

/// return the raw pointer of a device vector
///
template <typename T>
T* raw_pointer(thrust::device_vector<T>& vec) { return vec.size() ? thrust::raw_pointer_cast( &vec.front() ) : NULL; }

/// return the raw pointer of a device vector
///
template <typename T>
const T* raw_pointer(const thrust::device_vector<T>& vec) { return vec.size() ? thrust::raw_pointer_cast( &vec.front() ) : NULL; }

/// return the raw pointer of a device vector
///
template <typename T>
T* raw_pointer(thrust::host_vector<T>& vec) { return vec.size() ? thrust::raw_pointer_cast( &vec.front() ) : NULL; }

/// return the plain view of a device vector
///
template <typename T>
const T* raw_pointer(const thrust::host_vector<T>& vec) { return vec.size(), vec.size() ? thrust::raw_pointer_cast( &vec.front() ) : NULL; }

} // namespace nvbio
