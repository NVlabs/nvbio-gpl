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

/*! \file cached_iterator.h
 *   \brief CUDA-compatible iterator wrappers allowing to cache the dereferenced
 *   value of generic iterators
 */

#pragma once

#include <iterator>
#include <nvbio/basic/types.h>
#include <thrust/iterator/iterator_categories.h>

#if defined(__CUDACC__)

namespace std
{

// extend the std::iterator_traits with support for CUDA's __restrict__ pointers
template<class _Ty>
struct iterator_traits<const _Ty * __restrict__>
{	// get traits from const pointer
    typedef random_access_iterator_tag  iterator_category;
    typedef _Ty                         value_type;
    typedef ptrdiff_t                   difference_type;
    //typedef ptrdiff_t                   distance_type;	// retained
    typedef const _Ty* __restrict__     pointer;
    typedef const _Ty&                  reference;
};

} // namespace std

#endif // __CUDACC__

namespace nvbio {

typedef std::input_iterator_tag                     input_host_iterator_tag;
typedef std::output_iterator_tag                    output_host_iterator_tag;
typedef std::forward_iterator_tag                   forward_host_iterator_tag;
typedef std::bidirectional_iterator_tag             bidirectional_host_iterator_tag;
typedef std::random_access_iterator_tag             random_access_host_iterator_tag;

typedef thrust::input_device_iterator_tag           input_device_iterator_tag;
typedef thrust::output_device_iterator_tag          output_device_iterator_tag;
typedef thrust::forward_device_iterator_tag         forward_device_iterator_tag;
typedef thrust::bidirectional_device_iterator_tag   bidirectional_device_iterator_tag;
typedef thrust::random_access_device_iterator_tag   random_access_device_iterator_tag;

template <typename iterator_category> struct iterator_category_system {};
template <>                           struct iterator_category_system<input_host_iterator_tag>              { typedef host_tag   type; };
template <>                           struct iterator_category_system<output_host_iterator_tag>             { typedef host_tag   type; };
template <>                           struct iterator_category_system<forward_host_iterator_tag>            { typedef host_tag   type; };
template <>                           struct iterator_category_system<bidirectional_host_iterator_tag>      { typedef host_tag   type; };
template <>                           struct iterator_category_system<random_access_host_iterator_tag>      { typedef host_tag   type; };
template <>                           struct iterator_category_system<input_device_iterator_tag>            { typedef device_tag type; };
template <>                           struct iterator_category_system<output_device_iterator_tag>           { typedef device_tag type; };
template <>                           struct iterator_category_system<forward_device_iterator_tag>          { typedef device_tag type; };
template <>                           struct iterator_category_system<bidirectional_device_iterator_tag>    { typedef device_tag type; };
template <>                           struct iterator_category_system<random_access_device_iterator_tag>    { typedef device_tag type; };

template <typename iterator>
struct iterator_system
{
    typedef typename std::iterator_traits<iterator>::iterator_category  iterator_category;
    typedef typename iterator_category_system<iterator_category>::type  type;
};

} // namespace nvbio
