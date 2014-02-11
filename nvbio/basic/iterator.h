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
