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

#pragma once

#include <nvbio/basic/transform_iterator.h>
#include <nvbio/basic/numbers.h> // for cast_functor

namespace nvbio {

///@addtogroup Basic
///@{

///@addtogroup Iterators
///@{

/// make a cast_iterator
///
template <typename R, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
transform_iterator<T,cast_functor<typename std::iterator_traits<T>::value_type,R> > make_cast_iterator(const T it)
{
    return nvbio::make_transform_iterator( it, cast_functor<typename std::iterator_traits<T>::value_type,R>() );
}

///@} Iterators
///@} Basic

} // namespace nvbio
