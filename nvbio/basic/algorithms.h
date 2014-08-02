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

/*! \file algorithms.h
 *   \brief Defines some general purpose algorithms.
 */

#pragma once

#include <nvbio/basic/types.h>

namespace nvbio {

/// \page algorithms_page Algorithms
///
/// NVBIO provides a few basic algorithms which can be called either from the host or the device:
///
/// - find_pivot()
/// - lower_bound()
/// - upper_bound()
/// - merge()
/// - merge_sort()
///

/// find the first element in a sequence for which a given predicate evaluates to true
///
/// \param begin        sequence start iterator
/// \param n            sequence size
/// \param predicate    unary predicate
template <typename Iterator, typename Predicate>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE Iterator find_pivot(
    Iterator        begin,
    const uint32    n,
    const Predicate predicate)
{
    // if the range has a size of zero, let's just return the intial element
    if (n == 0)
        return begin;

    // check whether this segment contains only 0s or only 1s
    if (predicate( begin[0] ) == predicate( begin[n-1] ))
        return predicate( begin[0] ) ? begin + n : begin;

    // perform a binary search over the given range
    uint32 count = n;

    while (count > 0)
    {
        const uint32 count2 = count / 2;

        Iterator mid = begin + count2;

        if (predicate( *mid ) == false)
            begin = ++mid, count -= count2 + 1;
        else
            count = count2;
    }
	return begin;
}

/// find the lower bound in a sequence
///
/// \param x        element to find
/// \param begin    sequence start iterator
/// \param n        sequence size
template <typename Iterator, typename Value, typename index_type>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE Iterator lower_bound(
    const Value         x,
    Iterator            begin,
    const index_type    n)
{
    // if the range has a size of zero, let's just return the intial element
    if (n == 0)
        return begin;

    // check whether this segment is all left or right of x
    if (x < begin[0])
        return begin;

    if (begin[n-1] < x)
        return begin + n;

    // perform a binary search over the given range
    index_type count = n;

    while (count > 0)
    {
        const index_type count2 = count / 2;

        Iterator mid = begin + count2;

        if (*mid < x)
            begin = ++mid, count -= count2 + 1;
        else
            count = count2;
    }
	return begin;
}

/// find the upper bound in a sequence
///
/// \param x        element to find
/// \param begin    sequence start iterator
/// \param n        sequence size
template <typename Iterator, typename Value, typename index_type>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE Iterator upper_bound(
    const Value         x,
    Iterator            begin,
    const index_type    n)
{
    index_type count = n;
 
    while (count > 0)
    {
        const index_type step = count / 2;

        Iterator it = begin + step;

        if (!(x < *it))
        {
            begin = ++it;
            count -= step + 1;
        }
        else
            count = step;
    }
    return begin;
}

/// find the lower bound in a sequence
///
/// \param x        element to find
/// \param begin    sequence start iterator
/// \param n        sequence size
template <typename Iterator, typename Value, typename index_type>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE index_type lower_bound_index(
    const Value         x,
    Iterator            begin,
    const index_type    n)
{
    return uint32( lower_bound( x, begin, n ) - begin );
}

/// find the upper bound in a sequence
///
/// \param x        element to find
/// \param begin    sequence start iterator
/// \param n        sequence size
template <typename Iterator, typename Value, typename index_type>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE index_type upper_bound_index(
    const Value         x,
    Iterator            begin,
    const index_type    n)
{
    return index_type( upper_bound( x, begin, n ) - begin );
}

} // namespace nvbio
