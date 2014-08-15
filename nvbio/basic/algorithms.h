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
    // if the range has a size of zero, let's just return the intial element
    //if (n == 0)
    //    return begin;

    // check whether this segment is all left or right of x
    //if (x < begin[0])
    //    return begin;

    //if (begin[n-1] <= x)
    //    return begin + n;
    
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

/// merge two ranges
///
/// \param first1   beginning of the first range
/// \param end1     end of the first range
/// \param first2   beginning of the second range
/// \param end2     end of the second range
/// \param output   beginning of the output range
///
template <
    typename input_iterator1,
    typename input_iterator2,
    typename output_iterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE 
void merge(
    input_iterator1    first1,
    input_iterator1    end1,
    input_iterator2    first2,
    input_iterator2    end2,
    output_iterator    output)
{
    while (first1 != end1 &&
           first2 != end2)
    {
        if (*first2 < *first1)
        {
            *output = *first2;

            ++first2;
        }
        else
        {
            *output = *first1;

            ++first1;
        }

        ++output;
    }

    while (first1 != end1)
    {
        *output = *first1;

        ++first1;
        ++output;
    }

    while (first2 != end2)
    {
        *output = *first2;

        ++first2;
        ++output;
    }
}

/// merge two ranges
///
/// \param first1   beginning of the first range
/// \param end1     end of the first range
/// \param first2   beginning of the second range
/// \param end2     end of the second range
/// \param output   beginning of the output range
///
template <
    typename key_iterator1,
    typename key_iterator2,
    typename value_iterator1,
    typename value_iterator2,
    typename key_iterator,
    typename value_iterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE 
void merge_by_key(
    key_iterator1      first1,
    key_iterator1      end1,
    key_iterator2      first2,
    key_iterator2      end2,
    value_iterator1    values1,
    value_iterator2    values2,
    key_iterator       output_keys,
    value_iterator     output_values)
{
    while (first1 != end1 &&
           first2 != end2)
    {
        if (*first2 < *first1)
        {
            *output_keys   = *first2;
            *output_values = *values2;

            ++first2;
            ++values2;
        }
        else
        {
            *output_keys   = *first1;
            *output_values = *values1;

            ++first1;
            ++values1;
        }

        ++output_keys;
        ++output_values;
    }

    while (first1 != end1)
    {
        *output_keys   = *first1;
        *output_values = *values1;

        ++first1;
        ++values1;
        ++output_keys;
        ++output_values;
    }

    while (first2 != end2)
    {
        *output_keys   = *first2;
        *output_values = *values2;

        ++first2;
        ++values2;
        ++output_keys;
        ++output_values;
    }
}

} // namespace nvbio
