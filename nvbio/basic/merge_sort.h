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

#include <nvbio/basic/types.h>
#include <iterator>

namespace nvbio {

namespace mergesort {

/// merge two contiguous sorted sequences into a single sorted output sequence.
/// NOTE: the output buffer should not overlap the input buffer.
///
/// \param A            input buffer
/// \param int_left     beginning of the first interval
/// \param int_right    beginning of the second interval
/// \param int_end      end of the second interval
/// \param B            output buffer
///
template <typename Iterator, typename Compare>
NVBIO_HOST_DEVICE
void merge(Iterator A, uint32 int_left, uint32 int_right, uint32 int_end, Iterator B, const Compare cmp)
{
    uint32 i0 = int_left;
    uint32 i1 = int_right;

    // while there are elements in the left or right lists
    for (uint32 j = int_left; j < int_end; ++j)
    {
        // if left list head exists and is <= existing right list head
        if (i0 < int_right && (i1 >= int_end || (!(cmp( A[i1], A[i0] )))))
        {
            B[j] = A[i0];
            i0 = i0 + 1;
        }
        else
        {
            B[j] = A[i1];
            i1 = i1 + 1;
        }
    }
}

} // namespace merge_sort

///
/// Merge sort
///
/// \param n        number of entries
/// \param A        array to sort
/// \param B        temporary buffer
///
/// \return         true if the results are in B, false otherwise
///
template <typename Iterator, typename Compare>
NVBIO_HOST_DEVICE
bool merge_sort(uint32 n, Iterator A, Iterator B, const Compare cmp)
{
    if (n == 1)
        return false;

    // each 1-element run in A is already "sorted":
    // make successively longer sorted runs of length 2, 4, 8, 16...
    // until whole array is sorted

    // sort pairs in place to avoid unnecessary memory traffic
    const uint32 nn = (n & 1) ? n-1 : n;
    for (uint32 i = 0; i < nn; i += 2u)
    {
        if (cmp( A[i+1], A[i] ))
        {
            typename std::iterator_traits<Iterator>::value_type tmp = A[i];
            A[i]   = A[i+1];
            A[i+1] = tmp;
        }
    }

    // merge longer and longer runs in a loop
    bool swap = 0;
    for (uint32 width = 2u; width < n; width *= 2u, swap = !swap)
    {
        // array A is full of runs of length width
        for (uint32 i = 0; i < n; i += 2u * width)
        {
            // merge two runs: A[i:i+width-1] and A[i+width:i+2*width-1] to B[]
            //  or copy A[i:n-1] to B[] ( if(i+width >= n) )
            mergesort::merge( A, i, nvbio::min(i+width, n), nvbio::min(i+2*width, n), B, cmp );
        }

        // now work array B is full of runs of length 2*width, swap A and B.
        Iterator tmp = A;
        A = B;
        B = tmp;
    }
    return swap;
}

} // namespace nvbio
