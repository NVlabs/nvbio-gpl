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
#include <nvbio/fmindex/fmindex.h>

namespace nvbio {

///@addtogroup FMIndex
///@{

///
/// perform approximate matching using the Hamming distance by backtracking
///
/// \tparam String      a string iterator
/// \tparam Stack       an uint4 array to be used as a backtracking stack
/// \tparam Delegate    a delegate functor used to process hits, must implement the following interface:
///\code
/// struct Delegate
/// {
///     // process a series of hits identified by their SA range
///     void operator() (const uint2 range);
/// }
///\endcode
///
/// \param fmi          the FM-index
/// \param pattern      the search pattern
/// \param len          the pattern length
/// \param seed         the length of the seed to search with exact matching
/// \param mismatches   the maximum number of mismatches allowed
/// \param stack        the stack storage
/// \param delegate     the delegate functor invoked on hits
///
template <typename FMIndex, typename String, typename Stack, typename Delegate>
NVBIO_HOST_DEVICE
void hamming_backtrack(
    const FMIndex   fmi,
    const String    pattern,
    const uint32    len,
    const uint32    seed,
    const uint32    mismatches,
          Stack     stack,
          Delegate& delegate)
{
    if (mismatches == 0 || seed == len)
    {
        const uint2 range = match( fmi, pattern, len );

        if (range.x <= range.y)
            delegate( range );
    }
    else
    {
        uint2 root_range = match(
            fmi, pattern + len - seed, seed );

        // check if there is no seed match
        if (root_range.x > root_range.y)
            return;

        // compute the four children
        uint4 cnt_k, cnt_l;
        rank4( fmi, make_uint2( root_range.x-1, root_range.y ), &cnt_k, &cnt_l );

        // initialize the stack size
        uint32 sp = 0;

        const uint8 c_pattern = pattern[ len - seed - 1u ];

        for (uint32 c = 0; c < 4; ++c)
        {
            // check if the child node for character 'c' exists
            if (comp( cnt_k, c ) < comp( cnt_l, c ))
            {
                const uint8 cost = (c == c_pattern) ? 0u : 1u;

                const uint2 range_c = make_uint2(
                    fmi.L2(c) + comp( cnt_k, c ) + 1,
                    fmi.L2(c) + comp( cnt_l, c ) );

                stack[sp] = make_uint4( range_c.x, range_c.y, cost, len - seed - 1u );

                ++sp;
            }
        }

        while (sp)
        {
            // pop the stack
            --sp;

            const uint4  entry_info = stack[sp];
            const uint8  cost  = entry_info.z;
            const uint32 l     = entry_info.w;
                  uint2  range = make_uint2( entry_info.x, entry_info.y );

            // check if we reached the maximum extension length
            if (l == 0u)
            {
                if (range.x <= range.y)
                    delegate( range );
            }

            if (cost < mismatches)
            {
                // compute the four children
                uint4 cnt_k, cnt_l;
                rank4( fmi, make_uint2( range.x-1, range.y ), &cnt_k, &cnt_l );

                const uint8 c_pattern = pattern[l-1u];

                for (uint32 c = 0; c < 4; ++c)
                {
                    // check if the child node for character 'c' exists
                    if (comp( cnt_k, c ) < comp( cnt_l, c ))
                    {
                        const uint2 range_c = make_uint2(
                            fmi.L2(c) + comp( cnt_k, c ) + 1,
                            fmi.L2(c) + comp( cnt_l, c ) );

                        // compute the extension cost
                        const uint8 new_cost = cost + (c == c_pattern ? 0u : 1u);

                        // copy the band to the stack
                        stack[sp] = make_uint4( range_c.x, range_c.y, new_cost, l-1u );

                        ++sp;
                    }
                }
            }
            else
            {
                // perform exact matching of the rest of the string
                range = match( fmi, pattern, l, range );

                // update stats
                if (range.x <= range.y)
                    delegate( range );
            }
        }
    }
}

///@} FMIndex

} // namespace nvbio
