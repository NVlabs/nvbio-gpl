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

namespace nvbio {

///@addtogroup Basic
///@{

///@addtogroup BasicUtils Utilities
///@{

/// int32 popcount
///
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 popc(const int32 i);

/// uint32 popcount
///
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 popc(const uint32 i);

/// uint8 popcount
///
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 popc(const uint8 i);

/// find the n-th bit set in a 4-bit mask (n in [1,4])
///
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 find_nthbit4(const uint32 mask, const uint32 n);

/// compute the pop-count of 4-bit mask
///
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 popc4(const uint32 mask);

/// find the n-th bit set in a 8-bit mask (n in [1,8])
///
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 find_nthbit8(const uint32 mask, const uint32 n);

/// find the least significant bit set
///
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 ffs(const int32 x);


/// count the number of leading zeros
///
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 lzc(const uint32 x);


/// count the number of occurrences of a given 2-bit pattern in a given word
///
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 popc_2bit(const uint32 x, int c);

/// count the number of occurrences of all 2-bit patterns in a given word,
/// using an auxiliary table.
///
/// \param count_table      auxiliary table to perform the parallel bit-counting
///                         for all integers in the range [0,255].
///
/// \return                 the 4 pop counts shifted and OR'ed together
///
template <typename CountTable>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 popc_2bit_all(
    const uint32 b,
    const CountTable count_table);

/// given a 32-bit word encoding a set of 2-bit symbols, return a submask containing
/// all but the first 'i' entries.
///
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 hibits_2bit(const uint32 mask, const uint32 i);

/// count the number of occurrences of a given 2-bit pattern in all but the first 'i' symbols
/// of a 32-bit word mask.
///
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 popc_2bit(const uint32 mask, int c, const uint32 i);


/// count the number of occurrences of all 2-bit patterns in all but the first 'i' symbols
/// of a given word, using an auxiliary table.
///
/// \param count_table      auxiliary table to perform the parallel bit-counting
///                         for all integers in the range [0,255].
///
/// \return                 the 4 pop counts shifted and OR'ed together
///
template <typename CountTable>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 popc_2bit_all(
    const uint32     mask,
    const CountTable count_table,
    const uint32     i);

// generate table for counting 11,10,01,00(pattern) for 8 bits number
// table [no# ] = representation ( # of count-pattern, . , . , . )
// ---------------------------------------------------------------------------
// e.g cnt_table[11111111] = 0x04000000 ( 4-11, 0-10, 0-01, 0-00 )
// cnt_table[00100001] = 0x00010102 ( 0-11, 1-10, 1-01, 2-00 )
// cnt_table[00000001] = 0x00000103 ( 0-11, 0-10, 1-01, 3-00 )
inline void gen_2bit_count_table(uint32* count_table)
{
    for (int i = 0; i != 256; ++i)
    {
        uint32 x = 0;
        for (int j = 0; j != 4; ++j)
            x |= (((i&3) == j) + ((i>>2&3) == j) + ((i>>4&3) == j) + (i>>6 == j)) << (j<<3);

        count_table[i] = x;
    }
}

///@} BasicUtils
///@} Basic

} // namespace nvbio

#include <nvbio/basic/popcount_inl.h>
