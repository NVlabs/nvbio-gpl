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
#include <sais.h>

namespace nvbio {

/// helper function to generate a suffix array padded by 1, where
/// the 0-th entry is the SA size.
///
template <typename StreamIterator>
uint32 gen_sa(const uint32 n, const StreamIterator T, int32 *SA)
{
  SA[0] = n;
  if (n <= 1) {
      if (n == 1) SA[1] = 0;
      return 0;
  }
  return saisxx( T, SA+1, int32(n), 4 );
}

/// helper function to generate the BWT of a string given its suffix array.
///
template <typename StreamIterator>
uint32 gen_bwt_from_sa(const uint32 n, const StreamIterator T, const int32* SA, StreamIterator bwt)
{
    uint32 i, primary = 0;

    for (i = 0; i <= n; ++i)
    {
        if (SA[i] == 0) primary = i;
        else bwt[i] = T[SA[i] - 1];
    }
    for (i = primary; i < n; ++i) bwt[i] = bwt[i + 1];
    return primary;
}

/// helper function to generate the BWT of a string given a temporary buffer.
///
template <typename StreamIterator>
int32 gen_bwt(const uint32 n, const StreamIterator T, int32* buffer, StreamIterator bwt)
{
    return saisxx_bwt( T, bwt, buffer, int32(n), 4 );
}

/// helper function to generate the BWT of a string given a temporary buffer.
///
template <typename StreamIterator>
int64 gen_bwt(const uint32 n, const StreamIterator T, int64* buffer, StreamIterator bwt)
{
    return saisxx_bwt( T, bwt, buffer, int64(n), int64(4) );
}

// generate table for counting 11,10,01,00(pattern) for 8 bits number
// table [no# ] = representation ( # of count-pattern, . , . , . )
// ---------------------------------------------------------------------------
// e.g cnt_table[11111111] = 0x04000000 ( 4-11, 0-10, 0-01, 0-00 )
// cnt_table[00100001] = 0x00010102 ( 0-11, 1-10, 1-01, 2-00 )
// cnt_table[00000001] = 0x00000103 ( 0-11, 0-10, 1-01, 3-00 )
inline void gen_bwt_count_table(uint32* count_table)
{
    for (int i = 0; i != 256; ++i)
    {
        uint32 x = 0;
        for (int j = 0; j != 4; ++j)
            x |= (((i&3) == j) + ((i>>2&3) == j) + ((i>>4&3) == j) + (i>>6 == j)) << (j<<3);

        count_table[i] = x;
    }
}

} // namespace nvbio

