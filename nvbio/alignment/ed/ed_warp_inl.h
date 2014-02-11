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

#include <nvbio/alignment/sw/sw_warp_inl.h>


namespace nvbio {
namespace aln {
namespace priv {


// private dispatcher for the warp-parallel version of classic smith-waterman
template <
    uint32          BLOCKDIM,
    AlignmentType   TYPE,
    typename        pattern_string,
    typename        qual_string,
    typename        text_string,
    typename        column_type>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
int32 alignment_score(
    const EditDistanceAligner<TYPE>     aligner,
    const pattern_string                pattern,
    const qual_string                   quals,
    const text_string                   text,
    const  int32                        min_score,
          uint2*                        sink,
          column_type                   column)
{
#if defined(NVBIO_DEVICE_COMPILATION)
    return sw_alignment_score<BLOCKDIM,TYPE>(
        EditDistanceSWScheme(),
        pattern,
        quals,
        text,
        min_score,
        sink,
        column );
#else
    return Field_traits<int32>::min();
#endif
}

} // namespace priv
} // namespace aln
} // namespace nvbio
