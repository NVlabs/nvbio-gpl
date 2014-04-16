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

#include <nvbio/fmindex/fmindex.h>

namespace nvbio {

///@addtogroup FMIndex
///@{

/// forward extension using a bidirectional FM-index, extending the range
/// of a pattern P to the pattern Pc.
///\par
/// <b>Note:</b> extension can be performed without a sampled suffix array, so that
/// there's no need to store two of them; in practice, the FM-indices can
/// also be of type fm_index <RankDictionary,null_type>.
///
/// \param f_fmi    forward FM-index
/// \param r_fmi    reverse FM-index
/// \param f_range  current forward range
/// \param r_range  current reverse range
/// \param c        query character
///
template <
    typename TRankDictionary1,
    typename TSuffixArray1,
    typename TRankDictionary2,
    typename TSuffixArray2>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
void extend_forward(
    const fm_index<TRankDictionary1,TSuffixArray1>&                 f_fmi,
    const fm_index<TRankDictionary2,TSuffixArray2>&                 r_fmi,
    typename fm_index<TRankDictionary1,TSuffixArray1>::range_type&  f_range,
    typename fm_index<TRankDictionary2,TSuffixArray2>::range_type&  r_range,
    uint8                                                           c);

/// backwards extension using a bidirectional FM-index, extending the range
/// of a pattern P to the pattern cP
///\par
/// <b>Note:</b> extension can be performed without a sampled suffix array, so that
/// there's no need to store two of them; in practice, the FM-indices can
/// also be of type fm_index <RankDictionary,null_type>.
///
/// \param f_fmi    forward FM-index
/// \param r_fmi    reverse FM-index
/// \param f_range  current forward range
/// \param r_range  current reverse range
/// \param c        query character
///
template <
    typename TRankDictionary1,
    typename TSuffixArray1,
    typename TRankDictionary2,
    typename TSuffixArray2>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
void extend_backwards(
    const fm_index<TRankDictionary1,TSuffixArray1>&                 f_fmi,
    const fm_index<TRankDictionary2,TSuffixArray2>&                 r_fmi,
    typename fm_index<TRankDictionary1,TSuffixArray1>::range_type&  f_range,
    typename fm_index<TRankDictionary2,TSuffixArray2>::range_type&  r_range,
    uint8                                                           c);

///@} // end of the FMIndex group

} // namespace nvbio

#include <nvbio/fmindex/bidir_inl.h>
