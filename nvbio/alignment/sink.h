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
#include <nvbio/basic/simd.h>

namespace nvbio {
namespace aln {

///
///@addtogroup Alignment
///@{
///

///
///@addtogroup AlignmentSink Alignment Sinks
/// An alignment sink is an object passed to alignment functions to handle
/// the terminal (cell,score)- pairs of all valid alignments.
/// A particular sink might decide to discard or store all such alignments, while
/// another might decide to store only the best, or the best N, and so on.
///@{
///

///
/// A no-op sink for valid alignments
///
struct NullSink
{
    /// store a valid alignment
    ///
    /// \param _score    alignment's score
    /// \param _sink     alignment's end
    ///
    template <typename ScoreType>
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void report(const ScoreType _score, const uint2 _sink) {}
};

///
/// A sink for valid alignments, mantaining only a single best alignment
///
template <typename ScoreType>
struct BestSink
{
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    BestSink();

    /// store a valid alignment
    ///
    /// \param _score    alignment's score
    /// \param _sink     alignment's end
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void report(const ScoreType _score, const uint2 _sink);

    ScoreType score;
    uint2     sink;
};

///
/// A sink for valid alignments, mantaining only a single best alignment
///
template <>
struct BestSink<simd4u8>
{
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    BestSink() : score( uint32(0) ) {}

    /// store a valid alignment
    ///
    /// \param _score    alignment's score
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void report(const simd4u8 _score) { score = nvbio::max( score, _score ); }

    simd4u8 score;
};

///
/// A sink for valid alignments, mantaining the best two alignments
///
template <typename ScoreType>
struct Best2Sink
{
    /// constructor
    ///
    /// \param distinct_distance   the minimum text distance to consider two alignments distinct
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    Best2Sink(const uint32 distinct_dist = 0);

    /// store a valid alignment
    ///
    /// \param score    alignment's score
    /// \param sink     alignment's end
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void report(const ScoreType score, const uint2 sink);

    ScoreType score1;
    ScoreType score2;
    uint2     sink1;
    uint2     sink2;

private:
    uint32    m_distinct_dist;
};

///@} // end of the AlignmentSink group

///@} // end Alignment group

} // namespace aln
} // namespace nvbio

#include <nvbio/alignment/sink_inl.h>
