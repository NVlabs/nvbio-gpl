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

// A sink for valid alignments, mantaining only a single best alignment
//
template <typename ScoreType>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
BestSink<ScoreType>::BestSink() : score( Field_traits<ScoreType>::min() ), sink( make_uint2( uint32(-1), uint32(-1) ) ) {}

// store a valid alignment
//
// \param score    alignment's score
// \param sink     alignment's end
//
template <typename ScoreType>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
void BestSink<ScoreType>::report(const ScoreType _score, const uint2 _sink)
{
    // NOTE: we must use <= here because otherwise we won't pick the bottom-right most one
    // in case there's multiple optimal scores
    if (score <= _score)
    {
        score = _score;
        sink  = _sink;
    }
}

// A sink for valid alignments, mantaining the best two alignments
//
template <typename ScoreType>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Best2Sink<ScoreType>::Best2Sink(const uint32 distinct_dist) :
    score1( Field_traits<ScoreType>::min() ),
    score2( Field_traits<ScoreType>::min() ),
    sink1( make_uint2( uint32(-1), uint32(-1) ) ),
    sink2( make_uint2( uint32(-1), uint32(-1) ) ),
    m_distinct_dist( distinct_dist ) {}

// store a valid alignment
//
// \param score    alignment's score
// \param sink     alignment's end
//
template <typename ScoreType>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
void Best2Sink<ScoreType>::report(const ScoreType score, const uint2 sink)
{
    // NOTE: we must use <= here because otherwise we won't pick the bottom-right most one
    // in case there's multiple optimal scores
    if (score1 <= score)
    {
        score1 = score;
        sink1  = sink;
    }
    else if (score2 <= score && (sink.x + m_distinct_dist < sink1.x || sink.x > sink1.x + m_distinct_dist))
    {
        score2 = score;
        sink2  = sink;
    }
}

} // namespace aln
} // namespace nvbio
