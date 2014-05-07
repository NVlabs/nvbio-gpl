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

#include <nvBowtie/bowtie2/cuda/string_utils.h>
#include <nvbio/alignment/alignment.h>
#include <nvbio/alignment/batched.h>
#include <nvbio/io/utils.h>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

template <typename ScoringScheme> struct BaseScoringPipelineState;
template <typename ScoringScheme> struct BestApproxScoringPipelineState;

///@addtogroup nvBowtie
///@{

/// \defgroup Scoring
///
/// The functions in this module implement a pipeline stage in which all the seed hits currently
/// in the \ref ScoringQueues get "extended" and scored using DP alignment against the reference genome.
///
/// \b inputs:
///  - HitQueues::seed
///  - HitQueues::loc
///
/// \b outputs:
///  - HitQueues::score
///  - HitQueues::sink
///  - HitQueues::opposite_score
///  - HitQueues::opposite_loc
///  - HitQueues::opposite_sink
///

///@addtogroup Scoring
///@{

///
/// execute a batch of single-ended banded-alignment score calculations, best mapping
///
/// \b inputs:
///  - HitQueues::seed
///  - HitQueues::loc
///
/// \b outputs:
///  - HitQueues::score
///  - HitQueues::sink
///
template <typename scheme_type>
void score_best(
    const uint32                                        band_len,
    const BestApproxScoringPipelineState<scheme_type>&  pipeline,
    const ParamsPOD                                     params);

///
/// execute a batch of single-ended banded-alignment score calculations, all-mapping
///
/// \b inputs:
///  - HitQueues::seed
///  - HitQueues::loc
///
/// \b outputs:
///  - HitQueues::score
///  - HitQueues::sink
///
/// \param band_len             alignment band length
/// \param pipeline             all mapping pipeline
/// \param params               alignment params
/// \param buffer_offset        ring buffer offset
/// \param buffer_size          ring buffer size
/// \return                     number of valid alignments
template <typename scheme_type>
uint32 score_all(
    const uint32                                        band_len,
    const AllMappingPipelineState<scheme_type>&         pipeline,
    const ParamsPOD                                     params,
    const uint32                                        buffer_offset,
    const uint32                                        buffer_size);


///
/// execute a batch of banded-alignment score calculations for the anchor mates, best mapping
///
/// \b inputs:
///  - HitQueues::seed
///  - HitQueues::loc
///
/// \b outputs:
///  - HitQueues::score
///  - HitQueues::sink
///
template <typename scheme_type>
void anchor_score_best(
    const uint32                                        band_len,
    const BestApproxScoringPipelineState<scheme_type>&  pipeline,
    const ParamsPOD                                     params);

///
/// execute a batch of full-DP alignment score calculations for the opposite mates, best mapping
///
/// \b inputs:
///  - HitQueues::seed
///  - HitQueues::loc
///  - HitQueues::score
///  - HitQueues::sink
///
/// \b outputs:
///  - HitQueues::opposite_score
///  - HitQueues::opposite_loc
///  - HitQueues::opposite_sink
///
template <typename scheme_type>
void opposite_score_best(
    const BestApproxScoringPipelineState<scheme_type>&  pipeline,
    const ParamsPOD                                     params);

///@}  // group Scoring
///@}  // group nvBowtie

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio

#include <nvBowtie/bowtie2/cuda/score_inl.h>
