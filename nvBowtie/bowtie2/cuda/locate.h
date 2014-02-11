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

///
///\file locate.h
///

#pragma once

#include <nvBowtie/bowtie2/cuda/defs.h>
#include <nvBowtie/bowtie2/cuda/params.h>
#include <nvbio/io/alignments.h>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

template <typename ScoringScheme> struct BaseScoringPipelineState;
template <typename ScoringScheme> struct BestApproxScoringPipelineState;

///@addtogroup nvBowtie
///@{

/// \defgroup Locate
///
/// The functions in this module implement a pipeline stage in which all the previously selected hits (\ref Select)
/// locations are converted from Suffix Array coordinates to linear genome coordinates.
/// In terms of inputs and outputs, this stage takes the HitQueues::seed and HitQueues::loc fields, and rewrite
/// the HitQueues::loc field with the linear value.
///
/// \b inputs:
///  - HitQueues::seed
///  - HitQueues::loc
///
/// \b outputs:
///  - HitQueues::loc
///

///@addtogroup Locate
///@{

///
/// Locate the SA row of the the hits in the HitQueues.
/// Since the input might have been sorted to gather locality, the entries
/// in the HitQueues are now specified by an index (idx_queue).
/// This function reads HitQueues::seed and HitQueues::loc fields and rewrites
/// the HitQueues::loc field.
///
template <typename BatchType, typename FMType, typename rFMType>
void locate(
    const BatchType             read_batch, const FMType fmi, const rFMType rfmi,
    const uint32                in_count,
    const uint32*               idx_queue,
          HitQueuesDeviceView   hits,
    const ParamsPOD             params);

///
/// Locate the SA row of the the hits in the HitQueues.
/// Since the input might have been sorted to gather locality, the entries
/// in the HitQueues are now specified by an index (idx_queue).
/// This function reads HitQueues::seed and HitQueues::loc fields and writes
/// the HitQueues::loc and HitQueues::ssa fields with temporary values that
/// can be later consumed by locate_lookup.
///
template <typename BatchType, typename FMType, typename rFMType>
void locate_init(
    const BatchType             read_batch, const FMType fmi, const rFMType rfmi,
    const uint32                in_count,
    const uint32*               idx_queue,
          HitQueuesDeviceView   hits,
    const ParamsPOD             params);

///
/// Locate the SA row of the the hits in the HitQueues.
/// Since the input might have been sorted to gather locality, the entries
/// in the HitQueues are now specified by an index (idx_queue).
/// This function reads HitQueues::seed, HitQueues::loc and HitQueues::ssa fields
/// (which must have been previously produced by a call to locate_init) and rewrites
/// the HitQueues::loc field with the final linear coordinate value.
///
template <typename BatchType, typename FMType, typename rFMType>
void locate_lookup(
    const BatchType             read_batch, const FMType fmi, const rFMType rfmi,
    const uint32                in_count,
    const uint32*               idx_queue,
          HitQueuesDeviceView   hits,
    const ParamsPOD             params);

///
/// Locate the SA row of the the hits in the HitQueues.
/// This function reads HitQueues::seed and HitQueues::loc fields and writes
/// the HitQueues::loc and HitQueues::ssa fields with temporary values that
/// can be later consumed by locate_lookup.
///
template <typename ScoringScheme>
void locate_init(
    const BaseScoringPipelineState<ScoringScheme>&  pipeline,
    const ParamsPOD                                 params);

///
/// Locate the SA row of the the hits in the HitQueues.
/// This function reads HitQueues::seed, HitQueues::loc and HitQueues::ssa fields
/// (which must have been previously produced by a call to locate_init) and rewrites
/// the HitQueues::loc field with the final linear coordinate value.
///
template <typename ScoringScheme>
void locate_lookup(
    const BaseScoringPipelineState<ScoringScheme>&  pipeline,
    const ParamsPOD                                 params);

///@}  // group Locate
///@}  // group nvBowtie

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio

#include <nvBowtie/bowtie2/cuda/locate_inl.h>
