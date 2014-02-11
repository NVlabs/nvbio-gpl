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
///\file mapping.h
///

#pragma once

#include <nvBowtie/bowtie2/cuda/defs.h>
#include <nvBowtie/bowtie2/cuda/utils.h>
#include <nvBowtie/bowtie2/cuda/params.h>
#include <nvBowtie/bowtie2/cuda/seed_hit.h>
#include <nvBowtie/bowtie2/cuda/seed_hit_deque_array.h>
#include <nvbio/io/utils.h>
#include <nvbio/basic/cuda/pingpong_queues.h>
#include <nvbio/basic/cached_iterator.h>
#include <nvbio/basic/packedstream.h>
#include <nvbio/basic/priority_deque.h>
#include <nvbio/basic/vector_wrapper.h>
#include <nvbio/basic/strided_iterator.h>
#include <nvbio/basic/algorithms.h>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

///@addtogroup nvBowtie
///@{

/// \defgroup Mapping
///
/// The functions in this module implement the very first pipeline stage: seed mapping.
/// In this stage each read is broken up into many short, possibly overlapping seeds
/// which get mapped against the reference genome using an FM-index.
/// The output is a vector of variable-lenth "priority deques", one for each read,
/// containing the set of Suffix Array ranges (\ref SeedHits) where the seeds align,
/// prioritized by the inverse of the range size (see \ref SeedHitDequeArray).
///
/// The module implements many mapping algorithms:
///
/// - exact: allowing exact matches only
/// - approx-hybrid: allowing 0 mismatches in a subseed of the seed, and up to 1 mismatch in the rest
/// - approx-case-pruning: allowing 1 mismatch across the entire seed, using 2 FM-indices to perform
///   the search with case pruning (i.e. searching an alignment with 0 mismatches in the first half of
///   the seed in the forward FM-index, and an alignment with 0 mismatches in the second half in the
///   reverse FM-index).
///

///@addtogroup Mapping
///@{

///
/// For all the seed hit ranges, output the range size in out_ranges.
///
void gather_ranges(
    const uint32                    count,
    const uint32                    n_reads,
    SeedHitDequeArrayDeviceView     hits,
    const uint32*                   hit_counts_scan,
          uint64*                   out_ranges);

///
/// perform exact read mapping
///
template <typename BatchType, typename FMType, typename rFMType>
void map_whole_read(
    const BatchType&                                read_batch, const FMType fmi, const rFMType rfmi,
    const nvbio::cuda::PingPongQueuesView<uint32>   queues,
    SeedHitDequeArrayDeviceView                     hits,
    const ParamsPOD                                 params);

///
/// perform one run of exact seed mapping for all the reads in the input queue,
/// writing reads that need another run in the output queue
///
template <typename BatchType, typename FMType, typename rFMType>
void map_exact(
    const BatchType&                                read_batch, const FMType fmi, const rFMType rfmi,
    const uint32                                    retry,
    const nvbio::cuda::PingPongQueuesView<uint32>   queues,
    SeedHitDequeArrayDeviceView                     hits,
    const ParamsPOD                                 params);

///
/// perform multiple runs of exact seed mapping in one go and keep the best
///
template <typename BatchType, typename FMType, typename rFMType>
void map_exact(
    const BatchType&            read_batch, const FMType fmi, const rFMType rfmi,
    SeedHitDequeArrayDeviceView hits,
    const uint2                 seed_range,
    const ParamsPOD             params);

///
/// perform one run of approximate seed mapping using case pruning for all the reads in
/// the input queue, writing reads that need another run in the output queue
///
template <typename BatchType, typename FMType, typename rFMType>
void map_case_pruning(
    const BatchType&                                read_batch, const FMType fmi, const rFMType rfmi,
    const uint32                                    retry,
    const nvbio::cuda::PingPongQueuesView<uint32>   queues,
    SeedHitDequeArrayDeviceView                     hits,
    const ParamsPOD                                 params);

///
/// perform one run of approximate seed mapping for all the reads in the input queue,
/// writing reads that need another run in the output queue
///
template <typename BatchType, typename FMType, typename rFMType>
void map_approx(
    const BatchType&                                read_batch, const FMType fmi, const rFMType rfmi,
    const uint32                                    retry,
    const nvbio::cuda::PingPongQueuesView<uint32>   queues,
    SeedHitDequeArrayDeviceView                     hits,
    const ParamsPOD                                 params);

///
/// perform multiple runs of approximate seed mapping in one go and keep the best
///
template <typename BatchType, typename FMType, typename rFMType>
void map_approx(
    const BatchType&            read_batch, const FMType fmi, const rFMType rfmi,
    SeedHitDequeArrayDeviceView hits,
    const uint2                 seed_range,
    const ParamsPOD             params);

///
/// perform one run of seed mapping
///
template <typename BatchType, typename FMType, typename rFMType>
void map(
    const BatchType&                                read_batch, const FMType fmi, const rFMType rfmi,
    const uint32                                    retry,
    const nvbio::cuda::PingPongQueuesView<uint32>   queues,
    SeedHitDequeArrayDeviceView                     hits,
    const ParamsPOD                                 params);

///@}  // group Mapping
///@}  // group nvBowtie

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio

#include <nvBowtie/bowtie2/cuda/mapping_inl.h>
