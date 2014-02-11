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
///\file reduce.h
///

#pragma once

#include <nvBowtie/bowtie2/cuda/defs.h>
#include <nvbio/io/alignments.h>
#include <nvBowtie/bowtie2/cuda/seed_hit.h>
#include <nvBowtie/bowtie2/cuda/params.h>
#include <nvbio/io/utils.h>
#include <nvbio/basic/exceptions.h>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

template <typename ScoringScheme> struct BaseScoringPipelineState;
template <typename ScoringScheme> struct BestApproxScoringPipelineState;

///@addtogroup nvBowtie
///@{

///@addtogroup Reduce
///@{

///
/// A context class for the score_reduce_kernel to be used in best-approx pipeline.
///
/// \details
/// Implements the basic extension bail-out mechanisms of Bowtie2, namely stopping
/// when a given number of extensions of a read failed in a row.
/// This is done keeping a vector of per-read extension 'trys' counters, which
/// start from the maximum allowed number, and get decremented towards zero upon
/// each failure, or reset upon successful extensions.
///
struct ReduceBestApproxContext
{
    /// constructor
    ///
    /// \param trys        trys vector
    /// \param n_ext       total number of extensions (i.e. extension loops) already performed
    ///
    ReduceBestApproxContext(uint32* trys, const uint32 n_ext) : m_trys( trys ), m_ext( n_ext ) {}

    /// this method is called from score_reduce_kernel to report updates to the best score.
    ///
    NVBIO_FORCEINLINE NVBIO_DEVICE 
    void best_score(const uint32 read_id, const ParamsPOD& params) const
    {
        // reset the try counter
        m_trys[ read_id ] = params.max_effort;
    }
    /// this method is called from score_reduce_kernel to report updates to the second best score.
    ///
    NVBIO_FORCEINLINE NVBIO_DEVICE 
    void second_score(const uint32 read_id, const ParamsPOD& params) const
    {
        // reset the try counter
        m_trys[ read_id ] = params.max_effort;
    }
    /// this method is called from score_reduce_kernel to report extension failures.
    ///
    NVBIO_FORCEINLINE NVBIO_DEVICE 
    bool failure(const uint32 idx, const uint32 read_id, const uint32 top_flag, const ParamsPOD& params) const
    {
        if (m_trys[ read_id ] > 0)
        {
            if (((m_ext+idx >= params.min_ext) && (top_flag == 0) && (--m_trys[ read_id ] == 0)) || // bowtie2 does 1 more alignment than effort limit, we don't
                 (m_ext+idx >= params.max_ext))
                 return true;
        }
        return false;
    }

private:
    uint32* m_trys;
    uint32  m_ext;
};

///
/// A context class for the score_reduce_kernel to be used in best-exact pipeline.
///
/// \details
/// A trivial implementation, that never bails out.
///
struct ReduceBestExactContext
{
    ReduceBestExactContext() {}

    /// this method is called from score_reduce_kernel to report updates to the best score.
    ///
    NVBIO_FORCEINLINE NVBIO_DEVICE 
    void best_score(const uint32 read_id, const ParamsPOD& params) const {}

    /// this method is called from score_reduce_kernel to report updates to the second best score.
    ///
    NVBIO_FORCEINLINE NVBIO_DEVICE 
    void second_score(const uint32 read_id, const ParamsPOD& params) const {}

    /// this method is called from score_reduce_kernel to report extension failures.
    ///
    NVBIO_FORCEINLINE NVBIO_DEVICE 
    bool failure(const uint32 idx, const uint32 read_id, const uint32 top_flag, const ParamsPOD& params) const { return false; }
};

///
/// Reduce the scores associated to each read in the scoring queue to find the best 2 alignments.
///
template <typename ScoringScheme, typename ReduceContext>
void score_reduce(
    const ReduceContext                                     context,
    const BestApproxScoringPipelineState<ScoringScheme>&    pipeline,
    const ParamsPOD                                         params);

///
/// Reduce the scores associated to each paired-end read in the scoring queue to find the best 2 alignments.
///
template <typename ScoringScheme, typename ReduceContext>
void score_reduce_paired(
    const ReduceContext                                     context,
    const BestApproxScoringPipelineState<ScoringScheme>&    pipeline,
    const ParamsPOD                                         params);

///@}  // group Reduce
///@}  // group nvBowtie

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio

#include <nvBowtie/bowtie2/cuda/reduce_inl.h>
