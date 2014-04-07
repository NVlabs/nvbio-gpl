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

#include <nvBowtie/bowtie2/cuda/defs.h>
#include <nvBowtie/bowtie2/cuda/params.h>
#include <nvBowtie/bowtie2/cuda/string_utils.h>
#include <nvBowtie/bowtie2/cuda/scoring.h>
#include <nvbio/basic/exceptions.h>
#include <nvbio/io/alignments.h>
#include <nvbio/io/utils.h>
#include <nvbio/basic/dna.h> // for dna_to_char

namespace nvbio {
namespace bowtie2 {
namespace cuda {

///@addtogroup nvBowtie
///@{

///@addtogroup Traceback
///@{

template <typename vector_type>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE 
uint32 read_cigar_length(
    const vector_type cigar,
    const uint32      cigar_len)
{
    uint32 r = 0;
    for (uint32 i = 0; i < cigar_len; ++i)
    {
        const uint32 l  = cigar[ cigar_len - i - 1u ].m_len;
        const uint32 op = cigar[ cigar_len - i - 1u ].m_type;
        if (op != io::Cigar::DELETION) r += l;
    }
    return r;
}

enum MateType
{
    AnchorMate   = 0,
    OppositeMate = 1,
};

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
const char* mate_string(const MateType mate) { return mate == OppositeMate ? "opposite" : "anchor"; }

///
/// execute a batch of banded-alignment traceback calculations
///
template <uint32 ALN_IDX, typename pipeline_type>
void banded_traceback_best(
    const uint32                count,
    const uint32*               idx,
          io::BestAlignments*   best_data,
    const uint32                band_len,
    const pipeline_type&         pipeline,
    const ParamsPOD             params);

///
/// execute a batch of opposite alignment traceback calculations
///
template <uint32 ALN_IDX, typename pipeline_type>
void opposite_traceback_best(
    const uint32                count,
    const uint32*               idx,
          io::BestAlignments*   best_data,
    const pipeline_type&        pipeline,
    const ParamsPOD             params);

///
/// finish a batch of alignment calculations
///
template <uint32 ALN_IDX, typename scoring_scheme_type, typename pipeline_type>
void finish_alignment_best(
    const uint32                count,
    const uint32*               idx,
          io::BestAlignments*   best_data,
    const uint32                band_len,
    const pipeline_type&        pipeline,
    const scoring_scheme_type   scoring_scheme,
    const ParamsPOD             params);

///
/// finish a batch of opposite alignment calculations
///
template <uint32 ALN_IDX, typename scoring_scheme_type, typename pipeline_type>
void finish_opposite_alignment_best(
    const uint32                count,
    const uint32*               idx,
          io::BestAlignments*   best_data,
    const uint32                band_len,
    const pipeline_type&        pipeline,
    const scoring_scheme_type   scoring_scheme,
    const ParamsPOD             params);

///
/// finish a batch of alignment calculations, all-mapping
///
template <typename scoring_scheme_type, typename pipeline_type>
void finish_alignment_all(
    const uint32                count,
    const uint32*               idx,
    const uint32                buffer_offset,
    const uint32                buffer_size,
          io::Alignment*        alignments,
    const uint32                band_len,
    const pipeline_type&        pipeline,
    const scoring_scheme_type   scoring_scheme,
    const ParamsPOD             params);

///@}  // group Traceback
///@}  // group nvBowtie

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio

#include <nvBowtie/bowtie2/cuda/traceback_inl.h>
