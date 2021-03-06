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
#include <nvBowtie/bowtie2/cuda/stats.h>

#include <nvbio/io/alignments.h>
#include <nvbio/io/output/output_types.h>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

/// Bowtie2 Mapping Quality Calculator, V3
///
template <typename ScoringSchemeType>
struct BowtieMapq3
{
    typedef ScoringSchemeType scoring_scheme_type;

    /// constructor
    ///
	BowtieMapq3(const ScoringSchemeType sc) : m_sc(sc) {}

    /// compute mapping quality
    ///
	int operator() (
        const io::BestPairedAlignments& best_alignments,
        const uint32                    read_len,
        const uint32                    o_read_len) const
	{
        // no second best alignment, perfect score
        const int unpaired_one_perfect = 44;

        // no second best alignment
        const int unpaired_one[11] =
        {
	        43, 42, 41, 36, 32, 27, 20, 11, 4, 1, 0
        };

        // two alignment scores, the best of which has perfect score
        const int unpaired_two_perfect[11] =
        {
	        2, 16, 23, 30, 31, 32, 34, 36, 38, 40, 42
        };

        // two alignment scores, the best of which has a non perfect score
        const int unpaired_two[11][11] =
        {
	        {  2,  2,  2,  1,  1, 0, 0, 0, 0, 0, 0 },
	        { 20, 14,  7,  3,  2, 1, 0, 0, 0, 0, 0 },
	        { 20, 16, 10,  6,  3, 1, 0, 0, 0, 0, 0 },
	        { 20, 17, 13,  9,  3, 1, 1, 0, 0, 0, 0 },
	        { 21, 19, 15,  9,  5, 2, 2, 0, 0, 0, 0 },
	        { 22, 21, 16, 11, 10, 5, 0, 0, 0, 0, 0 },
	        { 23, 22, 19, 16, 11, 0, 0, 0, 0, 0, 0 },
	        { 24, 25, 21, 30,  0, 0, 0, 0, 0, 0, 0 },
	        { 30, 26, 29,  0,  0, 0, 0, 0, 0, 0, 0 },
	        { 30, 27,  0,  0,  0, 0, 0, 0, 0, 0, 0 },
	        { 30,  0,  0,  0,  0, 0, 0, 0, 0, 0, 0 },
        };

        // paired alignment, one perfect score
        const int paired_one_perfect = 44;

        //
        // TODO: if paired, return paired_one_perfect
        //
		if (best_alignments.is_paired())
			return paired_one_perfect;

        const float max_score = (float)m_sc.perfect_score( read_len );
        const float min_score = (float)m_sc.min_score( read_len );
        const float norm_factor = 10.0f / (max_score - min_score);

        // check whether the best score is beyond the minimum threshold
        if (best_alignments.best_score() < min_score)
            return 0;

        const int best     = std::max( (int)max_score - best_alignments.best_score(), 0 ); // negated best score
        const int best_bin = (int)(float(best) * norm_factor + 0.5f);

        const bool has_second = best_alignments.has_second();

//        if (info)
//        {
//            info->best        = best;
//            info->second_best = has_second ? best_alignments.second_score() : 255;
//        }

        // check whether there are two alignment scores or just one
        if (has_second)
        {
			const int diff     = best_alignments.best_score() - best_alignments.second_score();
			const int diff_bin = (int)(float(diff) * norm_factor + 0.5f);

			if (best == max_score)
				return unpaired_two_perfect[best_bin];
            else
				return unpaired_two[diff_bin][best_bin];
		}
        else
        {
			if (best == max_score)
				return unpaired_one_perfect;
            else
				return unpaired_one[best_bin];
		}
	}

private:
	ScoringSchemeType m_sc;
};

/// Bowtie2 Mapping Quality Calculator, V2
///
template <typename ScoringSchemeType>
struct BowtieMapq2
{
    typedef ScoringSchemeType scoring_scheme_type;

    /// constructor
    ///
	BowtieMapq2(const ScoringSchemeType sc) : m_sc(sc) {}

    /// compute mapping quality
    ///
	int operator() (
        const io::BestPairedAlignments& best_alignments,
        const uint32                    read_len,
        const uint32                    o_read_len) const
	{
		bool has_second = best_alignments.has_second();

        const float max_score = (float)m_sc.perfect_score( read_len ) +
            (best_alignments.is_paired() ? m_sc.perfect_score( o_read_len ) : 0);
        const float min_score = (float)m_sc.min_score( read_len ) +
            (best_alignments.is_paired() ? m_sc.min_score( o_read_len ) : 0);

		const float diff = (max_score - min_score);  // range of scores

        const float best        = (float)best_alignments.best_score();
              float second_best = min_score-1;

        if (best < min_score)
            return 0;

		// best score but normalized so that 0 = worst valid score
        const float best_over = best - min_score;

//        if (info)
//        {
//            info->best        = (int32)best;
//            info->second_best = (int32)second_best;
//        }

        if (m_sc.m_monotone)
        {
			// global alignment
			if (!has_second)
            {
				if      (best_over >= diff * 0.8f) return 42;
				else if (best_over >= diff * 0.7f) return 40;
				else if (best_over >= diff * 0.6f) return 24;
				else if (best_over >= diff * 0.5f) return 23;
				else if (best_over >= diff * 0.4f) return 8;
				else if (best_over >= diff * 0.3f) return 3;
				else                               return 0;
			}
            else 
            {
                second_best = (float)best_alignments.second_score();
//                if (info)
//                    info->second_best = (int32)second_best;

                const float best_diff = fabsf(fabsf( best ) - fabsf( second_best ));

                if (best_diff >= diff * 0.9f)
                    return  (best_over == diff) ? 39 : 33;
				else if (best_diff >= diff * 0.8f)
                    return  (best_over == diff) ? 38 : 27;
				else if (best_diff >= diff * 0.7f)
                    return  (best_over == diff) ? 37 : 26;
				else if (best_diff >= diff * 0.6f)
                    return  (best_over == diff) ? 36 : 22;
				else if (best_diff >= diff * 0.5f)
                {
					// top third is still pretty good
					if      (best_over == diff)         return 35;
					else if (best_over >= diff * 0.84f) return 25;
					else if (best_over >= diff * 0.68f) return 16;
					else                                return 5;
				}
                else if (best_diff >= diff * 0.4f)
                {
					// top third is still pretty good
					if      (best_over == diff)         return 34;
					else if (best_over >= diff * 0.84f) return 21;
					else if (best_over >= diff * 0.68f) return 14;
					else                                return 4;
				}
                else if (best_diff >= diff * 0.3f)
                {
					// top third is still pretty good
					if      (best_over == diff)         return 32;
					else if (best_over >= diff * 0.88f) return 18;
					else if (best_over >= diff * 0.67f) return 15;
					else                                return 3;
				}
                else if (best_diff >= diff * 0.2f)
                {
					// top third is still pretty good
					if      (best_over == diff)         return 31;
					else if (best_over >= diff * 0.88f) return 17;
					else if (best_over >= diff * 0.67f) return 11;
					else                                return 0;
				} 
                else if (best_diff >= diff * 0.1f)
                {
					// top third is still pretty good
					if      (best_over == diff)			return 30;
					else if (best_over >= diff * 0.88f) return 12;
					else if (best_over >= diff * 0.67f) return 7;
					else                                return 0;
				}
                else if (best_diff > 0)
                {
					// top third is still pretty good
                    return (best_over >= diff * 0.67f) ? 6 : 2;
				} 
                else
                {
					// top third is still pretty good
                    return (best_over >= diff * 0.67f) ? 1 : 0;
				}
			}
		}
        else
        {
			// local alignment
			if (!has_second)
            {
				if      (best_over >= diff * 0.8f)      return 44;
				else if (best_over >= diff * 0.7f)      return 42;
				else if (best_over >= diff * 0.6f)      return 41;
				else if (best_over >= diff * 0.5f)      return 36;
				else if (best_over >= diff * 0.4f)      return 28;
				else if (best_over >= diff * 0.3f)      return 24;
				else                                    return 22;
			}
            else
            {
				second_best = (float)best_alignments.second_score();
//                if (info)
//                    info->second_best = (int32)second_best;

                const float best_diff = fabsf(fabsf( best ) - fabsf( second_best ));

                if      (best_diff >= diff * 0.9f)      return 40;
				else if (best_diff >= diff * 0.8f)      return 39;
				else if (best_diff >= diff * 0.7f)      return 38;
				else if (best_diff >= diff * 0.6f)      return 37;
				else if (best_diff >= diff * 0.5f)
                {
					if      (best_over == diff)         return 35;
					else if (best_over >= diff * 0.50f) return 25;
					else                                return 20;
				}
                else if (best_diff >= diff * 0.4f)
                {
					if      (best_over == diff)         return 34;
					else if (best_over >= diff * 0.50f) return 21;
					else                                return 19;
				}
                else if (best_diff >= diff * 0.3f)
                {
					if      (best_over == diff)         return 33;
					else if (best_over >= diff * 0.5f)  return 18;
					else                                return 16;
				} 
                else if (best_diff >= diff * 0.2f)
                {
					if      (best_over == diff)         return 32;
					else if (best_over >= diff * 0.5f)  return 17;
					else                                return 12;
				}
                else if (best_diff >= diff * 0.1f)
                {
					if      (best_over == diff)         return 31;
					else if (best_over >= diff * 0.5f)  return 14;
					else                                return 9;
				}
                else if (best_diff > 0)
                    return (best_over >= diff * 0.5f) ? 11 : 2;
                else
                    return (best_over >= diff * 0.5f) ? 1 : 0;
			}
		}
	}

private:
	ScoringSchemeType m_sc;
};

// interface glue for io::MapQEvaluator
// wraps BowtieMapq2/3
// xxxnsubtil: this desperately needs cleaning up
template <typename OldBowtieMapq>
struct BowtieMapq : public io::MapQEvaluator
{
private:
    OldBowtieMapq old_mapq;

public:
    BowtieMapq(const typename OldBowtieMapq::scoring_scheme_type sc)
        : old_mapq(sc)
    { }

    // compute a single-end mapq
    // this is used for both single-end and paired-ends --- for PE, mate is invalid
    int compute_mapq(const io::AlignmentData& alignment,
                     const io::AlignmentData& mate) const
    {
        io::BestAlignments best_a, best_o;

        // stash the alignment into a BestPairedAlignments struct
        NVBIO_CUDA_ASSERT(alignment.valid);
        best_a = io::BestAlignments(*alignment.best, *alignment.second_best);

        if (mate.valid)
        {
            best_o = io::BestAlignments(*mate.best, *mate.second_best);
        } else {
            best_o = io::BestAlignments(io::Alignment::invalid(), io::Alignment::invalid());
        }

        io::BestPairedAlignments best_paired(best_a, best_o);

        // call into the old mapq code
        return old_mapq(best_paired, alignment.read_len, alignment.read_len);
    }
};

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio
