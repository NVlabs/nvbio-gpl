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

#include <nvbio-aln-diff/stats.h>
#include <nvbio-aln-diff/alignment.h>
#include <nvbio-aln-diff/filter.h>

namespace nvbio {
namespace alndiff {

struct SEAnalyzer
{
    SEAnalyzer(Filter& filter);

    void push(
        const Alignment& aln1,
        const Alignment& aln2);

    void generate_report(const char* aln_file_name1, const char* aln_file_name2, const char* report);

    // flush any open files
    //
    void flush() { m_filter.flush(); }

    float mismatched() const { return float(n_mismatched)/float(n + n_mismatched); }
    float different_ref() const { return float(n_different_ref.count)/float(n); }
    float distant() const { return float(n_distant.count)/float(n); }
    float discordant() const { return float(n_discordant.count)/float(n); }
    uint32 filtered() const { return m_filter.filtered(); }

    Filter& m_filter;

    BooleanStats mapped;
    BooleanStats unique;
    BooleanStats ambiguous;
    BooleanStats not_ambiguous;

    Histogram<8> mapped_L_not_R_by_mapQ;
    Histogram<8> mapped_R_not_L_by_mapQ;
    Histogram<8> unique_L_not_R_by_mapQ;
    Histogram<8> unique_R_not_L_by_mapQ;
    Histogram<8> ambiguous_L_not_R_by_mapQ;
    Histogram<8> ambiguous_R_not_L_by_mapQ;

    uint32 n;
    uint32 n_mismatched;

    Histogram<8> n_different_ref;
    Histogram<8> n_distant;
    Histogram<8> n_discordant;

    AlignmentStats al_stats;
    AlignmentStats distant_stats;
    AlignmentStats discordant_stats;
};

} // namespace alndiff
} // namespace nvbio