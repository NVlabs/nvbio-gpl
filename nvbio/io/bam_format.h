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

namespace nvbio {
namespace io {

// BAM format description: http://samtools.sourceforge.net/SAM1.pdf
// note that we don't actually use BGZF here, just plain gzip since we're reading everything sequentially
// (in other words, we don't use BAM indices)

// the BAM header section
struct BAM_header
{
    // header section
    uint8 magic[4];  // BAM magic string
    int32 l_text;    // length of the header text
    // header text comes next; we ignore it

    // reference sequence section
    int32 n_ref;     // number of reference sequences
};

// BAM reference sequence section
struct BAM_reference
{
    int32 l_name; // length of the reference name + 1 (including null)
    // reference sequence name goes here (null-terminated)
    int32 l_ref;  // length of the reference sequence
};

// BAM alignment section
struct BAM_alignment
{
    int32  block_size; // length of the remainder of the alignment record
    int32  refID;      // reference sequence ID, -1 <= refID < n_ref (-1 for a read without a mapping position)
    int32  pos;        // 0-based leftmost coordinate
    uint32 bin_mq_nl;  // bin << 16 | MAPQ << 8 | l_read_name
    uint32 flag_nc;    // FLAG << 16 | n_cigar_op
    int32  l_seq;      // length of the sequence
    int32  next_refID; // refID of the next segment (-1 <= next_refID < n_ref)
    int32  next_pos;   // 0-based leftmost pos of the next segment
    int32  tlen;       // template length

    // BAM_alignment_data_block follows
};

struct BAM_alignment_data_block
{
    const char *name;  // read name, NULL terminated
    uint32 cigar[1024]; // CIGAR string, encoded as op_len << 4 | op ; 'MIDNSHP=X' -> 012345678
    uint8 seq[1024];   // 4-bit encoded read: '=ACMGRSVTWYHKDBN' -> [0, 15]; other characters mapped to 'N'
                       //   high nibble first: 1st base in the highest 4-bit of the 1st byte
    uint8 qual[1024];  // Phred-base quality (a sequence of 0xFF if absent)

    // our own additional data, output as tags (only if read is mapped)
    int32               ed;                 // NM:i
    int32               score;              // AS:i
    int32               second_score;       // XS:i (optional)
    int32               mm;                 // XM:i
    int32               gapo;               // XO:i
    int32               gape;               // XG:i
    char                md_string[4096];    // MD:Z (mostly optional?)

    // extra data that's useful but not written out
    bool                second_score_valid; // do we have a second score?
};

} // namespace io
} // namespace nvbio
