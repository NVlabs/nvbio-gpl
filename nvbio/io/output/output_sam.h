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

#include <nvbio/io/output/output_types.h>
#include <nvbio/io/output/output_utils.h>
#include <nvbio/io/output/output_file.h>
#include <nvbio/io/output/output_batch.h>
#include <nvbio/io/sequence/sequence.h>

#include <stdio.h>

namespace nvbio {
namespace io {

struct SamOutput : public OutputFile
{
private:
    // SAM alignment flags
    // these are meant to be bitwised OR'ed together
    typedef enum {
        SAM_FLAGS_PAIRED        = 1,
        SAM_FLAGS_PROPER_PAIR   = 2,
        SAM_FLAGS_UNMAPPED      = 4,
        SAM_FLAGS_MATE_UNMAPPED = 8,
        SAM_FLAGS_REVERSE       = 16,
        SAM_FLAGS_MATE_REVERSE  = 32,
        SAM_FLAGS_READ_1        = 64,
        SAM_FLAGS_READ_2        = 128,
        SAM_FLAGS_SECONDARY     = 256,
        SAM_FLAGS_QC_FAILED     = 512,
        SAM_FLAGS_DUPLICATE     = 1024
    } SamAlignmentFlags;

    // struct to hold a SAM alignment
    // field names come from http://samtools.sourceforge.net/SAMv1.pdf (page 4)
    struct SamAlignment
    {
        // required fields
        const char *        qname;              // query template name
        uint32              flags;              // bitwise alignment flags from SamAlignmentFlags
        const char *        rname;              // reference sequence name
        uint32              pos;                // 1-based leftmost mapping position
        uint8               mapq;               // mapping quality
        char                cigar[4096];        // CIGAR string
        const char *        rnext;              // reference name of the mate/next read
        uint32              pnext;              // position of the mate/next read
        int32               tlen;               // observed template length
        char                seq[1024];          // segment sequence (xxxnsubtil: size this according to max read len)
        char                qual[1024];         // ASCII of phred-scaled base quality+33 (xxxnsubtil: same as above)

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

public:
    SamOutput(const char *file_name, AlignmentType alignment_type, BNT bnt);
    ~SamOutput();

    void process(struct GPUOutputBatch& gpu_batch,
                 const AlignmentMate mate,
                 const AlignmentScore score);
    void end_batch(void);

    void close(void);

private:
    // write a printf-style formatted string to the file (preceded by a \t)
    void write_formatted_string(const char *fmt, ...);
    // write a plain string
    // tab controls whether to output a \t before the string
    void write_string(const char *str, bool tab = true);
    // write an integer
    template <typename T> void write_int(T i, bool tab = true);
    // add a line break
    void linebreak();

    // write a SAM tag
    template <typename T>
    void write_tag(const char *name, T value);

    // output the SAM file header
    void output_header(void);
    // output an alignment
    void output_alignment(const struct SamAlignment& aln);

    // process a single alignment from the stream and output it
    uint32 process_one_alignment(const AlignmentData& alignment,
                                 const AlignmentData& mate);

    // generate a CIGAR string from the alignment data
    uint32 generate_cigar_string(SamAlignment& sam_align,
                                 const AlignmentData& alignment);
    // generate the MD string from the internal representation
    uint32 generate_md_string(SamAlignment& sam_align, const AlignmentData& alignment);

    // our file pointer
    FILE *fp;
    // CPU copy of the current alignment batch
    CPUOutputBatch cpu_batch;
};

} // namespace io
} // namespace nvbio
