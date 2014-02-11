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
#include <nvbio/io/output/output_databuffer.h>
#include <nvbio/io/output/output_gzip.h>

#include <nvbio/io/fmi.h>
#include <nvbio/io/reads/reads.h>

#include <nvbio/io/bam_format.h>

#include <stdio.h>

namespace nvbio {
namespace io {

struct BamOutput : public OutputFile
{
private:
    // BAM alignment flags
    // these are meant to be bitwised OR'ed together
    typedef enum {
        BAM_FLAGS_PAIRED        = 1 << 16,
        BAM_FLAGS_PROPER_PAIR   = 2 << 16,
        BAM_FLAGS_UNMAPPED      = 4 << 16,
        BAM_FLAGS_MATE_UNMAPPED = 8 << 16,
        BAM_FLAGS_REVERSE       = 16 << 16,
        BAM_FLAGS_MATE_REVERSE  = 32 << 16,
        BAM_FLAGS_READ_1        = 64 << 16,
        BAM_FLAGS_READ_2        = 128 << 16,
        BAM_FLAGS_SECONDARY     = 256 << 16,
        BAM_FLAGS_QC_FAILED     = 512 << 16,
        BAM_FLAGS_DUPLICATE     = 1024 << 16
    } BamAlignmentFlags;

public:
    BamOutput(const char *file_name, AlignmentType alignment_type, BNT bnt);
    ~BamOutput();

    void process(struct GPUOutputBatch& gpu_batch,
                 const AlignmentMate mate,
                 const AlignmentScore score);
    void end_batch(void);

    void close(void);

private:
    void output_header(void);
    uint32 process_one_alignment(DataBuffer& out, AlignmentData& alignment, AlignmentData& mate);
    void write_block(DataBuffer& block);

    uint32 generate_cigar(struct BAM_alignment& alnh,
                          struct BAM_alignment_data_block& alnd,
                          const AlignmentData& alignment);
    uint32 generate_md_string(BAM_alignment& alnh, BAM_alignment_data_block& alnd,
                              const AlignmentData& alignment);

    void output_tag_uint32(DataBuffer& out, const char *tag, uint32 val);
    void output_tag_uint8(DataBuffer& out, const char *tag, uint8 val);
    void output_tag_string(DataBuffer& out, const char *tag, const char *val);

    void output_alignment(DataBuffer& out, BAM_alignment& alnh, BAM_alignment_data_block& alnd);

    static uint8 encode_bp(uint8 bp);

    // our file pointer
    FILE *fp;
    // CPU copy of the current alignment batch
    CPUOutputBatch cpu_output;
    // text buffer that we're filling with data
    DataBuffer data_buffer;
    // our BGZF compressor
    BGZFCompressor bgzf;
};

} // namespace io
} // namespace nvbio
