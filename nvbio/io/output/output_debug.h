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

#include <zlib/zlib.h>

namespace nvbio {
namespace io {

struct DebugOutput : public OutputFile
{
    // structure that describes one read in the debug output format
    struct DbgAlignment
    {
        uint32 read_id;
        uint32 read_len;
        uint32 alignment_pos;   // if alignment_pos is 0, nothing else is written for this alignment

        // if alignment_pos is valid, struct Info follows
    };

    struct DbgInfo
    {
        enum {
            PAIRED        = 1,
            PROPER_PAIR   = 2,
            UNMAPPED      = 4,
            MATE_UNMAPPED = 8,
            REVERSE       = 16,
            MATE_REVERSE  = 32,
            READ_1        = 64,
            READ_2        = 128,
            SECONDARY     = 256,
            QC_FAILED     = 512,
            DUPLICATE     = 1024
        };

        uint32 flag;                // from enum above
        uint32 ref_id;
        // 8-bytes
        uint8  mate;
        uint8  mapQ;
        uint8  ed;
        uint8  pad;
        // 12-bytes
        uint16 subs:16;
        uint8  ins;
        uint8  dels;
        // 16-bytes
        uint8  mms;
        uint8  gapo;
        uint8  gape;
        uint8  has_second;
        // 20-bytes
        int32  score;
        int32  sec_score;
        // 24-bytes
    };

public:
    DebugOutput(const char *file_name, AlignmentType alignment_type, BNT bnt);
    ~DebugOutput();

    void process(struct GPUOutputBatch& gpu_batch,
                 const AlignmentMate mate,
                 const AlignmentScore score);
    void end_batch(void);

    void close(void);

private:
    void output_alignment(gzFile& fp, const struct DbgAlignment& al, const struct DbgInfo& info);
    void process_one_alignment(const AlignmentData& alignment, const AlignmentData& mate);
    void process_one_mate(DbgAlignment& al,
                          DbgInfo& info,
                          const AlignmentData& alignment,
                          const AlignmentData& mate,
                          const uint32 mapq);

    // our file pointers
    gzFile fp;
    gzFile fp_opposite_mate;

    // CPU copy of the current alignment batch
    CPUOutputBatch cpu_batch;
};

} // namespace io
} // namespace nvbio
