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

#include <zlib/zlib.h>

#include <nvbio/io/reads/reads.h>
#include <nvbio/io/reads/reads_priv.h>
#include <nvbio/basic/console.h>

namespace nvbio {
namespace io {

// SAM format description: http://samtools.sourceforge.net/SAM1.pdf

// flag comments come from SAMtools spec
// a better explanation is available at:
// http://genome.sph.umich.edu/wiki/SAM#What_Information_Does_SAM.2FBAM_Have_for_an_Alignment
enum AlignmentFlags
{
    // SAMtools: template having multiple segments in sequencing
    SAMFlag_MultipleSegments = 0x1,
    // each segment properly aligned according to the aligner
    SAMFlag_AllSegmentsAligned = 0x2,
    // segment unmapped
    SAMFlag_SegmentUnmapped = 0x4,
    // next segment in the template unmapped
    SAMFlag_NextSegmentUnmapped = 0x8,
    // SEQ being reverse complemented
    SAMFlag_ReverseComplemented = 0x10,
    // SEQ of the next segment in the template being reversed
    SAMFlag_NextSegmentReverseComplemented = 0x20,
    // the first segment in the template
    SAMFlag_FirstSegment = 0x40,
    // the last segment in the template
    SAMFlag_LastSegment = 0x80,
    // secondary alignment
    SAMFlag_SecondaryAlignment = 0x100,
    // not passing quality controls
    SAMFlag_FailedQC = 0x200,
    // PCR or optical duplicate
    SAMFlag_Duplicate = 0x400,
};


// ReadDataFile from a SAM file
struct ReadDataFile_SAM : public ReadDataFile
{
    enum { LINE_BUFFER_INIT_SIZE = 1024 };

    enum SortOrder
    {
        SortOrder_unknown,
        SortOrder_unsorted,
        SortOrder_queryname,
        SortOrder_coordinate,
    };

    ReadDataFile_SAM(const char *read_file_name,
                     const uint32 max_reads,
                     const uint32 max_read_len,
                     const ReadEncoding flags);

    virtual int nextChunk(ReadDataRAM *output, uint32 max_reads, uint32 max_bps);

    bool init(void);

private:
    bool readLine(void);
    void rewindLine(void);
    bool parseHeaderLine(char *start);

    gzFile fp;

    // a buffer for a full line; this will grow as needed
    char *linebuf;
    // current size of the buffer
    int linebuf_size;
    // length of the current line in the buffer
    int line_length;

    // how many lines we parsed so far
    int numLines;

    // info from the header
    char *version;
    SortOrder sortOrder;
};

}
}
