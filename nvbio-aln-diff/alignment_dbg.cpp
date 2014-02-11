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

#include <nvbio-aln-diff/alignment.h>
#include <zlib/zlib.h>

namespace nvbio {
namespace alndiff {

struct DebugAlignmentStream : public AlignmentStream
{
    struct Info
    {
        uint32 flag;
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

    DebugAlignmentStream(const char* file_name)
    {
        m_file = gzopen( file_name, "rb" );
    }

    // return if the stream is ok
    //
    bool is_ok() { return m_file != NULL; }

    // get the next batch
    //
    uint32 next_batch(
        const uint32    count,
        Alignment*      batch)
    {
        uint32 n_read = 0;

        Info info;

        while (n_read < count)
        {
            Alignment* aln = batch + n_read;

            // clean the alignment
            *aln = Alignment();

            // read id
            if (gzread( m_file, &aln->read_id, sizeof(uint32) ) == 0)
                break;

            // read length
            gzread( m_file, &aln->read_len, sizeof(uint32) );

            // alignment position
            gzread( m_file, &aln->pos, sizeof(uint32) );

            if (aln->pos != uint32(0))
            {
                gzread( m_file, &info, sizeof(Info) );

                aln->flag       = info.flag;
                aln->ref_id     = info.ref_id;
                aln->mate       = info.mate;
                aln->score      = info.score;
                aln->mapQ       = info.mapQ;
                aln->ed         = info.ed;
                aln->subs       = info.subs;
                aln->ins        = info.ins;
                aln->dels       = info.dels;
                aln->n_mm       = info.mms;
                aln->n_gapo     = info.gapo;
                aln->n_gape     = info.gape;
                aln->has_second = info.has_second;
                aln->sec_score  = info.sec_score;
            }

            ++n_read;
        }
        return n_read;
    }

    gzFile m_file;
};

AlignmentStream* open_dbg_file(const char* file_name)
{
    return new DebugAlignmentStream( file_name );
}

} // alndiff namespace
} // nvbio namespace


