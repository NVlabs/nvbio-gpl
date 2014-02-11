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

#include <nvbio-aln-diff/alignment.h>
#include <stdio.h>

namespace nvbio {
namespace alndiff {

struct Filter
{
    enum Flags      { DISTANT = 1u, DISCORDANT = 2u, DIFFERENT_REF = 4u, ALL = 0xFFFFFFFFu };
    enum Statistics { ED = 1u, MAPQ = 2u, MMS = 4u, INS = 8u, DELS = 16u, SCORE = 32u };

    // empty constructor
    //
    Filter() : m_file(NULL), m_filtered(0) {}

    // constructor
    //
    // \param file_name     output file name
    // \param flags         read flags (DISTANT | DISCORDANT | DIFFERENT_REF | ALL) accepted by the filter
    // \param stats         statistics accepted by the filter
    // \param delta         filtering threshold
    //
    Filter(const char* file_name, const uint32 flags, const uint32 stats, const int32 delta) :
        m_file( NULL ),
        m_flags( flags ),
        m_stats( stats ),
        m_delta( delta ),
        m_filtered(0)
    {
        if (file_name)
        {
            log_verbose(stderr, "opening filter file \"%s\"... done\n", file_name);
            m_file = fopen( file_name, "wb" );
            if (m_file == NULL)
                log_warning( stderr, "unable to open filter file \"%s\"\n", file_name );
        }
    }
    // destructor
    //
    ~Filter()
    {
        if (m_file)
        {
            fclose( m_file );
            log_verbose(stderr, "closing filter file... done\n");
        }
    }

    // push a statistic into the filter
    //
    void operator() (const int32 delta, const uint32 flags, const Statistics stat, const uint32 read_id)
    {
        if (m_file == NULL)
            return;

        if ((m_flags & flags) &&
            (m_stats & stat) &&
            (m_delta > 0 ? delta >= m_delta : delta <= m_delta))
        {
            fwrite( &read_id, sizeof(uint32), 1u, m_file );

            ++m_filtered;
        }
    }

    // get filtered count
    //
    uint32 filtered() const { return m_filtered; }

    // flush the file
    //
    void flush() { if (m_file) fflush( m_file ); }

    FILE*  m_file;
    uint32 m_flags;
    uint32 m_stats;
    int32  m_delta;
    uint32 m_filtered;
};

} // namespace alndiff
} // namespace nvbio
