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
#include <nvBowtie/bowtie2/cuda/stats.h>
#include <nvbio/basic/threads.h>
#include <nvbio/basic/timer.h>
#include <nvbio/io/sequence/sequence.h>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

//
// A class implementing a background input thread, providing
// a set of input read-streams which are read in parallel to the
// operations performed by the main thread.
//

struct InputThread : public Thread<InputThread>
{
    static const uint32 BUFFERS = 4;
    static const uint32 INVALID = 1u;

    InputThread(io::SequenceDataStream* read_data_stream, Stats& _stats, const uint32 batch_size) :
        m_read_data_stream( read_data_stream ), m_stats( _stats ), m_batch_size( batch_size ), m_set(0)
    {
        for (uint32 i = 0; i < BUFFERS; ++i)
            read_data[i] = NULL;
    }

    void run();

    io::SequenceDataStream* m_read_data_stream;
    Stats&              m_stats;
    uint32              m_batch_size;
    volatile uint32     m_set;

    io::SequenceDataHost<DNA_N>           read_data_storage[BUFFERS];
    io::SequenceDataHost<DNA_N>* volatile read_data[BUFFERS];
};

//
// A class implementing a background input thread, providing
// a set of input read-streams which are read in parallel to the
// operations performed by the main thread.
//

struct InputThreadPaired : public Thread<InputThreadPaired>
{
    static const uint32 BUFFERS = 4;
    static const uint32 INVALID = 1u;

    InputThreadPaired(io::SequenceDataStream* read_data_stream1, io::SequenceDataStream* read_data_stream2, Stats& _stats, const uint32 batch_size) :
        m_read_data_stream1( read_data_stream1 ), m_read_data_stream2( read_data_stream2 ), m_stats( _stats ), m_batch_size( batch_size ), m_set(0)
    {
        for (uint32 i = 0; i < BUFFERS; ++i)
            read_data1[i] = read_data2[i] = NULL;
    }

    void run();

    io::SequenceDataStream* m_read_data_stream1;
    io::SequenceDataStream* m_read_data_stream2;
    Stats&                  m_stats;
    uint32                  m_batch_size;
    volatile uint32         m_set;

    io::SequenceDataHost<DNA_N> read_data_storage1[BUFFERS];
    io::SequenceDataHost<DNA_N> read_data_storage2[BUFFERS];

    io::SequenceDataHost<DNA_N>* volatile read_data1[BUFFERS];
    io::SequenceDataHost<DNA_N>* volatile read_data2[BUFFERS];
};

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio
