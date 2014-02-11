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

#include <nvBowtie/bowtie2/cuda/input_thread.h>
#include <nvBowtie/bowtie2/cuda/defs.h>
#include <nvbio/io/alignments.h>
#include <nvBowtie/bowtie2/cuda/params.h>
#include <nvBowtie/bowtie2/cuda/stats.h>
#include <nvbio/io/output/output_utils.h>
#include <nvbio/basic/threads.h>
#include <nvbio/basic/timer.h>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

void InputThread::run()
{
    log_verbose( stderr, "starting background input thread\n" );

    while (1u)
    {
        // poll until the set is done reading & ready to be reused
        while (read_data[m_set] != NULL) {}

        //// lock the set to flush
        //ScopedLock lock( &m_lock[m_set] );

        Timer timer;
        timer.start();

        io::ReadData* data = m_read_data_stream->next( m_batch_size );

        timer.stop();

        if (data)
        {
            m_stats.read_io.add( data->size(), timer.seconds() );

            // mark the set as done
            read_data[ m_set ] = data;
        }
        else
        {
            // mark this as an invalid entry
            read_data[ m_set ] = (io::ReadData*)INVALID;
            break;
        }

        // switch to the next set
        m_set = (m_set + 1) % BUFFERS;
    }
}

void InputThreadPaired::run()
{
    log_verbose( stderr, "starting background paired-end input thread\n" );

    while (1u)
    {
        // poll until the set is done reading & ready to be reused
        while (read_data1[m_set] != NULL || read_data2[m_set] != NULL) {}

        //// lock the set to flush
        //ScopedLock lock( &m_lock[m_set] );

        Timer timer;
        timer.start();

        io::ReadData* data1 = m_read_data_stream1->next( m_batch_size );
        io::ReadData* data2 = m_read_data_stream2->next( m_batch_size );

        timer.stop();

        if (data1 && data2)
        {
            m_stats.read_io.add( data1->size(), timer.seconds() );

            // mark the set as done
            read_data1[ m_set ] = data1;
            read_data2[ m_set ] = data2;
        }
        else
        {
            // mark this as an invalid entry
            read_data1[ m_set ] = (io::ReadData*)INVALID;
            read_data2[ m_set ] = (io::ReadData*)INVALID;

            // delete unpaired segments
            if (data1) delete data1;
            if (data2) delete data2;
            break;
        }

        // switch to the next set
        m_set = (m_set + 1) % BUFFERS;
    }
}

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio
