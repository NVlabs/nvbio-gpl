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

#include <nvbio/basic/cuda/work_queue.h>
#include <nvbio/basic/types.h>
#include <nvbio/basic/numbers.h>
#include <thrust/device_vector.h>

namespace nvbio {
namespace cuda {

///@addtogroup WorkQueue
///@{

struct MultiPassQueueTag {};

/// Implements a multi-pass WorkQueue that uses thrust::copy_if to compact
/// continuations between kernel launches (see \ref work_queue_page).
/// Mantains input ordering in the thread assignment, hence mantaining memory access
/// coherence, but has potentially much higher overhead than the Ordered queue if the
/// queue capacity is low.
/// High continuation overhead at low queue capacities. At high queue capacities
/// (e.g. millions of elements), the kernel launch overheads are amortized.
///
/// see \ref WorkQueue
///
template <
    typename WorkUnitT,
    uint32   BLOCKDIM>
struct WorkQueue<
    MultiPassQueueTag,
    WorkUnitT,
    BLOCKDIM>
{
    typedef WorkUnitT   WorkUnit;

    /// constructor
    ///
    WorkQueue() : m_capacity(32*1024), m_separate_loads(false) {}

    /// set queue capacity
    ///
    void set_capacity(const uint32 capacity) { m_capacity = capacity; }

    /// enable separate loads
    ///
    void set_separate_loads(const bool flag) { m_separate_loads = flag; }

    /// consume a stream of work units
    ///
    template <typename WorkStream>
    void consume(const WorkStream stream, WorkQueueStats* stats = NULL) { consume( stream, DefaultMover(), stats ); }

    /// consume a stream of work units
    ///
    template <typename WorkStream, typename WorkMover>
    void consume(const WorkStream stream, const WorkMover mover, WorkQueueStats* stats = NULL);

    struct Context
    {
        WorkUnit*  m_work_queue;
        uint32*    m_continuations;
    };

private:
    /// get a context
    ///
    Context get_context()
    {
        Context context;
        context.m_work_queue      = thrust::raw_pointer_cast( &m_work_queue.front() );
        context.m_continuations   = thrust::raw_pointer_cast( &m_continuations.front() );
        return context;
    }

    uint32                          m_capacity;
    bool                            m_separate_loads;
    thrust::device_vector<WorkUnit> m_work_queue;
    thrust::device_vector<uint32>   m_continuations;
};

///@} // WorkQueue

} // namespace cuda
} // namespace nvbio

#include <nvbio/basic/cuda/work_queue_multipass_inl.h>
