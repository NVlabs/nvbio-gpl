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
#include <nvbio/basic/cuda/condition.h>
#include <nvbio/basic/cuda/syncblocks.h>
#include <thrust/device_vector.h>

namespace nvbio {
namespace cuda {

///@addtogroup WorkQueue
///@{

struct OrderedQueueTag {};

/// Implements a self-compacting parallel WorkQueue, using a single-kernel launch that
/// compacts continuations maintaining while input ordering in the thread assignment (see \ref work_queue_page).
/// Useful to maintain memory access coherence.
/// Relatively high continuation overhead, but much lower than the MultiPass queue
/// if the queue capacity is low.
///
/// see \ref WorkQueue
///
template <
    typename WorkUnitT,
    uint32   BLOCKDIM>
struct WorkQueue<
    OrderedQueueTag,
    WorkUnitT,
    BLOCKDIM>
{
    typedef WorkUnitT   WorkUnit;

    /// constructor
    ///
    WorkQueue() : m_capacity(32*1024) {}

    /// set queue capacity
    ///
    void set_capacity(const uint32 capacity) { m_capacity = capacity; }

    /// consume a stream of work units
    ///
    template <typename WorkStream, typename WorkMover>
    void consume(const WorkStream stream, const WorkMover mover);

    /// consume a stream of work units
    ///
    template <typename WorkStream>
    void consume(WorkStream stream) { consume( stream, DefaultMover() ); }

    struct Context
    {
                 WorkUnit*  m_work_queue;
        volatile uint32*    m_work_queue_size;
        volatile uint32*    m_partials;
        volatile uint32*    m_prefixes;
                 uint8*     m_continuations;
                 uint32*    m_source_ids;
        condition_set_view  m_conditions;
        syncblocks          m_syncblocks;
    };

private:
    /// get a context
    ///
    Context get_context()
    {
        Context context;
        context.m_work_queue      = thrust::raw_pointer_cast( &m_work_queue.front() );
        context.m_work_queue_size = thrust::raw_pointer_cast( &m_work_queue_size.front() );
        context.m_partials        = thrust::raw_pointer_cast( &m_partials.front() );
        context.m_prefixes        = thrust::raw_pointer_cast( &m_prefixes.front() );
        context.m_continuations   = thrust::raw_pointer_cast( &m_continuations.front() );
        context.m_source_ids      = thrust::raw_pointer_cast( &m_source_ids.front() );
        context.m_conditions      = m_condition_set.get();
        context.m_syncblocks      = m_syncblocks.get();
        return context;
    }

    uint32                          m_capacity;
    thrust::device_vector<WorkUnit> m_work_queue;
    thrust::device_vector<uint32>   m_work_queue_size;
    thrust::device_vector<uint32>   m_partials;
    thrust::device_vector<uint32>   m_prefixes;
    thrust::device_vector<uint8>    m_continuations;
    thrust::device_vector<uint32>   m_source_ids;
    condition_set_storage           m_condition_set;
    syncblocks_storage              m_syncblocks;
};

///@} // WorkQueue

} // namespace cuda
} // namespace nvbio

#include <nvbio/basic/cuda/work_queue_ordered_inl.h>
