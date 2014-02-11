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

#include <nvbio/basic/types.h>
#include <nvbio/basic/numbers.h>
#include <thrust/device_vector.h>

namespace nvbio {
namespace cuda {

///@addtogroup WorkQueue
///@{

// queue tags
struct PersistentWarpsQueueTag {};
struct PersistentThreadsQueueTag {};

/// Implements a WorkQueue using persistent warps to fetch more work at a warp's granularity.
/// with dynamic work assignment (see \ref work_queue_page).
/// Each work-unit is assigned to a single thread, which runs until completion of all its continuations.
/// Useful if some warps finish much earlier than others, but all threads within a warp execute
/// roughly the same amount of work. Potentially destroys intra-CTA memory coherence, although
/// it maintains any warp-level coherence in the input stream.
/// Very low continuation overhead.
///
/// see \ref WorkQueue
///
template <
    typename WorkUnitT,
    uint32   BLOCKDIM>
struct WorkQueue<
    PersistentWarpsQueueTag,
    WorkUnitT,
    BLOCKDIM>
{
    typedef WorkUnitT   WorkUnit;

    /// constructor
    ///
    WorkQueue() : m_capacity(uint32(-1)) {}

    /// set queue capacity
    ///
    void set_capacity(const uint32 capacity) { m_capacity = capacity; }

    /// consume a stream of work units
    ///
    template <typename WorkStream>
    void consume(const WorkStream stream, WorkQueueStats* stats = NULL) { consume( stream, DefaultMover(), stats ); }

    /// consume a stream of work units
    ///
    template <typename WorkStream, typename WorkMover>
    void consume(const WorkStream stream, const WorkMover mover, WorkQueueStats* stats = NULL);

private:
    thrust::device_vector<uint32>   m_pool;
    uint32                          m_capacity;
};

/// Implements a WorkQueue using persistent warps to fetch more work at a warp's granularity.
/// with dynamic work assignment.
/// Each work-unit is assigned to a single thread, which runs until completion of all its continuations.
/// Useful if the number of continuations is fairly random.
/// Potentially destroys intra-CTA memory coherence.
/// Very low continuation overhead.
///
/// The user of this class have to specify a WorkStream class responsible for feeding work
/// to the queue in the shape of a subclass WorkStream::WorkUnit.
/// The latter is responsible for specifying the data and execution of each unit.
/// WorkStream has to implement two methods:
///
///    uint32 size() const
///    void   get(const uint32 i, WorkUnit* unit, const uint2 queue_slot)
///
/// When the method WorkQueue::consume( stream ) is called, the queue will launch a kernel
/// to consume all WorkUnit's in the stream.
/// WorkUnit has to implement a single method:
///
///    bool WorkUnit::run(const WorkStream& context)
///
/// which should run the associated work and indicate whether the unit has finished execution,
/// or whether it has produced a continuation (stored in the WorkUnit itself), that has to be
/// run further. The WorkQueue will automatically queue the continuation for later execution.
///
/// Optionally, the class can also be passed a WorkMover which is responsible for moving
/// additional data attached to any WorkUnit. This must implement a method:
///
///    void move(
///        const WorkStream& stream,
///        const uint2 src_slot, WorkUnit* src_unit,
///        const uint2 dst_slot, WorkUnit* dst_unit) const;
///
template <
    typename WorkUnitT,
    uint32   BLOCKDIM>
struct WorkQueue<
    PersistentThreadsQueueTag,
    WorkUnitT,
    BLOCKDIM>
{
    typedef WorkUnitT   WorkUnit;

    /// constructor
    ///
    WorkQueue() : m_capacity(uint32(-1)), m_min_utilization(0.75f) {}

    /// set queue capacity
    ///
    void set_capacity(const uint32 capacity) { m_capacity = capacity; }

    /// set utilization threshold
    ///
    void set_min_utilization(const float min_utilization) { m_min_utilization = min_utilization; }

    /// consume a stream of work units
    ///
    template <typename WorkStream>
    void consume(const WorkStream stream, WorkQueueStats* stats = NULL) { consume( stream, DefaultMover(), stats ); }

    /// consume a stream of work units
    ///
    template <typename WorkStream, typename WorkMover>
    void consume(const WorkStream stream, const WorkMover mover, WorkQueueStats* stats = NULL);

private:
    thrust::device_vector<uint32> m_pool;
    uint32                        m_capacity;
    float                         m_min_utilization;
};

} // namespace cuda
} // namespace nvbio

#include <nvbio/basic/cuda/work_queue_persistent_inl.h>
