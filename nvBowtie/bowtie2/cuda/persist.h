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
#include <nvBowtie/bowtie2/cuda/scoring_queues.h>
#include <nvbio/basic/types.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <string>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

struct SeedHit;
struct HitQueue;
struct SeedHitDequeArray;

// clear the persisting files
//
void persist_clear(const std::string& file_name);

// persist a set of hits
//
void persist_hits(
    const std::string&                              file_name,
    const char*                                     name,
    const uint32                                    anchor,
    const uint32                                    count,
    const SeedHitDequeArray&                        hit_deques);

// persist a set of reads
//
void persist_reads(
    const std::string&                              file_name,
    const char*                                     name,
    const uint32                                    anchor,
    const uint32                                    count,
    const thrust::device_vector<uint32>::iterator   iterator);

// persist a set of selected hits
//
void persist_selection(
    const std::string&                              file_name,
    const char*                                     name,
    const uint32                                    anchor,
    const uint32                                    read_count,
    const packed_read*                              read_infos_dptr,
    const uint32                                    n_multi,
    const uint32                                    hits_queue_size,
    const ReadHitsIndex&                            hits_index,
    const HitQueues&                                hits_queue);

// persist a set of scores
//
void persist_scores(
    const std::string&                              file_name,
    const char*                                     name,
    const uint32                                    anchor,
    const uint32                                    read_count,
    const uint32                                    n_multi,
    const uint32                                    hits_queue_size,
    const ScoringQueues&                            scoring_queues);

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio
