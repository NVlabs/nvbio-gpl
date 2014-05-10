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
#include <nvbio/basic/console.h>
#include <nvbio/basic/exceptions.h>
#include <cuda_runtime.h>
#include <thrust/version.h>

// used for thrust_copy_dtoh only
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace nvbio {
namespace cuda {

struct Arch
{
    static const uint32 LOG_WARP_SIZE = 5;
    static const uint32 WARP_SIZE     = 1u << LOG_WARP_SIZE;
};

// granularity of shared memory allocation
inline void device_arch(uint32& major, uint32& minor);

// granularity of shared memory allocation
inline size_t smem_allocation_unit(const cudaDeviceProp& properties);

// granularity of register allocation
inline size_t reg_allocation_unit(const cudaDeviceProp& properties, const size_t regsPerThread);

// granularity of warp allocation
inline size_t warp_allocation_multiple(const cudaDeviceProp& properties);

// number of "sides" into which the multiprocessor is partitioned
inline size_t num_sides_per_multiprocessor(const cudaDeviceProp& properties);

// maximum number of blocks per multiprocessor
inline size_t max_blocks_per_multiprocessor(const cudaDeviceProp& properties);

// number of registers allocated per block
inline size_t num_regs_per_block(const cudaDeviceProp& properties, const cudaFuncAttributes& attributes, const size_t CTA_SIZE);

template <typename KernelFunction>
inline cudaFuncAttributes function_attributes(KernelFunction kernel);

template <typename KernelFunction>
size_t max_active_blocks(KernelFunction kernel, const size_t CTA_SIZE, const size_t dynamic_smem_bytes);

template <typename KernelFunction>
size_t num_registers(KernelFunction kernel);

template <typename KernelFunction>
size_t max_blocksize_with_highest_occupancy(KernelFunction kernel, size_t dynamic_smem_bytes_per_thread);

inline bool is_tcc_enabled();

inline void check_error(const char *message);

/// a generic syncthreads() implementation to synchronize contiguous
/// blocks of N threads at a time
///
template <uint32 N>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
void syncthreads();

// utility function to copy a thrust device vector to a thrust host vector
// the sole reason for this is to eliminate warnings from thrust when using the assignment operator
template<typename TTargetVector, typename TSourceVector>
static NVBIO_FORCEINLINE void thrust_copy_vector(TTargetVector& target, TSourceVector& source)
{
    if (target.size() != source.size())
    {
        target.clear();
        target.resize(source.size());
    }

    thrust::copy(source.begin(), source.end(), target.begin());
}

template<typename TTargetVector, typename TSourceVector>
static NVBIO_FORCEINLINE void thrust_copy_vector(TTargetVector& target, TSourceVector& source, uint32 count)
{
    if (target.size() != count)
    {
        target.clear();
        target.resize(count);
    }

    thrust::copy(source.begin(), source.begin() + count, target.begin());
}

} // namespace cuda
} // namespace nvbio

#include <nvbio/basic/cuda/arch_inl.h>
