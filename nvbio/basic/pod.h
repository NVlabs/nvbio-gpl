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

namespace nvbio {

template <typename T>
struct pod_type { typedef T type; };

template <typename T>
NVBIO_FORCEINLINE NVBIO_DEVICE void write(T* ptr, const T& e)
{
    *(typename pod_type<T>::type*)ptr = (const typename pod_type<T>::type&)e;
}
template <typename T>
NVBIO_FORCEINLINE NVBIO_DEVICE T read(const T* ptr)
{
    return binary_cast<T>( *(const typename pod_type<T>::type*)ptr );
}

template <typename T>
struct pod_writer
{
    NVBIO_FORCEINLINE NVBIO_DEVICE 
    pod_writer() {}

    NVBIO_FORCEINLINE NVBIO_DEVICE 
    pod_writer(T& e) : ptr(&e) {}

    NVBIO_FORCEINLINE NVBIO_DEVICE 
    void operator=(const T& e) { write( ptr, e ); }

    T* ptr;
};

template <typename T>
struct pod_reader
{
    NVBIO_FORCEINLINE NVBIO_DEVICE 
    pod_reader() {}

    NVBIO_FORCEINLINE NVBIO_DEVICE 
    pod_reader(const T& e) : ptr(&e) {}

    NVBIO_FORCEINLINE NVBIO_DEVICE 
    operator T() { return read( ptr ); }

    const T* ptr;
};

} // namespace nvbio
