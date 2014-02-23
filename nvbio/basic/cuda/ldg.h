/*
 * nvbio
 * Copyright (C) 2011-2013, NVIDIA Corporation
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

#if defined(__CUDACC__)
#include <cub/cub.cuh>
#endif

#include <nvbio/basic/types.h>
#include <nvbio/basic/iterator.h>

namespace nvbio {
namespace cuda {

///
/// Wrapper class to create an __ldg() iterator out of a raw pointer
///
template <typename T>
struct ldg_pointer
{
    typedef T                                                           value_type;
    typedef value_type                                                  reference;
    typedef value_type                                                  const_reference;
    typedef value_type*                                                 pointer;
    typedef typename std::iterator_traits<const T*>::difference_type    difference_type;
    typedef std::random_access_iterator_tag                             iterator_category;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    ldg_pointer() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    ldg_pointer(const T* base) : m_base( base ) {}

    /// copy constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    ldg_pointer(const ldg_pointer& it) : m_base( it.m_base ) {}

    /// const indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    value_type operator[](const uint32 i) const
    {
      #if __CUDA_ARCH__ >= 350
        return __ldg( m_base + i );
      #else
        return m_base[i];
      #endif
    }

    /// dereference operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    value_type operator*() const
    {
      #if __CUDA_ARCH__ >= 350
        return __ldg( m_base );
      #else
        return *m_base;
      #endif
    }

    /// pre-increment
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    ldg_pointer<T>& operator++()
    {
        ++m_base;
        return *this;
    }

    /// post-increment
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    ldg_pointer<T> operator++(int i)
    {
        ldg_pointer<T> r( m_base );
        ++m_base;
        return r;
    }

    /// pre-decrement
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    ldg_pointer<T>& operator--()
    {
        --m_base;
        return *this;
    }

    /// post-decrement
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    ldg_pointer<T> operator--(int i)
    {
        ldg_pointer<T> r( m_base );
        --m_base;
        return r;
    }

    /// addition
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    ldg_pointer<T> operator+(const difference_type i) const
    {
        return ldg_pointer( m_base + i );
    }

    /// subtraction
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    ldg_pointer<T> operator-(const difference_type i) const
    {
        return ldg_pointer( m_base - i );
    }

    /// addition
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    ldg_pointer<T>& operator+=(const difference_type i)
    {
        m_base += i;
        return *this;
    }

    /// subtraction
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    ldg_pointer<T>& operator-=(const difference_type i)
    {
        m_base -= i;
        return *this;
    }

    /// iterator subtraction
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    difference_type operator-(const ldg_pointer<T> it) const
    {
        return m_base - it.m_base;
    }

    /// assignment
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    ldg_pointer& operator=(const ldg_pointer<T>& it)
    {
        m_base = it.m_base;
        return *this;
    }

    const T* m_base;
};

/// make a ldg_pointer
///
template <typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
ldg_pointer<T> make_ldg_pointer(const T* it)
{
    return ldg_pointer<T>( it );
}


/**
 * \brief Enumeration of cache modifiers for memory load operations.
 */
enum CacheLoadModifier
{
    LOAD_DEFAULT,       ///< Default (no modifier)
    LOAD_CA,            ///< Cache at all levels
    LOAD_CG,            ///< Cache at global level
    LOAD_CS,            ///< Cache streaming (likely to be accessed once)
    LOAD_CV,            ///< Cache as volatile (including cached system lines)
    LOAD_LDG,           ///< Cache as texture
    LOAD_VOLATILE,      ///< Volatile (any memory space)
};

#if defined(__CUDACC__)
template <CacheLoadModifier MOD> struct cub_load_mod {};

template <> struct cub_load_mod<LOAD_DEFAULT> { static const cub::CacheLoadModifier MOD = cub::LOAD_DEFAULT; };
template <> struct cub_load_mod<LOAD_CA>      { static const cub::CacheLoadModifier MOD = cub::LOAD_CA;      };
template <> struct cub_load_mod<LOAD_CG>      { static const cub::CacheLoadModifier MOD = cub::LOAD_CG;      };
template <> struct cub_load_mod<LOAD_CS>      { static const cub::CacheLoadModifier MOD = cub::LOAD_CS;      };
template <> struct cub_load_mod<LOAD_CV>      { static const cub::CacheLoadModifier MOD = cub::LOAD_CV;      };
template <> struct cub_load_mod<LOAD_LDG>     { static const cub::CacheLoadModifier MOD = cub::LOAD_LDG;     };
#endif

///
/// Wrapper class to create a cub::ThreadLoad iterator out of a raw pointer
///
template <typename T, CacheLoadModifier MOD>
struct load_pointer
{
    typedef T                                                           value_type;
    typedef value_type                                                  reference;
    typedef value_type                                                  const_reference;
    typedef value_type*                                                 pointer;
    typedef typename std::iterator_traits<const T*>::difference_type    difference_type;
    typedef std::random_access_iterator_tag                             iterator_category;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    load_pointer() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    load_pointer(const T* base) : m_base( base ) {}

    /// copy constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    load_pointer(const load_pointer& it) : m_base( it.m_base ) {}

    /// const indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    value_type operator[](const uint32 i) const
    {
      #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
        return cub::ThreadLoad<cub_load_mod<MOD>::MOD>( const_cast<T*>(m_base + i) );
      #else
        return m_base[i];
      #endif
    }

    /// dereference operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    value_type operator*() const
    {
      #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
        return cub::ThreadLoad<cub_load_mod<MOD>::MOD>( const_cast<T*>(m_base) );
      #else
        return *m_base;
      #endif
    }

    /// pre-increment
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    load_pointer<T,MOD>& operator++()
    {
        ++m_base;
        return *this;
    }

    /// post-increment
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    load_pointer<T,MOD> operator++(int i)
    {
        load_pointer<T,MOD> r( m_base );
        ++m_base;
        return r;
    }

    /// pre-decrement
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    load_pointer<T,MOD>& operator--()
    {
        --m_base;
        return *this;
    }

    /// post-decrement
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    load_pointer<T,MOD> operator--(int i)
    {
        load_pointer<T,MOD> r( m_base );
        --m_base;
        return r;
    }

    /// addition
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    load_pointer<T,MOD> operator+(const difference_type i) const
    {
        return load_pointer( m_base + i );
    }

    /// subtraction
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    load_pointer<T,MOD> operator-(const difference_type i) const
    {
        return load_pointer( m_base - i );
    }

    /// addition
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    load_pointer<T,MOD>& operator+=(const difference_type i)
    {
        m_base += i;
        return *this;
    }

    /// subtraction
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    load_pointer<T,MOD>& operator-=(const difference_type i)
    {
        m_base -= i;
        return *this;
    }

    /// iterator subtraction
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    difference_type operator-(const load_pointer<T,MOD> it) const
    {
        return m_base - it.m_base;
    }

    /// assignment
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    load_pointer& operator=(const load_pointer<T,MOD>& it)
    {
        m_base = it.m_base;
        return *this;
    }

    const T* m_base;
};

/// make a load_pointer
///
template <CacheLoadModifier MOD, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
load_pointer<T,MOD> make_load_pointer(const T* it)
{
    return load_pointer<T,MOD>( it );
}

} // namespace cuda
} // namespace nvbio
