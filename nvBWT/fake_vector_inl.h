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

namespace nvbio {


template <typename ValueType, typename StorageType>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE ValueType fake_vector<ValueType,StorageType>::get(const index_type i) const
{
    return m_stream[i];
}
template <typename ValueType, typename StorageType>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void fake_vector<ValueType,StorageType>::set(const index_type i, const ValueType v)
{
    #ifndef __CUDA_ARCH__
    if (ValueType(StorageType(v)) != v)
    {
        fprintf(stderr,"fake_vector: %lld out of storage-type range\n", int64(v));
        throw fake_vector_out_of_range();
    }
    #endif

    m_stream[i] = StorageType(v);
}

// return begin iterator
//
template <typename ValueType, typename StorageType>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename fake_vector<ValueType,StorageType>::iterator
fake_vector<ValueType,StorageType>::begin() const
{
    return iterator( m_stream, 0 );
}

/*
// dereference operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename fake_vector_iterator<Stream>::Symbol
fake_vector_iterator<Stream>::operator* () const
{
    return m_stream.get( m_index );
}
*/
// dereference operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE typename fake_vector_iterator<Stream>::reference fake_vector_iterator<Stream>::operator* () const
{
    return reference( m_stream, m_index );
}

// indexing operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE typename fake_vector_iterator<Stream>::reference fake_vector_iterator<Stream>::operator[] (const index_type i) const
{
    return reference( m_stream, m_index + i );
}

// set value
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void fake_vector_iterator<Stream>::set(const value_type s)
{
    m_stream.set( m_index, s );
}

// pre-increment operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator<Stream>& fake_vector_iterator<Stream>::operator++ ()
{
    ++m_index;
    return *this;
}

// post-increment operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator<Stream> fake_vector_iterator<Stream>::operator++ (int dummy)
{
    This r( m_stream, m_index );
    ++m_index;
    return r;
}

// pre-decrement operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator<Stream>& fake_vector_iterator<Stream>::operator-- ()
{
    --m_index;
    return *this;
}

// post-decrement operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator<Stream> fake_vector_iterator<Stream>::operator-- (int dummy)
{
    This r( m_stream, m_index );
    --m_index;
    return r;
}

// add offset
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator<Stream>& fake_vector_iterator<Stream>::operator+= (const index_type distance)
{
    m_index += distance;
    return *this;
}

// subtract offset
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator<Stream>& fake_vector_iterator<Stream>::operator-= (const index_type distance)
{
    m_index -= distance;
    return *this;
}

// add offset
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator<Stream> fake_vector_iterator<Stream>::operator+ (const index_type distance) const
{
    return This( m_stream, m_index + distance );
}

// subtract offset
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_iterator<Stream> fake_vector_iterator<Stream>::operator- (const index_type distance) const
{
    return This( m_stream, m_index - distance );
}

// difference
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename fake_vector_iterator<Stream>::index_type
fake_vector_iterator<Stream>::operator- (const fake_vector_iterator it) const
{
    return index_type( m_index - it.m_index );
}

// less than
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator< (
    const fake_vector_iterator<Stream>& it1,
    const fake_vector_iterator<Stream>& it2)
{
    return it1.m_index < it2.m_index;
}

// greater than
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator>(
    const fake_vector_iterator<Stream>& it1,
    const fake_vector_iterator<Stream>& it2)
{
    return it1.m_index > it2.m_index;
}

// less than
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator<= (
    const fake_vector_iterator<Stream>& it1,
    const fake_vector_iterator<Stream>& it2)
{
    return it1.m_index <= it2.m_index;
}

// greater than
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator>=(
    const fake_vector_iterator<Stream>& it1,
    const fake_vector_iterator<Stream>& it2)
{
    return it1.m_index >= it2.m_index;
}

// equality test
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator== (
    const fake_vector_iterator<Stream>& it1,
    const fake_vector_iterator<Stream>& it2)
{
    return /*it1.m_stream == it2.m_stream &&*/ it1.m_index == it2.m_index;
}
// inequality test
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator!= (
    const fake_vector_iterator<Stream>& it1,
    const fake_vector_iterator<Stream>& it2)
{
    return /*it1.m_stream != it2.m_stream ||*/ it1.m_index != it2.m_index;
}

// assignment operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_ref<Stream>& fake_vector_ref<Stream>::operator= (const fake_vector_ref& ref)
{
    return (*this = value_type( ref ));
}

// assignment operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_ref<Stream>& fake_vector_ref<Stream>::operator= (const value_type v)
{
    m_stream.set( m_index, v );
    return *this;
}

// assignment operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_ref<Stream>& fake_vector_ref<Stream>::operator+= (const value_type v)
{
    m_stream.set( m_index, m_stream.get( m_index ) + v );
    return *this;
}

// assignment operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_ref<Stream>& fake_vector_ref<Stream>::operator-= (const value_type v)
{
    m_stream.set( m_index, m_stream.get( m_index ) - v );
    return *this;
}

// assignment operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_ref<Stream>& fake_vector_ref<Stream>::operator++ ()
{
    m_stream.set( m_index, m_stream.get( m_index )+1 );
    return *this;
}

// assignment operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_ref<Stream>& fake_vector_ref<Stream>::operator-- ()
{
    m_stream.set( m_index, m_stream.get( m_index )-1 );
    return *this;
}

// conversion operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE fake_vector_ref<Stream>::operator value_type() const
{
    return m_stream.get( m_index );
}

} // namespace nvbio
