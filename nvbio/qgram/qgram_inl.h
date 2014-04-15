/*
 * nvbio
 * Copyright (C) 2011-2014, NVIDIA Corporation
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

// build a q-group index from a given string
//
// \param q                the q parameter
// \param string_len       the size of the string
// \param string           the string iterator
//
template <uint32 SYMBOL_SIZE, typename string_type>
void QGramIndexDevice::build(
    const uint32        q,
    const uint32        string_len,
    const string_type   string)
{
    thrust::device_vector<uint8> d_temp_storage;

    Q = q;

    const uint32 ALPHABET_SIZE = 1u << SYMBOL_SIZE;

    uint64 n_qgrams = 1;
    for (uint32 i = 0; i < q; ++i)
        n_qgrams *= ALPHABET_SIZE;

    qgrams.resize( string_len );
    index.resize( string_len );
    slots.resize( string_len + 1u );

    thrust::device_vector<uint32> d_all_qgrams( string_len );

    // build the list of q-grams
    thrust::transform(
        thrust::make_counting_iterator<uint32>(0u),
        thrust::make_counting_iterator<uint32>(0u) + string_len,
        d_all_qgrams.begin(),
        string_qgram_functor<SYMBOL_SIZE,string_type>( Q, string_len, string ) );

    // build the list of q-gram indices
    thrust::copy(
        thrust::make_counting_iterator<uint32>(0u),
        thrust::make_counting_iterator<uint32>(0u) + string_len,
        index.begin() );

    // sort them
    thrust::sort_by_key(
        d_all_qgrams.begin(),
        index.begin() );

    // copy only the unique q-grams and count them
    n_unique_qgrams = cuda::runlength_encode(
        string_len,
        d_all_qgrams.begin(),
        qgrams.begin(),
        slots.begin(),
        d_temp_storage );

    // scan their counts
    cuda::exclusive_scan(
        n_unique_qgrams + 1u,
        slots.begin(),
        slots.begin(),
        thrust::plus<uint32>(),
        d_temp_storage );
}

} // namespace nvbio
