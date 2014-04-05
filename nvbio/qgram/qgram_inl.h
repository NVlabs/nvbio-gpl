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
    const string_type   string,
    const uint32        qlut)
{
    thrust::device_vector<uint8> d_temp_storage;

    Q   = q;
    QL  = qlut;
    QLS = (Q - QL) * SYMBOL_SIZE;

    qgrams.resize( string_len );
    index.resize( string_len );

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
        d_all_qgrams.begin() + string_len,
        index.begin() );

    // copy only the unique q-grams and count them
    thrust::device_vector<uint32> d_counts( string_len + 1u );

    n_unique_qgrams = cuda::runlength_encode(
        string_len,
        d_all_qgrams.begin(),
        qgrams.begin(),
        d_counts.begin(),
        d_temp_storage );

    // now we know how many unique q-grams there are
    slots.resize( n_unique_qgrams + 1u );

    // scan the counts to get the slots
    cuda::exclusive_scan(
        n_unique_qgrams + 1u,
        d_counts.begin(),
        slots.begin(),
        thrust::plus<uint32>(),
        uint32(0),
        d_temp_storage );

    // shrink the q-gram vector
    qgrams.resize( n_unique_qgrams );

    const uint32 n_slots = slots[ n_unique_qgrams ];
    if (n_slots != string_len)
        throw runtime_error( "mismatching number of q-grams: inserted %u q-grams, got: %u\n" );

    //
    // build a LUT
    //

    if (QL)
    {
        const uint32 ALPHABET_SIZE = 1u << SYMBOL_SIZE;

        uint64 lut_size = 1;
        for (uint32 i = 0; i < QL; ++i)
            lut_size *= ALPHABET_SIZE;

        // build a set of spaced q-grams
        thrust::device_vector<uint64> lut_qgrams( lut_size );
        thrust::transform(
            thrust::make_counting_iterator<uint32>(0),
            thrust::make_counting_iterator<uint32>(0) + lut_size,
            lut_qgrams.begin(),
            shift_left<uint64>( QLS ) );

        // and now search them
        lut.resize( lut_size+1 );

        thrust::lower_bound(
            qgrams.begin(),
            qgrams.begin() + n_unique_qgrams,
            lut_qgrams.begin(),
            lut_qgrams.begin() + lut_size,
            lut.begin() );

        // and write a sentinel value
        lut[ lut_size ] = n_unique_qgrams;
    }
    else
        lut.resize(0);
}

} // namespace nvbio
