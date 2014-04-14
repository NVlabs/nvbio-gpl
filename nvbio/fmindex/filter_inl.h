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

namespace fmindex {

// return the size of a given range
struct range_size
{
    typedef uint2  argument_type;
    typedef uint64 result_type;

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint64 operator() (const uint2 range) const { return 1u + range.y - range.x; }
};

template <typename index_type, typename string_set_type>
struct rank_functor
{
    typedef uint32  argument_type;
    typedef uint2   result_type;

    // constructor
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    rank_functor(
        const index_type        _index,
        const string_set_type   _string_set) :
    index       ( _index ),
    string_set  ( _string_set ) {}

    // functor operator
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    result_type operator() (const argument_type string_id) const
    {
        typedef typename string_set_type::string_type   string_type;

        // fetch the given string
        const string_type string = string_set[ string_id ];

        // and match it in the FM-index
        return match( index, string, length( string ) );
    }

    const index_type        index;
    const string_set_type   string_set;
};

struct filter_results
{
    typedef uint64  argument_type;
    typedef uint2   result_type;

    // constructor
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    filter_results(
        const uint32    _n_queries,
        const uint64*   _slots,
        const uint2*    _ranges) :
    n_queries   ( _n_queries ),
    slots       ( _slots ),
    ranges      ( _ranges ) {}

    // functor operator
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    result_type operator() (const uint64 output_index) const
    {
        // find the text q-gram slot corresponding to this output index
        const uint32 slot = uint32( upper_bound(
            output_index,
            slots,
            n_queries ) - slots );

        // fetch the corresponding text position
        const uint32 string_id   = slot;

        // locate the hit position
        const uint2  range       = ranges[ slot ];
        const uint64 base_slot   = slot ? slots[ slot-1 ] : 0u;
        const uint32 local_index = output_index - base_slot;

        // and write out the pair (qgram_pos,text_pos)
        return make_uint2(  range.x + local_index, string_id );
    }

    const uint32    n_queries;
    const uint64*   slots;
    const uint2*    ranges;
};


template <typename index_type>
struct locate_results
{
    typedef uint2   argument_type;
    typedef uint2   result_type;

    // constructor
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    locate_results(const index_type _index) : index( _index ) {}

    // functor operator
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    result_type operator() (const uint2 pair) const
    {
        return make_uint2( locate( index, pair.x ), pair.y );
    }

    const index_type index;
};

} // namespace fmindex


// enact the filter on an FM-index and a string-set
//
// \param fm_index         the FM-index
// \param string-set       the query string-set
//
// \return the total number of hits
//
template <typename fm_index_type>
template <typename string_set_type>
uint64 FMIndexFilter<host_tag, fm_index_type>::rank(
    const fm_index_type&    index,
    const string_set_type&  string_set)
{
    // save the query
    m_n_queries   = string_set.size();
    m_index       = index;

    // alloc enough storage for the results
    m_ranges.resize( m_n_queries );
    m_slots.resize( m_n_queries );

    // search the strings in the index, obtaining a set of ranges
    thrust::transform(
        thrust::make_counting_iterator<uint32>(0u),
        thrust::make_counting_iterator<uint32>(0u) + m_n_queries,
        m_ranges.begin(),
        fmindex::rank_functor<fm_index_type,string_set_type>( m_index, string_set ) );

    // scan their size to determine the slots
    thrust::inclusive_scan(
        thrust::make_transform_iterator( m_ranges.begin(), fmindex::range_size() ),
        thrust::make_transform_iterator( m_ranges.begin(), fmindex::range_size() ) + m_n_queries,
        m_slots.begin() );

    // determine the total number of occurrences
    m_n_occurrences = m_slots[ m_n_queries-1 ];
    return m_n_occurrences;
}

// enumerate all hits in a given range
//
// \tparam hits_iterator         a hit_type iterator
//
template <typename fm_index_type>
template <typename hits_iterator>
void FMIndexFilter<host_tag,fm_index_type>::locate(
    const uint64    begin,
    const uint64    end,
    hits_iterator   hits)
{
    // fill the output hits with (SA,string-id) coordinates
    thrust::transform(
        thrust::make_counting_iterator<uint64>(0u) + begin,
        thrust::make_counting_iterator<uint64>(0u) + end,
        hits,
        fmindex::filter_results(
            m_n_queries,
            nvbio::plain_view( m_slots ),
            nvbio::plain_view( m_ranges ) ) );

    // and locate the SA coordinates
    thrust::transform(
        hits,
        hits + (end - begin),
        hits,
        fmindex::locate_results<fm_index_type>( m_index ) );
}

// enact the filter on an FM-index and a string-set
//
// \param fm_index         the FM-index
// \param string-set       the query string-set
//
// \return the total number of hits
//
template <typename fm_index_type>
template <typename string_set_type>
uint64 FMIndexFilter<device_tag,fm_index_type>::rank(
    const fm_index_type&    index,
    const string_set_type&  string_set)
{
    // save the query
    m_n_queries   = string_set.size();
    m_index       = index;

    // alloc enough storage for the results
    m_ranges.resize( m_n_queries );
    m_slots.resize( m_n_queries );

    // search the strings in the index, obtaining a set of ranges
    thrust::transform(
        thrust::make_counting_iterator<uint32>(0u),
        thrust::make_counting_iterator<uint32>(0u) + m_n_queries,
        m_ranges.begin(),
        fmindex::rank_functor<fm_index_type,string_set_type>( m_index, string_set ) );

    // scan their size to determine the slots
    cuda::inclusive_scan(
        m_n_queries,
        thrust::make_transform_iterator( m_ranges.begin(), fmindex::range_size() ),
        m_slots.begin(),
        thrust::plus<uint64>(),
        d_temp_storage );

    // determine the total number of occurrences
    m_n_occurrences = m_slots[ m_n_queries-1 ];
    return m_n_occurrences;
}

// enumerate all hits in a given range
//
// \tparam hits_iterator         a hit_type iterator
//
template <typename fm_index_type>
template <typename hits_iterator>
void FMIndexFilter<device_tag,fm_index_type>::locate(
    const uint64    begin,
    const uint64    end,
    hits_iterator   hits)
{
#if 0
    const uint32 n_hits = end - begin;
    const uint32 buffer_size = align<32>( n_hits );

    if (m_hits.size() < buffer_size * 2u)
    {
        m_hits.clear();
        m_hits.resize( buffer_size * 2u );
    }

    // fill the output hits with (SA,string-id) coordinates
    thrust::transform(
        thrust::make_counting_iterator<uint64>(0u) + begin,
        thrust::make_counting_iterator<uint64>(0u) + end,
        m_hits.begin(),
        fmindex::filter_results(
            m_n_queries,
            nvbio::plain_view( m_slots ),
            nvbio::plain_view( m_ranges ) ) );

    // sort by the first 8 bits of the SA coordinates
    uint64* raw_hits( (uint64*)nvbio::plain_view( m_hits ) );

    cuda::SortBuffers<uint64*> sort_buffers;
    sort_buffers.keys[0] = raw_hits;
    sort_buffers.keys[1] = raw_hits + buffer_size;

    cuda::SortEnactor sort_enactor;
    sort_enactor.sort( n_hits, sort_buffers, 0u, 8u );

    // locate the SA coordinates, in place
    thrust::transform(
        m_hits.begin() + buffer_size * sort_buffers.selector,
        m_hits.begin() + buffer_size * sort_buffers.selector + n_hits,
        m_hits.begin() + buffer_size * sort_buffers.selector,
        fmindex::locate_results<fm_index_type>( m_index ) );

    // and sort back by string-id into final position
    sort_enactor.sort( n_hits, sort_buffers, 32u, 64u );

    thrust::copy(
        m_hits.begin() + buffer_size * sort_buffers.selector,
        m_hits.begin() + buffer_size * sort_buffers.selector + n_hits,
        device_iterator( hits ) );
#else
    // fill the output hits with (SA,string-id) coordinates
    thrust::transform(
        thrust::make_counting_iterator<uint64>(0u) + begin,
        thrust::make_counting_iterator<uint64>(0u) + end,
        device_iterator( hits ),
        fmindex::filter_results(
            m_n_queries,
            nvbio::plain_view( m_slots ),
            nvbio::plain_view( m_ranges ) ) );

    // and locate the SA coordinates
    thrust::transform(
        device_iterator( hits ),
        device_iterator( hits ) + (end - begin),
        device_iterator( hits ),
        fmindex::locate_results<fm_index_type>( m_index ) );
#endif
}

} // namespace nvbio
