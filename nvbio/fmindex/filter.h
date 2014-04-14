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

#include <nvbio/fmindex/fmindex.h>
#include <nvbio/basic/types.h>
#include <nvbio/basic/numbers.h>
#include <nvbio/basic/algorithms.h>
#include <nvbio/basic/exceptions.h>
#include <nvbio/basic/vector.h>
#include <nvbio/basic/cuda/sort.h>
#include <nvbio/basic/cuda/primitives.h>
#include <nvbio/strings/string.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

namespace nvbio {

///@addtogroup FMIndex
///@{

///
/// This class implements a FM-index filter which can be used to find and filter matches
/// between an arbitrary string set, and a \ref FMIndex "FM-index".
///\par
/// The filter will return an ordered set of <i>(index-pos,string-id)</i> pairs, where <i>string-id</i> is
/// the index into the string-set and <i>index-pos</i> is an index into the FM-index.
///\par
///
/// \tparam fm_index_type    the type of the fm-index
///
template <typename system_tag, typename fm_index_type>
struct FMIndexFilter {};

///
/// This class implements a FM-index filter which can be used to find and filter matches
/// between an arbitrary string set, and a \ref FMIndex "FM-index".
///\par
/// The filter will return an ordered set of <i>(index-pos,string-id)</i> pairs, where <i>string-id</i> is
/// the index into the string-set and <i>index-pos</i> is an index into the FM-index.
///\par
///
/// \tparam fm_index_type    the type of the fm-index
///
template <typename fm_index_type>
struct FMIndexFilter<host_tag, fm_index_type>
{
    typedef host_tag                                        system_tag;     ///< the backend system
    typedef fm_index_type                                   index_type;     ///< the index type

    //typedef typename index_type::coord_type               coord_type;     ///< the coordinate type of the fm-index, uint32|uint2
    typedef uint32                                          coord_type;     ///< the coordinate type of the fm-index, uint32|uint2
    static const uint32                                     coord_dim = vector_traits<coord_type>::DIM;

    static const uint32                                     hit_dim = coord_dim*2;  ///< hits are either uint2 or uint4
    typedef typename vector_type<coord_type,hit_dim>::type  hit_type;               ///< hits are either uint2 or uint4

    /// enact the filter on an FM-index and a string-set
    ///
    /// \param index            the FM-index
    /// \param string-set       the query string-set
    ///
    /// \return the total number of hits
    ///
    template <typename string_set_type>
    uint64 rank(
        const fm_index_type&    index,
        const string_set_type&  string_set);

    /// enumerate all hits in a given range
    ///
    /// \tparam hits_iterator         a hit_type iterator
    ///
    /// \param begin                  the beginning of the hits sequence to locate, in [0,n_hits)
    /// \param end                    the end of the hits sequence to locate, in [0,n_hits]
    ///
    template <typename hits_iterator>
    void locate(
        const uint64    begin,
        const uint64    end,
        hits_iterator   hits);

    /// return the number of hits from the last rank query
    ///
    uint64 n_hits() const { return m_n_occurrences; }

    /// return the individual ranges of the ranked queries
    ///
    const uint2* ranges() const { return nvbio::plain_view( m_ranges ); }

    /// return the global ranks of the output hits (i.e. the range <i>[ranks[i], ranks[i+1])</i>
    /// identifies the position of the hits corresponding to the i-th query in the locate output)
    ///
    const uint64* ranks() const { return nvbio::plain_view( m_slots ); }

    uint32                              m_n_queries;
    index_type                          m_index;
    uint64                              m_n_occurrences;
    thrust::host_vector<uint2>          m_ranges;
    thrust::host_vector<uint64>         m_slots;
};

///
/// This class implements a FM-index filter which can be used to find and filter matches
/// between an arbitrary string set, and a \ref FMIndex "FM-index".
///\par
/// The filter will return an ordered set of <i>(index-pos,string-id)</i> pairs, where <i>string-id</i> is
/// the index into the string-set and <i>index-pos</i> is an index into the FM-index.
///\par
///
/// \tparam fm_index_type    the type of the fm-index
///
template <typename fm_index_type>
struct FMIndexFilter<device_tag, fm_index_type>
{
    typedef device_tag                                      system_tag;     ///< the backend system
    typedef fm_index_type                                   index_type;     ///< the index type

    //typedef typename index_type::coord_type               coord_type;     ///< the coordinate type of the fm-index, uint32|uint2
    typedef uint32                                          coord_type;     ///< the coordinate type of the fm-index, uint32|uint2
    static const uint32                                     coord_dim = vector_traits<coord_type>::DIM;

    static const uint32                                     hit_dim = coord_dim*2;  ///< hits are either uint2 or uint4
    typedef typename vector_type<coord_type,hit_dim>::type  hit_type;               ///< hits are either uint2 or uint4

    /// enact thefilter on an FM-index and a string-set
    ///
    /// \param index            the FM-index
    /// \param string-set       the query string-set
    ///
    /// \return the total number of hits
    ///
    template <typename string_set_type>
    uint64 rank(
        const fm_index_type&    index,
        const string_set_type&  string_set);

    /// enumerate all hits in a given range
    ///
    /// \tparam hits_iterator         a hit_type iterator
    ///
    /// \param begin                  the beginning of the hits sequence to locate, in [0,n_hits)
    /// \param end                    the end of the hits sequence to locate, in [0,n_hits]
    ///
    template <typename hits_iterator>
    void locate(
        const uint64    begin,
        const uint64    end,
        hits_iterator   hits);

    /// return the number of hits from the last rank query
    ///
    uint64 n_hits() const { return m_n_occurrences; }

    /// return the individual ranges of the ranked queries
    ///
    const uint2* ranges() const { return nvbio::plain_view( m_ranges ); }

    /// return the global ranks of the output hits (i.e. the range <i>[ranks[i], ranks[i+1])</i>
    /// identifies the position of the hits corresponding to the i-th query in the locate output)
    ///
    const uint64* ranks() const { return nvbio::plain_view( m_slots ); }

    uint32                              m_n_queries;
    index_type                          m_index;
    uint64                              m_n_occurrences;
    thrust::device_vector<uint2>        m_ranges;
    thrust::device_vector<uint64>       m_slots;
    thrust::device_vector<uint2>        m_hits;
    thrust::device_vector<uint8>        d_temp_storage;
};

template <typename fm_index_type>
struct FMIndexFilterHost : public FMIndexFilter<host_tag, fm_index_type> {};

template <typename fm_index_type>
struct FMIndexFilterDevice : public FMIndexFilter<device_tag, fm_index_type> {};

///@} // end of the FMIndex group

} // namespace nvbio

#include <nvbio/fmindex/filter_inl.h>
