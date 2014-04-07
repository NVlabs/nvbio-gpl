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

#include <nvbio/qgram/qgram.h>
#include <nvbio/basic/types.h>
#include <nvbio/basic/numbers.h>
#include <nvbio/basic/algorithms.h>
#include <nvbio/basic/cuda/primitives.h>
#include <nvbio/basic/thrust_view.h>
#include <nvbio/basic/exceptions.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/for_each.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

namespace nvbio {

///@addtogroup QGram
///@{

///
/// This class implements a q-gram filter which can be used to find and filter matches
/// between an arbitrary set of indexed query q-grams, representing q-grams of a given
/// text, and a \ref QGramIndex "q-gram index".
/// The q-gram index can be either a simple string index or a string-set index.
///\par
/// For string q-gram indices, the filter will return an ordered set of <i>(qgram-pos,query-pos)</i>
/// pairs, where <i>qgram-pos</i> is the index of the hit into the string used to build qgram-index,
/// and <i>query-pos</i> corresponds to one of the input query q-gram indices.
///\par
/// For string-set q-gram indices, the filter will return an ordered set of <i>(string-id,query-diagonal)</i>
/// pairs, where <i>string-id</i> is the index of the hit into the string-set used to build qgram-index, and
/// <i>query-diagonal</i> corresponds to the matching diagonal of the input query text.
///
template <typename system_tag>
struct QGramFilter {};


///
/// This class implements a q-gram filter which can be used to find and filter matches
/// between an arbitrary set of indexed query q-grams, representing q-grams of a given
/// text, and a \ref QGramIndex "q-gram index".
/// The q-gram index can be either a simple string index or a string-set index.
///\par
/// For string q-gram indices, the filter will return an ordered set of <i>(qgram-pos,query-pos)</i>
/// pairs, where <i>qgram-pos</i> is the index of the hit into the string used to build qgram-index,
/// and <i>query-pos</i> corresponds to one of the input query q-gram indices.
///\par
/// For string-set q-gram indices, the filter will return an ordered set of <i>(string-id,query-diagonal)</i>
/// pairs, where <i>string-id</i> is the index of the hit into the string-set used to build qgram-index, and
/// <i>query-diagonal</i> corresponds to the matching diagonal of the input query text.
///
template <>
struct QGramFilter<host_tag>
{
    typedef host_tag            system_tag;

    /// enact the q-gram filter on a q-gram index and a set of indexed query q-grams
    ///
    /// \param qgram_index      the q-gram index
    /// \param n_queries        the number of query q-grams
    /// \param queries          the query q-grams
    /// \param indices          the query indices
    ///
    template <typename qgram_index_type, typename query_iterator, typename index_iterator>
    void enact(
        const qgram_index_type& qgram_index,
        const uint32            n_queries,
        const query_iterator    queries,
        const index_iterator    indices);

    /// return the number of matching hits
    ///
    uint32 n_hits() const { return m_output.size(); }

    /// return the output list of hits
    ///
    const uint2* hits() const { return nvbio::plain_view( m_output ); }

    // TODO: generalize to the host
    thrust::host_vector<uint2>  m_ranges;
    thrust::host_vector<uint32> m_slots;
    thrust::host_vector<uint2>  m_output;
};

////
/// This class implements a q-gram filter which can be used to find and filter matches
/// between an arbitrary set of indexed query q-grams, representing q-grams of a given
/// text, and a \ref QGramIndex "q-gram index".
/// The q-gram index can be either a simple string index or a string-set index.
///\par
/// For string q-gram indices, the filter will return an ordered set of <i>(qgram-pos,query-pos)</i>
/// pairs, where <i>qgram-pos</i> is the index of the hit into the string used to build qgram-index,
/// and <i>query-pos</i> corresponds to one of the input query q-gram indices.
///\par
/// For string-set q-gram indices, the filter will return an ordered set of <i>(string-id,query-diagonal)</i>
/// pairs, where <i>string-id</i> is the index of the hit into the string-set used to build qgram-index, and
/// <i>query-diagonal</i> corresponds to the matching diagonal of the input query text.
///
template <>
struct QGramFilter<device_tag>
{
    typedef device_tag          system_tag;

    /// enact the q-gram filter on a q-gram index and a set of indexed query q-grams
    ///
    /// \param qgram_index      the q-gram index
    /// \param n_queries        the number of query q-grams
    /// \param queries          the query q-grams
    /// \param indices          the query indices
    ///
    template <typename qgram_index_type, typename query_iterator, typename index_iterator>
    void enact(
        const qgram_index_type& qgram_index,
        const uint32            n_queries,
        const query_iterator    queries,
        const index_iterator    indices);

    /// return the number of matching hits
    ///
    uint32 n_hits() const { return m_output.size(); }

    /// return the output list of hits
    ///
    const uint2* hits() const { return nvbio::plain_view( m_output ); }

    // TODO: generalize to the host
    thrust::device_vector<uint2>  m_ranges;
    thrust::device_vector<uint32> m_slots;
    thrust::device_vector<uint2>  m_output;
    thrust::device_vector<uint8>  d_temp_storage;
};

typedef QGramFilter<host_tag>   QGramFilterHost;
typedef QGramFilter<device_tag> QGramFilterDevice;

///@} // end of the QGram group

} // namespace nvbio

#include <nvbio/qgram/filter_inl.h>
