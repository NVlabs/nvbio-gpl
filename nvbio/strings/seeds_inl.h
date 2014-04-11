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

template <typename in_coord_type, typename out_coord_type>
struct project_coords_functor {};

template <typename in_coord_type>
struct project_coords_functor<in_coord_type,in_coord_type>
{
    typedef in_coord_type argument_type;
    typedef in_coord_type result_type;

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    result_type operator() (const argument_type i) const { return i; }
};
template <typename in_coord_type>
struct project_coords_functor<in_coord_type,uint32>
{
    typedef in_coord_type argument_type;
    typedef uint32        result_type;

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    result_type operator() (const argument_type i) const { return i.x; }
};
template <typename in_coord_type>
struct project_coords_functor<in_coord_type,uint2>
{
    typedef in_coord_type argument_type;
    typedef uint2         result_type;

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    result_type operator() (const argument_type i) const { return make_uint2( i.x, i.y ); }
};
template <typename in_coord_type>
struct project_coords_functor<in_coord_type,uint3>
{
    typedef in_coord_type argument_type;
    typedef uint3         result_type;

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    result_type operator() (const argument_type i) const { return make_uint3( i.x, i.y, i.z ); }
};

/// project a given set of coordinates to a lower-dimensional object
///
template <typename out_coord_type, typename in_coord_type>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
out_coord_type project_coords(const in_coord_type i)
{
    const project_coords_functor<in_coord_type,out_coord_type> p;
    return p(i);
}

// A functor to return the coordinates given by a seed_functor
//
template <typename seed_functor, typename coord_type>
struct string_seed_functor
{
    typedef uint32      argument_type;
    typedef coord_type  result_type;

    // constructor
    //
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    string_seed_functor(const uint32 _string_len, const seed_functor _seeder) :
        string_len(_string_len), seeder(_seeder) {}

    // return the coordinate of the i-th seed
    //
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    coord_type operator() (const uint32 idx) const
    {
        return project_coords<coord_type>( seeder.seed( string_len, idx ) );
    }

    const uint32            string_len;
    const seed_functor      seeder;
};

// A functor to return the localized coordinates given by a seed_functor
//
template <typename string_set_type, typename seed_functor, typename coord_type>
struct localized_seed_functor
{
    typedef uint32      argument_type;
    typedef coord_type  result_type;

    // constructor
    //
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    localized_seed_functor(const string_set_type _string_set, const seed_functor _seeder, const uint32* _cum_seeds) :
        string_set(_string_set), seeder(_seeder), cum_seeds(_cum_seeds) {}

    // return the localized coordinate of the i-th seed
    //
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    coord_type operator() (const uint32 global_idx) const
    {
        // compute the string index
        const uint32 string_id = uint32( upper_bound( global_idx, cum_seeds, string_set.size() ) - cum_seeds );

        // fetch the string length
        const uint32 string_len = string_set[ string_id ].length();

        // compute the local string coordinate
        const uint32 base_offset = string_id ? cum_seeds[ string_id-1 ] : 0u;
        const uint32 seed_idx    = global_idx - base_offset;

        const uint2 seed = seeder.seed( string_len, seed_idx );
        return project_coords<coord_type>( make_uint4( string_id, seed.x, seed.y, 0u ) );
    }

    const string_set_type   string_set;
    const seed_functor      seeder;
    const uint32*           cum_seeds;
};

// extract a set of seed coordinates out of a string, according to a given seeding functor
//
template <typename seed_functor, typename index_vector_type>
uint32 enumerate_string_seeds(
    const uint32                string_len,
    const seed_functor          seeder,
          index_vector_type&    indices)
{
    typedef typename index_vector_type::value_type   coord_type;

    // fetch the total number of output q-grams
    const uint32 n_seeds = seeder( string_len );

    // reserve enough storage
    indices.resize( n_seeds );

    // build the list of q-gram indices
    thrust::transform(
        thrust::make_counting_iterator<uint32>(0u),
        thrust::make_counting_iterator<uint32>(0u) + n_seeds,
        indices.begin(),
        string_seed_functor<seed_functor,coord_type>( string_len, seeder ) );

    return n_seeds;
}

// extract a set of seed coordinates out of a string-set, according to a given seeding functor
//
template <typename string_set_type, typename seed_functor, typename index_vector_type>
uint32 enumerate_string_set_seeds(
    const string_set_type       string_set,
    const seed_functor          seeder,
          index_vector_type&    indices)
{
    // TODO: use some vector traits...
    typedef typename index_vector_type::system_tag   system_tag;
    typedef typename index_vector_type::value_type   coord_type;

    const uint32 n_strings = string_set.size();

    nvbio::vector<system_tag,uint32> cum_seeds( n_strings );

    // scan the number of q-grams produced per string
    thrust::inclusive_scan(
        thrust::make_transform_iterator(
            thrust::make_transform_iterator( thrust::make_counting_iterator<uint32>(0u), string_set_length_functor<string_set_type>( string_set ) ),
            seeder ),
        thrust::make_transform_iterator(
            thrust::make_transform_iterator( thrust::make_counting_iterator<uint32>(0u), string_set_length_functor<string_set_type>( string_set ) ),
            seeder ) + n_strings,
        cum_seeds.begin() );

    // fetch the total nunber of q-grams to output
    const uint32 n_seeds = cum_seeds[ n_strings-1 ];

    // reserve enough storage
    indices.resize( n_seeds );

    // build the list of q-gram indices
    thrust::transform(
        thrust::make_counting_iterator<uint32>(0u),
        thrust::make_counting_iterator<uint32>(0u) + n_seeds,
        indices.begin(),
        localized_seed_functor<string_set_type,seed_functor,coord_type>( string_set, seeder, nvbio::plain_view( cum_seeds ) ) );

    return n_seeds;
}

} // namespace nvbio
