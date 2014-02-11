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

namespace nvbio {

// constructor
//
template <typename Key, typename Container, typename Compare>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE priority_queue<Key,Container,Compare>::priority_queue(const Container cont, const Compare cmp)
    : m_size(0), m_queue(cont), m_cmp(cmp)
{
    // TODO: heapify if not empty!
    assert( m_queue.empty() == true );
}

// is queue empty?
//
template <typename Key, typename Container, typename Compare>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool priority_queue<Key,Container,Compare>::empty() const
{
    return m_size == 0;
}

// return queue size
//
template <typename Key, typename Container, typename Compare>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint32 priority_queue<Key,Container,Compare>::size() const
{
    return m_size;
}

// push an element
//
template <typename Key, typename Container, typename Compare>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void priority_queue<Key,Container,Compare>::push(const Key key)
{
    // check whether the queue is full
    m_size++;
    m_queue.resize( m_size+1 ); // we need one more entry for things to work out

	uint32 r = m_size;
	while (r > 1) // sift up new item
	{		
		const uint32 p = r/2;
        if (! m_cmp( m_queue[p], key )) // in proper order
			break;

        m_queue[r] = m_queue[p]; // else swap with parent
		r = p;
	}
    m_queue[r] = key; // insert new item at final location
}

// pop an element
//
template <typename Key, typename Container, typename Compare>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void priority_queue<Key,Container,Compare>::pop()
{
	Key dn = m_queue[m_size--];  // last item in queue
    m_queue.resize( m_size+1 );  // we need one more entry for things to work out

    uint32 p = 1;                // p points to item out of position
	uint32 r = p << 1;           // left child of p

	while (r <= m_size) // while r is still within the heap
	{
		// set r to smaller child of p
		if (r < m_size  && m_cmp( m_queue[r], m_queue[r+1] )) r++;
		if (! m_cmp( dn, m_queue[r] ))	// in proper order
			break;

		m_queue[p] = m_queue[r];    // else swap with child
		p = r;                      // advance pointers
		r = p<<1;
	}
    m_queue[p] = m_queue[m_size+1]; // insert last item in proper place
}

// top of the queue
//
template <typename Key, typename Container, typename Compare>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE Key priority_queue<Key,Container,Compare>::top() const
{
    return m_queue[1];
}

// return the i-th element in the heap
//
template <typename Key, typename Container, typename Compare>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE const Key& priority_queue<Key,Container,Compare>::operator[] (const uint32 i) const
{
    return m_queue[1+i];
}

// clear the queue
//
template <typename Key, typename Container, typename Compare>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void priority_queue<Key,Container,Compare>::clear()
{
    m_size = 0;
    m_queue.resize(0);
}

} // namespace nvbio
