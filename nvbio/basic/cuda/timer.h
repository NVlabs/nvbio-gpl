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
#include <cuda_runtime.h>

namespace nvbio {
namespace cuda {

///
/// A CUDA timer
///
struct Timer
{
    /// constructor
    ///
    inline Timer();

    /// destructor
    ///
    inline ~Timer();

	/// start timing
    ///
	inline void start();

	/// stop timing
    ///
	inline void stop();

    /// elapsed seconds
    ///
	inline float seconds() const;

    cudaEvent_t m_start, m_stop;
};

///
/// A helper timer which measures the time from its instantiation
/// to the moment it goes out of scope
///
template <typename T>
struct ScopedTimer
{
	 ScopedTimer(T* time) : m_time( time ), m_timer() { m_timer.start(); }
	~ScopedTimer() { m_timer.stop(); *m_time += m_timer.seconds(); }

	T*		m_time;
	Timer	m_timer;
};

// constructor
//
Timer::Timer()
{
    cudaEventCreate(&m_start);
    cudaEventCreate(&m_stop);
}

// destructor
//
Timer::~Timer()
{
    cudaEventDestroy(m_start);
    cudaEventDestroy(m_stop);
}

// start timing
//
void Timer::start()
{
    cudaEventRecord(m_start, 0);
}

// stop timing
//
void Timer::stop()
{
    cudaEventRecord(m_stop, 0);
    cudaEventSynchronize(m_stop);
}

// elapsed seconds
//
float Timer::seconds() const
{
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, m_start, m_stop);
    return elapsedTime * 1.0e-3f;
}

} // namespace cuda
} // namespace nvbio
