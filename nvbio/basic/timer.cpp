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

#include <nvbio/basic/timer.h>

#ifdef WIN32

#include <windows.h>
#include <winbase.h>

namespace nvbio {

Timer::Timer()
{
	LARGE_INTEGER freq;
	QueryPerformanceFrequency(&freq);

	m_freq = freq.QuadPart;
}

void Timer::start()
{
	LARGE_INTEGER tick;
	QueryPerformanceCounter(&tick);

	m_start = tick.QuadPart;
}
void Timer::stop()
{
	LARGE_INTEGER tick;
	QueryPerformanceCounter(&tick);

	m_stop = tick.QuadPart;
}

float Timer::seconds() const
{
	return float(double(m_stop - m_start) / double(m_freq));
}

} // namespace nvbio

#else

#include <time.h>
#include <sys/time.h>

namespace nvbio {

#if 0

void Timer::start()
{
    m_start = clock();
}
void Timer::stop()
{
    m_stop = clock();
}

float Timer::seconds() const
{
	return float(m_stop - m_start) / float(CLOCKS_PER_SEC);
}

#elif 0

void Timer::start()
{
    timespec _time;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&_time);
    m_start    = _time.tv_sec;
    m_start_ns = _time.tv_nsec;
}
void Timer::stop()
{
    timespec _time;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID,&_time);
    m_stop    = _time.tv_sec;
    m_stop_ns = _time.tv_nsec;
}

float Timer::seconds() const
{
    if (m_stop_ns < m_start_ns)
    	return float( double(m_stop - m_start - 1) + double(1000000000 + m_stop_ns - m_start_ns)*1.0e-9 );
    else
    	return float( double(m_stop - m_start) + double(m_stop_ns - m_start_ns)*1.0e-9 );
}

#else

void Timer::start()
{
    timeval _time;
    gettimeofday(&_time,NULL);
    m_start    = _time.tv_sec;
    m_start_ns = _time.tv_usec;
}
void Timer::stop()
{
    timeval _time;
    gettimeofday(&_time,NULL);
    m_stop    = _time.tv_sec;
    m_stop_ns = _time.tv_usec;
}

float Timer::seconds() const
{
    if (m_stop_ns < m_start_ns)
    	return float( double(m_stop - m_start - 1) + double(1000000 + m_stop_ns - m_start_ns)*1.0e-6 );
    else
    	return float( double(m_stop - m_start) + double(m_stop_ns - m_start_ns)*1.0e-6 );
}

#endif

} // namespace nvbio

#endif
