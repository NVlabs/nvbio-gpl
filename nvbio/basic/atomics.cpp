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

#include <nvbio/basic/atomics.h>

#ifdef WIN32

#include <windows.h>

namespace nvbio {

int32 atomic_increment(int32 volatile *value) { return InterlockedIncrement(reinterpret_cast<LONG volatile*>(value)); }
int64 atomic_increment(int64 volatile *value) { return InterlockedIncrement64(value); }

int32 atomic_decrement(int32 volatile *value) { return InterlockedDecrement(reinterpret_cast<LONG volatile*>(value)); }
int64 atomic_decrement(int64 volatile *value) { return InterlockedDecrement64(value); }

} // namespace nvbio

#endif
