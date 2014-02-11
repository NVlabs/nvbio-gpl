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

// sum_tree_test.cpp
//

#include <nvbio/basic/sum_tree.h>
#include <nvbio/basic/console.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

namespace nvbio {

int sum_tree_test()
{
    printf("sum tree... started\n");
    // 128 leaves
    {
        const uint32 n_leaves = 128;

        std::vector<int32> vec( SumTree<uint32*>::node_count( n_leaves ) );

        SumTree<int32*> sum_tree( n_leaves, &vec[0] );

        // test 1
        for (uint32 i = 0; i < n_leaves; ++i)
            vec[i] = i;
        {
            sum_tree.setup();

            uint32 c0 = sample( sum_tree, 0.0f );
            uint32 c1 = sample( sum_tree, 0.5f );
            uint32 c2 = sample( sum_tree, 1.0f );

            if (c0 != 1 || c1 != 90 || c2 != 127)
            {
                log_error( stderr, "error in test(1):\n  c(0.0) = %u (!= 1)\n  c(0.5) = %u (!= 90)  c(1.0) = %u (!= 127)\n", c0, c1, c2 );
                exit(1);
            }
        }

        // test 2
        for (uint32 i = 0; i < n_leaves; ++i)
            vec[i] = i < n_leaves/2 ? 0 : 1;
        {
            sum_tree.setup();

            uint32 c0 = sample( sum_tree, 0.0f );
            uint32 c1 = sample( sum_tree, 0.5f );
            uint32 c2 = sample( sum_tree, 1.0f );

            if (c0 != 64 || c1 != 96 || c2 != 127)
            {
                log_error( stderr, "error in test(2):\n  c(0.0) = %u (!= 64)\n  c(0.5) = %u (!= 90)  c(1.0) = %u (!= 127)\n", c0, c1, c2 );
                return 1;
            }
        }

        // test 3
        for (uint32 i = 0; i < n_leaves; ++i)
            vec[i] = i < n_leaves/2 ? 1 : 0;
        {
            sum_tree.setup();

            uint32 c0 = sample( sum_tree, 0.0f );
            uint32 c1 = sample( sum_tree, 0.5f );
            uint32 c2 = sample( sum_tree, 1.0f );

            if (c0 != 0 || c1 != 32 || c2 != 63)
            {
                log_error( stderr, "error in test(3):\n  c(0.0) = %u (!= 0)\n  c(0.5) = %u (!= 32)  c(1.0) = %u (!= 63)\n", c0, c1, c2 );
                exit(1);
            }
        }
    }
    // 80 leaves
    {
        const uint32 n_leaves = 80;

        std::vector<int32> vec( SumTree<uint32*>::node_count( n_leaves ) );

        SumTree<int32*> sum_tree( n_leaves, &vec[0] );

        if (sum_tree.padded_size() != 128)
        {
            log_error( stderr, "error: wrong padded size: %u != 128\n", sum_tree.padded_size() );
            exit(1);
        }

        // test 4
        for (uint32 i = 0; i < n_leaves; ++i)
            vec[i] = i < n_leaves/2 ? 0 : 1;
        {
            sum_tree.setup();

            uint32 c0 = sample( sum_tree, 0.0f );
            uint32 c1 = sample( sum_tree, 0.5f );
            uint32 c2 = sample( sum_tree, 1.0f );

            if (c0 != 40 || c1 != 60 || c2 != 79)
            {
                log_error( stderr, "error in test(3):\n  c(0.0) = %u (!= 40)\n  c(0.5) = %u (!= 60)  c(1.0) = %u (!= 79)\n", c0, c1, c2 );
                exit(1);
            }
        }

        // test 5
        for (uint32 i = 0; i < n_leaves; ++i)
            vec[i] = i < n_leaves/2 ? 1 : 0;
        {
            sum_tree.setup();

            uint32 c0 = sample( sum_tree, 0.0f );
            uint32 c1 = sample( sum_tree, 0.5f );
            uint32 c2 = sample( sum_tree, 1.0f );

            if (c0 != 0 || c1 != 20 || c2 != 39)
            {
                log_error( stderr, "error in test(5):\n  c(0.0) = %u (!= 0)\n  c(0.5) = %u (!= 20)  c(1.0) = %u (!= 39)\n", c0, c1, c2 );
                exit(1);
            }
        }

        // remove the last leaf
        sum_tree.add( 39, -1 );

        // test 6
        uint32 c = sample( sum_tree, 1.0f );
        if (c != 38)
        {
            log_error( stderr, "error in test(6):\n  c(1.0) = %u (!= 38)\n", c );
            exit(1);
        }

        // remove one more leaf
        sum_tree.add( 38, -1 );

        c = sample( sum_tree, 1.0f );
        if (c != 37)
        {
            log_error( stderr, "error in test(7):\n  c(1.0) = %u (!= 37)\n", c );
            exit(1);
        }

        // add it back
        sum_tree.add( 38, 1 );

        // and remove it using set
        sum_tree.set( 38, 0 );

        c = sample( sum_tree, 1.0f );
        if (c != 37)
        {
            log_error( stderr, "error in test(8):\n  c(1.0) = %u (!= 37)\n", c );
            exit(1);
        }

        // remove the first ten leaves
        for (uint32 i = 0; i < 10; ++i)
        {
            sum_tree.set( i, 0 );

            c = sample( sum_tree, 0.0f );
            if (c != i+1)
            {
                log_error( stderr, "error in test(9):\n  c(0.0) = %u (!= %u)\n", c, i );
                exit(1);
            }
        }
    }
    printf("sum tree... done\n");

    return 0;
}

}