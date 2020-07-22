//
//  random.metal
//  Bronze
//
//  Created by Bradley on 7/20/20.
//  Copyright © 2020 Bradley. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;


float rand(int x, int y, int z) // Apple wood number generator
{
    int seed = x + y * 57 + z * 241;
    seed= (seed<< 13) ^ seed;
    return (( 1.0 - ( (seed * (seed * seed * 15731 + 789221) + 1376312589) & 2147483647) / 1073741824.0f) + 1.0f) / 2.0f;
}
