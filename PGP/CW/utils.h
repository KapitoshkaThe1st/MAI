#ifndef UTILS_H
#define UTILS_H

#include "constants.h"

__host__ __device__ bool approx_equal(float a, float b, float e = eps){
    return (abs(a - b) < e);
}

__host__ __device__ float clamp(float mn, float val, float mx){
    return max(min(val, mx), mn);
}

__device__ void swap(float *a, float *b){
    float t = *a;
    *a = *b;
    *b = t;
}

#endif