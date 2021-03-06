#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <stdio.h>

#include "error.h"

#ifdef BENCHMARK

#define MEASURE(KERNEL)                                              \
    {                                                                \
        cudaEvent_t start, end;                                      \
        CHECK_CUDA_CALL_ERROR(cudaEventCreate(&start));              \
        CHECK_CUDA_CALL_ERROR(cudaEventCreate(&end));                \
        CHECK_CUDA_CALL_ERROR(cudaEventRecord(start));               \
        KERNEL;                                                      \
        CHECK_CUDA_CALL_ERROR(cudaGetLastError());                   \
        CHECK_CUDA_CALL_ERROR(cudaEventRecord(end));                 \
        CHECK_CUDA_CALL_ERROR(cudaEventSynchronize(end));            \
        float t;                                                     \
        CHECK_CUDA_CALL_ERROR(cudaEventElapsedTime(&t, start, end)); \
        CHECK_CUDA_CALL_ERROR(cudaEventDestroy(start));              \
        CHECK_CUDA_CALL_ERROR(cudaEventDestroy(end));                \
        printf("%f\n", t);                                           \
    }

// fprintf(stderr, "Kernel time (milliseconds): %f\n", t);                                  x
#else

#define MEASURE(KERNEL) (KERNEL)

#endif

#endif