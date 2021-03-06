#include <stdio.h>
#include <stdlib.h>

#include "error.h"
#include "benchmark.h"

#ifndef BENCHMARK

#define GRID (1024)
#define BLOCK (256)

#endif

__global__ void kernel(float *dev_vec, int n){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    for(int i = idx; i < n; i += offset){
        dev_vec[i] = fabsf(dev_vec[i]);
    }
}

int main() {
    init_error_handling();

    int n = 0;
    scanf("%d", &n);

    if(n == 0)
        return 0;

    int size = sizeof(float) * n;
    float *vec = (float*)malloc(size);

    for(int i = 0; i < n; ++i)
        scanf("%f", &vec[i]);
    
    float *dev_vec;
    CHECK_CUDA_CALL_ERROR(cudaMalloc(&dev_vec, size));

    CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_vec, vec, size, cudaMemcpyHostToDevice));

    MEASURE((kernel<<<GRID, BLOCK>>>(dev_vec, n)));
    CHECK_CUDA_KERNEL_ERROR();

    CHECK_CUDA_CALL_ERROR(cudaMemcpy(vec, dev_vec, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL_ERROR(cudaFree(dev_vec));

    for(int i = 0; i < n; ++i)
        printf("%10.10e ", vec[i]);

    free(vec);
}
