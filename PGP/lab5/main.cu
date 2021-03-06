#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>

#include "error.h"

#define BLOCK_SIZE 1024
#define WARP_SIZE 32

#define idx(ind) \
    ((WARP_SIZE + 1) * (ind / WARP_SIZE) + (ind % WARP_SIZE))

int next_multiple(int n, int m){
    int r = n % m;
    if(r == 0)
        return n;
    return n + (m - r);
}

__device__ void conditional_swap(int *x, int *y){
    int x_val = *x;
    int y_val = *y;
    if (x_val > y_val) {
        *y = x_val;
        *x = y_val;
    }
}

__global__ void int_memset(int *dev_arr, int n, int val){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = gridDim.x * blockDim.x;

    for(int i = idx; i < n; i += offset)
        dev_arr[i] = val;
}

__global__ void sort_blocks(int *dev_arr){
    __shared__ int sarr[BLOCK_SIZE + (BLOCK_SIZE / WARP_SIZE) + 1];
    
    sarr[idx(threadIdx.x)] = dev_arr[threadIdx.x + blockIdx.x * BLOCK_SIZE];
    int second_half_idx = threadIdx.x + BLOCK_SIZE / 2;
    sarr[idx(second_half_idx)] = dev_arr[threadIdx.x + blockIdx.x * BLOCK_SIZE + BLOCK_SIZE / 2];

    if(threadIdx.x == 0)
        sarr[idx(BLOCK_SIZE)] = INT_MAX;
    
    __syncthreads();

    int swap1_idx1 = 2 * threadIdx.x;
    int swap1_idx2 = 2 * threadIdx.x + 1;
    int swap2_idx2 = 2 * threadIdx.x + 2;

    for(int i = 0; i < BLOCK_SIZE; ++i){
        conditional_swap(sarr + idx(swap1_idx1), sarr + idx(swap1_idx2));
        __syncthreads();
        
        conditional_swap(sarr + idx(swap1_idx2), sarr + idx(swap2_idx2));
        __syncthreads();
    }

    __syncthreads();

    dev_arr[threadIdx.x + blockIdx.x * BLOCK_SIZE] = sarr[idx(threadIdx.x)];
    dev_arr[threadIdx.x + blockIdx.x * BLOCK_SIZE + BLOCK_SIZE / 2] = sarr[idx(second_half_idx)];
}

__global__ void merge(int *dev_arr, int iter, int type){
    __shared__ int sarr[BLOCK_SIZE + (BLOCK_SIZE / WARP_SIZE)];

    sarr[idx(threadIdx.x)] = dev_arr[threadIdx.x + blockIdx.x * BLOCK_SIZE];
    int load_second_half_idx = BLOCK_SIZE - threadIdx.x - 1;
    sarr[idx(load_second_half_idx)] = dev_arr[threadIdx.x + blockIdx.x * BLOCK_SIZE + BLOCK_SIZE / 2];

    __syncthreads();

    for(int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2){
        int i = threadIdx.x / stride;
        int j = threadIdx.x % stride;

        int swap_idx1 = 2 * stride * i + j;
        int swap_idx2 = 2 * stride * i + j + stride;

        __syncthreads();
        conditional_swap(sarr + idx(swap_idx1), sarr + idx(swap_idx2));
    }

    __syncthreads();

    dev_arr[threadIdx.x + blockIdx.x * BLOCK_SIZE] = sarr[idx(threadIdx.x)];
    int store_second_half_idx = threadIdx.x + BLOCK_SIZE / 2;;
    dev_arr[threadIdx.x + blockIdx.x * BLOCK_SIZE + BLOCK_SIZE / 2] = sarr[idx(store_second_half_idx)];
}

void block_odd_even_sort(int *arr, int n){
    int dev_n = next_multiple(n, BLOCK_SIZE);
    int n_blocks = dev_n / BLOCK_SIZE;

    int *dev_arr;
    CHECK_CUDA_CALL_ERROR(cudaMalloc(&dev_arr, dev_n * sizeof(int)));
    CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_arr, arr, n * sizeof(int), cudaMemcpyHostToDevice));

    int_memset<<<1, BLOCK_SIZE>>>(dev_arr + n, (dev_n - n), INT_MIN);
    CHECK_CUDA_KERNEL_ERROR();

    sort_blocks<<<n_blocks, BLOCK_SIZE / 2>>>(dev_arr);
    CHECK_CUDA_KERNEL_ERROR();

    if(n_blocks == 1){
        CHECK_CUDA_CALL_ERROR(cudaMemcpy(arr, dev_arr + (dev_n - n), n * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA_CALL_ERROR(cudaFree(dev_arr));
        return;
    }

    for(int iter = 0; iter < n_blocks; ++iter){
        merge<<<n_blocks - 1, BLOCK_SIZE / 2>>>(dev_arr + BLOCK_SIZE / 2, iter, 0);
        CHECK_CUDA_KERNEL_ERROR();

        merge<<<n_blocks, BLOCK_SIZE / 2>>>(dev_arr, iter, 1);
        CHECK_CUDA_KERNEL_ERROR();
    }

    CHECK_CUDA_CALL_ERROR(cudaMemcpy(arr, dev_arr + (dev_n - n), n * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL_ERROR(cudaFree(dev_arr));
}

int main(){
    init_error_handling();

    unsigned int n;
    assert(fread(&n, sizeof(int), 1, stdin) == 1);

    if(n == 0)
        return 0;

    int *arr = (int*)malloc(n * sizeof(int));
    assert(fread(arr, sizeof(int), n, stdin) == n);
    
    block_odd_even_sort(arr, n);

    assert(fwrite(arr, sizeof(int), n, stdout) == n);

    free(arr);
}