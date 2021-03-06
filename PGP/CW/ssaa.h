#ifndef SSAA_H
#define SSAA_H

#include "vec.h"
#include "error.h"

__host__ __device__ vec3 avg_kernel(vec3 *img, int i, int j, int w, int h, int kernel_w, int kernel_h){
    vec3 sum(0.0f);
    for(int y = i; y < i + kernel_h; ++y){
        for(int x = j; x < j + kernel_w; ++x){
            sum += img[y * w + x];
        }
    }

    int n_pixels = kernel_w * kernel_h;
    float coef = 1.0f / n_pixels;

    return coef * sum;
}

__global__ void SSAA_kernel(vec3 *ref, vec3 *res, int w, int h, int new_w, int new_h){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;
    
    int kernel_w = w / new_w;
    int kernel_h = h / new_h;

    for(int i = idy; i < new_h; i += offset_y){
        for(int j = idx; j < new_w; j += offset_x){
            int pix_i = i * kernel_h;
            int pix_j = j * kernel_w; 

            // res[i * new_w + j] = vec3((float)i / new_h, (float)j / new_w, 0.0f);
            // res[j * new_h + i] = avg_kernel(ref, pix_i, pix_j, w, h, kernel_w, kernel_h);
            res[i * new_w + j] = avg_kernel(ref, pix_i, pix_j, w, h, kernel_w, kernel_h);
        }
    }
}

void SSAA_gpu(vec3 *ref, vec3 *res, int w, int h, int new_w, int new_h){
    vec3 *dev_res, *dev_ref;
    CHECK_CUDA_CALL_ERROR(cudaMalloc(&dev_ref, w * h * sizeof(vec3)));
    CHECK_CUDA_CALL_ERROR(cudaMalloc(&dev_res, new_w * new_h * sizeof(vec3)));
    
    CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_ref, ref, w * h * sizeof(vec3), cudaMemcpyHostToDevice));
    
    SSAA_kernel<<<dim3(32, 32), dim3(16, 16)>>>(dev_ref, dev_res, w, h, new_w, new_h);
    // SSAA_kernel<<<dim3(32, 32), dim3(32, 32)>>>(dev_ref, dev_res, w, h, new_w, new_h);
    CHECK_CUDA_KERNEL_ERROR();

    CHECK_CUDA_CALL_ERROR(cudaMemcpy(res, dev_res, new_w * new_h * sizeof(vec3), cudaMemcpyDeviceToHost));
}

void SSAA(vec3 *ref, vec3 *res, int w, int h, int new_w, int new_h){
    int kernel_w = w / new_w, kernel_h = h / new_h;
    for(int i = 0; i < new_h; ++i){
        for(int j = 0; j < new_w; ++j){
            int pix_i = i * kernel_h;
            int pix_j = j * kernel_w; 

            res[i * new_w + j] = avg_kernel(ref, pix_i, pix_j, w, h, kernel_w, kernel_h);
        }
    }
}

#endif