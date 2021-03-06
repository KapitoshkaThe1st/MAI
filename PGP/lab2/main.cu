#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "error.h"
#include "benchmark.h"

#ifndef BENCHMARK

#define GRID (1024)
#define BLOCK (256)

#endif

texture<uchar4, 2, cudaReadModeElementType> tex;

__device__ uchar4 avg_kernel(int i, int j, int w, int h, int kernel_w, int kernel_h){
    int sum_r = 0, sum_g = 0, sum_b = 0, sum_a = 0;
    for(int y = i; y < i + kernel_h; ++y){
        for(int x = j; x < j + kernel_w; ++x){
            uchar4 pixel = tex2D(tex, x, y);
            sum_r += pixel.x;
            sum_g += pixel.y;
            sum_b += pixel.z;
            sum_a += pixel.w;
        }
    }

    int n_pixels = kernel_w * kernel_h;
    uchar4 result;
    result.x = (unsigned char)(sum_r / n_pixels);
    result.y = (unsigned char)(sum_g / n_pixels);
    result.z = (unsigned char)(sum_b / n_pixels);
    result.w = (unsigned char)(sum_a / n_pixels);

    return result;
}

__global__ void SSAA(uchar4 *res, int w, int h, int new_w, int new_h){
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

            res[i * new_w + j] = avg_kernel(pix_i, pix_j, w, h, kernel_w, kernel_h);
        }
    }
}

int main() {
    init_error_handling();

    char *input_file_path = (char*)malloc(PATH_MAX * sizeof(char));
    char *output_file_path = (char*)malloc(PATH_MAX * sizeof(char));

    scanf("%s", input_file_path);
    scanf("%s", output_file_path);

    int new_width = 0, new_height = 0;
    scanf("%d", &new_width);
    scanf("%d", &new_height);

    FILE *input_file;
    if((input_file = fopen(input_file_path, "rb")) == NULL) {
        printf("ERROR: can't open input file\n");
        exit(0);
    }
    
    free(input_file_path);

    int width = 0, height = 0;
    fread(&width, sizeof(int), 1, input_file);
    fread(&height, sizeof(int), 1, input_file);

    if(width == 0 || height == 0)
        return 0;

    int ref_n_pixels = width * height;
    int res_n_pixels = new_height * new_width;

    int ref_size = sizeof(uchar4) * ref_n_pixels;
    int res_size = sizeof(uchar4) * res_n_pixels;

    uchar4 *ref = (uchar4*)malloc(ref_size);

    fread(ref, sizeof(uchar4), ref_n_pixels, input_file);

    fclose(input_file);

    uchar4 *dev_res;
    cudaArray *dev_ref;

    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
    
    CHECK_CUDA_CALL_ERROR(cudaMallocArray(&dev_ref, &ch, width, height));
    
    CHECK_CUDA_CALL_ERROR(cudaMemcpyToArray(dev_ref, 0, 0, ref, ref_size, cudaMemcpyHostToDevice));
    
	tex.addressMode[0] = cudaAddressModeClamp;
	tex.addressMode[1] = cudaAddressModeClamp;
	tex.channelDesc = ch;
	tex.filterMode = cudaFilterModePoint;
	tex.normalized = false;	

    CHECK_CUDA_CALL_ERROR(cudaBindTextureToArray(tex, dev_ref, ch));
    
    CHECK_CUDA_CALL_ERROR(cudaMalloc(&dev_res, res_size));
    
    int grid_single_dim = (int)sqrt(GRID);
    int block_single_dim = (int)sqrt(BLOCK);

    dim3 gridDim(grid_single_dim, grid_single_dim);
    dim3 blockDim(block_single_dim, block_single_dim);

    MEASURE((SSAA<<<gridDim, blockDim>>>(dev_res, width, height, new_width, new_height)));
    CHECK_CUDA_KERNEL_ERROR();
    
    uchar4 *res = (uchar4*)realloc(ref, res_size);
    CHECK_CUDA_CALL_ERROR(cudaMemcpy(res, dev_res, res_size, cudaMemcpyDeviceToHost));

    CHECK_CUDA_CALL_ERROR(cudaUnbindTexture(tex));

    CHECK_CUDA_CALL_ERROR(cudaFreeArray(dev_ref));
    CHECK_CUDA_CALL_ERROR(cudaFree(dev_res));

    FILE *output_file;
    if((output_file = fopen(output_file_path, "wb")) == NULL) {
        printf ("ERROR: can't open output file\n");
        exit(0);
    }

    free(output_file_path);

    fwrite(&new_width, sizeof(int), 1, output_file);
    fwrite(&new_height, sizeof(int), 1, output_file);

    fwrite(res, sizeof(uchar4), res_n_pixels, output_file);

    fclose(output_file);

    free(res);
}