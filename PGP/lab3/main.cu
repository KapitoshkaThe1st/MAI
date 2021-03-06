#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include "error.h"
#include "benchmark.h"

#define GRID_DIM (1024)
#define BLOCK_DIM (256)

#define MAX_N_CLUSTERS (32)

__constant__ float4 centroid[MAX_N_CLUSTERS];

float eps = 0.01;

// squared distance between pixel and k-th centroid
__device__ float dev_dist(uchar4 pixel, int k){
    float dx = pixel.x - centroid[k].x;
    float dy = pixel.y - centroid[k].y;
    float dz = pixel.z - centroid[k].z;

    return dx*dx + dy*dy + dz*dz;
}

float dist(float4 a, float4 b){
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;

    return dx*dx + dy*dy + dz*dz;
}

__global__ void belong_to_cluster(uchar4 *pixels, int n_pixels, int n_clusters){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    for(int i = idx; i < n_pixels; i += offset){
        int nearest_cluster_index = 0;
        float min_dist = DBL_MAX;
        for(int j = 0; j < n_clusters; ++j){
            float temp_dist = dev_dist(pixels[i], j);
            if(temp_dist < min_dist){
                nearest_cluster_index = j;
                min_dist = temp_dist;
            }
        }
        pixels[i].w = nearest_cluster_index;
    }
}

void k_means(uchar4 *pixels, int width, int height, int *cluster_center_x, int *cluster_center_y, int n_clusters){
    int n_pixels = width * height;
    
    uchar4 *dev_pixels;
    CHECK_CUDA_CALL_ERROR(cudaMalloc(&dev_pixels, sizeof(uchar4) * n_pixels));
    
    CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_pixels, pixels, sizeof(uchar4) * n_pixels, cudaMemcpyHostToDevice));

    ulonglong4 color_by_cluster_idx[MAX_N_CLUSTERS];
    float4 local_centroid[MAX_N_CLUSTERS];
    
    for(int i = 0; i < n_clusters; ++i){
        uchar4 center_pixel = pixels[cluster_center_x[i] + cluster_center_y[i] * width];
        local_centroid[i].x = center_pixel.x;
        local_centroid[i].y = center_pixel.y;
        local_centroid[i].z = center_pixel.z;
        local_centroid[i].w = 0.0f;
    }
    
    int stop = 0;
    while(!stop){
        stop = 1;
        CHECK_CUDA_CALL_ERROR(cudaMemcpyToSymbol(centroid, local_centroid, sizeof(float4) * n_clusters, 0, cudaMemcpyHostToDevice));
        
        // belong to cluster
        belong_to_cluster<<<GRID_DIM, BLOCK_DIM>>>(dev_pixels, n_pixels, n_clusters);
        CHECK_CUDA_KERNEL_ERROR();

        CHECK_CUDA_CALL_ERROR(cudaMemcpy(pixels, dev_pixels, sizeof(uchar4) * n_pixels, cudaMemcpyDeviceToHost));

        // recompute centroids
        memset(color_by_cluster_idx, 0, sizeof(ulonglong4) * MAX_N_CLUSTERS);
        for(int i = 0; i < n_pixels; ++i){
            uchar4 cur_pixel = pixels[i];
            color_by_cluster_idx[cur_pixel.w].x += cur_pixel.x;
            color_by_cluster_idx[cur_pixel.w].y += cur_pixel.y;
            color_by_cluster_idx[cur_pixel.w].z += cur_pixel.z;
            ++color_by_cluster_idx[cur_pixel.w].w;
        }
        
        for(int i = 0; i < n_clusters; ++i){
            int pixel_count_for_cluster = color_by_cluster_idx[i].w;
            float4 temp = make_float4((float)color_by_cluster_idx[i].x / pixel_count_for_cluster,
                (float)color_by_cluster_idx[i].y / pixel_count_for_cluster,
                (float)color_by_cluster_idx[i].z / pixel_count_for_cluster,
                0.0f);

            if(dist(local_centroid[i], temp) > eps * eps)
                stop = 0;

            local_centroid[i] = temp;
        }
    }

    CHECK_CUDA_CALL_ERROR(cudaFree(dev_pixels));
}

int main(int argc, char **argv){
    char *input_file_path = (char*)malloc(PATH_MAX * sizeof(char));
    char *output_file_path = (char*)malloc(PATH_MAX * sizeof(char));

    scanf("%s", input_file_path);
    scanf("%s", output_file_path);

    int n_clusters = 0;
    scanf("%d", &n_clusters);
    int cluster_center_x[32], cluster_center_y[32];
    for(int i = 0; i < n_clusters; ++i){
        scanf("%d", &cluster_center_x[i]);        
        scanf("%d", &cluster_center_y[i]);
    }

    FILE *input_file;
    if((input_file = fopen(input_file_path, "rb")) == NULL){
        printf("ERROR: can't open input file\n");
        exit(0);
    }
    free(input_file_path);

    int width = 0, height = 0;
    fread(&width, sizeof(int), 1, input_file);
    fread(&height, sizeof(int), 1, input_file);

    if (width == 0 || height == 0)
        return 0;

    int n_pixels = width * height;
    
    uchar4 *pixels = (uchar4*)malloc(n_pixels * sizeof(uchar4));
    if(pixels == NULL){
        printf("ERROR: allocation error\n");
        exit(0);
    }

    fread(pixels, sizeof(uchar4), n_pixels, input_file);
    fclose(input_file);

    k_means(pixels, width, height, cluster_center_x, cluster_center_y, n_clusters);

    FILE *output_file;
    if((output_file = fopen(output_file_path, "wb")) == NULL){
        printf("ERROR: can't open output file\n");
        exit(0);
    }
    free(output_file_path);

    fwrite(&width, sizeof(int), 1, output_file);
    fwrite(&height, sizeof(int), 1, output_file);

    fwrite(pixels, sizeof(uchar4), n_pixels, output_file);

    fclose(output_file);

    free(pixels);
}