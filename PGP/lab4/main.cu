#include <stdio.h>
#include <math.h>

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

#include "error.h"

double eps = 1e-17;

#define MAX_BLOCKS 512
#define MAX_THREADS 1024

#define MAX_BLOCKS_2D sqrt(MAX_BLOCKS)
#define MAX_THREADS_2D sqrt(MAX_THREADS)

__global__ void swap_rows(double *mat, int n, int a, int b){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    for(int i = idx; i < n; i += offset){
        double temp = mat[i * n + a];
        mat[i * n + a] = mat[i * n + b];
        mat[i * n + b] = temp;
    }
}

__global__ void compute_column(double *mat, int n, int i){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    for(int j = i + 1 + idx; j < n; j += offset){
        mat[i * n + j] /= mat[i * n + i];
    }
}

__global__ void recompute(double *mat, int n, int i){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    for(int j = i + 1 + idx; j < n; j += offset_x){
        double coef = mat[i * n + j];
        for(int k = i + 1 + idy; k < n; k += offset_y){
            mat[k * n + j] -= coef * mat[k * n + i];
        }
    }
}

int approx_equal(double a, double b){ 
    return fabs(a - b) < eps;
}

__host__ __device__ bool comp(double a, double b){
    return fabs(a) < fabs(b);
}

struct compare{
    __host__ __device__ bool operator()(double lhs, double rhs){
        return fabs(lhs) < fabs(rhs);
    }
};

int closest_power2_upper(int x){
    int p2 = 1;
    while(1)
    {
        if (p2 >= x)
            return p2;
        p2 <<= 1;
    }
}

int closest_power2_lower(int x){
    return closest_power2_upper(x) >> 1;
}

dim3 grid_configuration2d(int n1dim){
    int g1dim = closest_power2_upper(n1dim) / MAX_THREADS_2D;
    g1dim = min((int)MAX_BLOCKS_2D, max(1, g1dim));
    return dim3(g1dim, g1dim, 1);
}

int grid_configuration1d(int n1dim){
    int g1dim = closest_power2_upper(n1dim) / MAX_THREADS;
    return min((int)MAX_BLOCKS, max(1, g1dim));
}

int lu_decomposition(double *mat, int *p, int n){
    double *dev_mat = NULL;
    CHECK_CUDA_CALL_ERROR(cudaMalloc(&dev_mat, n * n * sizeof(double)));
    CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_mat, mat, n * n * sizeof(double), cudaMemcpyHostToDevice));
    
    dim3 gridDim2d = grid_configuration2d(n);
    dim3 blockDim(MAX_THREADS_2D, MAX_THREADS_2D, 1);

    int gridDim1d = grid_configuration1d(n);
    
    int lower_power = closest_power2_lower(n);

    compare comp;

    for(int i = 0; i < n-1; ++i){
        if(n - i == lower_power){
            gridDim2d = grid_configuration2d(n-i);
            gridDim1d = grid_configuration1d(n-i);

            lower_power = closest_power2_lower(n-i);
        }

        thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(dev_mat + i * n);
        thrust::device_ptr<double> it = thrust::max_element(dev_ptr + i, dev_ptr + n, comp);
        CHECK_CUDA_KERNEL_ERROR();
        int maxInd = it - dev_ptr;
        double maxVal = *it;

        if(approx_equal(maxVal, 0.0)){
            CHECK_CUDA_CALL_ERROR(cudaFree(dev_mat));
            return 0;
        }

        if(maxInd != i){
            swap_rows<<<gridDim1d, MAX_THREADS>>>(dev_mat, n, i, maxInd);
            CHECK_CUDA_KERNEL_ERROR();
        }

        p[i] = maxInd;

        compute_column<<<gridDim1d, MAX_THREADS>>>(dev_mat, n, i);
        recompute<<<gridDim2d, blockDim>>>(dev_mat, n, i);
    }

    CHECK_CUDA_CALL_ERROR(cudaMemcpy(mat, dev_mat, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA_CALL_ERROR(cudaFree(dev_mat));

    p[n-1] = n-1;
    return 1;
}

int main(){
    int n;
    assert(scanf("%d", &n) == 1);

    double *mat = (double*)malloc(n * n * sizeof(double));
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            assert(scanf("%lf", &mat[j * n + i]) == 1);
        }
    }

    int *p = (int*)malloc(n * sizeof(int));

    if(!lu_decomposition(mat, p, n)){
        printf("ERROR: LU-decomposition can't be computed.\n");
    }

    int double_len = snprintf(NULL, 0, "% .10e ", mat[0]);
    int int_len = snprintf(NULL, 0, "%d ", n);
    char *buffer = (char*)malloc((n * (n * double_len + 1) + n * int_len + 1) * sizeof(char));

    int offset = 0;
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j)
            offset += sprintf(buffer + offset, "%.10e ", mat[j * n + i]);
        offset += sprintf(buffer + offset, "\n");
    }
    
    for(int i = 0; i < n; ++i)
        offset += sprintf(buffer + offset, "%d ", p[i]);
    offset += sprintf(buffer + offset, "\n");

    printf("%s", buffer);

    free(mat);
    free(buffer);
}
