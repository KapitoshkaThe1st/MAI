
#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>
#include <limits.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

// -- error handling --

#include <signal.h>

#define CHECK_CUDA_CALL_ERROR(ERR)                                                                             \
    do                                                                                                         \
    {                                             \
		cudaDeviceSynchronize();                                                             \
        cudaError_t err = ERR;                                                                                 \
		cudaDeviceSynchronize();                                                             \
        if (err != cudaSuccess)                                                                                \
        {                                                                                                      \
            printf("ERROR: %s (error_code: %d) at %s:%d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
            fprintf(stderr, "ERROR: %s (error_code: %d) at %s:%d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(0);                                                                                           \
        }                                                                                                      \
    } while (0)

#define CHECK_CUDA_KERNEL_ERROR()                                                                              \
    do                                                                                                         \
	{                                                                                                          \
		cudaDeviceSynchronize();                                                             \
        cudaError_t err = cudaGetLastError();                                                                  \
		cudaDeviceSynchronize();                                                             \
        if (err != cudaSuccess)                                                                                \
        {                                                                                                      \
            printf("ERROR: %s (error_code: %d) at %s:%d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
            fprintf(stderr, "ERROR: %s (error_code: %d) at %s:%d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(0);                                                                                           \
        }                                                                                                      \
    } while (0)

void _sigsegv_handler(int signo)
{
	printf("ERROR: segmentation fault (error code: %d)\n", signo);
    fprintf(stderr, "ERROR: segmentation fault (error code: %d)\n", signo);
	
    exit(0);
}

void _sigabrt_handler(int signo)
{
	printf("ERROR: aborted (error code: %d)\n", signo);
    fprintf(stderr, "ERROR: aborted (error code: %d)\n", signo);
	
    exit(0);
}

void init_error_handling()
{
    signal(SIGSEGV, _sigsegv_handler);
    signal(SIGABRT, _sigabrt_handler);
}

// -- \error handling --

// перевод из трехмерной индексации к одномерной, используется в вызовах send\recv
#define block_index(i, j, k) ((k) * (n_blocks_x * n_blocks_y)) + (j) * n_blocks_x + (i)

// перевод из одномерной индексации в трехмерную
#define block_index_i(n) ((n) % (n_blocks_x * n_blocks_y) % n_blocks_x)
#define block_index_j(n) ((n) % (n_blocks_x * n_blocks_y) / n_blocks_x)
#define block_index_k(n) ((n) / (n_blocks_x * n_blocks_y))

// перевод из трехмерной индексации к одномерной, используется в вызовах send\recv
#define cell_index(i, j, k) (((k) + 1) * ((block_size_x + 2) * (block_size_y + 2)) + ((j) + 1) * (block_size_x + 2) + ((i) + 1)) 

// перевод из одномерной индексации в трехмерную
#define cell_index_i(n) ((n) % ((block_size_x + 2) * (block_size_y + 2)) % (block_size_x + 2) - 1)
#define cell_index_j(n) ((n) % ((block_size_x + 2) * (block_size_y + 2)) / (block_size_x + 2) - 1)
#define cell_index_k(n) ((n) / ((block_size_x + 2) * (block_size_y + 2)) - 1)

#define LEFT 0
#define RIGHT 1
#define BACK 2
#define FRONT 3
#define DOWN 4
#define UP 5

#define UPDOWN 0
#define LEFTRIGHT 1
#define FRONTBACK 2

#define BORDER_OPERATIONS_GRID_DIM 32
#define BORDER_OPERATIONS_BLOCK_DIM 32

#define GRID_DIM 8

#define MULTIPLE_GPU_CRITERIA 100000

__global__ void get_updown_border_kernel(double *dst, double *src, int block_size_x, int block_size_y, int block_size_z, int border_index){
	int id_x = threadIdx.x + blockIdx.x * blockDim.x;
	int id_y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;
	
	for(int j = id_y; j < block_size_y; j += offset_y){
        for(int i = id_x; i < block_size_x; i += offset_x){
            dst[j * block_size_x + i] = src[cell_index(i, j, border_index)];
        }
    }
}

__global__ void get_leftright_border_kernel(double *dst, double *src, int block_size_x, int block_size_y, int block_size_z, int border_index){
	int id_x = threadIdx.x + blockIdx.x * blockDim.x;
	int id_y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    for(int k = id_y; k < block_size_z; k += offset_y){
        for(int j = id_x; j < block_size_y; j += offset_x){
            dst[k * block_size_y + j] = src[cell_index(border_index, j, k)];
        }
    }
}

__global__ void get_frontback_border_kernel(double *dst, double *src, int block_size_x, int block_size_y, int block_size_z, int border_index){
	int id_x = threadIdx.x + blockIdx.x * blockDim.x;
	int id_y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    for(int k = id_y; k < block_size_z; k += offset_y){
        for(int i = id_x; i < block_size_x; i += offset_x){
            dst[k * block_size_x + i] = src[cell_index(i, border_index, k)];
        }
    }
}

__global__ void set_updown_border_kernel(double *dst, double *src, int block_size_x, int block_size_y, int block_size_z, int border_index){
	int id_x = threadIdx.x + blockIdx.x * blockDim.x;
	int id_y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    for(int j = id_y; j < block_size_y; j += offset_y){
        for(int i = id_x; i < block_size_x; i += offset_x){
			dst[cell_index(i, j, border_index)] = src[j * block_size_x + i];
        }
    }
}

__global__ void set_leftright_border_kernel(double *dst, double *src, int block_size_x, int block_size_y, int block_size_z, int border_index){
	int id_x = threadIdx.x + blockIdx.x * blockDim.x;
	int id_y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    for(int k = id_y; k < block_size_z; k += offset_y){
        for(int j = id_x; j < block_size_y; j += offset_x){
            dst[cell_index(border_index, j, k)] = src[k * block_size_y + j];
        }
    }
}

__global__ void set_frontback_border_kernel(double *dst, double *src, int block_size_x, int block_size_y, int block_size_z, int border_index){
	int id_x = threadIdx.x + blockIdx.x * blockDim.x;
	int id_y = threadIdx.y + blockIdx.y * blockDim.y;

	int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;

    for(int k = id_y; k < block_size_z; k += offset_y){
        for(int i = id_x; i < block_size_x; i += offset_x){
            dst[cell_index(i, border_index, k)] = src[k * block_size_x + i];
        }
    }
}

void horizontal_stack(double *dst, double **buffers, int *horizontal_sizes, int vertical_size, int device_count){
    int total_horizontal_size = 0;
    for(int d = 0; d < device_count; ++d)
        total_horizontal_size += horizontal_sizes[d];
    
    for(int d = 0; d < device_count; ++d){
        for(int j = 0; j < vertical_size; ++j){
            for(int i = 0; i < horizontal_sizes[d]; ++i){
                dst[total_horizontal_size * j + d * horizontal_sizes[0] + i] = buffers[d][j * horizontal_sizes[d] + i];
            }
        }
    }
}

void vertical_stack(double *dst, double **buffers, int horizontal_size, int *vertical_sizes, int device_count){
    int offset = 0;
    for(int d = 0; d < device_count; ++d){
        for(int j = 0; j < vertical_sizes[d]; ++j){
            for(int i = 0; i < horizontal_size; ++i){
                dst[offset + j * horizontal_size + i] = buffers[d][j * horizontal_size + i];
            }
        }
        offset += vertical_sizes[d] * horizontal_size;
    }
}

void horizontal_unstack(double *src, double **buffers, int *horizontal_sizes, int vertical_size, int device_count){
    int total_horizontal_size = 0;
    for(int d = 0; d < device_count; ++d)
        total_horizontal_size += horizontal_sizes[d];

    for(int d = 0; d < device_count; ++d){
        for(int j = 0; j < vertical_size; ++j){
            for(int i = 0; i < horizontal_sizes[d]; ++i){
                buffers[d][j * horizontal_sizes[d] + i] = src[total_horizontal_size * j + d * horizontal_sizes[0] + i];
            }
        }
    }
}

void vertical_unstack(double *src, double **buffers, int horizontal_size, int *vertical_sizes, int device_count){
    int offset = 0;
    for(int d = 0; d < device_count; ++d){
        for(int j = 0; j < vertical_sizes[d]; ++j){
            for(int i = 0; i < horizontal_size; ++i){
                buffers[d][j * horizontal_size + i] = src[offset + j * horizontal_size + i];
            }
        }
        offset += vertical_sizes[d] * horizontal_size;
    }
}

void get_border_multigpu(double *dst, double *dev_dst, double **src, double **buffers, double **dev_buffers, 
	int border, int *block_sizes_x, int *block_sizes_y, int *block_sizes_z, int device_count, int split_type)
{
	dim3 kernel_launch_grid = dim3(BORDER_OPERATIONS_GRID_DIM, BORDER_OPERATIONS_GRID_DIM);
	dim3 kernel_launch_block = dim3(BORDER_OPERATIONS_BLOCK_DIM, BORDER_OPERATIONS_BLOCK_DIM);

    if(split_type == UPDOWN){
		if(border == UP || border == DOWN){
            int src_index = border == DOWN ? 0 : device_count-1;
			int border_index = border == DOWN ? 0 : block_sizes_z[device_count-1]-1;
			CHECK_CUDA_CALL_ERROR(cudaSetDevice(src_index));
			get_updown_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_dst, src[src_index],
				block_sizes_x[src_index], block_sizes_y[src_index], block_sizes_z[src_index], border_index);
			CHECK_CUDA_CALL_ERROR(cudaMemcpy(dst, dev_dst, sizeof(double) * block_sizes_x[0] * block_sizes_y[0], cudaMemcpyDeviceToHost));
		}
		else if(border == LEFT || border == RIGHT){
            int border_index = border == LEFT ? 0 : block_sizes_x[device_count-1]-1;
            for(int d = 0; d < device_count; ++d){
				CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
				get_leftright_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d], 
					block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index);
				CHECK_CUDA_CALL_ERROR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * block_sizes_y[d] * block_sizes_z[d], cudaMemcpyDeviceToHost));
			}
            vertical_stack(dst, buffers, block_sizes_y[0], block_sizes_z, device_count);
		}
		else if(border == FRONT || border == BACK){
            int border_index = border == FRONT ? 0 : block_sizes_y[device_count-1]-1;
            for(int d = 0; d < device_count; ++d){
				CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
				get_frontback_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
					block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index);
				CHECK_CUDA_CALL_ERROR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_z[d], cudaMemcpyDeviceToHost));
			}
            vertical_stack(dst, buffers, block_sizes_x[0], block_sizes_z, device_count);
		}
	}
	else if(split_type == LEFTRIGHT){
		if(border == UP || border == DOWN){
            int border_index = border == DOWN ? 0 : block_sizes_z[device_count-1]-1;
			for(int d = 0; d < device_count; ++d){
				CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
				get_updown_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
					block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index);
				CHECK_CUDA_CALL_ERROR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_y[d], cudaMemcpyDeviceToHost));
			}
            horizontal_stack(dst, buffers, block_sizes_x, block_sizes_y[0], device_count);
		}
		else if(border == LEFT || border == RIGHT){
			int src_index = border == LEFT ? 0 : device_count-1;
			int border_index = border == LEFT ? 0 : block_sizes_x[device_count-1]-1;
			CHECK_CUDA_CALL_ERROR(cudaSetDevice(src_index));
			get_leftright_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_dst, src[src_index],
				block_sizes_x[src_index], block_sizes_y[src_index], block_sizes_z[src_index], border_index);
			CHECK_CUDA_CALL_ERROR(cudaMemcpy(dst, dev_dst, sizeof(double) * block_sizes_y[0] * block_sizes_z[0], cudaMemcpyDeviceToHost));
		}
		else if(border == FRONT || border == BACK){
            int border_index = border == FRONT ? 0 : block_sizes_y[device_count-1]-1;
			for(int d = 0; d < device_count; ++d){
				CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
				get_frontback_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
					block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index);
				CHECK_CUDA_CALL_ERROR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_z[d], cudaMemcpyDeviceToHost));
			}
            horizontal_stack(dst, buffers, block_sizes_x, block_sizes_z[0], device_count);
		}
	}
	else if(split_type == FRONTBACK){
		if(border == UP || border == DOWN){
            int border_index = border == DOWN ? 0 : block_sizes_z[device_count-1]-1;
			for(int d = 0; d < device_count; ++d){
				CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
				get_updown_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
					block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index);
				CHECK_CUDA_CALL_ERROR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_y[d], cudaMemcpyDeviceToHost));
			}
            vertical_stack(dst, buffers, block_sizes_x[0], block_sizes_y, device_count);
		}
		else if(border == LEFT || border == RIGHT){
            int border_index = border == LEFT ? 0 : block_sizes_x[device_count-1]-1;
			for(int d = 0; d < device_count; ++d){
				CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
				get_leftright_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffers[d], src[d],
					block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index);
				CHECK_CUDA_CALL_ERROR(cudaMemcpy(buffers[d], dev_buffers[d], sizeof(double) * block_sizes_y[d] * block_sizes_z[d], cudaMemcpyDeviceToHost));
			}
            horizontal_stack(dst, buffers, block_sizes_y, block_sizes_z[0], device_count);
		}
		else if(border == FRONT || border == BACK){
			int src_index = border == FRONT ? 0 : device_count-1;
			int border_index = border == FRONT ? 0 : block_sizes_y[device_count-1]-1;
			CHECK_CUDA_CALL_ERROR(cudaSetDevice(src_index));
			get_frontback_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_dst, src[src_index],
				block_sizes_x[src_index], block_sizes_y[src_index], block_sizes_z[src_index], border_index);
			CHECK_CUDA_CALL_ERROR(cudaMemcpy(dst, dev_dst, sizeof(double) * block_sizes_x[0] * block_sizes_z[0], cudaMemcpyDeviceToHost));
		}
	}
	else{
        printf("ERROR OCCURED\n");
        exit(0);
	}
}

void set_border_multigpu(double **dst, double *src, double *dev_src, double **buffers, double **dev_buffers,
	int border, int *block_sizes_x, int *block_sizes_y, int *block_sizes_z, int device_count, int split_type)
{
	dim3 kernel_launch_grid = dim3(BORDER_OPERATIONS_GRID_DIM, BORDER_OPERATIONS_GRID_DIM);
	dim3 kernel_launch_block = dim3(BORDER_OPERATIONS_BLOCK_DIM, BORDER_OPERATIONS_BLOCK_DIM);

	if(split_type == UPDOWN){
		if(border == UP || border == DOWN){
            int src_index = border == DOWN ? 0 : device_count-1;
			int border_index = border == DOWN ? -1 : block_sizes_z[device_count-1];
			CHECK_CUDA_CALL_ERROR(cudaSetDevice(src_index));
			CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_src, src, sizeof(double) * block_sizes_x[0] * block_sizes_y[0], cudaMemcpyHostToDevice));
			set_updown_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[src_index], dev_src,
				block_sizes_x[src_index], block_sizes_y[src_index], block_sizes_z[src_index], border_index);
		}
		else if(border == LEFT || border == RIGHT){
            int border_index = border == LEFT ? -1 : block_sizes_x[device_count-1];
            vertical_unstack(src, buffers, block_sizes_y[0], block_sizes_z, device_count);
            for(int d = 0; d < device_count; ++d){
				CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
				CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizes_y[d] * block_sizes_z[d], cudaMemcpyHostToDevice));
				set_leftright_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
					block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index);
			}
		}
		else if(border == FRONT || border == BACK){
            int border_index = border == FRONT ? -1 : block_sizes_y[device_count-1];
            vertical_unstack(src, buffers, block_sizes_x[0], block_sizes_z, device_count);
            for(int d = 0; d < device_count; ++d){
				CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
				CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_z[d], cudaMemcpyHostToDevice));
				set_frontback_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
					block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index);
			}
		}
	}
	else if(split_type == LEFTRIGHT){
		if(border == UP || border == DOWN){
            int border_index = border == DOWN ? -1 : block_sizes_z[device_count-1];
            horizontal_unstack(src, buffers, block_sizes_x, block_sizes_y[0], device_count);
			for(int d = 0; d < device_count; ++d){
				CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
				CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_y[d], cudaMemcpyHostToDevice));
				set_updown_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
					block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index);
			}
		}
		else if(border == LEFT || border == RIGHT){
			int src_index = border == LEFT ? 0 : device_count-1;
			int border_index = border == LEFT ? -1 : block_sizes_x[device_count-1];
			CHECK_CUDA_CALL_ERROR(cudaSetDevice(src_index));
			CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_src, src, sizeof(double) * block_sizes_y[0] * block_sizes_z[0], cudaMemcpyHostToDevice));
			set_leftright_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[src_index], dev_src,
				block_sizes_x[src_index], block_sizes_y[src_index], block_sizes_z[src_index], border_index);
		}
		else if(border == FRONT || border == BACK){
            int border_index = border == FRONT ? -1 : block_sizes_y[device_count-1];
            horizontal_unstack(src, buffers, block_sizes_x, block_sizes_z[0], device_count);
			for(int d = 0; d < device_count; ++d){
				CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
				CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_z[d], cudaMemcpyHostToDevice));
				set_frontback_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
					block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index);
			}
		}
	}
	else if(split_type == FRONTBACK){
		if(border == UP || border == DOWN){
            int border_index = border == DOWN ? -1 : block_sizes_z[device_count-1];
            vertical_unstack(src, buffers, block_sizes_x[0], block_sizes_y, device_count);
			for(int d = 0; d < device_count; ++d){
				CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
				CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizes_x[d] * block_sizes_y[d], cudaMemcpyHostToDevice));
				set_updown_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
					block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index);
			}
		}
		else if(border == LEFT || border == RIGHT){
            int border_index = border == LEFT ? -1 : block_sizes_x[device_count-1];
            horizontal_unstack(src, buffers, block_sizes_y, block_sizes_z[0], device_count);
			for(int d = 0; d < device_count; ++d){
				CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
				CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_buffers[d], buffers[d], sizeof(double) * block_sizes_y[d] * block_sizes_z[d], cudaMemcpyHostToDevice));
				set_leftright_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[d], dev_buffers[d],
					block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], border_index);
			}
		}
		else if(border == FRONT || border == BACK){
			int src_index = border == FRONT ? 0 : device_count-1;
			int border_index = border == FRONT ? -1 : block_sizes_y[device_count-1];
			CHECK_CUDA_CALL_ERROR(cudaSetDevice(src_index));
			CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_src, src, sizeof(double) * block_sizes_x[0] * block_sizes_z[0], cudaMemcpyHostToDevice));
			set_frontback_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst[src_index], dev_src,
				block_sizes_x[src_index], block_sizes_y[src_index], block_sizes_z[src_index], border_index);
		}
	}
	else{
        printf("ERROR OCCURED\n");
        exit(0);
	}
}

int get_last_index(int split_type, int *block_sizes_x, int *block_sizes_y, int *block_sizes_z, int device){
	int last_index;
	if(split_type == UPDOWN)
		last_index = block_sizes_z[device]-1;
	if(split_type == LEFTRIGHT)
		last_index = block_sizes_x[device]-1;
	if(split_type == FRONTBACK)
		last_index = block_sizes_y[device]-1;
	return last_index;
}

void get_intergpu_border(double *dst, double *src, double *dev_buffer, int index, int block_size_x, int block_size_y, int block_size_z, int split_type){
	dim3 kernel_launch_grid = dim3(BORDER_OPERATIONS_GRID_DIM, BORDER_OPERATIONS_GRID_DIM);
	dim3 kernel_launch_block = dim3(BORDER_OPERATIONS_BLOCK_DIM, BORDER_OPERATIONS_BLOCK_DIM);

	if(split_type == UPDOWN){
		get_updown_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffer, src, block_size_x, block_size_y, block_size_z, index);	
		CHECK_CUDA_CALL_ERROR(cudaMemcpy(dst, dev_buffer, sizeof(double) * block_size_x * block_size_y, cudaMemcpyDeviceToHost));
		CHECK_CUDA_KERNEL_ERROR();
	}
	else if(split_type == LEFTRIGHT){
		get_leftright_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffer, src, block_size_x, block_size_y, block_size_z, index);	
		CHECK_CUDA_CALL_ERROR(cudaMemcpy(dst, dev_buffer, sizeof(double) * block_size_y * block_size_z, cudaMemcpyDeviceToHost));
		CHECK_CUDA_KERNEL_ERROR();
	}
	else if(split_type == FRONTBACK){		
		get_frontback_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dev_buffer, src, block_size_x, block_size_y, block_size_z, index);	
		CHECK_CUDA_CALL_ERROR(cudaMemcpy(dst, dev_buffer, sizeof(double) * block_size_x * block_size_z, cudaMemcpyDeviceToHost));
		CHECK_CUDA_KERNEL_ERROR();
	}
	else{
        printf("ERROR OCCURED\n");
        exit(0);
	}
}

void set_intergpu_border(double *dst, double *src, double *dev_buffer, int index, int block_size_x, int block_size_y, int block_size_z, int split_type){
	dim3 kernel_launch_grid = dim3(BORDER_OPERATIONS_GRID_DIM, BORDER_OPERATIONS_GRID_DIM);
	dim3 kernel_launch_block = dim3(BORDER_OPERATIONS_BLOCK_DIM, BORDER_OPERATIONS_BLOCK_DIM);

	if(split_type == UPDOWN){
		CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_buffer, src, sizeof(double) * block_size_x * block_size_y, cudaMemcpyHostToDevice));
		set_updown_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst, dev_buffer, block_size_x, block_size_y, block_size_z, index);	
		CHECK_CUDA_KERNEL_ERROR();
	}
	else if(split_type == LEFTRIGHT){
		CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_buffer, src, sizeof(double) * block_size_y * block_size_z, cudaMemcpyHostToDevice));
		set_leftright_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst, dev_buffer, block_size_x, block_size_y, block_size_z, index);	
		CHECK_CUDA_KERNEL_ERROR();
	}
	else if(split_type == FRONTBACK){		
		CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_buffer, src, sizeof(double) * block_size_x * block_size_z, cudaMemcpyHostToDevice));
		set_frontback_border_kernel<<<kernel_launch_grid, kernel_launch_block>>>(dst, dev_buffer, block_size_x, block_size_y, block_size_z, index);	
		CHECK_CUDA_KERNEL_ERROR();
	}
	else{
        printf("ERROR OCCURED\n");
        exit(0);
	}
}

__global__ void compute_kernel(double *u_new, double *u, int block_size_x, int block_size_y, int block_size_z, double hx, double hy, double hz){
	int id_x = threadIdx.x + blockIdx.x * blockDim.x;
	int id_y = threadIdx.y + blockIdx.y * blockDim.y;
	int id_z = threadIdx.z + blockIdx.z * blockDim.z;

	int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;
    int offset_z = blockDim.z * gridDim.z;

	for(int i = id_x; i < block_size_x; i += offset_x)
		for(int j = id_y; j < block_size_y; j += offset_y)
			for(int k = id_z; k < block_size_z; k += offset_z){
				double inv_hxsqr = 1.0 / (hx * hx);
				double inv_hysqr = 1.0 / (hy * hy);
				double inv_hzsqr = 1.0 / (hz * hz);

				double num = (u[cell_index(i+1, j, k)] + u[cell_index(i-1, j, k)]) * inv_hxsqr
					+ (u[cell_index(i, j+1, k)] + u[cell_index(i, j-1, k)]) * inv_hysqr
					+ (u[cell_index(i, j, k+1)] + u[cell_index(i, j, k-1)]) * inv_hzsqr;
				double denum = 2.0 * (inv_hxsqr + inv_hysqr + inv_hzsqr);

				u_new[cell_index(i, j, k)] = num / denum;
			}
}

__global__ void abs_error_kernel(double *u1, double *u2, int block_size_x, int block_size_y, int block_size_z){
	int id_x = threadIdx.x + blockIdx.x * blockDim.x;
	int id_y = threadIdx.y + blockIdx.y * blockDim.y;
	int id_z = threadIdx.z + blockIdx.z * blockDim.z;

	int offset_x = blockDim.x * gridDim.x;
    int offset_y = blockDim.y * gridDim.y;
    int offset_z = blockDim.z * gridDim.z;

	for(int i = id_x - 1; i < block_size_x + 1; i += offset_x)
		for(int j = id_y - 1; j < block_size_y + 1; j += offset_y)
			for(int k = id_z - 1; k < block_size_z + 1; k += offset_z){
				u1[cell_index(i, j, k)] = fabsf(u1[cell_index(i, j, k)] - u2[cell_index(i, j, k)]);
			}
}

int main(int argc,char **argv) {
	init_error_handling();

	MPI_Init(&argc, &argv);

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int n_processes;
	MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

	int n_blocks_x, n_blocks_y, n_blocks_z;
	int block_size_x, block_size_y, block_size_z;
	double eps;
	double lx, ly, lz;
	double u_down, u_up, u_left, u_right, u_front, u_back, u0;

	char *output_file_path = (char*)malloc(PATH_MAX * sizeof(char));
	int output_file_path_len;
	if(rank == 0){
		assert(scanf("%d %d %d", &n_blocks_x, &n_blocks_y, &n_blocks_z) == 3);
		assert(scanf("%d %d %d", &block_size_x, &block_size_y, &block_size_z) == 3);

		assert(scanf("%s", output_file_path) == 1);
		assert(scanf("%lf", &eps) == 1);
		assert(scanf("%lf %lf %lf", &lx, &ly, &lz) == 3);
		assert(scanf("%lf %lf %lf %lf %lf %lf %lf", &u_down, &u_up, &u_left, &u_right, &u_front, &u_back, &u0) == 7);

		output_file_path_len = strlen(output_file_path);

		fprintf(stderr, "n_blocks_x: %d n_blocks_y: %d n_blocks_z: %d ", n_blocks_x, n_blocks_y, n_blocks_z);

		fprintf(stderr, "block_size_x: %d block_size_y: %d block_size_z: %d ", block_size_x, block_size_y, block_size_z);

		fprintf(stderr, "file: %s ", output_file_path);
		fprintf(stderr, "eps: %e ", eps);

		fprintf(stderr, "lx: %lf ly: %lf lz: %lf ", lx, ly, lz);

		fprintf(stderr, "u_down: %lf ", u_down);
		fprintf(stderr, "u_up: %lf ", u_up);
		fprintf(stderr, "u_left: %lf ", u_left);
		fprintf(stderr, "u_right: %lf ", u_right);
		fprintf(stderr, "u_front: %lf ", u_front);
		fprintf(stderr, "u_back: %lf ", u_back);
		fprintf(stderr, "u0: %lf\n", u0);
	}
	
	MPI_Status status;

	MPI_Bcast(&n_blocks_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n_blocks_y, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n_blocks_z, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&block_size_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&block_size_y, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&block_size_z, 1, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Bcast(&output_file_path_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(output_file_path, output_file_path_len + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
	
	MPI_Bcast(&eps, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lx, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ly, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&lz, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u_down, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u_up, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u_left, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u_right, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u_front, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u_back, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&u0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double hx = lx / (n_blocks_x * block_size_x), hy = ly / (n_blocks_y * block_size_y), hz = lz / (n_blocks_z * block_size_z);

	int block_i = block_index_i(rank), block_j = block_index_j(rank), block_k = block_index_k(rank);

	MPI_Barrier(MPI_COMM_WORLD);

	int n_cells_per_block = (block_size_x + 2) * (block_size_y + 2) * (block_size_z + 2);
	double *u = (double*)malloc(sizeof(double) * n_cells_per_block);

	printf("line %d rank %d\n", __LINE__, rank);

	for(int i = 0; i < n_cells_per_block; ++i)
		u[i] = 0.0;

	for(int k = 0; k < block_size_z; ++k){
		for(int j = 0; j < block_size_y; ++j){
			for(int i = 0; i < block_size_x; ++i){
				u[cell_index(i, j, k)] = u0;
			}
		}
	}

	int max_dim = max(block_size_x, max(block_size_y, block_size_z));
	double *buffer = (double*)malloc(sizeof(double) * max_dim * max_dim);
	double *buffer1 = (double*)malloc(sizeof(double) * max_dim * max_dim);
	assert(buffer != NULL);
	assert(buffer1 != NULL);

	int device_count;
	CHECK_CUDA_CALL_ERROR(cudaGetDeviceCount(&device_count));
	

	int split_type;
	if(max_dim == block_size_x){
		split_type = LEFTRIGHT;
		printf("split_type: LEFTRIGHT\n");
	}
	else if(max_dim == block_size_y){
		split_type = FRONTBACK;
		printf("split_type: FRONTBACK\n");
	}
	else{
		split_type = UPDOWN;
		printf("split_type: UPDOWN\n");
	}

	int n_used_devices = min(device_count, max_dim);

	if(block_size_x * block_size_y * block_size_z < MULTIPLE_GPU_CRITERIA)
		n_used_devices = 1;

	double* dev_u[n_used_devices];
	double* dev_u1[n_used_devices]; 

	int block_sizes_x[n_used_devices];
	int block_sizes_y[n_used_devices];
	int block_sizes_z[n_used_devices];

	if(split_type == LEFTRIGHT){
		int per_block_size = block_size_x / n_used_devices;
		int rest = block_size_x % n_used_devices;
		for(int i = 0; i < n_used_devices; ++i){
			block_sizes_x[i] = i != n_used_devices-1 ? per_block_size : (per_block_size + rest);
			block_sizes_y[i] = block_size_y;
			block_sizes_z[i] = block_size_z;
		}
	}
	else if(split_type == FRONTBACK) {
		int per_block_size = block_size_y / n_used_devices;
		int rest = block_size_y % n_used_devices;
		for(int i = 0; i < n_used_devices; ++i){
			block_sizes_x[i] = block_size_x;
			block_sizes_y[i] = i != n_used_devices-1 ? per_block_size : (per_block_size + rest);
			block_sizes_z[i] = block_size_z;
		}
	}
	else if(split_type == UPDOWN){
		int per_block_size = block_size_z / n_used_devices;
		int rest = block_size_z % n_used_devices;
		for(int i = 0; i < n_used_devices; ++i){
			block_sizes_x[i] = block_size_x;
			block_sizes_y[i] = block_size_y;
			block_sizes_z[i] = i != n_used_devices-1 ? per_block_size : (per_block_size + rest);
		}
	}

	printf("line %d rank %d\n", __LINE__, rank);
	
	// -- copying data on devices -- 
	int n_cells = (block_sizes_x[0] + 2) * (block_sizes_y[0] + 2) * (block_sizes_z[0] + 2);
	int last_n_cells = (block_sizes_x[n_used_devices-1] + 2) * (block_sizes_y[n_used_devices-1] + 2) * (block_sizes_z[n_used_devices-1] + 2);

	double *temp = (double*)malloc(last_n_cells * sizeof(double));

    for(int d = 0; d < n_used_devices; ++d){
		int cur_n_cells = d < n_used_devices-1 ? n_cells : last_n_cells;
		
		CHECK_CUDA_CALL_ERROR(cudaMalloc(&dev_u[d], cur_n_cells * sizeof(double)));
		CHECK_CUDA_CALL_ERROR(cudaMalloc(&dev_u1[d], cur_n_cells * sizeof(double)));
        for(int k = -1; k < block_sizes_z[d]+1; ++k){
            for(int j = -1; j < block_sizes_y[d]+1; ++j){
                for(int i = -1; i < block_sizes_x[d]+1; ++i){
                    int index = (((k) + 1) * ((block_sizes_x[d] + 2) * (block_sizes_y[d] + 2)) + ((j) + 1) * (block_sizes_x[d] + 2) + ((i) + 1));
                    if(split_type == LEFTRIGHT){
                        temp[index] = u[cell_index(i + block_sizes_x[0] * d, j, k)];
                    }
                    else if(split_type == FRONTBACK){
                        temp[index] = u[cell_index(i, j + block_sizes_y[0] * d, k)];
                    }
                    else if(split_type == UPDOWN){
                        temp[index] = u[cell_index(i, j, k + block_sizes_z[0] * d)];
                    }
                }
            }
        }

		CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_u[d], temp, cur_n_cells * sizeof(double), cudaMemcpyHostToDevice));
		CHECK_CUDA_CALL_ERROR(cudaMemcpy(dev_u1[d], dev_u[d], cur_n_cells * sizeof(double), cudaMemcpyDeviceToDevice));
	}
	printf("line %d rank %d\n", __LINE__, rank);
	
	double* buffers[n_used_devices];
	double* dev_buffers[n_used_devices];
    for(int d = 0; d < n_used_devices; ++d){
		buffers[d] = NULL;
		dev_buffers[d] = NULL;
		buffers[d] = (double*)malloc(sizeof(double) * max_dim * max_dim);
		CHECK_CUDA_CALL_ERROR(cudaMalloc(&dev_buffers[d], sizeof(double) * max_dim * max_dim));
		assert(buffers[d] != NULL);
		assert(dev_buffers[d] != NULL);
		// assert(0);
	}
	double *dev_buffer = NULL;
	CHECK_CUDA_CALL_ERROR(cudaMalloc(&dev_buffer, sizeof(double) * max_dim * max_dim));
	assert(dev_buffer != NULL);

	double max_error = 100.0;
	int iter = 0;
	while(max_error > eps){
		if(block_i > 0){
			printf("line %d rank %d iter: %d\n", __LINE__, rank, iter);
			// отсылка и прием левого граничного условия
			get_border_multigpu(buffer, dev_buffer, dev_u, buffers, dev_buffers, LEFT, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);

			int count = block_size_y * block_size_z;
			int exchange_process_rank = block_index(block_i-1, block_j, block_k);

			MPI_Sendrecv(buffer, count, MPI_DOUBLE, exchange_process_rank, LEFT,
				buffer1, count, MPI_DOUBLE, exchange_process_rank, RIGHT,
				MPI_COMM_WORLD, &status);
			
			set_border_multigpu(dev_u, buffer1, dev_buffer, buffers, dev_buffers, LEFT, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);
			set_border_multigpu(dev_u1, buffer1, dev_buffer, buffers, dev_buffers, LEFT, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);
		}
		else{
			for(int j = 0; j < block_size_y; ++j)
				for(int k = 0; k < block_size_z; ++k)
					buffer1[j * block_size_z + k] = u_left;
			set_border_multigpu(dev_u, buffer1, dev_buffer, buffers, dev_buffers, LEFT, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);
			set_border_multigpu(dev_u1, buffer1, dev_buffer, buffers, dev_buffers, LEFT, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);
		}

		if(block_i < n_blocks_x-1){
			printf("line %d rank %d\n", __LINE__, rank);

			// отсылка и прием правого граничного условия
			get_border_multigpu(buffer, dev_buffer, dev_u, buffers, dev_buffers, RIGHT, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);

			int count = block_size_y * block_size_z;
			int exchange_process_rank = block_index(block_i+1, block_j, block_k);
			
			MPI_Sendrecv(buffer, count, MPI_DOUBLE, exchange_process_rank, RIGHT,
				buffer1, count, MPI_DOUBLE, exchange_process_rank, LEFT,
				MPI_COMM_WORLD, &status);

			set_border_multigpu(dev_u, buffer1, dev_buffer, buffers, dev_buffers, RIGHT, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);
			set_border_multigpu(dev_u1, buffer1, dev_buffer, buffers, dev_buffers, RIGHT, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);

		}
		else{
			for(int j = 0; j < block_size_y; ++j)
				for(int k = 0; k < block_size_z; ++k)
					buffer1[j * block_size_z + k] = u_right;
			set_border_multigpu(dev_u, buffer1, dev_buffer, buffers, dev_buffers, RIGHT, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);
			set_border_multigpu(dev_u1, buffer1, dev_buffer, buffers, dev_buffers, RIGHT, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);
		}

		if(block_j > 0){
			// отсылка и прием переднего граничного условия
			printf("line %d rank %d\n", __LINE__, rank);

			get_border_multigpu(buffer, dev_buffer, dev_u, buffers, dev_buffers, FRONT, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);

			int count = block_size_x * block_size_z;
			int exchange_process_rank = block_index(block_i, block_j-1, block_k);
			
			MPI_Sendrecv(buffer, count, MPI_DOUBLE, exchange_process_rank, FRONT,
				buffer1, count, MPI_DOUBLE, exchange_process_rank, BACK,
				MPI_COMM_WORLD, &status);

			set_border_multigpu(dev_u, buffer1, dev_buffer, buffers, dev_buffers, FRONT, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);
			set_border_multigpu(dev_u1, buffer1, dev_buffer, buffers, dev_buffers, FRONT, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);

		}
		else{
			for(int i = 0; i < block_size_x; ++i)
				for(int k = 0; k < block_size_z; ++k)
					buffer1[i * block_size_z + k] = u_front;
			set_border_multigpu(dev_u, buffer1, dev_buffer, buffers, dev_buffers, FRONT, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);
			set_border_multigpu(dev_u1, buffer1, dev_buffer, buffers, dev_buffers, FRONT, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);
		}

		if(block_j < n_blocks_y-1){
			// отсылка и прием заднего граничного условия
			printf("line %d rank %d\n", __LINE__, rank);
			
			get_border_multigpu(buffer, dev_buffer, dev_u, buffers, dev_buffers, BACK, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);

			int count = block_size_x * block_size_z;
			int exchange_process_rank = block_index(block_i, block_j+1, block_k);
			
			MPI_Sendrecv(buffer, count, MPI_DOUBLE, exchange_process_rank, BACK,
				buffer1, count, MPI_DOUBLE, exchange_process_rank, FRONT,
				MPI_COMM_WORLD, &status);

			set_border_multigpu(dev_u, buffer1, dev_buffer, buffers, dev_buffers, BACK, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);
			set_border_multigpu(dev_u1, buffer1, dev_buffer, buffers, dev_buffers, BACK, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);
		}
		else{
			for(int i = 0; i < block_size_x; ++i)
				for(int k = 0; k < block_size_z; ++k)
					buffer1[i * block_size_z + k] = u_back;
			set_border_multigpu(dev_u, buffer1, dev_buffer, buffers, dev_buffers, BACK, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);
			set_border_multigpu(dev_u1, buffer1, dev_buffer, buffers, dev_buffers, BACK, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);
		}

		if(block_k > 0){
			// отсылка и прием нижнего граничного условия
			printf("line %d rank %d\n", __LINE__, rank);

			get_border_multigpu(buffer, dev_buffer, dev_u, buffers, dev_buffers, DOWN, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);

			int count = block_size_x * block_size_y;
			int exchange_process_rank = block_index(block_i, block_j, block_k-1);
			
			MPI_Sendrecv(buffer, count, MPI_DOUBLE, exchange_process_rank, DOWN,
				buffer1, count, MPI_DOUBLE, exchange_process_rank, UP,
				MPI_COMM_WORLD, &status);

			set_border_multigpu(dev_u, buffer1, dev_buffer, buffers, dev_buffers, DOWN, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);
			set_border_multigpu(dev_u1, buffer1, dev_buffer, buffers, dev_buffers, DOWN, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);
		}
		else{
			for(int i = 0; i < block_size_x; ++i)
				for(int j = 0; j < block_size_y; ++j)
					buffer1[i * block_size_y + j] = u_down;
			set_border_multigpu(dev_u, buffer1, dev_buffer, buffers, dev_buffers, DOWN, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);
			set_border_multigpu(dev_u1, buffer1, dev_buffer, buffers, dev_buffers, DOWN, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);
		}

		if(block_k < n_blocks_z-1){
			// отсылка и прием верхнего граничного условия
			printf("line %d rank %d\n", __LINE__, rank);

			get_border_multigpu(buffer, dev_buffer, dev_u, buffers, dev_buffers, UP, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);

			int count = block_size_x * block_size_y;
			int exchange_process_rank = block_index(block_i, block_j, block_k+1);
			
			MPI_Sendrecv(buffer, count, MPI_DOUBLE, exchange_process_rank, UP,
				buffer1, count, MPI_DOUBLE, exchange_process_rank, DOWN,
				MPI_COMM_WORLD, &status);

			set_border_multigpu(dev_u, buffer1, dev_buffer, buffers, dev_buffers, UP, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);
			set_border_multigpu(dev_u1, buffer1, dev_buffer, buffers, dev_buffers, UP, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);
		}
		else{
			for(int i = 0; i < block_size_x; ++i)
				for(int j = 0; j < block_size_y; ++j)
					buffer1[i * block_size_y + j] = u_up;
			set_border_multigpu(dev_u, buffer1, dev_buffer, buffers, dev_buffers, UP, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);
			set_border_multigpu(dev_u1, buffer1, dev_buffer, buffers, dev_buffers, UP, block_sizes_x, block_sizes_y, block_sizes_z, n_used_devices, split_type);
		}

		for(int d = 0; d < n_used_devices-1; ++d){
			CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
			int last_index = get_last_index(split_type, block_sizes_x, block_sizes_y, block_sizes_z, d);
			get_intergpu_border(buffer, dev_u[d], dev_buffer, last_index, block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], split_type);
			get_intergpu_border(buffer1, dev_u[d+1], dev_buffer, 0, block_sizes_x[d+1], block_sizes_y[d+1], block_sizes_z[d+1], split_type);
			CHECK_CUDA_CALL_ERROR(cudaSetDevice(d+1));
			set_intergpu_border(dev_u[d+1], buffer, dev_buffer, -1, block_sizes_x[d+1], block_sizes_y[d+1], block_sizes_z[d+1], split_type);
			set_intergpu_border(dev_u1[d+1], buffer, dev_buffer, -1, block_sizes_x[d+1], block_sizes_y[d+1], block_sizes_z[d+1], split_type);
			CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
			set_intergpu_border(dev_u[d], buffer1, dev_buffer, last_index+1, block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], split_type);
			set_intergpu_border(dev_u1[d], buffer1, dev_buffer, last_index+1, block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], split_type);
		}

		MPI_Barrier(MPI_COMM_WORLD);

		for(int d = 0; d < n_used_devices; ++d){
			CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
			
			compute_kernel<<<dim3(GRID_DIM, GRID_DIM, GRID_DIM), dim3(GRID_DIM, GRID_DIM, GRID_DIM)>>>(dev_u1[d], dev_u[d],
				block_sizes_x[d], block_sizes_y[d], block_sizes_z[d], hx, hy, hz);
				CHECK_CUDA_CALL_ERROR(cudaDeviceSynchronize());
				CHECK_CUDA_KERNEL_ERROR();	
		}
			
		double block_error = 0.0;
		for(int d = 0; d < n_used_devices; ++d){
			CHECK_CUDA_CALL_ERROR(cudaSetDevice(d));
			int cur_gpu_block_size = (block_sizes_x[d] + 2) * (block_sizes_y[d] + 2) * (block_sizes_z[d] + 2);

			CHECK_CUDA_CALL_ERROR(cudaDeviceSynchronize());
			abs_error_kernel<<<dim3(GRID_DIM, GRID_DIM, GRID_DIM), dim3(GRID_DIM, GRID_DIM, GRID_DIM)>>>(dev_u[d], dev_u1[d],
				block_sizes_x[d], block_sizes_y[d], block_sizes_z[d]);
			CHECK_CUDA_KERNEL_ERROR();

			CHECK_CUDA_CALL_ERROR(cudaDeviceSynchronize());

			thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(dev_u[d]);
			double error = *thrust::max_element(dev_ptr, dev_ptr + cur_gpu_block_size);
			CHECK_CUDA_KERNEL_ERROR();

			if(error > block_error)
				block_error = error;

			double *temp = dev_u1[d];
			dev_u1[d] = dev_u[d];
			dev_u[d] = temp;
		}

		// вычисление максимальной ошибки
		MPI_Allreduce(&block_error, &max_error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
		iter++;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	for(int d = 0; d < n_used_devices; ++d){
		int cur_n_cells = d < n_used_devices-1 ? n_cells : last_n_cells;
		
		CHECK_CUDA_CALL_ERROR(cudaMemcpy(temp, dev_u[d], cur_n_cells * sizeof(double), cudaMemcpyDeviceToHost));

        for(int k = -1; k < block_sizes_z[d]+1; ++k){
            for(int j = -1; j < block_sizes_y[d]+1; ++j){
                for(int i = -1; i < block_sizes_x[d]+1; ++i){
                    int index = (((k) + 1) * ((block_sizes_x[d] + 2) * (block_sizes_y[d] + 2)) + ((j) + 1) * (block_sizes_x[d] + 2) + ((i) + 1));
                    if(split_type == LEFTRIGHT){
                        u[cell_index(i + block_sizes_x[0] * d, j, k)] = temp[index];
                    }
                    else if(split_type == FRONTBACK){
                        u[cell_index(i, j + block_sizes_y[0] * d, k)] = temp[index];
                    }
                    else if(split_type == UPDOWN){
                        u[cell_index(i, j, k + block_sizes_z[0] * d)] = temp[index];
                    }
                }
            }
        }
	}

	int n_outputs_per_block = block_size_y * block_size_z;
	
    MPI_Datatype string_type;
 	double d = 1.234567890;
    int len = snprintf(NULL, 0, "% e ", d);
    int str_length = len * block_size_x;

    MPI_Type_contiguous(str_length, MPI_CHAR, &string_type);
    MPI_Type_commit(&string_type);

    MPI_Datatype pattern_type;
    int *lens = (int*)malloc(n_outputs_per_block * sizeof(int));
    MPI_Aint *disps = (MPI_Aint*)malloc(n_outputs_per_block * sizeof(MPI_Aint));

	int y_stride = block_size_x * n_blocks_x;
    int z_stride = y_stride * block_size_y * n_blocks_y;

	int global_y = block_j * block_size_y;
	int global_z = block_k * block_size_z;

	int disp = (z_stride * global_z + y_stride * global_y + block_i * block_size_x) / block_size_x * str_length * sizeof(char);

	// умножаем на string_length т.к. для hindex смещения в байтах
	lens[0] = 1;
	disps[0] = 0;

	for(int i = 1; i < n_outputs_per_block; ++i){
        int increase = ((i % block_size_y != 0) ? n_blocks_x : n_blocks_x + (n_blocks_y - 1) * n_blocks_x * block_size_y);
		lens[i] = 1;
		disps[i] = disps[i-1] + increase * str_length;
	}

    MPI_Type_create_hindexed(n_outputs_per_block, lens, disps, string_type, &pattern_type);
    MPI_Type_commit(&pattern_type);

    char *str = (char*)malloc((str_length * block_size_y * block_size_z + 1) * sizeof(char));

	for(int k = 0; k < block_size_z; ++k){
        for(int j = 0; j < block_size_y; ++j){
            for(int i = 0; i < block_size_x-1; ++i){
				int offset = len * (i + block_size_x * j + (block_size_x * block_size_y) * k);
                sprintf(str + offset, "% e ", u[cell_index(i, j, k)]);
            }
			int offset = len * ((block_size_x - 1) + block_size_x * j + (block_size_x * block_size_y) * k);
			if(block_i == n_blocks_x - 1)
				sprintf(str + offset, "% e\n", u[cell_index(block_size_x-1, j, k)]);
			else
				sprintf(str + offset, "% e ", u[cell_index(block_size_x-1, j, k)]);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_File fd;

    MPI_File_open(MPI_COMM_WORLD, output_file_path, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fd);

    MPI_File_set_view(fd, disp, string_type, pattern_type, "native", MPI_INFO_NULL);
	
    MPI_File_write(fd, str, n_outputs_per_block, string_type, MPI_STATUS_IGNORE);

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_File_close(&fd);

    MPI_Type_free(&string_type);
    MPI_Type_free(&pattern_type);

    free(lens);
    free(disps);
	free(str);
	free(temp);

	free(output_file_path);
	free(buffer);
	free(buffer1);
	free(u);

	for(int d = 0; d < n_used_devices; ++d){
        free(buffers[d]);
		CHECK_CUDA_CALL_ERROR(cudaFree(dev_buffers[d]));
		CHECK_CUDA_CALL_ERROR(cudaFree(dev_u[d]));
		CHECK_CUDA_CALL_ERROR(cudaFree(dev_u1[d]));
	}
	CHECK_CUDA_CALL_ERROR(cudaFree(dev_buffer));

    MPI_Finalize();	
}
