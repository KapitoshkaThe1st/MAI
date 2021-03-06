#ifndef CUDA_ERR
#define CUDA_ERR

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>

#define CHECK_CUDA_CALL_ERROR(ERR) _check_cuda_call_error(ERR, __FILE__, __LINE__)
#define CHECK_CUDA_KERNEL_ERROR() _check_cuda_kernel_error(__FILE__, __LINE__)

void _check_cuda_call_error(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("ERROR: %s (error_code: %d) at %s:%d\n", cudaGetErrorString(err), err, file, line);
        exit(0);
    }
}

void _check_cuda_kernel_error(const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    _check_cuda_call_error(err, file, line);
}

void sigsegv_handler(int signo) {
    printf("ERROR: segmentation fault\n");
    exit(0);
}

void sigabrt_handler(int signo) {
    printf("ERROR: aborted\n");
    exit(0);
}

void init_error_handling(){
    signal(SIGSEGV, sigsegv_handler);
    signal(SIGABRT, sigabrt_handler);
}

#endif