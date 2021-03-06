#ifndef ERROR_H
#define ERROR_H

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA_CALL_ERROR(ERR)                                                                             \
    do {                                                                                                       \
        cudaError_t err = ERR;                                                                                 \
        if (err != cudaSuccess) {                                                                              \
            printf("ERROR: %s (error_code: %d) at %s:%d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(0);                                                                                           \
        }                                                                                                      \
    } while (0)

#define CHECK_CUDA_KERNEL_ERROR()                                                                              \
    do {                                                                                                       \
        cudaError_t err = cudaGetLastError();                                                                  \
        if (err != cudaSuccess) {                                                                              \
            printf("ERROR: %s (error_code: %d) at %s:%d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(0);                                                                                           \
        }                                                                                                      \
    } while (0)

void _sigsegv_handler(int signo) {
    printf("ERROR: segmentation fault\n");
    exit(0);
}

void _sigabrt_handler(int signo) {
    printf("ERROR: aborted\n");
    exit(0);
}

void init_error_handling() {
    signal(SIGSEGV, _sigsegv_handler);
    signal(SIGABRT, _sigabrt_handler);
}

#endif