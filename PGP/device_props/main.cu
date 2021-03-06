#include <stdio.h>

void display_device_proprties(){
    const int kb = 1024;
    const int mb = kb * kb;

    printf("CUDA version:   v%d\n", CUDART_VERSION);    

    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("CUDA Devices:\n\n");

    int deviceCount;
    cudaDeviceProp deviceProp;

    cudaGetDeviceCount(&deviceCount);
    printf("device count: %d\n", deviceCount);
    for (int i = 0; i < deviceCount; i++)
    {
        //Получаем информацию об устройстве
        cudaGetDeviceProperties(&deviceProp, i);

        //Выводим иформацию об устройстве
        printf("Device name: %s\n", deviceProp.name);
        printf("Total global memory: %lu mb\n", deviceProp.totalGlobalMem / mb);
        printf("Shared memory per block: %lu kb\n", deviceProp.sharedMemPerBlock / kb);
        printf("Total constant memory: %lu kb\n", deviceProp.totalConstMem / kb);
        printf("Registers per block: %d\n", deviceProp.regsPerBlock);
        printf("Warp size: %d\n", deviceProp.warpSize);
        printf("Memory pitch: %lu\n", deviceProp.memPitch);
        printf("Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        
        printf("Max threads dimensions: x = %d, y = %d, z = %d\n",
        deviceProp.maxThreadsDim[0],
        deviceProp.maxThreadsDim[1],
        deviceProp.maxThreadsDim[2]);
        
        printf("Max grid size: x = %d, y = %d, z = %d\n",
        deviceProp.maxGridSize[0],
        deviceProp.maxGridSize[1],
        deviceProp.maxGridSize[2]);

        printf("Clock rate: %d\n", deviceProp.clockRate);
        printf("Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("Texture alignment: %lu\n", deviceProp.textureAlignment);
        printf("Device overlap: %d\n", deviceProp.deviceOverlap);
        printf("Multiprocessor count: %d\n", deviceProp.multiProcessorCount);

        printf("Kernel execution timeout enabled: %s\n",
        deviceProp.kernelExecTimeoutEnabled ? "true" : "false");
    }

    for(int i = 0; i < devCount; ++i)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        printf("%d: %s: (compute capability: %d.%d)\n", i, props.name, props.major, props.minor);
        printf("\tGlobal memory:   %lu mb\n", props.totalGlobalMem / mb);

        printf("\tShared memory:   %lu kb\n", props.sharedMemPerBlock / kb);
        printf("\tConstant memory:   %lu kb\n", props.totalConstMem / kb);
        printf("\tBlock registers:   %d\n\n", props.regsPerBlock);
        printf("\tWarp size:   %d\n", props.warpSize);
        printf("\tMax threads per block:   %d\n", props.maxThreadsPerBlock);
        // printf("\tMax blocks per multiprocessor:   %d\n", props.maxBlocksPerMultiProcessor);
        printf("\tMax threads all dimensions:   %d\n", props.maxThreadsDim[0] * props.maxThreadsDim[1] * props.maxThreadsDim[2]);


        printf("\tMultiprocessor count:   %d\n", props.multiProcessorCount);

        printf("\tMax block dimensions: [ %d, %d, %d ]\n", props.maxThreadsDim[0], props.maxThreadsDim[1], props.maxThreadsDim[2]);
        printf("\tMax grid dimensions:  [ %d, %d, %d ]\n\n", props.maxGridSize[0], props.maxGridSize[1], props.maxGridSize[2]);
    }
} 

int main(){
    display_device_proprties();
}
