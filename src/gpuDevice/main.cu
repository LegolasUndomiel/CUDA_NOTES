#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

int main(int argc, char const *argv[]) {
    const int REQUIRED_MAJOR = 6;
    const int REQUIRED_MINOR = 0;

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    // Check for errors in getting device count
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device count: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    } else if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        exit(EXIT_FAILURE);
    }

    // Iterate through all devices and check their properties
    int bestDevice = -1;
    size_t maxGlobalMem = 0;
    cudaDeviceProp bestProp;

    for (int device = 0; device < deviceCount; ++device) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, device);
        // Check for errors in getting device properties
        if (err != cudaSuccess) {
            fprintf(stderr, "Error getting properties for device %d: %s\n", device, cudaGetErrorString(err));
            continue; // Skip to next device
        }
        // 根据计算能力和全局内存大小选择最优设备
        if (prop.major > REQUIRED_MAJOR || (prop.major == REQUIRED_MAJOR && prop.minor >= REQUIRED_MINOR)) {
            if (prop.totalGlobalMem > maxGlobalMem || (prop.totalGlobalMem == maxGlobalMem && device < bestDevice)) {
                maxGlobalMem = prop.totalGlobalMem;
                bestDevice = device;
                bestProp = prop;
            }
        }
    }

    if (bestDevice == -1) {
        fprintf(stderr, "No suitable CUDA device found with compute capability %d.%d or higher.\n", REQUIRED_MAJOR, REQUIRED_MINOR);
        exit(EXIT_FAILURE);
    }

    cudaSetDevice(bestDevice);
    printf("-------- General Information for device %d --------\n", bestDevice);
    printf("Device name: %s\n", bestProp.name);
    printf("Compute Capability: %d.%d\n", bestProp.major, bestProp.minor);
    printf("Clock Rate: %d MHz\n", bestProp.clockRate / 1000);
    printf("Device copy overlap: %s\n", bestProp.deviceOverlap ? "Enabled" : "Disabled");
    printf("Kernel execution timeout: %s\n", bestProp.kernelExecTimeoutEnabled ? "Enabled" : "Disabled");

    printf("-------- Memory Information for device %d --------\n", bestDevice);
    printf("Total Global Memory: %.2f GB\n", bestProp.totalGlobalMem / (1024 * 1024 * 1024.0));
    printf("Total Constant Memory: %d KB\n", (int)(bestProp.totalConstMem / 1024));
    printf("Max Memory Pitch: %.2f GB\n", bestProp.memPitch / (1024 * 1024 * 1024.0));
    printf("Texture Alignment: %zu bytes\n", bestProp.textureAlignment);

    printf("--------  MP Information for device %d   --------\n", bestDevice);
    printf("Multiprocessor Count: %d\n", bestProp.multiProcessorCount);
    printf("Shared Memory per Block: %d KB\n", (int)(bestProp.sharedMemPerBlock / 1024));
    printf("Registers per Block: %d\n", bestProp.regsPerBlock);
    printf("Warp Size: %d\n", bestProp.warpSize);
    printf("Max Threads per Block: %d\n", bestProp.maxThreadsPerBlock);
    printf("Max Threads Dimensions: [%d, %d, %d]\n", bestProp.maxThreadsDim[0], bestProp.maxThreadsDim[1], bestProp.maxThreadsDim[2]);
    printf("Max Grid Size: [%d, %d, %d]\n", bestProp.maxGridSize[0], bestProp.maxGridSize[1], bestProp.maxGridSize[2]);

    // 剩余内存查询
    size_t freeMem = 0, totalMem = 0;
    err = cudaMemGetInfo(&freeMem, &totalMem);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting memory info: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("Free Memory: %.2f GB\n", freeMem / (1024 * 1024 * 1024.0));
    printf("Total Memory: %.2f GB\n", totalMem / (1024 * 1024 * 1024.0));

    return 0;
}
