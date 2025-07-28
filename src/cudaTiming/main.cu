#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                                     \
    {                                                                                        \
        do {                                                                                 \
            const cudaError_t error = call;                                                  \
            if (error != cudaSuccess) {                                                      \
                fprintf(stderr, "CUDA Error: %s:%d\n", __FILE__, __LINE__);                  \
                fprintf(stderr, "code: %d\nreason: %s\n", error, cudaGetErrorString(error)); \
                exit(EXIT_FAILURE);                                                          \
            }                                                                                \
        } while (0);                                                                         \
    }

int main(int argc, char const *argv[]) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaEventSynchronize(start));
    // 此处不能使用 CUDA_CHECK，因为返回值可能是 cudaErrorNotReady
    // cudaEventQuery(start);

    // 计时

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Elapsed time: %g ms\n", milliseconds);

    // 销毁事件
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
