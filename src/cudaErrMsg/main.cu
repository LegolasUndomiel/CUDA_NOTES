#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// 使用 do-while 循环是必要的，避免在使用宏时出现语法错误。
// 确保宏展开后始终作为单个语句执行
// 避免在 if 语句中使用宏时出现问题
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
    double *h_data = (double *)malloc(1024 * sizeof(double));
    double *d_data = nullptr;
    // 检查 CUDA 运行时 API 调用是否成功
    CUDA_CHECK(cudaMalloc((void **)&d_data, 1024 * sizeof(double)));
    // 故意写错 cudaMemcpyKind 以触发错误
    CUDA_CHECK(cudaMemcpy(d_data, h_data, 1024 * sizeof(double), cudaMemcpyDeviceToHost));

    // 运行一些 CUDA 内核（假设有一个内核函数）
    // kernel<<<1, 1>>>(d_data); // 假设有一个
    CUDA_CHECK(cudaGetLastError());      // 检查内核执行是否有错误
    CUDA_CHECK(cudaDeviceSynchronize()); // 确保所有 CUDA 操作完成

    // 释放设备内存
    CUDA_CHECK(cudaFree(d_data));
    // 释放主机内存
    free(h_data);
    return 0;
}
