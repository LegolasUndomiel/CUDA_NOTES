#include "cudaTimer.cuh"
#include <cublas_v2.h>

#define N (1024 * 1024)

// 使用 __restrict 关键字来优化内存访问
// __restrict 告诉编译器指针指向的内存不会被其他指针所指向
template <typename T> __global__ void addKernel(T *__restrict a, T *__restrict b, T *__restrict c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < n) {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

void test01() {
    // 使用自定义的 CUDA 核函数来执行向量加法
    printf("Running test01: Vector Addition\n");
    int n = N;
    // CPU内存
    int *h_a = (int *)malloc(n * sizeof(int));
    int *h_b = (int *)malloc(n * sizeof(int));
    int *h_c = (int *)malloc(n * sizeof(int));
    for (int i = 0; i < n; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // GPU显存
    cudaTimer timer;
    float elapsedTime;
    int *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void **)&d_a, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_b, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_c, n * sizeof(int)));

    // 将数据从CPU内存复制到GPU显存
    timer.start();
    CUDA_CHECK(cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice));
    elapsedTime = timer.stop();
    printf("Copy data from host to device time: %g ms\n", elapsedTime);
    timer.start();
    CUDA_CHECK(cudaMemcpy(d_b, h_b, n * sizeof(int), cudaMemcpyHostToDevice));
    elapsedTime = timer.stop();
    printf("Copy data from host to device time: %g ms\n", elapsedTime);

    // 执行核函数
    timer.start();
    addKernel<<<128, 256>>>(d_a, d_b, d_c, n);

    // 确保核函数执行没有错误
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    elapsedTime = timer.stop();
    printf("Add kernel time: %g ms\n", elapsedTime);

    // 将结果从GPU显存复制回CPU内存
    timer.start();
    CUDA_CHECK(cudaMemcpy(h_c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost));
    elapsedTime = timer.stop();
    printf("Copy data from device to host time: %g ms\n", elapsedTime);

    // 验证结果
    for (int i = 0; i < n; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            fprintf(stderr, "Error at index %d: expected %d, got %d\n", i, h_a[i] + h_b[i], h_c[i]);
            exit(EXIT_FAILURE);
        }
    }
    printf("All results are correct!\n");

    // 释放 GPU显存
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    // 释放 CPU内存
    free(h_a);
    free(h_b);
    free(h_c);
}

void test02() {
    // 使用cuBLAS库来执行向量加法
    printf("Running test02: Vector Addition with cuBLAS\n");
    int n = N;
    // CPU内存
    float *h_a = (float *)malloc(n * sizeof(float));
    float *h_b = (float *)malloc(n * sizeof(float));
    float *h_c = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; ++i) {
        h_a[i] = i * 1.0f;
        h_b[i] = i * i * 2.0f;
    }

    // GPU显存
    cudaTimer timer;
    float elapsedTime;
    float *d_a, *d_b;
    CUDA_CHECK(cudaMalloc((void **)&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_b, n * sizeof(float)));

    // 创建 cuBLAS 句柄
    cublasHandle_t handle;
    cublasCreate(&handle);

    // 将数据从CPU内存复制到GPU显存
    // cublasSetVector(int n, int elemSize, const void *x, int incx, void *devicePtr, int incy)
    timer.start();
    cublasSetVector(n, sizeof(float), h_a, 1, d_a, 1);
    elapsedTime = timer.stop();
    printf("Copy data from host to device time: %g ms\n", elapsedTime);
    timer.start();
    cublasSetVector(n, sizeof(float), h_b, 1, d_b, 1);
    elapsedTime = timer.stop();
    printf("Copy data from host to device time: %g ms\n", elapsedTime);

    // 使用 cuBLAS 执行向量加法
    float alpha = 1.0f;
    // cublasSaxpy(cublasHandle_t handle, int n, const float *alpha, const float *x, int incx, float *y, int incy)
    timer.start();
    cublasSaxpy(handle, n, &alpha, d_a, 1, d_b, 1); // d_b = d_a + 1.0f * d_b
    elapsedTime = timer.stop();
    printf("cublasSaxpy time: %g ms\n", elapsedTime);

    // 将结果从GPU显存复制回CPU内存
    // cublasGetVector(int n, int elemSize, const void *x, int incx, void *y, int incy)
    timer.start();
    cublasGetVector(n, sizeof(float), d_b, 1, h_c, 1);
    elapsedTime = timer.stop();
    printf("Copy data from device to host time: %g ms\n", elapsedTime);

    // 验证结果
    for (int i = 0; i < n; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            fprintf(stderr, "Error at index %d: expected %f, got %f\n", i, h_a[i] + h_b[i], h_c[i]);
            exit(EXIT_FAILURE);
        }
    }
    printf("All results are correct!\n");

    // 释放 GPU显存
    cudaFree(d_a);
    cudaFree(d_b);
    // 释放 CPU内存
    free(h_a);
    free(h_b);
    free(h_c);

    // 销毁cuBLAS 句柄
    cublasDestroy(handle);
}

int main(int argc, char const *argv[]) {
    test01();
    test02();
    return 0;
}
