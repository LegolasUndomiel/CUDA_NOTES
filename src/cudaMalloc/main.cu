#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

void cpuMalloc() {
    int *h_ptr = nullptr;
    // CPU端申请内存
    h_ptr = static_cast<int *>(malloc(1024 * sizeof(int)));

    // 检查内存分配是否成功
    if (h_ptr == nullptr) {
        // 内存分配失败异常处理
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    } else {
        // 内存分配成功
        printf("Host memory allocated successfully\n");

        // 初始化并使用内存(此处省略代码)

        // 释放内存
        free(h_ptr);
        printf("Host memory freed successfully\n");
    }

    int **h_ptr2 = nullptr;
    // CPU端申请内存，保存 int* 指针变量
    h_ptr2 = (int **)malloc(1024 * sizeof(int *));
    // 检查内存分配是否成功
    if (h_ptr2 == nullptr) {
        // 内存分配失败异常处理
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    } else {
        // 内存分配成功
        printf("Host memory allocated successfully\n");
        // 初始化并使用内存(此处省略代码)

        // 释放内存
        free(h_ptr2);
        printf("Host memory freed successfully\n");
    }
}

void gpuCUDAMalloc() {
    int *d_ptr = nullptr;
    // GPU端申请内存
    cudaError_t err = cudaMalloc((void **)&d_ptr, 1024 * sizeof(int));

    // 检查内存分配是否成功
    if (err != cudaSuccess) {
        // 内存分配失败异常处理
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    } else {
        // 内存分配成功
        printf("Device memory allocated successfully\n");

        // 拷贝内存(此处省略代码)

        // 使用内存(此处省略代码)

        // 释放内存
        cudaFree(d_ptr);
        printf("Device memory freed successfully\n");
    }

    int **d_ptr2 = nullptr;
    // GPU端申请内存，保存 int* 指针变量
    cudaError_t err2 = cudaMalloc((void **)&d_ptr2, 1024 * sizeof(int *));

    // 检查内存分配是否成功
    if (err2 != cudaSuccess) {
        // 内存分配失败异常处理
        fprintf(stderr, "Failed to allocate device memory for pointer: %s\n", cudaGetErrorString(err2));
        exit(EXIT_FAILURE);
    } else {
        // 内存分配成功
        printf("Device memory for pointer allocated successfully\n");

        // 拷贝内存(此处省略代码)

        // 使用内存(此处省略代码)

        // 释放内存
        cudaFree(d_ptr2);
        printf("Device memory for pointer freed successfully\n");
    }
}

enum class myCudaError_t {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
};

myCudaError_t myCudaMalloc(void **devPtr, size_t size) {
    // 使用malloc函数模仿 cudaMalloc 的行为
    // 以此说明 cudaMalloc 的第一个参数是一个指向指针的指针
    *devPtr = malloc(size);
    if (*devPtr == nullptr) {
        // 如果分配失败，返回错误代码
        return myCudaError_t::cudaErrorInvalidValue;
    }
    // 如果分配成功，返回成功代码
    return myCudaError_t::cudaSuccess;
}

void test() {
    // malloc 函数的作用是申请一块内存，并返回一个指向这块内存首地址的指针
    // 以此类比，cudaMalloc 函数的作用是申请一块 GPU 内存，并返回一个指向这块内存首地址的指针
    // 为了接收任意类型的指针，应该使用 void* 类型，因为 void* 可以接收任何类型的指针
    // 并且，malloc的返回值是 void* 类型，可以直接利用返回值，将申请的内存地址返回给申请者
    // 但是 cudaMalloc 的返回值是 cudaError_t 类型，已经占用了返回值的位置
    // 所以 cudaMalloc 函数需要一个参数来接收返回值
    // 这个参数需要是 void* 类型，可以保存任意类型的指针
    // 并且为了能够修改这个参数的值，需要传入指向这个参数的指针，即 void**
    // 这样就可以在函数内部修改 void* 参数的值
    // myCudaMalloc函数利用 malloc 函数模拟了 cudaMalloc 的行为，从而演示了 cudaMalloc 的参数和返回值的使用方式

    // 从Fortran语法中也可以看出，如果需要在函数中修改一个变量的值，就需要传入这个变量的地址
    // Fortran中函数的参数传递变量的地址，但是这样会导致传入函数的所有参数都可以在函数内部被修改
    // C/C++中函数的参数传递方式是值传递，即传入参数的副本
    // 所以在C/C++中，如果想在函数内部修改变量的值，需要传入变量的地址
    // 我猜测这也是C/C++中引入指针的原因之一
    int *d_ptr = nullptr;
    myCudaError_t err = myCudaMalloc((void **)&d_ptr, 1024 * sizeof(int));

    if (err != myCudaError_t::cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory\n");
        exit(EXIT_FAILURE);
    }
    printf("Device memory allocated successfully using myCudaMalloc\n");

    // 使用 d_ptr 进行其他操作...

    // 释放内存
    free(d_ptr);
    printf("Device memory freed successfully using myCudaMalloc\n");

    // 注意：由于 void* 可以接收任意类型的指针，所以不论所申请的内存是保存 int* 还是其他类型，都可以使用 void* 类型保存地址
    // 因为要修改 void* 参数变量的值，所以需要传入指向 void* 的指针，即 void**
    // 客观地讲，对于申请内存的函数来说，不论是 malloc 还是 cudaMalloc，这些函数并不关心内存中存储的具体类型
    // 它们只关心申请的内存大小，申请一块内存并返回一个指向这块内存首地址的指针
    // 至于这块内存中存储的是什么类型的数据，后续如何使用，这些完全是调用者的事情，内存申请函数并不关心
    int **d_ptr2 = nullptr;
    myCudaError_t err2 = myCudaMalloc((void **)&d_ptr2, 1024 * sizeof(int *));
    if (err2 != myCudaError_t::cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for pointer\n");
        exit(EXIT_FAILURE);
    }
    printf("Device memory for pointer allocated successfully using myCudaMalloc\n");

    // 使用 d_ptr2 进行其他操作...

    // 释放内chmod +x .husky/pre-commit存
    free(d_ptr2);
    printf("Device memory for pointer freed successfully using myCudaMalloc\n");
}

int main(int argc, char const *argv[]) {
    printf("CUDA Memory Allocation Example\n");
    cpuMalloc();
    gpuCUDAMalloc();
    test();
    return 0;
}
