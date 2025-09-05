#ifndef __CUDA_ERR_MSG_H__
#define __CUDA_ERR_MSG_H__
#pragma once

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

#endif // __CUDA_ERR_MSG_H__
