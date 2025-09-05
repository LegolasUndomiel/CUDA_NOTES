#ifndef __CUDA_TIMING_H__
#define __CUDA_TIMING_H__
#pragma once

#include "cudaErrMsg.cuh"
#include <stdexcept>

class cudaTimer {
  private:
    cudaEvent_t startEvent_, stopEvent_;
    cudaStream_t stream_;
    bool running_ = false;

  public:
    explicit cudaTimer(cudaStream_t stream = 0) : stream_(stream), running_(false) {
        CUDA_CHECK(cudaEventCreate(&startEvent_));
        CUDA_CHECK(cudaEventCreate(&stopEvent_));
    }

    ~cudaTimer() {
        CUDA_CHECK(cudaEventDestroy(startEvent_));
        CUDA_CHECK(cudaEventDestroy(stopEvent_));
    }

    // 禁用拷贝构造函数和赋值运算符
    cudaTimer(const cudaTimer &) = delete;
    cudaTimer &operator=(const cudaTimer &) = delete;

    // 开始计时
    void start() {
        if (running_)
            throw std::runtime_error("Timer is already running.");
        CUDA_CHECK(cudaEventRecord(startEvent_, stream_));
        running_ = true;
    }

    // 停止计时
    float stop(bool synchronize = true) {
        if (!running_)
            throw std::runtime_error("Timer is not running.");
        CUDA_CHECK(cudaEventRecord(stopEvent_, stream_));
        if (synchronize)
            CUDA_CHECK(cudaEventSynchronize(stopEvent_));

        float elapsedTime;
        CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, startEvent_, stopEvent_));
        running_ = false;
        return elapsedTime;
    }

    // 重置计时器
    void reset(cudaStream_t new_stream) {
        /* 注意：
        1. cudaEvent_t本质是时间标记点，记录时自动关联当前流
        2. 重用现有事件对象不会影响新流的时间测量
        3. 销毁重建反而会导致额外开销 */
        stream_ = new_stream;
        running_ = false;
    }
};

#endif // __CUDA_TIMING_H__