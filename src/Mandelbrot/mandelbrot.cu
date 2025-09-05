#include "cudaTimer.h"
#include "mandelbrot.h"
#include <chrono>
#include <cuda_runtime.h>
#include <iostream>
#include <omp.h>
#include <string>

using std::cout;
using std::endl;
using std::string;

using namespace std::chrono;

__host__ __device__ unsigned short int mandelbrot(float real, float imag, unsigned short int maxIterations) {
    float r = real;
    float i = imag;
    for (int iter = 0; iter < maxIterations; ++iter) {
        float r2 = r * r;
        float i2 = i * i;
        if (r2 + i2 > 4.0f) {
            return iter;
        }
        i = 2.0f * r * i + imag;
        r = r2 - i2 + real;
    }
    return maxIterations;
}

__global__ void Kernel(unsigned short int *data, unsigned int WIDTH, unsigned int HEIGHT, unsigned int maxIterations) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < WIDTH && y < HEIGHT) {
        float real = (x - WIDTH / 2.0) * 3.84 / WIDTH;
        float imag = (y - HEIGHT / 2.0) * 2.16 / HEIGHT;
        unsigned short int value = mandelbrot(real, imag, maxIterations);
        data[y * WIDTH + x] = value;
    }
}

Mandelbrot::Mandelbrot(unsigned int width, unsigned int height, unsigned int maxIterations)
    : width_(width), height_(height), maxIterations_(maxIterations) {
    h_data_.resize(width_ * height_);
    CUDA_CHECK(cudaMalloc((void **)&d_data_, width_ * height_ * sizeof(unsigned short int)));
}

Mandelbrot::~Mandelbrot() {
    // free memory on host
    h_data_.clear();         // delete data but keep the capacity of the vector
    h_data_.shrink_to_fit(); // free the capacity of the vector

    // free memory on device
    if (d_data_ != nullptr) {
        CUDA_CHECK(cudaFree(d_data_));
        d_data_ = nullptr;
    }
}

void Mandelbrot::pixelCalculation() {
    unsigned short int value;

    // 启动计时器
    auto start = high_resolution_clock::now();
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            float real = (x - width_ / 2.0) * 3.84 / width_;
            float imag = (y - height_ / 2.0) * 2.16 / height_;

            value = mandelbrot(real, imag, maxIterations_);

            h_data_[y * width_ + x] = value;
        }
    }

    // 停止计时器
    auto end = high_resolution_clock::now();

    // 计算持续时间
    duration<double, std::milli> duration = end - start;
    cout << "CPU Single Thread Version:" << duration.count() << "ms" << endl;
}

void Mandelbrot::pixelCalculationOMP() {
    unsigned short int value;

    int threads = omp_get_max_threads();
    // 启动计时器
    auto start = high_resolution_clock::now();
    omp_set_num_threads(threads);
#pragma omp parallel for
    for (int y = 0; y < height_; ++y) {
        for (int x = 0; x < width_; ++x) {
            float real = (x - width_ / 2.0) * 3.84 / width_;
            float imag = (y - height_ / 2.0) * 2.16 / height_;

            value = mandelbrot(real, imag, maxIterations_);

            h_data_[y * width_ + x] = value;
        }
    }

    // 停止计时器
    auto end = high_resolution_clock::now();

    // 计算持续时间
    duration<double, std::milli> duration = end - start;
    cout << "OpenMP " << threads << " Threads Version:" << duration.count() << "ms" << endl;
}

void Mandelbrot::pixelCalculationCUDA() {
    cudaTimer timer;
    dim3 blockDim(32, 32);
    dim3 gridDim((width_ + blockDim.x - 1) / blockDim.x, (height_ + blockDim.y - 1) / blockDim.y);

    timer.start();

    Kernel<<<gridDim, blockDim>>>(d_data_, width_, height_, maxIterations_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float elapsedTime = timer.stop();
    cout << "CUDA Kernel Execution Time: " << elapsedTime << "ms" << endl;

    // 将结果从GPU显存复制回CPU内存
    CUDA_CHECK(
        cudaMemcpy(h_data_.data(), d_data_, width_ * height_ * sizeof(unsigned short int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize()); // 确保数据传输完成
}

const vector<unsigned short int> &Mandelbrot::getData() { return h_data_; }

void Mandelbrot::renderWithOpenGL() {
    // This is a placeholder and not implemented in this snippet
    // OpenGL rendering code would go here
    cout << "Rendering with OpenGL (functionality not implemented in this snippet)." << endl;
}