#include "mandelbrot.h"

Mandelbrot::Mandelbrot(int width, int height, int maxIterations)
    : width_(width), height_(height), maxIterations_(maxIterations) {
    this->h_data_ = new unsigned short int[width_ * height_];
    for (int i = 0; i < width_ * height_; i++)
        this->h_data_[i] = 0;
    CUDA_CHECK(cudaMalloc((void **)&this->d_data_, width_ * height_ * sizeof(unsigned short int)));
}

Mandelbrot::~Mandelbrot() {
    delete[] this->h_data_;
    CUDA_CHECK(cudaFree(this->d_data_));
}
