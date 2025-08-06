#ifndef __MANDELBROT_H__
#define __MANDELBROT_H__
#pragma once

#include "cudaTimer.h"
#include <GLFW/glfw3.h>
#include <glad/glad.h>
#include <matplotlibcpp.h>

class Mandelbrot {
  private:
    /* data */
    int width_, height_, maxIterations_;
    unsigned short int *h_data_;
    unsigned short int *d_data_;

  public:
    Mandelbrot(int, int, int);
    ~Mandelbrot();
    void pixelCalculation();
    void save();
    void copyBack();
    void renderWithOpenGL();
};

#endif
