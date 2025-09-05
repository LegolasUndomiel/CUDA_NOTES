#ifndef __MANDELBROT_H__
#define __MANDELBROT_H__
#pragma once

#include <vector>

using std::vector;

class Mandelbrot {
  private:
    /* data */
    unsigned int width_, height_, maxIterations_;
    vector<unsigned short int> h_data_;
    unsigned short int *d_data_;

  public:
    /* method */
    Mandelbrot(unsigned int, unsigned int, unsigned int);
    ~Mandelbrot();

    void pixelCalculation();     // CPU version
    void pixelCalculationOMP();  // OpenMP version
    void pixelCalculationCUDA(); // CUDA version

    const vector<unsigned short int> &getData(); // Get data, render with python-matplotlib
    void renderWithOpenGL();                     // Render with OpenGL directly
};

#endif
