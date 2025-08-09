#ifndef __MANDELBROT_H__
#define __MANDELBROT_H__
#pragma once

class Mandelbrot {
  private:
    /* data */
    int width_, height_;
    unsigned short int maxIterations_;
    unsigned short int *h_data_;
    unsigned short int *d_data_;

  public:
    Mandelbrot(int, int, unsigned short int);
    ~Mandelbrot();
    void pixelCalculation();
    void copyBack();
    void renderWithOpenGL();
};

#endif
