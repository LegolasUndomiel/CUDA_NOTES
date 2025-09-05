#include "mandelbrot.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(mandelbrot, m) {
    m.doc() = "mandelbrot set";
    py::class_<Mandelbrot>(m, "Mandelbrot")
        .def(py::init<unsigned int, unsigned int, unsigned int>()) // (width, height, maxIterations)
        .def("pixelCalculation", &Mandelbrot::pixelCalculation, "CPU version")
        .def("pixelCalculationOMP", &Mandelbrot::pixelCalculationOMP, "OpenMP version")
        .def("pixelCalculationCUDA", &Mandelbrot::pixelCalculationCUDA, "CUDA version")
        .def("getData", &Mandelbrot::getData, "Get data, render with python-matplotlib");
}
