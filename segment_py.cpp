#include <iostream>
#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include "segment/segment-image.h"

boost::numpy::ndarray segment(const boost::numpy::ndarray& input_image, float sigma, float c, int min_size)
{
    const int nd = input_image.get_nd();
    if(nd != 3)
    {
        throw std::runtime_error("input_image must be 3-dimensional");
    }

    const int h = input_image.shape(0);
    const int w = input_image.shape(1);
    const int depth = input_image.shape(2);

    if(depth != 3)
    {
        throw std::runtime_error("input_image must have rgb channel");
    }

    if(input_image.get_dtype() != boost::numpy::dtype::get_builtin<unsigned char>())
    {
        throw std::runtime_error("dtype of input_image must be uint8");
    }

    if(!input_image.get_flags() & boost::numpy::ndarray::C_CONTIGUOUS)
    {
        throw std::runtime_error("input_image must be C-style contiguous");
    }

    return input_image;
}

BOOST_PYTHON_MODULE(segment)
{
    boost::numpy::initialize();
    boost::python::def("segment", segment);
}

