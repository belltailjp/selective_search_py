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

    // Convert to internal format
    image<rgb> seg_input_img(w, h);
    rgb* p = reinterpret_cast<rgb*>(input_image.get_data());
    std::copy(p, p + w * h, seg_input_img.data);

    int num_css;
    image<rgb> *seg_result_img = segment_image(&seg_input_img, sigma, c, min_size, &num_css);

    // Convert from internal format
    boost::numpy::ndarray result_image = boost::numpy::empty(nd, input_image.get_shape(), input_image.get_dtype());
    std::copy(seg_result_img->data, seg_result_img->data + w * h, reinterpret_cast<rgb*>(result_image.get_data()));

    delete seg_result_img;
    return result_image;
}

BOOST_PYTHON_MODULE(segment)
{
    boost::numpy::initialize();
    boost::python::def("segment", segment);
}

