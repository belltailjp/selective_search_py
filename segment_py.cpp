#include <iostream>
#include <boost/python.hpp>
#include <boost/numpy.hpp>
#include "segment/segment-image.h"

boost::numpy::ndarray segment(const boost::numpy::ndarray& img, float sigma, float c, int min_size)
{
    return img;
}

BOOST_PYTHON_MODULE(segment)
{
    boost::numpy::initialize();
    boost::python::def("segment", segment);
}

