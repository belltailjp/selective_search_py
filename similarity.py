#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

def size(size1, size2, size_img):
    return 1. - float(size1 + size2) / size_img

def __histogram_intersection(vec1, vec2):
    return numpy.sum(numpy.minimum(vec1, vec2), axis = 1)

def color(hist1, hist2):
    return __histogram_intersection(hist1, hist2)

