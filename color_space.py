#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import skimage.io
import skimage.color


def to_grey(I):
    grey_img = (255 * skimage.color.rgb2grey(I)).astype(numpy.uint8)
    return numpy.dstack([grey_img, grey_img, grey_img])

def to_Lab(I):
    lab = skimage.color.rgb2lab(I)
    l = 255 * lab[:, :, 0] / 100    # L component ranges from 0 to 100
    a = 127 + lab[:, :, 1]          # a component ranges from -127 to 127
    b = 127 + lab[:, :, 2]          # b component ranges from -127 to 127
    return numpy.dstack([l, a, b]).astype(numpy.uint8)

def to_rgI(I):
    rgi = I.copy()
    rgi[:, :, 2] = to_grey(I)[:, :, 0]
    return rgi

def to_HSV(I):
    return (255 * skimage.color.rgb2hsv(I)).astype(numpy.uint8)

def to_nRGB(I):
    _I = I / 255.0
    norm_I = numpy.sqrt(_I[:, :, 0] ** 2 + _I[:, :, 1] ** 2 + _I[:, :, 2] ** 2)
    norm_r = (_I[:, :, 0] / norm_I * 255).astype(numpy.uint8)
    norm_g = (_I[:, :, 1] / norm_I * 255).astype(numpy.uint8)
    norm_b = (_I[:, :, 2] / norm_I * 255).astype(numpy.uint8)
    return numpy.dstack([norm_r, norm_g, norm_b])

def to_Hue(I):
    I_h = to_HSV(I)[:, :, 0]
    return numpy.dstack([I_h, I_h, I_h])

