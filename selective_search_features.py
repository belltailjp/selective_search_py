#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy
import skimage
import skimage.filters
import scipy.ndimage.filters

def color_histgram(input_img, label_img, n_region):
    n_bin = 25
    bin_width = int(math.ceil(255.0 / n_bin))

    bins_color = [i * bin_width for i in range(n_bin + 1)]
    bins_label = range(n_region + 1)
    bins = [bins_label, bins_color]

    r_hist = numpy.histogram2d(label_img.ravel(), input_img[:, :, 0].ravel(), bins=bins)[0] #shape=(n_region, n_bin)
    g_hist = numpy.histogram2d(label_img.ravel(), input_img[:, :, 1].ravel(), bins=bins)[0]
    b_hist = numpy.histogram2d(label_img.ravel(), input_img[:, :, 2].ravel(), bins=bins)[0]
    hist = numpy.hstack([r_hist, g_hist, b_hist])
    l1_norm = numpy.sum(hist, axis = 1).reshape((n_region, 1))

    return numpy.nan_to_num(hist / l1_norm)


def size(label_img, n_region):
    return numpy.bincount(label_img.ravel(), minlength = n_region)


def bounding_box(label_img, n_region):
    B = numpy.full((n_region, 4), fill_value = float('NaN'))
    for ((i, j), label) in numpy.ndenumerate(label_img):
        (i1, j1, i2, j2) = B[label]
        B[label] = min(i, i1), min(j, j1), max(i, i2), max(j, j2)

    return B


def __gradient_histogram(label_img, gaussian, n_region, nbins_orientation = 8, nbins_inten = 10):
    op = numpy.array([[-1, 0, 1]], dtype=numpy.float32)
    h = scipy.ndimage.filters.convolve(gaussian, op)
    v = scipy.ndimage.filters.convolve(gaussian, op.transpose())
    g = numpy.arctan2(v, h)

    # define each axis for texture histogram
    bin_width = 2 * math.pi / 8
    bins_label = range(n_region + 1)
    bins_angle = numpy.linspace(-math.pi, math.pi, nbins_orientation + 1)
    bins_inten = numpy.linspace(.0, 1., nbins_inten + 1)
    bins = [bins_label, bins_angle, bins_inten]

    # calculate 3 dimensional histogram
    ar = numpy.vstack([label_img.ravel(), g.ravel(), gaussian.ravel()]).transpose()
    hist = numpy.histogramdd(ar, bins = bins)[0]

    # orientation_wise intensity histograms are serialized for each region
    return numpy.reshape(hist, (n_region, nbins_orientation * nbins_inten))

def texture(input_img, label_img, n_region):
    gaussian = skimage.filters.gaussian_filter(input_img, sigma = 1.0, multichannel = True).astype(numpy.float32)
    r_hist = __gradient_histogram(label_img, gaussian[:, :, 0], n_region)
    g_hist = __gradient_histogram(label_img, gaussian[:, :, 1], n_region)
    b_hist = __gradient_histogram(label_img, gaussian[:, :, 2], n_region)

    hist = numpy.hstack([r_hist, g_hist, b_hist])
    l1_norm = numpy.sum(hist, axis = 1).reshape((n_region, 1))

    return numpy.nan_to_num(hist / l1_norm)

