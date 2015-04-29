#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy

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
