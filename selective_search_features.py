#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy

def color_histgram(input_img, label_img, n_region):
    n_bin = 25
    bin_width = int(math.ceil(255.0 / n_bin))

    h, w = label_img.shape
    hist = numpy.zeros((n_region, 25 * 3))
    for y in range(h):
        for x in range(w):
            label = label_img[y, x]
            r, g, b = input_img[y, x] / bin_width
            hist[label, int(r) + 50] += 1
            hist[label, int(g) + 25] += 1
            hist[label, int(b)     ] += 1
    return hist

