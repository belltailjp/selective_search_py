#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import segment

def calc_adjacency_matrix(label_img, n_region):
    adjacency = numpy.diag([1] * n_region)
    h, w = label_img.shape[0], label_img.shape[1]
    for y in range(h):
        for x in range(w):
            here = label_img[y, x]
            if y < h - 1:
                b = label_img[y + 1, x]
                adjacency[here, b] = adjacency[b, here] = 1
            if x < w - 1:
                r = label_img[y, x + 1]
                adjacency[here, r] = adjacency[r, here] = 1

    return adjacency

