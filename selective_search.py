#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import segment

def calc_adjacency_matrix(label_img, n_region):
    adjacency = numpy.zeros((n_region, n_region))
    for y in range(label_img.shape[0] - 1):
        for x in range(label_img.shape[1] - 1):
            here, right, down = label_img[y, x], label_img[y, x + 1], label_img[y + 1, x]
            adjacency[here, right] = adjacency[right, here] = 1

    return adjacency

