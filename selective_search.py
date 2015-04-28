#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import segment
import collections

def calc_adjacency_matrix(label_img, n_region):
    A = numpy.diag([1] * n_region)
    h, w = label_img.shape[0], label_img.shape[1]
    for y in range(h):
        for x in range(w):
            here = label_img[y, x]
            if y < h - 1:
                b = label_img[y + 1, x]
                A[here, b] = A[b, here] = 1
            if x < w - 1:
                r = label_img[y, x + 1]
                A[here, r] = A[r, here] = 1

    dic = {i : {j for (j, Aij) in enumerate(Ai) if Aij == 1 and i != j} for (i, Ai) in enumerate(A)}

    Adjacency = collections.namedtuple('Adjacency', ['matrix', 'dictionary'])
    return Adjacency(matrix = A, dictionary = dic)

