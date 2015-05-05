#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import scipy.sparse
import segment
import collections

def calc_adjacency_matrix(label_img, n_region):
    r = numpy.vstack([label_img[:, :-1].ravel(), label_img[:, 1:].ravel()])
    b = numpy.vstack([label_img[:-1, :].ravel(), label_img[1:, :].ravel()])
    t = numpy.hstack([r, b, r[::-1], b[::-1]])
    A = scipy.sparse.coo_matrix((numpy.ones(t.shape[1]), (t[0], t[1]))).todense()
    A = numpy.minimum(A, numpy.ones((n_region, n_region))).astype(int)

    for i in range(n_region):
        A[i, i] = 1

    A = numpy.triu(A)
    dic = {i : {j for (j, Aij) in enumerate(Ai) if Aij == 1 and i != j} for (i, Ai) in enumerate(A)}

    Adjacency = collections.namedtuple('Adjacency', ['matrix', 'dictionary'])
    return Adjacency(matrix = A, dictionary = dic)

