#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import scipy.sparse
import segment
import collections

def calc_adjacency_matrix(label_img, n_region):
    r = numpy.vstack([label_img[:, :-1].ravel(), label_img[:, 1:].ravel()])
    b = numpy.vstack([label_img[:-1, :].ravel(), label_img[1:, :].ravel()])
    t = numpy.hstack([r, b])
    A = scipy.sparse.coo_matrix((numpy.ones(t.shape[1]), (t[0], t[1])), shape=(n_region, n_region), dtype=bool).todense().getA()
    A = A | A.transpose()

    for i in range(n_region):
        A[i, i] = True

    dic = {i : {i} ^ set(numpy.flatnonzero(A[i])) for i in range(n_region)}

    Adjacency = collections.namedtuple('Adjacency', ['matrix', 'dictionary'])
    return Adjacency(matrix = A, dictionary = dic)

