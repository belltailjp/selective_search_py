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
    A = scipy.sparse.coo_matrix((numpy.ones(t.shape[1]), (t[0], t[1])), shape=(n_region, n_region), dtype=bool).todense()
    A = A | A.transpose()

    for i in range(n_region):
        A[i, i] = True

    A = numpy.triu(A)
    dic = {i : {j for (j, Aij) in enumerate(Ai) if Aij and i != j} for (i, Ai) in enumerate(A)}

    Adjacency = collections.namedtuple('Adjacency', ['matrix', 'dictionary'])
    return Adjacency(matrix = A, dictionary = dic)

