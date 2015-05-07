#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import numpy
import scipy.sparse
import segment
import collections

def _calc_adjacency_matrix(label_img, n_region):
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

def _new_adjacency_dict(A, i, j, t):
    Ak = copy.deepcopy(A)
    Ak[t] = (Ak[i] | Ak[j]) - {i, j}
    del Ak[i], Ak[j]
    for (p, Q) in Ak.items():
        if i in Q:
            Q.remove(i)
            Q.add(t)
        if j in Q:
            Q.remove(j)
            Q.add(t)

    return Ak

def _new_label_image(L, i, j, t):
    Lk = numpy.copy(L)
    Lk[Lk == i] = Lk[Lk == j] = t
    return Lk

def _build_initial_similarity_set(A0, feature_extractor):
    S = list()
    for (i, J) in A0.items():
        for j in J:
            if i < j:
                S.append((feature_extractor.similarity(i, j), (i, j)))

    return sorted(S)

