#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import itertools
import copy
import numpy
import scipy.sparse
import segment
import collections
import skimage.io
import features
import color_space

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
        if i in Q or j in Q:
            Q -= {i, j}
            Q.add(t)

    return Ak

def _new_label_image(F, i, j, t):
    Fk = numpy.copy(F)
    Fk[Fk == i] = Fk[Fk == j] = t
    return Fk

def _build_initial_similarity_set(A0, feature_extractor):
    S = list()
    for (i, J) in A0.items():
        S += [(feature_extractor.similarity(i, j), (i, j)) for j in J if i < j]

    return sorted(S)

def _merge_similarity_set(feature_extractor, Ak, S, i, j, t):
    # remove entries which have i or j
    S = list(filter(lambda x: not(i in x[1] or j in x[1]), S))

    # calculate similarity between region t and its adjacencies
    St = [(feature_extractor.similarity(t, x), (t, x)) for x in Ak[t] if t < x] +\
         [(feature_extractor.similarity(x, t), (x, t)) for x in Ak[t] if x < t]

    return sorted(S + St)

def hierarchical_segmentation(I, k = 100, feature_mask = features.SimilarityMask(1, 1, 1, 1)):
    F0, n_region = segment.segment_label(I, 0.8, k, 100)
    adj_mat, A0 = _calc_adjacency_matrix(F0, n_region)
    feature_extractor = features.Features(I, F0, n_region)

    # stores list of regions sorted by their similarity
    S = _build_initial_similarity_set(A0, feature_extractor)

    # stores region label and its parent (empty if initial).
    R = {i : set() for i in range(n_region)}

    A = [A0]    # stores adjacency relation for each step
    F = [F0]    # stores label image for each step

    # greedy hierarchical grouping loop
    while len(S):
        (s, (i, j)) = S.pop()
        t = feature_extractor.merge(i, j)
        R[t] = {i, j}

        Ak = _new_adjacency_dict(A[-1], i, j, t)
        A.append(Ak)

        S = _merge_similarity_set(feature_extractor, Ak, S, i, j, t)

        F.append(_new_label_image(F[-1], i, j, t))

    # bounding boxes for each hierarchy
    L = feature_extractor.bbox

    return (R, F, L)

def _generate_regions(R, L):
    n_ini = sum(not parent for parent in R.values())
    n_all = len(R)

    regions = list()
    for label in R.keys():
        i = min(n_all - n_ini + 1, n_all - label)
        vi = numpy.random.rand() * i
        regions.append((vi, L[i]))

    return sorted(regions)

def selective_search(I, color_spaces = ['rgb'], ks = [100], feature_masks = [features.SimilarityMask(1, 1, 1, 1)]):
    regions = list()
    for color_name in color_spaces:
        I_color = color_space.convert_color(I, color_name)
        for k in ks:
            for feature_mask in feature_masks:
                (R, F, L) = hierarchical_segmentation(I_color, k, feature_mask)
                regions += _generate_regions(R, L)
    return sorted(regions)

