#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nose
import nose.tools

import numpy
import selective_search

class TestCalcAdjecencyMatrix:
    def setup_method(self, method):
        self.label = numpy.zeros((4, 4), dtype=int)

    def test_only_1_segment(self):
        # 0, 0, 0, 0
        # 0, 0, 0, 0
        # 0, 0, 0, 0
        # 0, 0, 0, 0
        (adj_mat, adj_dic) = selective_search._calc_adjacency_matrix(self.label, 1)
        assert type(adj_mat) == numpy.ndarray
        assert adj_mat.shape == (1, 1) and adj_mat.dtype == bool
        assert adj_mat[0, 0] == True
        assert adj_dic[0] == set()

    def test_fully_adjacent(self):
        # 1, 1, 0, 0
        # 1, 1, 0, 0
        # 1, 1, 0, 0
        # 1, 1, 0, 0
        self.label[:2, :] = 1
        expected_mat = numpy.array([[True, True],\
                                    [True, True]])

        (adj_mat, adj_dic) = selective_search._calc_adjacency_matrix(self.label, 2)
        assert adj_mat.shape == (2, 2) and adj_mat.dtype == bool
        assert numpy.array_equal(adj_mat, expected_mat)
        assert adj_dic[0] == {1}
        assert adj_dic[1] == {0}

    def test_partially_adjacent(self):
        # 0, 0, 1, 1
        # 0, 0, 1, 1
        # 2, 2, 3, 3
        # 2, 2, 3, 3
        self.label[:2, :2] = 0
        self.label[:2, 2:] = 1
        self.label[2:, :2] = 2
        self.label[2:, 2:] = 3
        expected_mat = numpy.array([[True, True, True, False],\
                                    [True, True, False, True],\
                                    [True, False, True, True],\
                                    [False, True, True, True]])

        (adj_mat, adj_dic) = selective_search._calc_adjacency_matrix(self.label, 4)
        assert numpy.diag(adj_mat).all()
        assert numpy.array_equal(adj_mat.transpose(), adj_mat)
        assert numpy.array_equal(adj_mat, expected_mat)

        assert adj_dic[0] == {1, 2}
        assert adj_dic[1] == {0, 3}
        assert adj_dic[2] == {0, 3}
        assert adj_dic[3] == {1, 2}

    def test_edge_case_vertical(self):
        # 0, 0, 0, 1
        # 0, 0, 0, 1
        # 0, 0, 0, 2
        # 0, 0, 0, 2
        self.label[:2, -1:] = 1
        self.label[2:, -1:] = 2
        expected_mat = numpy.array([[True, True, True],\
                                    [True, True, True],\
                                    [True, True, True]])

        (adj_mat, adj_dic) = selective_search._calc_adjacency_matrix(self.label, 3)
        assert numpy.array_equal(expected_mat, adj_mat)
        assert adj_dic[0] == {1, 2}
        assert adj_dic[1] == {0, 2}
        assert adj_dic[2] == {0, 1}

    def test_edge_case_horizontal(self):
        # 0, 0, 0, 0
        # 0, 0, 0, 0
        # 0, 0, 0, 0
        # 1, 1, 2, 2
        self.label[-1:, :2] = 1
        self.label[-1:, 2:] = 2
        expected_mat = numpy.array([[True, True, True],\
                                    [True, True, True],\
                                    [True, True, True]])

        (adj_mat, adj_dic) = selective_search._calc_adjacency_matrix(self.label, 3)
        assert numpy.array_equal(expected_mat, adj_mat)
        assert adj_dic[0] == {1, 2}
        assert adj_dic[1] == {0, 2}
        assert adj_dic[2] == {0, 1}

    def test_extreme_example(self):
        # 0, 1, 2, 3
        # 4, 5, 6, 7
        # 8, 9,10,11
        #12,13,14,15
        self.label = numpy.array(range(16)).reshape((4,4))
        (adj_mat, adj_dic) = selective_search._calc_adjacency_matrix(self.label, 16)
        assert numpy.array_equal(adj_mat.transpose(), adj_mat)
        assert set(numpy.flatnonzero(adj_mat[ 0])) == { 0,  1,  4}
        assert set(numpy.flatnonzero(adj_mat[ 1])) == { 0,  1,  2,  5}
        assert set(numpy.flatnonzero(adj_mat[ 2])) == { 1,  2,  3,  6}
        assert set(numpy.flatnonzero(adj_mat[ 3])) == { 2,  3,  7} 
        assert set(numpy.flatnonzero(adj_mat[ 4])) == { 0,  4,  5,  8}
        assert set(numpy.flatnonzero(adj_mat[ 5])) == { 1,  4,  5,  6,  9}
        assert set(numpy.flatnonzero(adj_mat[ 6])) == { 2,  5,  6,  7, 10}
        assert set(numpy.flatnonzero(adj_mat[ 7])) == { 3,  6,  7, 11}
        assert set(numpy.flatnonzero(adj_mat[ 8])) == { 4,  8,  9, 12}
        assert set(numpy.flatnonzero(adj_mat[ 9])) == { 5,  8,  9, 10, 13}
        assert set(numpy.flatnonzero(adj_mat[10])) == { 6,  9, 10, 11, 14}
        assert set(numpy.flatnonzero(adj_mat[11])) == { 7, 10, 11, 15} 
        assert set(numpy.flatnonzero(adj_mat[12])) == { 8, 12, 13}
        assert set(numpy.flatnonzero(adj_mat[13])) == { 9, 12, 13, 14}
        assert set(numpy.flatnonzero(adj_mat[14])) == {10, 13, 14, 15}
        assert set(numpy.flatnonzero(adj_mat[15])) == {11, 14, 15}

        for (i, adj_labels) in adj_dic.items():
            assert set(numpy.flatnonzero(adj_mat[i])) - {i} == adj_labels


class TestNewAdjacencyDict:
    def setup_method(self, method):
        # from:
        #   000000
        #   122334
        #   122334
        #   555555
        # to:
        #   000000
        #   166664
        #   166664
        #   555555
        self.A = {0: {1, 2, 3, 4},\
                  1: {0, 2, 5},\
                  2: {0, 1, 3, 5},\
                  3: {0, 2, 4, 5},\
                  4: {0, 3, 5},\
                  5: {1, 2, 3, 4}}

    def test_exclusiveness(self):
        """
        It should never violate source dictionary A
        """
        assert self.A[0] == {1, 2, 3, 4}
        assert self.A[1] == {0, 2, 5}
        assert self.A[2] == {0, 1, 3, 5}
        assert self.A[3] == {0, 2, 4, 5}
        assert self.A[4] == {0, 3, 5}
        assert self.A[5] == {1, 2, 3, 4}
        assert 6 not in self.A

    def test_label(self):
        Ak = selective_search._new_adjacency_dict(self.A, 2, 3, 6)
        assert 2 not in Ak
        assert 3 not in Ak
        assert Ak[0] == {1, 4, 6}
        assert Ak[1] == {0, 5, 6}
        assert Ak[4] == {0, 5, 6}
        assert Ak[5] == {1, 4, 6}
        assert Ak[6] == {0, 1, 4, 5}

