#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nose
import nose.tools

import numpy
import selective_search

class TestCalcAdjecencyMatrix:
    def setup_method(self, method):
        self.label = numpy.zeros((10, 10), dtype=int)

    def test_only_1_segment(self):
        adj_mat = selective_search.calc_adjacency_matrix(self.label, 1)
        assert adj_mat.shape == (1, 1) and adj_mat.dtype == int
        assert adj_mat[0, 0] == 1

    def test_fully_adjacent(self):
        self.label[:5, :] = 1
        adj_mat = selective_search.calc_adjacency_matrix(self.label, 2)
        assert adj_mat.shape == (2, 2) and adj_mat.dtype == int
        assert numpy.array_equal(adj_mat, [[1, 1], [1, 1]])

    def test_partially_adjacent(self):
        #make checker pattern
        self.label[:5, :5] = lu = 0
        self.label[:5, 5:] = ru = 1
        self.label[5:, :5] = lb = 2
        self.label[5:, 5:] = rb = 3
        adj_mat = selective_search.calc_adjacency_matrix(self.label, 4)
        assert adj_mat.shape == (4, 4) and adj_mat.dtype == int
        assert numpy.array_equal(numpy.diag(adj_mat), [1, 1, 1, 1])
        assert numpy.array_equal(adj_mat.transpose(), adj_mat)
        assert adj_mat[lu, ru] == adj_mat[lu, lb] == adj_mat[ru, rb] == adj_mat[lb, rb] == 1
        assert adj_mat[lu, rb] == adj_mat[ru, lb] == 0

    def test_edge_case_vertical(self):
        self.label[:5, -1:] = 1
        self.label[5:, -1:] = 2
        adj_mat = selective_search.calc_adjacency_matrix(self.label, 3)
        assert adj_mat[0, 1] == adj_mat[0, 2] == 1
        assert adj_mat[1, 2] == 1

    def test_edge_case_horizontal(self):
        self.label[-1:, :5] = 1
        self.label[-1:, 5:] = 2
        adj_mat = selective_search.calc_adjacency_matrix(self.label, 3)
        assert adj_mat[0, 1] == adj_mat[0, 2] == 1
        assert adj_mat[1, 2] == 1

    def test_extreme_example(self):
        # create label matrix like multiplication table
        self.label = numpy.array([[(i * 10 + j) for j in range(10)] for i in range(10)], dtype=int)
        adj_mat = selective_search.calc_adjacency_matrix(self.label, 100)
        for i in range(10):
            for j in range(10):
                if i == j or i == j + 1 or i == j - 1 or i == j + 10 or i == j - 10:
                    assert adj_mat[i, j] == 1
                else:
                    assert adj_mat[i, j] == 0

