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
        (adj_mat, adj_dic) = selective_search.calc_adjacency_matrix(self.label, 1)
        assert adj_mat.shape == (1, 1) and adj_mat.dtype == bool
        assert adj_mat[0, 0] == True
        assert adj_dic[0] == set()

    def test_fully_adjacent(self):
        # 1, 1, 0, 0
        # 1, 1, 0, 0
        # 1, 1, 0, 0
        # 1, 1, 0, 0
        self.label[:2, :] = 1
        (adj_mat, adj_dic) = selective_search.calc_adjacency_matrix(self.label, 2)
        assert adj_mat.shape == (2, 2) and adj_mat.dtype == bool
        assert numpy.array_equal(adj_mat, numpy.array([[1, 1], [0, 1]], dtype=bool))
        assert adj_dic[0] == {1}
        assert adj_dic[1] == set()

    def test_partially_adjacent(self):
        # 0, 0, 1, 1
        # 0, 0, 1, 1
        # 2, 2, 3, 3
        # 2, 2, 3, 3
        self.label[:2, :2] = lu = 0
        self.label[:2, 2:] = ru = 1
        self.label[2:, :2] = lb = 2
        self.label[2:, 2:] = rb = 3
        (adj_mat, adj_dic) = selective_search.calc_adjacency_matrix(self.label, 4)
        assert adj_mat.shape == (4, 4) and adj_mat.dtype == bool
        assert numpy.array_equal(numpy.diag(adj_mat), numpy.array([1, 1, 1, 1], dtype=bool))
        assert numpy.array_equal(numpy.triu(adj_mat), adj_mat)
        assert adj_mat[lu, ru] == adj_mat[lu, lb] == adj_mat[ru, rb] == adj_mat[lb, rb] == True
        assert adj_mat[lu, rb] == adj_mat[ru, lb] == False
        assert adj_dic[lu] == {ru, lb}
        assert adj_dic[ru] == {rb}
        assert adj_dic[lb] == {rb}
        assert adj_dic[rb] == set()

    def test_edge_case_vertical(self):
        # 0, 0, 0, 1
        # 0, 0, 0, 1
        # 0, 0, 0, 2
        # 0, 0, 0, 2
        self.label[:2, -1:] = 1
        self.label[2:, -1:] = 2
        (adj_mat, adj_dic) = selective_search.calc_adjacency_matrix(self.label, 3)
        assert numpy.array_equal(numpy.triu(adj_mat), adj_mat)
        assert adj_mat[0, 1] == adj_mat[0, 2] == True
        assert adj_mat[1, 2] == True
        assert adj_dic[0] == {1, 2}
        assert adj_dic[1] == {2}
        assert adj_dic[2] == set()

    def test_edge_case_horizontal(self):
        # 0, 0, 0, 0
        # 0, 0, 0, 0
        # 0, 0, 0, 0
        # 1, 1, 2, 2
        self.label[-1:, :2] = 1
        self.label[-1:, 2:] = 2
        (adj_mat, adj_dic) = selective_search.calc_adjacency_matrix(self.label, 3)
        assert numpy.array_equal(numpy.triu(adj_mat), adj_mat)
        assert adj_mat[0, 1] == adj_mat[0, 2] == True
        assert adj_mat[1, 2] == True
        assert adj_dic[0] == {1, 2}
        assert adj_dic[1] == {2}
        assert adj_dic[2] == set()

    def test_extreme_example(self):
        # 0, 1, 2, 3
        # 4, 5, 6, 7
        # 8, 9,10,11
        #12,13,14,15
        self.label = numpy.array(range(16)).reshape((4,4))
        (adj_mat, adj_dic) = selective_search.calc_adjacency_matrix(self.label, 16)
        assert numpy.array_equal(numpy.triu(adj_mat), adj_mat)
        assert (adj_dic[ 0]|{ 0}) == set(numpy.flatnonzero(adj_mat[ 0])) == { 0,  1,  4}
        assert (adj_dic[ 1]|{ 1}) == set(numpy.flatnonzero(adj_mat[ 1])) == { 1,  2,  5}
        assert (adj_dic[ 2]|{ 2}) == set(numpy.flatnonzero(adj_mat[ 2])) == { 2,  3,  6}
        assert (adj_dic[ 3]|{ 3}) == set(numpy.flatnonzero(adj_mat[ 3])) == { 3,  7} 
        assert (adj_dic[ 4]|{ 4}) == set(numpy.flatnonzero(adj_mat[ 4])) == { 4,  5,  8}
        assert (adj_dic[ 5]|{ 5}) == set(numpy.flatnonzero(adj_mat[ 5])) == { 5,  6,  9}
        assert (adj_dic[ 6]|{ 6}) == set(numpy.flatnonzero(adj_mat[ 6])) == { 6,  7, 10}
        assert (adj_dic[ 7]|{ 7}) == set(numpy.flatnonzero(adj_mat[ 7])) == { 7, 11}
        assert (adj_dic[ 8]|{ 8}) == set(numpy.flatnonzero(adj_mat[ 8])) == { 8,  9, 12}
        assert (adj_dic[ 9]|{ 9}) == set(numpy.flatnonzero(adj_mat[ 9])) == { 9, 10, 13}
        assert (adj_dic[10]|{10}) == set(numpy.flatnonzero(adj_mat[10])) == {10, 11, 14}
        assert (adj_dic[11]|{11}) == set(numpy.flatnonzero(adj_mat[11])) == {11, 15} 
        assert (adj_dic[12]|{12}) == set(numpy.flatnonzero(adj_mat[12])) == {12, 13}
        assert (adj_dic[13]|{13}) == set(numpy.flatnonzero(adj_mat[13])) == {13, 14}
        assert (adj_dic[14]|{14}) == set(numpy.flatnonzero(adj_mat[14])) == {14, 15}
        assert (adj_dic[15]|{15}) == set(numpy.flatnonzero(adj_mat[15])) == {15}
                                                                      
