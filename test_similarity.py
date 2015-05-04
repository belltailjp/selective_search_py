#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import similarity

class TestSimilaritySize:
    def test_type(self):
        ret = similarity.size(int(10), int(10), int(1))
        assert type(ret) == float

    def test_value(self):
        ret = similarity.size(10, 20, 100)
        assert ret == 0.7

class TestSimilarityColor:
    def test_1region(self):
        ar1 = numpy.array([[1] * 75])
        ar2 = numpy.array([[2] * 75])
        t = similarity.color(ar1, ar2)
        assert t.shape == (1, )
        assert numpy.array_equal(t, [75])

    def test_multiregion(self):
        # build (5, 75) array whose i-th row is filled with i (or 2i)
        ar1 = numpy.array([[i * 2] * 75 for i in range(5)])
        ar2 = numpy.array([[i]     * 75 for i in range(5)])
        t = similarity.color(ar1, ar2)
        assert t.shape == (5, )
        assert numpy.array_equal(t, [0, 75, 150, 225, 300])

class TestSimilarityColor:
    def test_multiregion(self):
        ar1 = numpy.array([[i * 2] * 240 for i in range(5)])
        ar2 = numpy.array([[i]     * 240 for i in range(5)])
        t = similarity.color(ar1, ar2)
        assert t.shape == (5, )
        assert numpy.array_equal(t, [0, 240, 240 * 2, 240 * 3, 240 * 4])

