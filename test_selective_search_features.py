#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import selective_search_features

class TestSelectiveSearchFeaturesColorHistogram:
    def setup_method(self, method = None, w = 10, h = 10):
        self.h, self.w = h, w
        self.input_img = numpy.zeros((self.h, self.w, 3), dtype=int)
        self.label_img = numpy.zeros((self.h, self.w), dtype=int)

    def test_1region_1color(self):
        hist = selective_search_features.color_histgram(self.input_img, self.label_img, 1)
        assert hist.shape == (1, 75)
        r_expected = [0.333333333] + [0] * 24
        g_expected = [0.333333333] + [0] * 24
        b_expected = [0.333333333] + [0] * 24
        numpy.testing.assert_array_almost_equal(hist.ravel(), r_expected + g_expected + b_expected)

    def test_1region_255color(self):
        self.setup_method(self, w = 1, h = 256)
        for y in range(self.h):
            self.input_img[y, :, :] = y

        hist = selective_search_features.color_histgram(self.input_img, self.label_img, 1)
        assert hist.shape == (1, 75)
        r_expected = [11] * 23 + [3, 0] # because bin width equals 11
        g_expected = [11] * 23 + [3, 0]
        b_expected = [11] * 23 + [3, 0]
        expected = numpy.array(r_expected + g_expected + b_expected)
        numpy.testing.assert_array_almost_equal(hist.ravel(), expected / numpy.sum(expected))

    def test_2region_1color(self):
        self.setup_method(self, w = 1, h = 2)
        for y in range(self.h):
            self.label_img[y, :] = y

        hist = selective_search_features.color_histgram(self.input_img, self.label_img, 2)
        assert hist.shape == (2, 75)
        r1_expected = ([1.0/3] + [0] * 24) + ([1.0/3] + [0] * 24) + ([1.0/3] + [0] * 24)
        r2_expected = ([1.0/3] + [0] * 24) + ([1.0/3] + [0] * 24) + ([1.0/3] + [0] * 24)
        numpy.testing.assert_array_almost_equal(hist[0].ravel(), r1_expected)
        numpy.testing.assert_array_almost_equal(hist[1].ravel(), r2_expected)

class TestSelectiveSearchFeaturesSize:
    def setup_method(self, method):
        self.label_img = numpy.zeros((10, 10), dtype=int)

    def test_1region(self):
        sizes = selective_search_features.size(self.label_img, 2)
        assert sizes.shape == (2,)
        assert sizes[0] == 100
        assert sizes[1] == 0

    def test_2region(self):
        self.label_img[:5, :] = 1
        sizes = selective_search_features.size(self.label_img, 2)
        assert sizes.shape == (2,)
        assert sizes[0] == 50
        assert sizes[1] == 50

