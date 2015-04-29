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
        assert hist[0, 0]  == 10 * 10
        assert hist[0, 25] == 10 * 10
        assert hist[0, 50] == 10 * 10

    def test_1region_255color(self):
        self.setup_method(self, w = 1, h = 256)
        for y in range(self.h):
            self.input_img[y, :, :] = y

        hist = selective_search_features.color_histgram(self.input_img, self.label_img, 1)
        assert hist[0, 0]  == 11    # because bin width equals 11
        assert hist[0, 25] == 11
        assert hist[0, 50] == 11

    def test_2region_1color(self):
        self.setup_method(self, w = 1, h = 2)
        for y in range(self.h):
            self.label_img[y, :] = y

        hist = selective_search_features.color_histgram(self.input_img, self.label_img, 2)
        assert hist[0, 0]  == hist[0, 25] == hist[0, 50] == 1
        assert hist[1, 0]  == hist[1, 25] == hist[1, 50] == 1

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

