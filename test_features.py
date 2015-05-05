#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import features

class TestFeaturesColorHistogram:
    def setup_method(self, method = None, w = 10, h = 10):
        self.h, self.w = h, w
        image = numpy.zeros((self.h, self.w, 3), dtype=numpy.uint8)
        label = numpy.zeros((self.h, self.w), dtype=int)
        self.f = features.Features(image, label, 1, None)

    def test_1region_1color(self):
        hist = self.f._Features__init_color(1)
        assert len(hist) == 1
        assert hist[0].shape == (75,)
        r_expected = [0.333333333] + [0] * 24
        g_expected = [0.333333333] + [0] * 24
        b_expected = [0.333333333] + [0] * 24
        numpy.testing.assert_array_almost_equal(hist[0].ravel(), r_expected + g_expected + b_expected)

    def test_1region_255color(self):
        self.setup_method(self, w = 1, h = 256)
        for y in range(self.h):
            self.f.image[y, :, :] = y

        hist = self.f._Features__init_color(1)
        assert len(hist) == 1
        assert hist[0].shape == (75,)
        r_expected = [11] * 23 + [3, 0] # because bin width equals 11
        g_expected = [11] * 23 + [3, 0]
        b_expected = [11] * 23 + [3, 0]
        expected = numpy.array(r_expected + g_expected + b_expected)
        numpy.testing.assert_array_almost_equal(hist[0].ravel(), expected / numpy.sum(expected))

    def test_2region_1color(self):
        self.setup_method(self, w = 1, h = 2)
        for y in range(self.h):
            self.f.label[y, :] = y

        hist = self.f._Features__init_color(2)
        assert len(hist) == 2
        assert hist[0].shape == (75,)
        r1_expected = ([1.0/3] + [0] * 24) + ([1.0/3] + [0] * 24) + ([1.0/3] + [0] * 24)
        r2_expected = ([1.0/3] + [0] * 24) + ([1.0/3] + [0] * 24) + ([1.0/3] + [0] * 24)
        numpy.testing.assert_array_almost_equal(hist[0].ravel(), r1_expected)
        numpy.testing.assert_array_almost_equal(hist[1].ravel(), r2_expected)


class TestFeaturesSize:
    def setup_method(self, method):
        image = numpy.zeros((10, 10, 3), dtype=numpy.uint8)
        label = numpy.zeros((10, 10), dtype=int)
        self.f = features.Features(image, label, 1, None)

    def test_1region(self):
        sizes = self.f._Features__init_size(1)
        assert len(sizes) == 1
        assert sizes[0] == 100

    def test_2region(self):
        self.f.label[:5, :] = 1
        sizes = self.f._Features__init_size(2)
        assert len(sizes) == 2
        assert sizes[0] == 50
        assert sizes[1] == 50

class TestFeaturesBoundingBox:
    def setup_method(self, method):
        image = numpy.zeros((10, 10, 3), dtype=numpy.uint8)
        label = numpy.zeros((10, 10), dtype=int)
        self.f = features.Features(image, label, 1, None)

    def test_1region(self):
        bb = self.f._Features__init_bounding_box(1)
        assert len(bb) == 1
        assert bb[0].shape == (4,)
        assert numpy.array_equal(bb[0], [0, 0, 9, 9])

    def test_4region(self):
        self.f.label[:5, :5] = 0
        self.f.label[:5, 5:] = 1
        self.f.label[5:, :5] = 2
        self.f.label[5:, 5:] = 3
        bb = self.f._Features__init_bounding_box(4)
        assert len(bb) == 4
        assert numpy.array_equal(bb[0], [0, 0, 4, 4])
        assert numpy.array_equal(bb[1], [0, 5, 4, 9])
        assert numpy.array_equal(bb[2], [5, 0, 9, 4])
        assert numpy.array_equal(bb[3], [5, 5, 9, 9])

