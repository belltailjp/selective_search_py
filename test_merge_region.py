#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import features

class TestMerge:
    def setup_method(self, method):
        dummy_image = numpy.zeros((10, 10, 3), dtype=numpy.uint8)
        dummy_label = numpy.zeros((10, 10), dtype=int)
        self.f = features.Features(dummy_image, dummy_label, 1)

    def test_merge_size(self):
        self.f.size = {0: 10, 1: 20}
        self.f._Features__merge_size(0, 1, 2)
        assert self.f.size[2] == 30

    def test_merge_color(self):
        self.f.color[0] = numpy.array([1.] * 75)
        self.f.size[0]  = 100
        self.f.color[1] = numpy.array([2.] * 75)
        self.f.size[1]  = 50
        self.f._Features__merge_color(0, 1, 2)

        expected = (100 * 1. + 50 * 2.) / (100 + 50)
        assert numpy.array_equal(self.f.color[2], [expected] * 75)

    def test_merge_texture(self):
        self.f.texture[0] = numpy.array([1.] * 240)
        self.f.size[0]    = 100
        self.f.texture[1] = numpy.array([2.] * 240)
        self.f.size[1]    = 50
        self.f._Features__merge_texture(0, 1, 2)

        expected = (100 * 1. + 50 * 2.) / (100 + 50)
        assert numpy.array_equal(self.f.texture[2], [expected] * 240)

    def test_merge_bbox(self):
        self.f.bbox[0] = numpy.array([10, 10, 20, 20])
        self.f.size[0] = 100
        self.f.bbox[1] = numpy.array([20, 20, 30, 30])
        self.f.size[1] = 50
        self.f.imsize  = 1000
        self.f._Features__merge_bbox(0, 1, 2)

        assert numpy.array_equal(self.f.bbox[2], [10, 10, 30, 30])

    def test_merge(self):
        self.f.imsize  = 1000
        self.f.size    = {0: 10, 1: 20}
        self.f.color   = {0: numpy.array([1.] * 75), 1: numpy.array([2.] * 75)}
        self.f.texture = {0: numpy.array([1.] * 240), 1: numpy.array([2.] * 240)}
        self.f.bbox    = {0: numpy.array([10, 10, 20, 20]), 1: numpy.array([20, 20, 30, 30])}
        self.f.merge(0, 1)
        assert len(self.f.size) == 3
        assert len(self.f.color) == 3
        assert len(self.f.texture) == 3
        assert len(self.f.bbox) == 3

