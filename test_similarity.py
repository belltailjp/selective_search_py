#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import features

class TestSimilarity:
    def setup_method(self, method):
        self.dummy_image = numpy.zeros((10, 10, 3), dtype=numpy.uint8)
        self.dummy_label = numpy.zeros((10, 10), dtype=int)
        self.f = features.Features(self.dummy_image, self.dummy_label, 1)

    def test_similarity_size(self):
        self.f.size = {0 : 10, 1 : 20}

        s = self.f._Features__sim_size(0, 1)
        assert s == 0.7

    def test_similarity_color_simple(self):
        self.f.color[0] = numpy.array([1] * 75)
        self.f.color[1] = numpy.array([2] * 75)
        s = self.f._Features__sim_color(0, 1)
        assert s == 75

    def test_similarity_color_complex(self):
        # build 75-dimensional arrays as color histogram
        self.f.color[0] = numpy.array([1, 2, 1, 2, 1] * 15)
        self.f.color[1] = numpy.array([2, 1, 2, 1, 2] * 15)
        s = self.f._Features__sim_color(0, 1)
        assert s == 75

    def test_similarity_texture(self):
        # build 240-dimensional arrays as texture histogram
        self.f.texture[0] = numpy.array([1, 2, 1, 2, 1, 2] * 40)
        self.f.texture[1] = numpy.array([2, 1, 2, 1, 2, 1] * 40)
        s = self.f._Features__sim_texture(0, 1)
        assert s == 240

    def test_similarity_fill(self):
        self.f.bbox[0] = numpy.array([10, 10, 20, 20])
        self.f.size[0] = 100
        self.f.bbox[1] = numpy.array([20, 20, 30, 30])
        self.f.size[1] = 100
        s = self.f._Features__sim_fill(0, 1)
        assert s == 1. - float(400 - 200) / 100

    def test_similarity_user_all(self, monkeypatch):
        monkeypatch.setattr(features.Features, '_Features__sim_size',   lambda self, i, j: 1)
        monkeypatch.setattr(features.Features, '_Features__sim_texture',lambda self, i, j: 1)
        monkeypatch.setattr(features.Features, '_Features__sim_color',  lambda self, i, j: 1)
        monkeypatch.setattr(features.Features, '_Features__sim_fill',   lambda self, i, j: 1)
        w = features.SimilarityWeight(1, 1, 1, 1)
        f = features.Features(self.dummy_image, self.dummy_label, 1, w)
        assert f.similarity(0, 1) == 4

