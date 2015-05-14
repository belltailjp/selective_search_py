#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
from color_space import *

class TestColorSpace:
    def _assert_range(self, img):
        assert img.dtype == numpy.uint8
        assert img.shape == (10, 10, 3)
        assert 0 <= numpy.min(img)
        assert 1 < numpy.max(img) <= 255

    def setup_method(self, method):
        self.I = numpy.ndarray((10, 10, 3), dtype=numpy.uint8)
        self.I[:, :, 0] = 50
        self.I[:, :, 1] = 100
        self.I[:, :, 2] = 150
        self.Irand = numpy.random.randint(0, 256, (10, 10, 3)).astype(numpy.uint8)


    def test_to_grey_range(self):
        self._assert_range(to_grey(self.Irand))
        
    def test_to_grey_value(self):
        img = to_grey(self.I)
        grey_value = int(0.2125 * 50 + 0.7154 * 100 + 0.0721 * 150)
        assert ((img == grey_value).all())


    def test_to_Lab_range(self):
        self._assert_range(to_Lab(self.Irand)) 

    def test_to_Lab_value(self):
        img = to_Lab(self.I)


    def test_to_rgI_range(self):
        self._assert_range(to_rgI(self.Irand))

    def test_to_rgI_value(self):
        img = to_rgI(self.I)
        grey_value = int(0.2125 * 50 + 0.7154 * 100 + 0.0721 * 150)
        assert ((img[:, :, 0] == 50).all())
        assert ((img[:, :, 1] == 100).all())
        assert ((img[:, :, 2] == grey_value).all())


    def test_to_HSV_range(self):
        self._assert_range(to_HSV(self.Irand))

    def test_to_HSV_value(self):
        img = to_HSV(self.I)
        h, s, v = 148, 170, 150
        assert ((img[:, :, 0] == h).all())
        assert ((img[:, :, 1] == s).all())
        assert ((img[:, :, 2] == v).all())


    def test_to_nRGB_range(self):
        self._assert_range(to_nRGB(self.Irand))

    def test_to_nRGB_value(self):
        img = to_nRGB(self.I)
        denom = numpy.sqrt(50 ** 2 + 100 ** 2 + 150 ** 2) / 255.0
        r, g, b = 50 / denom, 100 / denom, 150 / denom
        assert ((img[:, :, 0] == int(r)).all())
        assert ((img[:, :, 1] == int(g)).all())
        assert ((img[:, :, 2] == int(b)).all())


    def test_to_Hue_range(self):
        self._assert_range(to_Hue(self.Irand))

    def test_to_Hue_value(self):
        img = to_Hue(self.I)
        expected_h = 148
        assert ((img[:, :, 0] == expected_h).all())
        assert ((img[:, :, 1] == expected_h).all())
        assert ((img[:, :, 2] == expected_h).all())

