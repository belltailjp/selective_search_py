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

