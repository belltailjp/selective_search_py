#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy

def size(size1, size2, size_img):
    return 1. - float(size1 + size2) / size_img

