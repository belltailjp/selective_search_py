#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import skimage.io

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, required=True, help='filename of the image')
    args = parser.parse_args()

    img = skimage.io.imread(args.image)
