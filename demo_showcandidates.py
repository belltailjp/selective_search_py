#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy
import skimage.io
import selective_search

import matplotlib.pyplot
import PIL.Image
import PIL.ImageDraw

def draw_rects(img, regions):
    pil_img = PIL.Image.fromarray(img)
    draw = PIL.ImageDraw.Draw(pil_img)
    for (value, (y0, x0, y1, x1)) in regions:
        draw.rectangle([x0, y0, x1, y1], outline = (0, 255, 0))

    return numpy.asarray(pil_img)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, required=True, help='filename of the image')
    args = parser.parse_args()

    img = skimage.io.imread(args.image)
    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)

    regions = selective_search.selective_search(img)
    result = draw_rects(img, regions)

    skimage.io.imshow(result)
    skimage.io.show()

