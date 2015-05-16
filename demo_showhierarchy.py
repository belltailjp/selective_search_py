#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import warnings
import numpy
import skimage.io
import features
import color_space
import selective_search

def generate_color_table(R):
    # generate initial color
    colors = numpy.random.randint(0, 255, (len(R), 3))

    # merged-regions are colored same as larger parent
    for region, parent in R.items():
        if not len(parent) == 0:
            colors[region] = colors[parent[0]]

    return colors

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('image',            type=str,   help='filename of the image')
    parser.add_argument('-k', '--k',        type=int,   default=100, help='threshold k for initial segmentation')
    parser.add_argument('-c', '--color',    nargs=1,    default='rgb', choices=['rgb', 'lab', 'rgi', 'hsv', 'nrgb', 'hue'], help='color space')
    parser.add_argument('-f', '--feature',  nargs="+",  default=['texture', 'fill'], choices=['size', 'color', 'texture', 'fill'], help='feature for similarity calculation')
    parser.add_argument('-o', '--output',   type=str,   default='result', help='prefix of resulting images')
    args = parser.parse_args()

    img = skimage.io.imread(args.image)
    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)

    print('k:', args.k)
    print('color:', args.color)
    print('feature:', ' '.join(args.feature))

    mask = features.SimilarityMask('size' in args.feature, 'color' in args.feature, 'texture' in args.feature, 'fill' in args.feature)
    (R, F, L) = selective_search.hierarchical_segmentation(img, args.k, mask)
    print('result filename: %s_[0000-%04d].png' % (args.output, len(F) - 1))

    # suppress warning when saving result images
    warnings.filterwarnings("ignore", category = UserWarning)

    colors = generate_color_table(R)
    for depth, label in enumerate(F):
        result = colors[label].astype(numpy.uint8)
        fn = "%s_%04d.png" % (args.output, depth)
        skimage.io.imsave(fn, result)
        print('.', end="")
        sys.stdout.flush()

    print()

