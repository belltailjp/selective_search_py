# Overview

This is a python implementation of the Selective Search [[1]](#selective_search).

The Selective Search is used as a preprocess of object detection pipeline.<br/>
It finds regions likely to contain any objects from input image regardless of its scale and location,
allows detector to concentrate only for such regions.<br/>
Therefore you can configure more computationally efficient detector,
or use more rich feature representation and classification method [[3]](#deeplearning)
compared to the conventional exhaustive search scheme.


# Requirements

* CMake (>= 2.8)
* GCC (>= 4.8.2)
* Python (>= 3.4.3)
    * numpy (>= 1.9.2)
    * scikit-image (>= 0.11.3)
* Boost (>= 1.58.0) built with python support
* [Boost.NumPy](https://github.com/ndarray/Boost.NumPy)

In addition, this is only tested on x64 Linux environment.


# Preparation

This implementation contains a few C++ code which wraps the Efficient Graph-Based Image Segmentation [[2]](#segmentation) used for generating an initial value.
It works as a python module, so build it first.

```sh
% git clone https://github.com/belltailjp/selective_search_py.git
% cd selective_search_py
% wget http://cs.brown.edu/~pff/segment/segment.zip; unzip segment.zip; rm segment.zip
% cmake .
% make
```

Then you will see `segment.so` in the directory.


# References

\[1\] <a name="selective_search"> [J. R. R. Uijlings et al., Selective Search for Object Recognition, IJCV, 2013](https://ivi.fnwi.uva.nl/isis/publications/bibtexbrowser.php?key=UijlingsIJCV2013&bib=all.bib) <br/>
\[2\] <a name="segmentation"> [P. Felzenszwalb et al., Efficient Graph-Based Image Segmentation, IJCV, 2004](http://cs.brown.edu/~pff/segment/) <br/>
\[3\] <a name="deeplearning"> [R. Girshick et al., Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation, CVPR, 2014](http://www.cs.berkeley.edu/~rbg/papers/r-cnn-cvpr.pdf)
