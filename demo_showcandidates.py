#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import numpy
import skimage.io
import selective_search

from PySide.QtCore import *
from PySide.QtGui import *

class Demo(QWidget):
    def __init__(self, ndimg, regions):
        super().__init__()
        h, w = ndimg.shape[:2]
        self.qimg = QImage(ndimg.flatten(), w, h, QImage.Format_RGB888)
        self.regions = regions

        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.__init_imagearea()
        self.__init_slider()

    def __init_imagearea(self):
        self.label = QLabel()
        self.layout.addWidget(self.label, 0, 0)

    def __init_slider(self):
        hbox = QHBoxLayout()

        label = QLabel()
        label.setText('count:')
        hbox.addWidget(label)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(len(regions))
        slider.valueChanged.connect(self.count_changed)
        slider.setValue(int(len(regions) / 4))
        hbox.addWidget(slider)

        self.count_label = QLabel()
        hbox.addWidget(self.count_label)

        self.layout.addLayout(hbox, 1, 0)


    def count_changed(self, value):
        self.__draw(value)
        self.count_label.setText(str(value))

    def __draw(self, count):
        self.pixmap = QPixmap(self.qimg)
        painter = QPainter(self.pixmap)
        painter.setPen(QColor(0, 255, 0))
        for v, (y0, x0, y1, x1) in self.regions[:count]:
            painter.drawRect(x0, y0, x1, y1)

        self.label.setPixmap(self.pixmap)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, required=True, help='filename of the image')
    args = parser.parse_args()

    img = skimage.io.imread(args.image)
    if len(img.shape) == 2:
        img = skimage.color.gray2rgb(img)

    regions = selective_search.selective_search(img)
    app = QApplication(sys.argv)
    wnd = Demo(img, regions)
    wnd.show()
    app.exec_()

