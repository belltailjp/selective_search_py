#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import itertools
import numpy
import skimage.io
import color_space
import features
import selective_search

from PySide.QtCore import *
from PySide.QtGui import *

#color_choises = ["RGB", "Lab", "rgI", "HSV", "nRGB", "Hue"]
color_choises = ["RGB", "rgI", "HSV", "nRGB", "Hue"]
k_choises = ["50", "100", "150", "300"]
similarity_choises = ["C", "T", "S", "F",\
                      "C+T", "C+S", "C+F", "T+S", "T+F", "S+F",\
                      "C+T+S", "C+T+F", "C+S+F", "T+S+F",\
                      "C+T+S+F"]

class Demo(QWidget):
    chosen_colors = {"RGB"}
    chosen_ks = {"100"}
    chosen_similarities = {"C+T+S+F", "T+S+F", "F", "S"}
    regions = list()

    def __init__(self, ndimg):
        super().__init__()
        self.ndimg = ndimg
        h, w = ndimg.shape[:2]
        self.qimg = QImage(ndimg.flatten(), w, h, QImage.Format_RGB888)

        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.__init_parameter_choises()
        self.__init_imagearea()
        self.__init_slider()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape:
            self.close()

    def __init_imagearea(self):
        self.label = QLabel()
        self.layout.addWidget(self.label, 0, 2)

    def __init_parameter_choises(self):
        color_checkbox = self.__init_choises('Color space', color_choises, self.chosen_colors, self.color_selected)
        k_checkbox = self.__init_choises('k', k_choises, self.chosen_ks, self.k_selected)
        sim_checkbox = self.__init_choises('Similarity measure', similarity_choises, self.chosen_similarities, self.similarity_selected)

        color_k_vbox = QVBoxLayout()
        color_k_vbox.addWidget(color_checkbox)
        color_k_vbox.addWidget(k_checkbox)

        self.layout.addLayout(color_k_vbox, 0, 0)
        self.layout.addWidget(sim_checkbox, 0, 1)

    def __init_choises(self, title, choises, default_choises, handler):
        group = QGroupBox(title)
        group.setFlat(False)

        vbox = QVBoxLayout()
        for choise in choises:
            checkbox = QCheckBox(choise)
            if choise in default_choises:
                checkbox.setCheckState(Qt.Checked)
            checkbox.stateChanged.connect(handler)
            vbox.addWidget(checkbox)

        group.setLayout(vbox)
        return group

    def __init_slider(self):
        hbox = QHBoxLayout()

        label = QLabel()
        label.setText('count:')
        hbox.addWidget(label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.count_changed)
        hbox.addWidget(self.slider)

        self.count_label = QLabel()
        hbox.addWidget(self.count_label)

        self.layout.addLayout(hbox, 1, 2)


    def count_changed(self, value):
        self.__draw(value)
        self.count_label.setText(str(value))

    def color_selected(self, value):
        color = self.sender().text()
        if value:
            self.chosen_colors.add(color)
        else:
            self.chosen_colors.remove(color)
            if len(self.chosen_colors) == 0:
                self.sender().setCheckState(Qt.Checked)
        self.__parameter_changed()

    def k_selected(self, value):
        k = self.sender().text()
        if value:
            self.chosen_ks.add(k)
        else:
            self.chosen_ks.remove(k)
            if len(self.chosen_ks) == 0:
                self.sender().setCheckState(Qt.Checked)
        self.__parameter_changed()

    def similarity_selected(self, value):
        similarity = self.sender().text()
        if value:
            self.chosen_similarities.add(similarity)
        else:
            self.chosen_similarities.remove(similarity)
            if len(self.chosen_similarities) == 0:
                self.sender().setCheckState(Qt.Checked)
        self.__parameter_changed()

    def __parameter_changed(self):
        # obtain parameters
        color_spaces = [color.lower() for color in self.chosen_colors]
        ks = [float(k) for k in self.chosen_ks]
        similarity_masks = [features.SimilarityMask('S' in mask, 'C' in mask, 'T' in mask, 'F' in mask) for mask in self.chosen_similarities]

        self.regions = selective_search.selective_search(self.ndimg, color_spaces, ks, similarity_masks)
        self.slider.setMaximum(len(self.regions))
        self.slider.setValue(int(len(self.regions) / 4))
        self.__draw()

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

    app = QApplication(sys.argv)
    wnd = Demo(img)
    wnd.show()
    app.exec_()

