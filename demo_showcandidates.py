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

color_choises = ["RGB", "Lab", "rgI", "HSV", "nRGB", "Hue"]
k_choises = ["50", "100", "150", "250", "500"]
similarity_choises = ["Texture", "Color", "Fill", "Size"]

class Demo(QWidget):
    chosen_colors = {"RGB"}
    chosen_ks = {"100"}
    chosen_similarities = set(similarity_choises)

    def __init__(self, ndimg, regions):
        super().__init__()
        h, w = ndimg.shape[:2]
        self.qimg = QImage(ndimg.flatten(), w, h, QImage.Format_RGB888)
        self.regions = regions

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
        self.layout.addWidget(self.label, 0, 1)

    def __init_parameter_choises(self):
        choise_vbox = QVBoxLayout()
        self.__init_choises(choise_vbox, 'Color space', color_choises, self.chosen_colors, self.color_selected)
        self.__init_choises(choise_vbox, 'k', k_choises, self.chosen_ks, self.k_selected)
        self.__init_choises(choise_vbox, 'Similarity measure', similarity_choises, self.chosen_similarities, self.similarity_selected)
        self.layout.addLayout(choise_vbox, 0, 0)

    def __init_choises(self, choise_vbox, title, choises, default_choises, handler):
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
        choise_vbox.addWidget(group)

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

        self.layout.addLayout(hbox, 1, 1)


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
        print(self.chosen_colors, self.chosen_ks, self.chosen_similarities)

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

