# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: sumex
@software: PyCharm
@time: 2022/8/2 13:50
@file: lasiesta.py
"""
import os
import numpy as np
import cv2


def get_gt_path(path):
    """
    获取gt
    """
    return os.path.join(path, os.path.basename(path) + "-GT")


def get_bin_img(binaryPath, gt_filename):
    bin_path = os.path.join(binaryPath, gt_filename.replace("gt", "bin"))

    return cv2.imread(bin_path, 0)


def to_gray_img(path):
    """
    GT to gray
    """
    img = cv2.imread(path)

    # BGR
    img[np.all(img == (0, 0, 255), axis=-1)] = (255, 255, 255)
    img[np.all(img == (0, 255, 255), axis=-1)] = (255, 255, 255)
    img[np.all(img == (0, 255, 0), axis=-1)] = (255, 255, 255)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img