# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: sumex
@software: PyCharm
@time: 2022/8/2 13:49
@file: cdnet.py
"""
import os

import cv2


def is_valid_video_folder(path):
    """
    视频文件夹是否有效
    A valid video folder must have \\groundtruth, \\input, ROI.bmp, temporalROI.txt
    :param path:
    :return:
    """
    return os.path.exists(os.path.join(path, 'groundtruth')) and \
           os.path.exists(os.path.join(path, 'input')) and \
           os.path.exists(os.path.join(path, 'ROI.bmp')) and \
           os.path.exists(os.path.join(path, 'temporalROI.txt'))


def get_temporalROI(path):
    """
    获取检测帧的范围
    :param path:dataset/baseline/baseline/highway
    :return:['470', '1700'] [起始帧，结束帧]
    """
    path = os.path.join(path, 'temporalROI.txt')
    with open(path, 'r') as f:
        avail_frames = f.read()

    return avail_frames.split(' ')


def get_roi(video_path):
    roi_path = os.path.join(video_path, "ROI.bmp")
    roi = cv2.imread(roi_path, 0)
    if "traffic" in video_path:
        # 数据集中 traffic 视频序列 的ROI.bmp 与数据集尺寸不同
        roi_path_jpg = os.path.join(video_path, "ROI.jpg")
        roi_size = cv2.imread(roi_path_jpg, 0).shape
        roi = cv2.resize(cv2.imread(roi_path, 0), (roi_size[1], roi_size[0]))

    return roi


def get_gt(video_path, filename):
    gt_path = os.path.join(video_path, "groundtruth", filename.replace("bin", "gt").replace("jpg", "png"))

    return cv2.imread(gt_path, 0)
