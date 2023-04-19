# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: sumex
@software: PyCharm
@time: 2021/4/8 12:28
@file: util.py
@brief: 
"""
import os
import numpy as np
import time
from pprint import pprint

import cv2

from folder_process import folder_for_save

def get_category_video(category, video):
    """
    csv 索引
    :param category_video:
    :return: dict
    """
    return {'category': category, 'video': video}


def get_categories(dataset_dir):
    """
    Stores the list of categories as string and the videos of each
    category in a dictionary.
    """
    categories = sorted(os.listdir(dataset_dir), key=lambda v: v.upper())

    videos = dict()

    for category in categories:
        category_dir = os.path.join(dataset_dir, category)
        videos[category] = sorted(os.listdir(category_dir), key=lambda v: v.upper())

    return categories, videos


def img2uint8(frame):
    return cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)


def saveimg(frame, filename, input_path, output_path, sub_rootpath=None):
    """
    Args:
        frame: 1. 正常帧， 2.字典形式的帧
            {
                mask: 帧  # []
                masks：多个帧的集合  # [[], [], []]
            }
        filename:
        input_path:
        output_path:
        sub_rootpath: 子根目录。同一算法下记录不同操作的检测结果
    Returns:

    """
    if type(frame) == dict:
        output_path_mask = output_path + '_mask'
        output_path_masks = output_path + '_masks'
        save_dir_path_mask = folder_for_save(input_path, output_path_mask, sub_rootpath)
        save_dir_path_masks = folder_for_save(input_path, output_path_masks, sub_rootpath)
        save_dir_path_masks = os.path.join(save_dir_path_masks, filename[:-4])
        if not os.path.exists(save_dir_path_masks):
            os.mkdir(save_dir_path_masks)

        save_path_mask = os.path.join(save_dir_path_mask, filename)
        cv2.imwrite(save_path_mask, img2uint8(frame["mask"]))

        for id_m, f in enumerate(frame["masks"]):
            filename = str(id_m) + '.jpg'
            save_path_masks = os.path.join(save_dir_path_masks, filename)
            cv2.imwrite(save_path_masks, img2uint8(f))
    else:
        save_dir_path = folder_for_save(input_path, output_path, sub_rootpath)
        save_path = os.path.join(save_dir_path, filename)
        # print(save_path)
        cv2.imwrite(save_path, img2uint8(frame))


def save_fps(path, name, fps):
    """
    保存fps
    :param path:
    :param name:
    :param fps:
    :return:
    """
    with open(path, 'a+') as f:
        f.write('{}, {:.2f} \n'.format(name, fps))



if __name__ == '__main__':
    pass


