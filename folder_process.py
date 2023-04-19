# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: sumex
@software: PyCharm
@time: 2021/5/5 13:05
@file: folder_process.py
@brief: 
"""
import csv
import os
import shutil
import numpy as np
import pandas as pd


# 结果保存
def write_result_tocsv(stats_root, filename, data):
    """
    保存数据，依次追加
    :param stats_root:
    :param filename:
    :param data:
    :return:
    """
    if not os.path.exists(stats_root):
        os.makedirs(stats_root)
    stats_path = os.path.join(stats_root, filename)
    is_stats = True
    try:
        pd.read_csv(stats_path)
    except Exception as e:
        is_stats = False

    header = list(data.keys())
    with open(stats_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)  # 提前预览列名，当下面代码写入数据时，会将其一一对应。
        if not is_stats:  # 第一次写入将创建 header
            writer.writeheader()  # 写入列名
        writer.writerows([data])  # 写入数据


def write_category_and_overall_tocsv(stats_root, save_filename):
    """
    保存类统计
    :param stats_root:
    :param save_filename:
    :return:
    """
    stats_path = os.path.join(stats_root, save_filename)
    stats_data = pd.read_csv(stats_path)
    category_col = stats_data['category']
    # 去重，不改变顺序
    categories = []
    for cat in category_col:
        if cat not in categories:
            categories.append(cat)

    overall = []
    write_result_tocsv(stats_root, save_filename, {})  # 空三行
    write_result_tocsv(stats_root, save_filename, {})
    write_result_tocsv(stats_root, save_filename, {})

    # category
    for category in categories:
        category_dict = {'category': category, 'video': ''}
        category_stats = stats_data[stats_data['category'] == category]

        category_dict['Recall'] = category_stats['Recall'].mean()
        category_dict['Precision'] = category_stats['Precision'].mean()
        category_dict['Specificity'] = category_stats['Specificity'].mean()
        category_dict['FPR'] = category_stats['FPR'].mean()
        category_dict['FNR'] = category_stats['FNR'].mean()
        category_dict['PWC'] = category_stats['PWC'].mean()
        category_dict['FMeasure'] = category_stats['FMeasure'].mean()

        overall.append(category_dict)
        write_result_tocsv(stats_root, save_filename, category_dict)

    write_result_tocsv(stats_root, save_filename, {})

    # overall
    overall_stats = pd.DataFrame(overall)
    category_dict = {'category': 'overall', 'video': ''}

    category_dict['Recall'] = overall_stats['Recall'].mean()
    category_dict['Precision'] = overall_stats['Precision'].mean()
    category_dict['Specificity'] = overall_stats['Specificity'].mean()
    category_dict['FPR'] = overall_stats['FPR'].mean()
    category_dict['FNR'] = overall_stats['FNR'].mean()
    category_dict['PWC'] = overall_stats['PWC'].mean()
    category_dict['FMeasure'] = overall_stats['FMeasure'].mean()

    write_result_tocsv(stats_root, save_filename, category_dict)


# 文件路径操作
def get_input_path(dataset_rootpath):
    """
    文件操作，获取输入数据的路径
    :param dataset_rootpath:
    :return:
    """
    input_list = []
    for category in get_directories(dataset_rootpath):
        category_path = os.path.join(dataset_rootpath, category, category)

        for video in get_directories(category_path):
            video_path = os.path.join(category_path, video, 'input')

            input_list.append(video_path)

    return input_list


def folder_for_save(input_path, output_rootpath, sub_rootpath=None):
    """
    根据输入图片的路径，保存结果图片到指定文件夹内
    :param sub_rootpath:
    :param output_rootpath:'../results/mb_K=200_t=t0.2_'
    :param input_path:../../dataset/baseline/baseline/{}/input'.format(video)
    :return: ../../results/gmm/gmm/baseline/{}/bin%06d.jpg
    """
    input_path_list = input_path.replace('\\', '/').split('/')
    output_rootpath_list = output_rootpath.replace('\\', '/').split('/')
    # 20210812 添加无 input 时的路径处理
    if input_path_list[-1] == "input":
        path_list = input_path_list[-4:-1]  # 去掉input之后 及 dataset及之前
    else:
        # 'data\\bs\\baseline\\office'
        path_list = input_path_list[-3:]
    # 有 sub_rootpath 则拼接，没有则创建一个，以免后期在 output_rootpath
    output_rootpath = os.path.join(output_rootpath, sub_rootpath) if sub_rootpath else os.path.join(output_rootpath, output_rootpath_list[-1])
    path_list[0] = output_rootpath
    output_path = list2path(path_list)  # 输出路径
    # output_path_ = path_is_exist(output_path)  # 判断路径是否存在，不存在则创建
    if not os.path.exists(output_path):
        # os.mkdirs(output_path)
        os.makedirs(output_path)

    return output_path


def dir_is_null(path):
    """
    文件夹下是否为空，不为空删除文件
    :param path:
    :return:
    """
    flag = os.listdir(path)
    if flag:
        # 强制删除文件夹
        shutil.rmtree(path)
        path_is_exist(path)


def path_is_exist(path):
    """
    判断路径是否存在，不存在则创建
    :param path:
    :return:
    """
    output_path_ = ''
    path_list = path.replace('\\', '/').split('/')
    for path_name in path_list:  # 查看输出路径是否存在
        output_path_ = os.path.join(output_path_, path_name)
        if '.\\' not in path_name:
            if not os.path.exists(output_path_):
                os.mkdir(output_path_)

    return output_path_


def list2path(path_list):
    """
    根据list返回 path
    :param path_list:
    :return:
    """
    return os.path.join(*path_list) if path_list else ''


def get_directories(path):
    """
    Return a list of directories name on the specifed path
    :param path:
    :return:
    """
    return [file for file in os.listdir(path) if os.path.isdir(os.path.join(path, file))]


if __name__ == '__main__':
    PATH_ROOT = r"E:\00_Datasets\LASIESTA\Boost\I_BS_01\I_BS_01"
    # start_time = time.time()
    # create_dir_is_exist(PATH_ROOT)

    p = folder_for_save(PATH_ROOT, '../../results/gmm_', 'medain')

    print(p)







