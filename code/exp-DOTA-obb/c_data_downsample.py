# -*- coding: utf-8 -*-
"""
# @file name  : c_data_downsample.py
# @author     : https://github.com/TingsongYu
# @date       :
# @brief      :
"""

import os
import shutil
import random

BASE_DIR = os.path.dirname(__file__)
random.seed(42)

if __name__ == '__main__':

    data_root = r"G:\deep_learning_data\DOTAv1\DOTA-sub-split"
    new_data_root = data_root + '-downsample'
    if os.path.exists(new_data_root):
        shutil.rmtree(new_data_root)

    # 采样比例
    sampling_ratio = 0.1

    # 处理 train 和 val 目录
    for split in ['train', 'val']:
        # 原始图片和标签路径
        split_image_dir = os.path.join(data_root, 'images', split)
        split_label_dir = os.path.join(data_root, 'labels', split)

        # 新图片和标签路径
        new_split_image_dir = os.path.join(new_data_root, 'images', split)
        new_split_label_dir = os.path.join(new_data_root, 'labels', split)
        os.makedirs(new_split_image_dir, exist_ok=True)
        os.makedirs(new_split_label_dir, exist_ok=True)

        # 获取当前 split 的所有图片文件
        label_files = [f for f in os.listdir(split_label_dir) if f.endswith(('.txt'))]
        random.shuffle(label_files)  # 随机打乱顺序

        # 计算需要采样的数量
        sample_size = int(len(label_files) * sampling_ratio)
        sampled_files = label_files[:sample_size]  # 取前10%

        # 复制采样后的图片和标签文件到新目录
        for label_file in sampled_files:
            # 复制图片
            image_file = label_file.replace('txt', 'jpg')

            original_image_path = os.path.join(split_image_dir, image_file)
            new_image_path = os.path.join(new_split_image_dir, image_file)
            shutil.copy(original_image_path, new_image_path)

            # 复制对应的标签文件
            original_label_path = os.path.join(split_label_dir, label_file)
            new_label_path = os.path.join(new_split_label_dir, label_file)

            if os.path.exists(original_label_path):
                shutil.copy(original_label_path, new_label_path)

        print(f"{split} 下采样完成！共采样 {len(sampled_files)} 张图片和标签文件。")
