# -*- coding: utf-8 -*-
"""
# @file name  : a_cut_data.py
# @author     : https://github.com/TingsongYu
# @date       : 2025/01/13
# @brief      : 对DOTAv1 数据集进行类别裁剪，选择需要训练的类别，重新生产数据集
"""

import os
import shutil


def cut_data(data_dir, out_dir, setname='train'):
    original_image_dir = os.path.join(data_dir, 'images', setname)
    original_label_dir = os.path.join(data_dir, 'labels', setname)
    new_image_dir = os.path.join(out_dir, 'images', setname)
    new_label_dir = os.path.join(out_dir, 'labels', setname)
    os.makedirs(new_image_dir, exist_ok=True)
    os.makedirs(new_label_dir, exist_ok=True)

    counter = 0
    for label_file in os.listdir(original_label_dir):
        label_path = os.path.join(original_label_dir, label_file)
        new_label_path = os.path.join(new_label_dir, label_file)

        with open(label_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])

            # 只保留指定类别的目标
            if class_id in keep_classes:
                # 修改类别索引
                parts[0] = str(keep_classes[class_id])
                new_lines.append(' '.join(parts) + '\n')

        # 如果有保留的目标，则保存新的标签文件和对应的图片
        if new_lines:
            with open(new_label_path, 'w') as f:
                f.writelines(new_lines)

            image_file = label_file.replace('.txt', '.jpg')
            original_image_path = os.path.join(original_image_dir, image_file)
            new_image_path = os.path.join(new_image_dir, image_file)

            shutil.copy(original_image_path, new_image_path)
            counter += 1

    print("筛选和映射完成！获得{}张图片，位于:{}".format(counter, out_dir))


if __name__ == '__main__':
    # 原始数据路径
    # 下载链接：https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip)
    data_dir = r"G:\deep_learning_data\DOTAv1"
    out_dir = os.path.join(data_dir, "DOTA-sub")  # 输出目录

    # 需要保留的类别及其新映射
    keep_classes_l = [0, 1]  # 挑选的类别，这里选择第0类和第1类。
    keep_classes = {ori_class_idx: trg_index for trg_index, ori_class_idx in enumerate(keep_classes_l)}

    cut_data(data_dir, out_dir, setname='train')
    cut_data(data_dir, out_dir, setname='val')
