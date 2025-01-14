# -*- coding: utf-8 -*-
"""
# @file name  : b_data_crop.py
# @author     : https://github.com/TingsongYu
# @date       : 2025/01/13
# @brief      : 对图像进行切块处理
"""

import os
from ultralytics.data.split_dota import split_test, split_trainval

BASE_DIR = os.path.dirname(__file__)

if __name__ == '__main__':

    data_dir = r"G:\deep_learning_data\DOTAv1\DOTA-sub"  # 上一步获取得到的数据目录
    out_dir = os.path.join(os.path.dirname(data_dir), 'DOTA-sub-split')

    # 对数据集图片进行切块处理，“降低”图片分辨率
    split_trainval(data_root=data_dir, save_dir=out_dir, rates=[1.0], crop_size=640, gap=50)  # gap表示重叠的像素

