# -*- coding: utf-8 -*-
"""
# @file name  :
# @author     : https://github.com/TingsongYu
# @date       :
# @brief      :
"""

import os
from ultralytics import YOLO
BASE_DIR = os.path.dirname(__file__)

if __name__ == '__main__':

    dataset_path = os.path.join(BASE_DIR, 'cfg', 'DOTAv1-sub.yaml')  # 数据集
    weights_path = os.path.join(BASE_DIR, 'weights', 'yolo11n-obb.pt')  # 模型
    yolo11n_obb_path = os.path.join(BASE_DIR, 'cfg', 'yolo11-obb.yaml')  # 模型

    # Load a model
    # model = YOLO(weights_path)
    model = YOLO(yolo11n_obb_path)

    # Train the model
    results = model.train(data=dataset_path, epochs=50, amp=False, batch=8)

