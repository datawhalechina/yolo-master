# -*- coding: utf-8 -*-
"""
# @file name  :
# @author     : https://github.com/TingsongYu
# @date       :
# @brief      :
"""
import cv2
import os
import os
from ultralytics import YOLO
BASE_DIR = os.path.dirname(__file__)

if __name__ == '__main__':
    dataset_path = os.path.join(BASE_DIR, 'cfg', 'DOTAv1-sub.yaml')  # 数据集
    weights_path = os.path.join(BASE_DIR, 'runs', 'obb', 'train8', 'weights', 'best.pt')  # 模型

    # ========================== step1: 初始化YOLO对象 ==========================
    model = YOLO(weights_path)
    metrics = model.val(data=dataset_path, imgsz=640, batch=16, conf=0.25, iou=0.6)


