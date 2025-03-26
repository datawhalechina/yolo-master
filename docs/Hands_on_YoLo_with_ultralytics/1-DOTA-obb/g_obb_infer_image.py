# -*- coding: utf-8 -*-
"""
# @file name  : g_obb_infer_image.py
# @author     : https://github.com/TingsongYu
# @date       : 2025/01/13
# @brief      : 模型推理及分析
"""
import cv2
import numpy as np
import os
from ultralytics import YOLO
BASE_DIR = os.path.dirname(__file__)

if __name__ == '__main__':

    # weights_path = os.path.join(BASE_DIR, 'weights', 'yolo11m-obb.pt')  # 模型
    weights_path = os.path.join(BASE_DIR, 'runs', 'obb', 'train6', 'weights', 'best.pt')  # 模型
    image_path = r"G:\deep_learning_data\DOTAv1\DOTA-sub-split-downsample\images\val\P2231__640__590___1180.jpg"

    # ========================== step1: 初始化YOLO对象 ==========================
    model = YOLO(weights_path)

    # ========================== step2: 执行推理 ==========================
    results = model(image_path, conf=0.1, iou=0.7)  # 推理配置
    annotated_frame = results[0].plot()  # 检测框结果绘制

    # ========================== step3: OBB预测分析 ==========================
    xywhr_pred = np.array(results[0].obb[0].xywhr.to('cpu'))[0]
    print(xywhr_pred)

    # 获取旋转矩形框的四个顶点
    cx, cy, w, h, angle = xywhr_pred
    angle = angle * 180 / np.pi
    rect = ((cx, cy), (w, h), angle)
    box = cv2.boxPoints(rect)  # 获取四个顶点坐标
    box = np.int0(box)  # 转换为整数
    print("xywhr转换后得到四个顶点：", box)

    # 在图像上绘制旋转矩形框
    cv2.polylines(annotated_frame, [box], isClosed=True, color=(0, 255, 0), thickness=3)

    cv2.imshow("YOLO Inference", annotated_frame)
    cv2.waitKey(0)

