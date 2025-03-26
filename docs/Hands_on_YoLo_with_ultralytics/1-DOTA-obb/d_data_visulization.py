# -*- coding: utf-8 -*-
"""
# @file name  : d_data_visulization.py
# @author     : https://github.com/TingsongYu
# @date       : 2025/01/13
# @brief      : 对DOTAv1 数据集标签进行可视化
"""
import math
import os
import random
import cv2
import matplotlib.pyplot as plt
BASE_DIR = os.path.dirname(__file__)


def parse_annotation(annotation_path):
    """
    解析 YOLO 格式的标注文件，返回标注信息
    """
    annotations = []
    with open(annotation_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) == 9:  # YOLO 格式：class_id x1 y1 x2 y2 x3 y3 x4 y4
                class_id = int(parts[0])  # 类别 ID
                coords = list(map(float, parts[1:]))  # 提取归一化的 OBB 顶点坐标
                annotations.append((class_id, coords))
    return annotations


def draw_obb(image, annotations, img_width, img_height):
    """
    在图像上绘制有向边界框（OBB）和类别标签
    """
    for ann in annotations:
        class_id, coords = ann
        # 将归一化坐标转换为实际像素坐标
        points = [(int(coords[i] * img_width), int(coords[i + 1] * img_height)) for i in range(0, 8, 2)]
        # 绘制 OBB
        for i in range(4):
            cv2.line(image, points[i], points[(i + 1) % 4], (0, 255, 0), 2)
        # 绘制类别标签
        cv2.putText(image, str(class_id), points[0], cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
    return image


def visualize_random_images(images_dir, labels_dir, num_images=4):
    """
    随机选择并可视化数据集中的图像
    """
    # 获取所有图像文件名
    labels_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    selected_images = random.sample(labels_files, min(num_images, len(labels_files)))  # 随机选择图像

    # 动态调整 subplot 布局
    num_rows = math.ceil(math.sqrt(num_images))  # 行数
    num_cols = math.ceil(num_images / num_rows)  # 列数

    # 创建画布
    plt.figure(figsize=(num_cols * 5, num_rows * 5))  # 根据图像数量调整画布大小

    # 可视化选中的图像
    for i, label_file in enumerate(selected_images):
        # 加载图像
        image_file = label_file.replace('.txt', '.jpg')  # 标签文件名与图片文件名一致
        image_path = os.path.join(images_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
        img_height, img_width, _ = image.shape  # 获取图像的宽度和高度

        # 加载标注
        annotation_path = os.path.join(labels_dir, label_file)
        annotations = parse_annotation(annotation_path)

        # 绘制 OBB 和类别标签
        image_with_boxes = draw_obb(image, annotations, img_width, img_height)

        # 显示图像
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(image_with_boxes)
        plt.title(image_file)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 设置数据集路径
    # https://github.com/ultralytics/assets/releases/download/v0.0.0/DOTAv1.zip

    images_dir = r'G:\deep_learning_data\DOTAv1\DOTA-sub-split\images\train'  # 图片路径
    labels_dir = r'G:\deep_learning_data\DOTAv1\DOTA-sub-split\labels\train'  # 标签路径

    # 可视化随机N张图像
    visualize_random_images(images_dir, labels_dir, num_images=4)


