# -*- coding: utf-8 -*-
"""
# @file name  : i_logs_plot.py
# @author     : https://github.com/TingsongYu
# @date       : 2025/01/20
# @brief      : 对比实验日志可视化
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import re
BASE_DIR = os.path.dirname(__file__)


def plot_log_model_size(log_dir):
    # 定义模型版本和对应的颜色
    models = ['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange']

    # 初始化存储数据的字典
    data = {model: {'imgsz': [], 'map50_95': []} for model in models}

    # 遍历日志文件夹
    for folder in os.listdir(log_dir):
        #
        re_1 = r'(yolo11|yolov8)([nslm])_imgsz_(\d+)_mosaic_(on)_degrees_(0)'
        # 使用正则表达式解析文件夹名称
        match = re.match(re_1, folder)
        if match:
            model_type = match.group(1) + match.group(2)  # 例如 'yolo11n'
            imgsz = int(match.group(3))  # 图像尺寸
            mosaic = match.group(4)  # mosaic开关
            degrees = int(match.group(5))  # 旋转角度

            # 读取results.csv文件
            csv_path = os.path.join(log_dir, folder, 'results.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # 获取metrics/mAP50-95(B)列的最高值
                max_map = df['metrics/mAP50-95(B)'].max()
                # 将数据存储到字典中
                data[model_type]['imgsz'].append(imgsz)
                data[model_type]['map50_95'].append(max_map)

    # 绘制曲线
    plt.figure(figsize=(10, 6))
    for model, color in zip(models, colors):
        if data[model]['imgsz']:  # 确保有数据
            # 按图像尺寸排序
            sorted_imgsz, sorted_map = zip(*sorted(zip(data[model]['imgsz'], data[model]['map50_95'])))
            # 设置线型：yolo11用实线，yolov8用虚线
            linestyle = '-' if model.startswith('yolo11') else '--'
            plt.plot(sorted_imgsz, sorted_map, label=model, color=color, linestyle=linestyle, marker='o')

    # 设置图表标题和标签
    plt.title('mAP50-95 vs Image Size')
    plt.xlabel('Image Size')
    plt.ylabel('mAP50-95')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_mosaic(log_dir, model_version='yolov8'):
    # 定义模型版本和对应的颜色
    model_sizes = "n s m l".split()
    models = [model_version + size for size in model_sizes]
    colors = ['b', 'g', 'r', 'c']  # 颜色用于区分模型版本
    line_styles = ['-', '--']  # 实线表示mosaic开启，虚线表示mosaic关闭

    # 初始化存储数据的字典
    data = {model: {'imgsz_mosaic_on': [], 'map50_95_mosaic_on': [], 'imgsz_mosaic_off': [], 'map50_95_mosaic_off': []} for
            model in models}

    # 遍历日志文件夹
    for folder in os.listdir(log_dir):
        # 使用正则表达式解析文件夹名称
        pattern = r'({})([nslm])_imgsz_(\d+)_mosaic_(on|off)_degrees_(0)'.format(re.escape(model_version))
        match = re.match(pattern, folder)
        # match = re.match(r'(model_version)([nslm])_imgsz_(\d+)_mosaic_(on|off)_degrees_(0)', folder)
        if match:
            model_type = match.group(1) + match.group(2)  # 例如 'yolov8n'
            imgsz = int(match.group(3))  # 图像尺寸
            mosaic = match.group(4)  # mosaic开关状态
            degrees = int(match.group(5))  # 旋转角度

            # 读取results.csv文件
            csv_path = os.path.join(log_dir, folder, 'results.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # 获取metrics/mAP50-95(B)列的最高值
                max_map = df['metrics/mAP50-95(B)'].max()
                # 根据mosaic状态存储数据
                if mosaic == 'on':
                    data[model_type]['imgsz_mosaic_on'].append(imgsz)
                    data[model_type]['map50_95_mosaic_on'].append(max_map)
                else:
                    data[model_type]['imgsz_mosaic_off'].append(imgsz)
                    data[model_type]['map50_95_mosaic_off'].append(max_map)

    # 绘制曲线
    plt.figure(figsize=(10, 6))
    for model, color in zip(models, colors):
        # 绘制mosaic开启的曲线
        if data[model]['imgsz_mosaic_on']:
            sorted_imgsz, sorted_map = zip(*sorted(zip(data[model]['imgsz_mosaic_on'], data[model]['map50_95_mosaic_on'])))
            plt.plot(sorted_imgsz, sorted_map, label=f'{model} (mosaic on)', color=color, linestyle=line_styles[0],
                     marker='o')
        # 绘制mosaic关闭的曲线
        if data[model]['imgsz_mosaic_off']:
            sorted_imgsz, sorted_map = zip(
                *sorted(zip(data[model]['imgsz_mosaic_off'], data[model]['map50_95_mosaic_off'])))
            plt.plot(sorted_imgsz, sorted_map, label=f'{model} (mosaic off)', color=color, linestyle=line_styles[1],
                     marker='x')

    # 设置图表标题和标签
    plt.title(f'mAP50-95 vs Image Size ({model_version}with Mosaic On/Off)')
    plt.xlabel('Image Size')
    plt.ylabel('mAP50-95')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_degrees(log_dir, model_version='yolov8'):
    # 定义模型版本和对应的颜色
    model_sizes = "n s m l".split()
    models = [model_version + size for size in model_sizes]
    # 定义模型版本和对应的颜色
    colors = ['b', 'g', 'r', 'c']  # 颜色用于区分模型版本
    line_styles = ['-', '--']  # 实线表示degrees=0，虚线表示degrees=180

    # 初始化存储数据的字典
    data = {
        model: {'imgsz_degrees_0': [], 'map50_95_degrees_0': [], 'imgsz_degrees_180': [], 'map50_95_degrees_180': []}
        for model in models}

    # 遍历日志文件夹
    for folder in os.listdir(log_dir):
        # 使用正则表达式解析文件夹名称
        pattern = r'({})([nslm])_imgsz_(\d+)_mosaic_(on)_degrees_(\d+)'.format(re.escape(model_version))
        match = re.match(pattern, folder)
        # match = re.match(r'(yolov8)([nslm])_imgsz_(\d+)_mosaic_(on)_degrees_(\d+)', folder)
        if match:
            model_type = match.group(1) + match.group(2)  # 例如 'yolov8n'
            imgsz = int(match.group(3))  # 图像尺寸
            mosaic = match.group(4)  # mosaic开关状态
            degrees = int(match.group(5))  # 旋转角度

            # 读取results.csv文件
            csv_path = os.path.join(log_dir, folder, 'results.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                # 获取metrics/mAP50-95(B)列的最高值
                max_map = df['metrics/mAP50-95(B)'].max()
                # 根据degrees状态存储数据
                if degrees == 0:
                    print(folder)
                    data[model_type]['imgsz_degrees_0'].append(imgsz)
                    data[model_type]['map50_95_degrees_0'].append(max_map)
                elif degrees == 180:
                    print(folder)
                    data[model_type]['imgsz_degrees_180'].append(imgsz)
                    data[model_type]['map50_95_degrees_180'].append(max_map)

    # 绘制曲线
    plt.figure(figsize=(10, 6))
    for model, color in zip(models, colors):
        # 绘制degrees=0的曲线
        if data[model]['imgsz_degrees_0']:
            sorted_imgsz, sorted_map = zip(
                *sorted(zip(data[model]['imgsz_degrees_0'], data[model]['map50_95_degrees_0'])))
            plt.plot(sorted_imgsz, sorted_map, label=f'{model} (degrees=0)', color=color, linestyle=line_styles[0],
                     marker='o')
        # 绘制degrees=180的曲线
        if data[model]['imgsz_degrees_180']:
            sorted_imgsz, sorted_map = zip(
                *sorted(zip(data[model]['imgsz_degrees_180'], data[model]['map50_95_degrees_180'])))
            plt.plot(sorted_imgsz, sorted_map, label=f'{model} (degrees=180)', color=color, linestyle=line_styles[1],
                     marker='x')

    # 设置图表标题和标签
    plt.title(f'mAP50-95 vs Image Size ({model_version} with Degrees=0 vs Degrees=180)')
    plt.xlabel('Image Size')
    plt.ylabel('mAP50-95')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':

    # 定义文件夹路径
    log_dir = os.path.join(BASE_DIR, 'runs', 'csv_files', 'obb')

    # 实验一：观察不同图像尺寸、不同模型版本之间的差异
    plot_log_model_size(log_dir)

    # 实验二：观察mosaic数据增强开启与关闭的差异
    plot_mosaic(log_dir, 'yolov8')

    # 实验三：观察旋转角度是否开启的差异
    plot_degrees(log_dir, 'yolov8')










