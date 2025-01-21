# -*- coding: utf-8 -*-
"""
# @file name  : utils_exp.py
# @author     : https://github.com/TingsongYu
# @date       : 2025/01/13
# @brief      : DOTA 模型训练工具函数库
"""
import os
import glob
import shutil
import math
import os
import random
import cv2
import yaml
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor


def cut_data(data_dir, out_dir, keep_classes, setname='train'):
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


def count_classes_in_file(label_path):
    """
    统计单个标签文件中的类别
    :param label_path: 标签文件路径
    :return: 包含类别ID的列表
    """
    with open(label_path, "r") as f:
        lines = f.readlines()
    return [int(line.split()[0]) for line in lines if line.strip()]


def count_classes_in_files(label_files):
    """
    统计多个标签文件中的类别
    :param label_files: 标签文件路径列表
    :return: 包含所有类别ID的列表
    """
    class_ids = []
    with ThreadPoolExecutor() as executor:
        results = executor.map(count_classes_in_file, label_files)
        for result in results:
            class_ids.extend(result)
    return class_ids


def print_yolo_dataset_info(dataset_path):
    """
    打印YOLO数据集的基本信息
    :param dataset_path: YOLO数据集的根目录路径
    """
    # 定义路径
    images_train_path = os.path.join(dataset_path, "images", "train")
    images_val_path = os.path.join(dataset_path, "images", "val")
    labels_train_path = os.path.join(dataset_path, "labels", "train")
    labels_val_path = os.path.join(dataset_path, "labels", "val")

    # 检查路径是否存在
    if not all(
            os.path.exists(path) for path in [images_train_path, images_val_path, labels_train_path, labels_val_path]):
        print("错误：数据集目录结构不符合YOLO格式！")
        print([images_train_path, images_val_path, labels_train_path, labels_val_path])
        return

    # 统计图片数量
    train_images = glob.glob(os.path.join(images_train_path, "*"))
    val_images = glob.glob(os.path.join(images_val_path, "*"))
    total_images = len(train_images) + len(val_images)

    # 统计标签数量
    train_labels = glob.glob(os.path.join(labels_train_path, "*"))
    val_labels = glob.glob(os.path.join(labels_val_path, "*"))
    total_labels = len(train_labels) + len(val_labels)

    # 统计类别数量
    all_label_files = train_labels + val_labels
    class_ids = count_classes_in_files(all_label_files)
    class_counts = defaultdict(int)
    for class_id in class_ids:
        class_counts[class_id] += 1

    # 打印基本信息
    print(f"数据集{dataset_path}基本信息：")
    print(f"训练集图片数量: {len(train_images)}")
    print(f"验证集图片数量: {len(val_images)}")
    print(f"总图片数量: {total_images}")
    print(f"训练集标签数量: {len(train_labels)}")
    print(f"验证集标签数量: {len(val_labels)}")
    print(f"总标签数量: {total_labels}")
    print(f"类别数量: {len(class_counts)}")
    print("类别分布：")
    for class_id, count in sorted(class_counts.items()):
        print(f"  类别 {class_id}: {count} 个实例")


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


def visualize_random_images(yolo_data_dir, setname='train', num_images=4):
    """
    随机选择并可视化数据集中的图像
    """
    images_dir = os.path.join(yolo_data_dir, 'images', setname)  # 图片路径
    labels_dir = os.path.join(yolo_data_dir, 'labels', setname)  # 标签路径

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


def check_yolo_yaml_paths(yaml_path):
    """
    检查YOLO YAML文件中的路径是否存在
    :param yaml_path: YAML文件的路径
    """
    # 检查YAML文件是否存在
    if not os.path.exists(yaml_path):
        print(f"错误：YAML文件 '{yaml_path}' 不存在！")
        return

    # 读取YAML文件（指定编码为utf-8）
    with open(yaml_path, "r", encoding="utf-8") as f:
        try:
            yaml_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"错误：无法解析YAML文件 '{yaml_path}'，原因：{e}")
            return
        except UnicodeDecodeError as e:
            print(f"错误：文件编码不是utf-8，请检查文件 '{yaml_path}' 的编码格式。")
            return

    # 检查训练、验证路径是否存在
    root_dir = yaml_data['path']
    for key in ["train", "val"]:
        if key not in yaml_data:
            print(f"警告：YAML文件缺少字段 '{key}'")
            continue

        paths = yaml_data[key]
        if paths is None:
            print(f"警告：字段 '{key}' 的值为 None")
            continue

        # 如果路径是字符串，转换为列表
        if isinstance(paths, str):
            paths = [paths]

        # 检查每个路径是否存在
        for path in paths:
            abs_path = os.path.abspath(os.path.join(root_dir, path))  # 转换为绝对路径
            if not os.path.exists(abs_path):
                print(f"警告!：路径 '{abs_path}'!")
                print(f"警告!：路径 '{abs_path}'!!")
                print(f"警告!：路径 '{abs_path}'!!!")


def visualize_yolo_logs(log_dir, setname='train'):
    """
    可视化 YOLO 训练日志中的图片文件
    :param log_dir: 日志目录路径（例如 'runs/train/exp'）
    """
    # 检查日志目录是否存在
    if not os.path.exists(log_dir):
        print(f"错误：日志目录 '{log_dir}' 不存在！")
        return

    # 定义需要可视化的图片文件名
    log_images = {
        "results.png": "results.png",
        "PR_curve.png": "PR_curve.png",
        "confusion_matrix.png": "confusion_matrix.png"
    }
    if setname == 'val':
        del log_images["results.png"]

    # 创建一个 figure，用于绘制所有图片
    fig, axes = plt.subplots(1, len(log_images), figsize=(24, 8))
    fig.suptitle("YOLO logs", fontsize=16)

    # 遍历图片文件并绘制
    for i, (image_name, title) in enumerate(log_images.items()):
        image_path = os.path.join(log_dir, image_name)
        if not os.path.exists(image_path):
            print(f"警告：图片文件 '{image_name}' 不存在！")
            continue

        # 读取图片
        img = mpimg.imread(image_path)

        # 绘制图片
        axes[i].imshow(img)
        axes[i].set_title(title)
        axes[i].axis("off")  # 关闭坐标轴

    # 调整布局并显示
    # plt.tight_layout()
    plt.show()
