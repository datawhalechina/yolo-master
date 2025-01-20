# -*- coding: utf-8 -*-
"""
# @file name  : e_train.py
# @author     : https://github.com/TingsongYu
# @date       : 2025/01/20
# @brief      : 模型对比实验，包含不同模版版本、不同模型大小、不同图片输入尺寸、是否开启mosaic、是否开启角度旋转
"""

from ultralytics import YOLO
import itertools
import os
BASE_DIR = os.path.dirname(__file__)


def train_with_config(model_cfg, imgsz, data_aug_config, aug_id):
    # 生成实验名称
    model_name = model_cfg.replace("-obb.yaml", "")
    mosaic_status = "mosaic_on" if data_aug_config["mosaic"] == 1.0 else "mosaic_off"
    degrees_status = "degrees_180" if data_aug_config["degrees"] == 180 else "degrees_0"
    experiment_name = f"{model_name}_imgsz_{imgsz}_{mosaic_status}_{degrees_status}"  # 实验名称

    print(f"Training with config: model={model_name}, imgsz={imgsz}, "
          f"mosaic={data_aug_config['mosaic']}, degrees={data_aug_config['degrees']}, "
          f"name={experiment_name}")

    # 加载模型
    model = YOLO(model_cfg)

    # 训练模型
    results = model.train(
        data=dataset_path,  # 数据集配置文件
        epochs=100,  # 训练轮数
        imgsz=imgsz,  # 输入分辨率
        batch=4,  # 批量大小, 采用90%显存
        mosaic=data_aug_config["mosaic"],  # Mosaic 数据增强
        degrees=data_aug_config["degrees"],  # 随机旋转增强
        name=experiment_name,  # 实验名称
        amp=True,  # 禁用混合精度训练
    )

    return results


if __name__ == '__main__':

    # 数据集路径
    dataset_path = os.path.join(BASE_DIR, 'cfg', 'DOTAv1-sub.yaml')  # 数据集配置文件

    # 定义模型配置文件和输入分辨率列表
    versions = "v8 11".split()[::-1]  # 从最大的开始跑，避免出现OOM
    sizes = "n s m l".split()[::-1]
    model_cfgs = [f"yolo{version}{size}-obb.yaml" for version in versions for size in sizes]
    imgszs = [640, 960, 1280][::-1]

    # 生成所有组合
    experiment_configs = list(itertools.product(model_cfgs, imgszs))

    # 数据增强配置
    data_aug_configs = [
        {"mosaic": 0.0, "degrees": 0},  # 实验 1：关闭 Mosaic，关闭随机旋转
        {"mosaic": 0.0, "degrees": 180},  # 实验 2：关闭 Mosaic，开启随机旋转
        {"mosaic": 1.0, "degrees": 0},  # 实验 3：开启 Mosaic，关闭随机旋转
        {"mosaic": 1.0, "degrees": 180},  # 实验 4：开启 Mosaic，开启随机旋转
    ]

    # 运行消融实验
    for model_cfg, imgsz in experiment_configs:
        for aug_id, data_aug_config in enumerate(data_aug_configs):
            train_with_config(model_cfg, imgsz, data_aug_config, aug_id)