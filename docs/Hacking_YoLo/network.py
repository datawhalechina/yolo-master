"""
该程序用于快速打印模型网络结构，适用于魔改时快速检查模型结构
"""

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

yaml_path = "docs/Hacking_YoLo/ultralytics/cfg/models/11_Hacking/yolo11x_RMT.yaml"

# 打印网络结构图
DetectionModel(cfg=yaml_path)

# 详细网络结构参数
# model = YOLO(yaml_path)
# # print(model.info())   # 缩略版
# print(model.info(detailed=True))  # 详尽版