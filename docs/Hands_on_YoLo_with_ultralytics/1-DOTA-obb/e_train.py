# -*- coding: utf-8 -*-
"""
# @file name  : e_train.py
# @author     : https://github.com/TingsongYu
# @date       : 2025/01/13
# @brief      : 模型训练
"""

import os
from ultralytics import YOLO
BASE_DIR = os.path.dirname(__file__)

"""
由于ultralytics旋转目标检测的代码问题，在训练中的验证代码里的NMS可能导致显存激增.
这块目前没有好的方法，建议大家修改max_nms数量来避免。
具体操作：
首先找到ultralytics的安装目录,然后找到对应的.py文件：ultralytics/models/yolo/obb/val.py。
对于ops.non_max_suppression函数，新增一个入参，max_nms=1920。
"""
import ultralytics
install_dir = os.path.dirname(ultralytics.__file__)
val_path = os.path.join(install_dir, "models", 'yolo', 'obb', 'val.py')
print(f"ultralytics 安装目录: {install_dir} \n所需修改的文件路径:{val_path}")

if __name__ == '__main__':

    dataset_path = os.path.join(BASE_DIR, 'cfg', 'DOTAv1-sub.yaml')  # 数据集
    # weights_path = os.path.join(BASE_DIR, 'weights', 'yolo11n-obb.pt')  # 模型
    yolo11n_obb_path = os.path.join(BASE_DIR, 'cfg', 'yolo11s-obb.yaml')  # 模型

    # Load a model
    # model = YOLO(weights_path)
    model = YOLO(yolo11n_obb_path)

    # Train the model
    results = model.train(data=dataset_path, epochs=100, amp=False, batch=8)

