# YoLo Master

## YOLO Models from Scratch

本项目旨在采用**手撸YOLO模型**的方式，来带大家一起在**小型数据集上复现YOLO算法**，目前计划里打算对YOLO [**V1**](../v1/YOLOv1.ipynb)，[**V3**](../v3/YOLOv3.ipynb)，[**V5**](../v5/YOLOv5.ipynb)，V8等进行手撸。

我们计划首先编写一套可以共用的数据 pipeline ( **`dataset`** , **`dataloader`** ), 最终采取较为统一的基于 **Pytorch**的简单算法接口，并进行在关键通用数据集上的精度对齐。

欢迎感兴趣的有算力的小伙伴加入我们, 一起手撸YOLO模型!!

### YOLO V3 from scratch

[YOLO V3 from scratch Notebook](./YOLOv3.ipynb) 中对使用的coco8和coco128数据集进行了**探索性数据分析(EDA)**，大家可以根据自己需要查看.

内容大纲如下

- 手撸YOLOv3
    - 主要参考代码
    - 执行环境与关键python库
    - 数据集检查
        - COCOYOLODataset
        - dataset和dataloaders
            - COCODataset
            - create_dataloaders
    - 测试dataloader和可视化
        - 添加matplotlib中文支持
        - plot_image_with_boxes
        - test_visualization
    - YOLOLoss
    - YOLOv3 Model
        - ConvLayer
        - ResBlock
        - make_conv_and_res_block
        - YoloLayer（Used in DetectionBlock ）
        - DetectionBlock（类似Neck）
        - DarkNet53BackBone（Backbone）
        - YoloNetTail（Neck+Head）
        - YoloNetV3（Model）
        - test_yolov3_output_shape测试函数
    - YOLOEvaluator
        - low_confidence_suppression
        - non_max_suppression
        - calculate_map
        - calculate_ap
        - box_iou
        - evaluate_model
    - YOLOV3_Lightning与可视化预测框
        - YOLOV3_Lightning
        - visualize_predictions
        - test_nms_pipeline
        - training main函数
    - YOLOModule（对上面代码的改进）
        - 问题
        - plot_training_metrics

###  本节参考资料

- [YOLOv3-Object-Detection-from-Scratch](https://github.com/williamcfrancis/YOLOv3-Object-Detection-from-Scratch/blob/main/YOLO_object_detection.ipynb)
- [YOLOv3-in-PyTorch](https://github.com/westerndigitalcorporation/YOLOv3-in-PyTorch/blob/release/src/model.py)
- [coco128 in kaggle](https://www.kaggle.com/datasets/ultralytics/coco128)
