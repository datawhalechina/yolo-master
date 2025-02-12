# YoLo Master

![image](https://github.com/user-attachments/assets/f23752e3-e440-4fd6-a2ee-33b64bdc6544)

## YOLO Models from scratch

本项目旨在采用**手撸YOLO模型**的方式，来带大家一起在**小型数据集上复现YOLO算法**，目前计划里打算对YOLO V1，[**V3**](./v3/YOLOv3_Hong.ipynb)，[**V5**](./v5/YOLOv5_NetModel.ipynb)，V8等进行手撸。

我们计划首先编写一套可以共用的数据 pipeline ( dataset , dataloader ), 最终采取较为统一的基于torch的简单算法接口，并进行在关键通用数据集上的精度对齐。

欢迎感兴趣的有算力的小伙伴加入我们, 一起手撸YOLO模型!!

### 项目结构
```
.
├── datasets
│   ├── coco128.zip
│   └── coco8.zip
├── resource
├── v1
├── v3
│   ├── YOLOv3.ipynb
│   ├── config.py
│   └── metrics
└── v5
    ├── C3模块.png
    ├── ConBNSiLU.png
    ├── Conv2d模块.png
    ├── SPPF.png
    ├── Upsample模块.png
    ├── YOLOv5.png
    ├── YOLOv5_NetModel.ipynb
    ├── bottleneck1.png
    ├── bottleneck2.png
    ├── coco8
    └── readme.md
```


### 飞书教程

【todo】飞书教程期待大家共建~

---

【以下为基于ultralytics的YOLO系列入门和进阶教程】

- [0-dog-breed-detection YOLO系列**入门实操教程**](https://wvet00aj34c.feishu.cn/docx/Ojcfd0ZF5olk4Yxwt9ZcjgSenUD?from=from_copylink)

    - 【Stanford Dogs Dataset basic tutorial with yoloV8】 本章通过dog-breed-detection, 演示快速上手使用ultralytics完成一个狗品种检测的实战项目，实现YOLOV8模型的训练及推理，来帮助学习者通过实操掌握对yolo系列模型的**入门操作**。
    - 数据集来源于[**Stanford Dogs Dataset**](http://vision.stanford.edu/aditya86/ImageNetDogs/)


- [1-DOTA-obb YOLO系列**进阶实操教程**](https://wvet00aj34c.feishu.cn/docx/IPHFddAZmoBTr3xrRS0cW0Yanof?from=from_copylink)

    - 【DOTA-obb tutorial with yoloV11】 本章通过DOTAv1样例数据集演示如何在ultralytics项目中，实现**YOLOV11-n OBB模型**的训练及推理，来帮助学习者通过实操掌握对yolo系列模型的**进阶操作**。
    - 数据集来源于[**DOTA-v1.0**](https://captain-whu.github.io/DOTA/dataset.html)


### YOLO V3 from scratch


[YOLO V3 from scratch Notebook](./v3/YOLOv3_Hong.ipynb) 中对使用的coco8和coco128数据集进行了**探索性数据分析(EDA)**，大家可以优先查看.

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


##  本节参考资料

- [YOLOv3-Object-Detection-from-Scratch](https://github.com/williamcfrancis/YOLOv3-Object-Detection-from-Scratch/blob/main/YOLO_object_detection.ipynb)
- [YOLOv3-in-PyTorch](https://github.com/westerndigitalcorporation/YOLOv3-in-PyTorch/blob/release/src/model.py)
- [coco128 in kaggle](https://www.kaggle.com/datasets/ultralytics/coco128)



## 参与贡献

- 如果你发现了一些问题，可以提Issue进行反馈，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你想参与贡献本项目，可以提Pull request，如果提完没有人回复你可以联系[保姆团队](https://github.com/datawhalechina/DOPMC/blob/main/OP.md)的同学进行反馈跟进~
- 如果你对 Datawhale 很感兴趣并想要发起一个新的项目，请按照[Datawhale开源项目指南](https://github.com/datawhalechina/DOPMC/blob/main/GUIDE.md)进行操作即可~

## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。

*注：默认使用CC 4.0协议，也可根据自身项目情况选用其他协议*
