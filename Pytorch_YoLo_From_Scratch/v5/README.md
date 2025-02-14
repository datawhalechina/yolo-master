# YOLOV5 from scratch 代码文件说明

代码贡献者：蔡鋆捷

## 已完成部分
**1.** 已完成coco8数据集导入

**2.** 各模块的代码撰写与网络结构的拼接

## 需要完善部分
**1.** 补充YOLOv5的损失函数

**2.** 补充train与test代码(补充的时候注意电脑显存问题)

## 图片说明
**1.** 下面图片为ConBNSiLU模块的结构图

<img src="./image/ConBNSiLU.png" alt="ConBNSiLU模块" width="300" />

**2.** 下面图片为BottleNeck1模块的结构图

<img src="./image/bottleneck1.png" alt="图片描述" width="300" />

**3.** 下面图片为BottleNeck2模块的结构图

<img src="./image/bottleneck2.png" alt="图片描述" width="300" />

**4.** 下面图片为SPPF模块的结构图

<img src="./image/SPPF.png" alt="图片描述" width="600" />

## 其他实践

[YOLO V3 from scratch Notebook](../v3/YOLOv3_Hong.ipynb) 中对使用的coco8和coco128数据集进行了较多**探索性数据分析(EDA)**，大家可以优先查看.
