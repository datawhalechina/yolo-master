# YoLo Master

## YOLO Models from Scratch

本项目旨在采用**手撸YOLO模型**的方式，来带大家一起在**小型数据集上复现YOLO算法**，目前计划里打算对YOLO [**V1**](../v1/YOLOv1.ipynb)，[**V3**](../v3/YOLOv3.ipynb)，[**V5**](../v5/YOLOv5.ipynb)，V8等进行手撸。

我们计划首先编写一套可以共用的数据 pipeline ( **`dataset`** , **`dataloader`** ), 最终采取较为统一的基于 **Pytorch**的简单算法接口，并进行在关键通用数据集上的精度对齐。

欢迎感兴趣的有算力的小伙伴加入我们, 一起手撸YOLO模型!!

## YOLO v1: 从零开始的 PyTorch 实现
本仓库实现了论文 [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) 中的内容，使用 PyTorch。代码遵循 [Darknet](https://github.com/pjreddie/darknet) 仓库的官方实现，与论文相比有一些细微差别：

- 最重要的区别在于模型的架构。具体来说，第一个全连接层被局部连接层取代。在论文中，YOLO 模型的架构如下：
<p align="center" width="100%"> <img src="images/model_architecture.png"/> </p>

- 在每个卷积层中，卷积操作后和激活函数之前使用了批量归一化操作。
- 学习率调度和网络训练的最大批次。

本仓库从零开始实现了论文中的内容，包括：
+ 使用 VOC 训练集（train/val 2007 + train/val 2012）进行训练，以及
+ 使用 VOC 测试集（test 2007）进行评估

### 已完成部分
1. 已完成VOC数据集导入
2. 各模块的代码撰写与网络结构的拼接

### 需要完善部分
1. 还需要适配COCO数据集


### 安装环境
```
pip install -r requirements.txt
```


### 数据集准备
#### PASCAL VOC 2007 + PASCAL VOC 2012 数据集
要下载和准备 VOC 数据集，请按以下顺序运行脚本：


```
./download_voc.sh ./data/voc
./organize_voc.sh ./data/voc
python3 simplify_voc_targets.py ./data/voc
```


如果需要小数据集进行训练（可选）
- --source_voc表示完整的voc原始数据的路径
- --target_voc表示筛选过后的小数据集存储路径
- --classes表示需要筛选的标签使用英文逗号隔开，例如： car,person,bus
```
python scripts/toy_voc.py --source_voc ./data/voc/VOC_Detection --target_voc ./data/voc/small_voc --classes car,person,bus
```

数据转换后的csv如下图所示：

![](images/data.png)


### [Notebook](./yolov1.ipynb)大纲

![image](https://github.com/user-attachments/assets/d7141cd5-ba30-4ba6-9e71-b26db6dd524b)

#### 训练细节

采用默认的Config，第一个epoch耗时140s，156epoch大约需要6h。

batch_size=64显存需要约14G。

```
训练Epoch:   1%|          | 1/156 [02:20<6:02:08, 140.18s/epoch, 训练损失=7.125, 测试损失=4.574, 当前学习率=0.000500, 耗时=140.18s]
Epoch 1/156:
  训练损失: 7.125, 测试损失: 4.574
  当前学习率: 0.000500
  本轮训练耗时: 140.18s
```

#### 训练结果

MAP(Mean Average Precision) = `79.4%`

![image](https://github.com/user-attachments/assets/8b849673-d8a8-465b-9b04-6054d163e095)



