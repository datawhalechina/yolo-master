# 目录

2-Beverages-Labeling/
├── 任务数据集介绍与准备/
│ └── VOC_to_YOLO.py # VOC 格式转 YOLO 格式转换脚本
├── 合并数据集/
│ ├── dataset_merge.py # 多数据集合并脚本
│ └── label_id_update.py # 标签 ID 更新映射脚本
├── 基于 ultralytics 训练自定义模型/
│ ├── data.yaml # 数据集配置文件
│ └── train.py # 模型训练脚本
└── 使用 X-AnyLabeling 和微调后的模型为私有数据集打标/
├── export.py # 模型导出脚本
└── train.py # 模型训练脚本

# 程序文件详细介绍

## 任务数据集介绍与准备

- **VOC_to_YOLO.py**：**此脚本用于将 VOC 格式的数据集转换为 YOLO 格式。**在实际应用中，不同来源的数据集可能采用不同的标注格式，VOC 是常见的一种，而 YOLO 训练需要特定的格式，该脚本解决了格式转换的问题。

## 合并数据集

- **dataset_merge.py: 合并两个不同的饮料数据集**（Drink_284_Detection_Labelme和鱼眼镜头_智能销售数据集），处理类别重合问题，并保证数据集比例平衡。脚本优先选择包含重合类别的样本，确保模型能够学习到这些共享特征。

- **label_id_update.py: 更新标签ID**，使不同数据集的标签ID统一。脚本会创建索引映射字典，并据此更新所有标签文件中的类别索引。

## 基于ultralytics训练自定义模型。

- **data.yaml**: 数据集配置文件，定义了训练和验证数据路径，以及所有类别名称（共113个类别）,YOLOv8 训练时需要通过该文件来获取数据集的相关信息。

- **train.py**: 模型训练脚本，使用预训练的YOLOv8m权重进行迁移学习，可以设置各种训练参数如轮数、图像尺寸、批次大小等。

## 使用 X-AnyLabeling 和微调后的模型为私有数据集打标

- **export.py**：用于将训练好的模型导出为 ONNX 格式，为使用X-AnyLabeling工具和微调后的模型为私有数据集打标做准备。

# 飞书教程

**本教程手把手教你如何处理私有数据集，包括数据集格式转化、合并和采样，结合YOLOv8模型与X-AnyLabeling工具实现自动打标。通过鱼眼镜头_智能销售数据集与饮料数据集的实战案例，实现YOLO目标检测，快速上手YOLO模型训练与优化！**

地址：https://wvet00aj34c.feishu.cn/docx/R04QdmQMMoaA44xyDYkcA0AfnOd

## 1. 数据集格式转换

运行 `任务数据集介绍与准备/VOC_to_YOLO.py`，将 VOC 格式的数据集转换为 YOLO 格式。

## 2. 数据集合并与标签更新

1. 运行 `合并数据集/label_id_update.py`，更新标签 ID，使不同数据集的标签 ID 统一。

2. 运行 `合并数据集/label_id_update.py`，更新标签 ID，使不同数据集的标签 ID 统一。

## 3. 模型训练

运行 `基于ultralytics训练自定义模型/train.py`，使用预训练的 YOLOv8m 权重进行迁移学习，根据需要调整训练参数。

## 4. 模型导出

运行 `使用 X-AnyLabeling 和微调后的模型为私有数据集打标/export.py`，将训练好的模型导出为 ONNX 格式。

## 5. 私有数据集打标

**使用 X-AnyLabeling 工具和导出的 ONNX 模型为私有数据集打标。**

# 魔搭社区

我们在**魔搭社区(ModelScope)**中也有组织：**yolo_master**。

该组织地址为：https://www.modelscope.cn/organization/yolo_master

本教程的数据集，模型存放在[组织合集](https://www.modelscope.cn/collections/YOLO-shijianzhishujujihebingyuzi-54143625553f44)中，你可以通过访问上述链接获取所需资源。
