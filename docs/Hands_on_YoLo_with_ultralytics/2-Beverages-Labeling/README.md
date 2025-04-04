# 2-Beverages-Labeling

在本章教程中，我们对选取的[鱼眼镜头_智能销售数据集](https://aistudio.baidu.com/datasetdetail/91732/0) 和[饮料数据集](https://zihao-download.obs.cn-east-3.myhuaweicloud.com/yolov8/datasets/Drink_284_Detection_Dataset/Drink_284_Detection_YOLO.zip)两个数据集作为示例数据集，来手把手的教大家如何进一步处理私有数据集并进行自动打标。以下为主要步骤和效果截图
- 进行格式转化处理（VOC2YOLO）
- 数据集的合并和采样
- 训练自定义模型（YOLOv8m）
- 结合打标工具 X-AnyLabeling 进行自动打标

![image](https://github.com/user-attachments/assets/3f46b897-1fee-4ac3-8a2f-92538ba60e32)

![image](https://github.com/user-attachments/assets/6ec7848b-fb39-4087-9c86-c5b577fd2a25)


## 项目结构
```python
2-Beverages-Labeling/

├── 00-dataset                          # 任务数据集介绍与准备
│   └── VOC_to_YOLO.py                    # # VOC 格式转 YOLO 格式转换脚本
├── 01-data_merging                     # 合并数据集
│   ├── dataset_merge.py                  # # 多数据集合并脚本
│   └── label_id_update.py                # # 标签 ID 更新映射脚本 
├── 02-ulralytics_based_model_training  # 基于 ultralytics 训练自定义模型使用 X-AnyLabeling 和微调后的模型为私有数据集打标
│   ├── data.yaml                         # # 数据集配置文件
│   ├── train.py                          # # 模型训练脚本
│   └── export.py                         # # 模型导出脚本
└── README.md
```


## 程序文件详细介绍

### 任务数据集介绍与准备

- **[VOC_to_YOLO.py](./00-dataset/VOC_to_YOLO.py)**：**此脚本用于将 VOC 格式的数据集转换为 YOLO 格式。** 在实际应用中，不同来源的数据集可能采用不同的标注格式，VOC 是常见的一种，而 YOLO 训练需要特定的格式，该脚本解决了格式转换的问题。

### 合并数据集

![image](https://github.com/user-attachments/assets/06e117c5-c651-43fa-bdad-60c97e3bb147)

![image](https://github.com/user-attachments/assets/7e5801a9-fe8f-4ce2-b890-76882276e750)


- **[dataset_merge.py](./01-data_merging/dataset_merge.py): 合并两个不同的饮料数据集**（*Drink_284_Detection_Labelme*和*鱼眼镜头_智能销售数据集*），处理类别重合问题，并保证数据集比例平衡。脚本优先选择包含重合类别的样本，确保模型能够学习到这些共享特征。

- **[label_id_update.py](./01-data_merging/label_id_update.py): 更新标签 ID**，使不同数据集的标签 ID 统一。脚本会创建索引映射字典，并据此更新所有标签文件中的类别索引。

### 基于ultralytics训练自定义模型并导出 ONNX。

- **[data.yaml](./02-ulralytics_based_model_training/data.yaml)**: 数据集配置文件，定义了训练和验证数据路径，以及所有类别名称（共 113 个类别）, YOLOv8m 训练时需要通过该文件来获取数据集的相关信息。

- **[train.py](./02-ulralytics_based_model_training/train.py)**: 模型训练脚本，微调预训练的 YOLOv8m 权重进行迁移学习，可以设置各种训练参数如轮数、图像尺寸、批次大小等。

- **[export.py](./02-ulralytics_based_model_training/export.py)**: 导出 ONNX 格式的微调后模型用于私有数据集的打标。

## 飞书教程

在本章教程中，我们选取的示例数据集来自*鱼眼镜头_智能销售数据集*与*饮料数据集*两个数据集，我们会对两个数据集进行格式转化处理，合并和采样，来手把手的教大家如何进一步的处理私有数据集，并训练自定义模型（ YOLOv8m ），结合打标工具 X-AnyLabeling 进行自动打标。


**教程地址**：https://wvet00aj34c.feishu.cn/docx/R04QdmQMMoaA44xyDYkcA0AfnOd

以下是简要使用方法，更多细节请查看飞书教程。

### 00. 数据集格式转换

运行 `./00-dataset/VOC_to_YOLO.py)`，将 VOC 格式的数据集转换为 YOLO 格式。

### 01. 数据集合并与标签更新

1. 运行 `./01-data_merging/dataset_merge.py`，合并两个不同的饮料数据集，处理类别重合问题，并保证数据集比例平衡。

2. 运行 `./01-data_merging/label_id_update.py`，更新标签 ID，使不同数据集的标签 ID 统一。

### 02. 模型训练与导出

1. 运行 `./02-ulralytics_based_model_training/train.py`，使用预训练的 YOLOv8m 权重进行迁移学习，根据需要调整`./02-ulralytics_based_model_training/data.yaml`中的训练参数。

2. 运行 `./02-ulralytics_based_model_training/export.py`，将训练好的模型导出为 ONNX 格式。

### 03. 私有数据集打标

- **使用 X-AnyLabeling 工具和导出的 ONNX 模型为私有数据集打标。** 具体操作，详见[**飞书教程**](https://wvet00aj34c.feishu.cn/docx/R04QdmQMMoaA44xyDYkcA0AfnOd)。

## 数据集和训练后的模型下载

> YOLO Master 在魔搭社区建立了专门服务于 YOLO Master 项目的 ModelScope 组织，将会搬运 YOLO 各个版本的官方模型权重在这里备份，方便国内的学习者使用，也能支持贡献者们存放占用空间较大的实验结果和权重。

> YOLO Master ModelScope 组织地址：https://modelscope.cn/organization/yolo_master

本教程的数据集，模型存放在 **[YOLO 实践之数据集合并与自动打标教程文件合集](https://www.modelscope.cn/collections/YOLO-shijianzhishujujihebingyuzi-54143625553f44)** 中，学习者可以通过访问上述链接获取所需资源。

![image](https://github.com/user-attachments/assets/f039c2c6-c642-4ce2-a13e-0f57ad7c70d2)

![image](https://github.com/user-attachments/assets/847bec8e-c842-4a33-83a4-f5e52c03f8be)


