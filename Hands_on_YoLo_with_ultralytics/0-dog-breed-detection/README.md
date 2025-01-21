# labs-dog-breed-detection

![dogs1.jpg](./runs/detect/predict4/dogs1.jpg)

这个教程是为了从数据集处理的角度，帮助大家快速上手基于ultralytics YOLOv8，并完成一个狗品种检测的实战项目。

数据集来源于[**Stanford Dogs Dataset**](http://vision.stanford.edu/aditya86/ImageNetDogs/) (kaggle上也有人上传整理，[链接](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset))，包含120种狗品种，每个品种有100张图片。

## 数据集Context
斯坦福狗数据集包含来自世界各地的 120 种狗的图像。该数据集使用 ImageNet 中的图像和注释构建而成，用于细粒度图像分类任务。它最初是为了细粒度图像分类而收集的，这是一个具有挑战性的问题，因为某些狗品种具有几乎相同的特征或颜色和年龄不同。 

![val_batch2_labels.jpg](./yolov8m-v1/val_batch2_labels.jpg)

## 数据集Content
- 类别数量：120 
- 图像数量：20,580 
- 注释：类别标签、边界框

## 文件介绍

### 文件结构
```
├── README.md
├── dog-breed-detection_kaggle.ipynb
├── dog-breed-detection_local.ipynb
├── runs
│   └── detect
│       └── predict4
│           ├── dogs1.jpg
│           └── dogs2.jpg
├── test_dogs
│   ├── dogs1.png
│   └── dogs2.png
└── yolov8m-v1
    ├── F1_curve.png
    ├── PR_curve.png
    ├── P_curve.png
    ├── R_curve.png
    ├── args.yaml
    ├── confusion_matrix.png
    ├── confusion_matrix_normalized.png
    ├── labels.jpg
    ├── labels_correlogram.jpg
    ├── results.csv
    ├── results.png
    ├── train_batch0.jpg
    ├── train_batch1.jpg
    ├── train_batch2.jpg
    ├── train_batch23250.jpg
    ├── train_batch23251.jpg
    ├── train_batch23252.jpg
    ├── val_batch0_labels.jpg
    ├── val_batch0_pred.jpg
    ├── val_batch1_labels.jpg
    ├── val_batch1_pred.jpg
    ├── val_batch2_labels.jpg
    ├── val_batch2_pred.jpg
    └── weights
        └── best.pt
```

- `dog-breed-detection_kaggle.ipynb` 是基于kaggle平台P100 GPU （16G）进行训练的notebook，需要使用kaggle的GPU资源，请大家直接看我修改后[**dog-breed-detection-hong-fixed**](https://www.kaggle.com/code/chg0901/dog-breed-detection/notebook) (修改调整自[vineetmahajan/dog-breed-detection](https://www.kaggle.com/code/vineetmahajan/dog-breed-detection)，这个文件会有点环境问题，所以需要修改)。

- **`dog-breed-detection_local.ipynb`** 是基于本地GPU（A6000 32G, 三卡似乎有点问题，大家可以根据自己的计算资源调整batch size）进行训练的notebook，需要本地有GPU资源，并下载数据集（notebook会全部提供），修改调整自上面的kaggle notebook。

- **`yolov8m-v1`** 是基于`dog-breed-detection_local.ipynb`训练好的模型路径，包括训练过程中的权重、训练结果等。
  - `yolov8m-v1/weights` 是训练好的模型权重，只保留了`best.pt`
  - `args.yaml` 是训练过程中的参数记录。
  - `results.csv` 是训练过程中的记录。
  - `results.png` 是训练过程中的记录的可视化。
  - `confusion_matrix.png` ，`confusion_matrix_normalized.png` 是训练过程中的混淆矩阵的可视化和可视化归一化。
  - `labels.jpg`，`labels_correlogram.jpg` 是训练过程中的标签和标签的correlogram。
  - `F1_curve.png`，`PR_curve.png`，`P_curve.png`，`R_curve.png` 是训练过程中的F1曲线、PR曲线、P曲线、R曲线。
  - `train_batch0.jpg`，`train_batch1.jpg`，`train_batch2.jpg` 是训练过程中的训练图片，采用了mosaic的数据增强方式。
  - `train_batch23250.jpg`，`train_batch23251.jpg`，`train_batch23252.jpg` 是训练过程中的训练图片，不再采用mosaic的数据增强方式。
  - `val_batch0_labels.jpg`，`val_batch1_labels.jpg` ，`val_batch2_labels.jpg` 是训练过程中的验证图片的标签。
  - `val_batch0_labels.jpg`，`val_batch1_labels.jpg` ，`val_batch2_labels.jpg` 是训练过程中的验证图片的标签。
  - `val_batch0_pred.jpg`，`val_batch1_pred.jpg` ，`val_batch2_pred.jpg` 是训练过程中的验证图片的预测值。

- `test_dogs` 是测试图片，用于测试模型效果。
  - 包含`dogs1.jpg` 和 `dogs2.jpg` 两张图片
  
- `runs` 是测试图片的预测结果
  - `runs/detect/predict4` 里存放了`test_dogs`中的`dogs1.jpg`和`dogs2.jpg`的预测结果。`4`是预测时的index。  



## 0. 环境准备 [todo]

## 1. 数据集准备

## 2. 模型训练

## 3. 模型推理

## 4. 模型评估
5. 模型部署