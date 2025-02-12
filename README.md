# YOLO Master

![image](https://github.com/user-attachments/assets/f23752e3-e440-4fd6-a2ee-33b64bdc6544)

## 项目简介

- 本项目主要对yolo系列模型进行介绍，包括各版本模型的结构，进行的改进等

- 本课程内容，在传统的DL课程中，大致位于深度视觉模型里的ResNet之后，transformer等squence model之前的位置

- 本课程旨在帮助学习者们可以了解和掌握主要yolo模型的发展脉络，以期在各自的应用领域可以进一步创新并在自己的任务上达到较好的效果。

- [飞书白皮书规划文档](https://sxwqtaijh4.feishu.cn/docx/WNLJdo0wxoFPuExt6rbcvB8MnPg?from=from_copylink)

- [内测文档](https://wvet00aj34c.feishu.cn/docx/FwivdWGqMoYQPSxMotMcYVIrnOh?from=from_copylink) 

![image](./Images/Main_README/Image_1.png)

- **目录结构说明**

  Hacking_Yolo: YOLO系列模型魔改教程

  Hands_on_YoLo_with_ultralytics: YOLO系列模型应用教程

  Images: 项目图片

  Pytorch_YoLo_From_Scratch: YOLO系列模型从零开始实现教程



## 项目受众

- 本课程面向有一定的机器学习基础的，上过**deep learning**和**图像图形学课程**的学生、工程师或者研究者
- 应用领域为基于YOLO的**目标检测**、**图像分类**、**图像分割**、**姿态检测**和**目标跟踪**(如 **[ultralytics](https://github.com/chg0901/yolo_master/tree/main/Hands_on_YoLo_with_ultralytics)** )
- 期待**手动实现YOLO算法（[From Scratch](https://github.com/chg0901/yolo_master/tree/main/Pytorch_YoLo_From_Scratch)）** 的学习者，将**YOLO系列模型应用到所在领域数据或者提高表现（[Hacking](https://github.com/chg0901/yolo_master/tree/main/Hacking_YoLo)）** 的工程师，研究者


## 目录

### 第一部分 YOLO 全系列模型详解 ###

1. [YOLOv1-YOLO11概述](https://sxwqtaijh4.feishu.cn/docx/Yc40ddMGIo7nOyxSXVZc6KztnYd?from=from_copylink) 

2. [YOLOv1详解](https://wvet00aj34c.feishu.cn/docx/U8STd5txXod1R5xhrrmcZh9fnTf?from=from_copylink) @刘伟鸿

3. [YOLOv2详解](https://wvet00aj34c.feishu.cn/docx/OHEhdwqXYoe8LIxwkRWcG0FLnnf?from=from_copylink) @蔡鋆捷

4. [YOLOv3详解](https://wvet00aj34c.feishu.cn/docx/U1e2dVfN3oFMUcxqkTWcNrNEnHr?from=from_copylink) @蔡鋆捷 @程宏

5. [YOLOv4详解](https://wvet00aj34c.feishu.cn/docx/IqGJdDvXsoNIGBxLsEWcGQGNnng?from=from_copylink) @蔡鋆捷

6. [YOLOv5详解](https://wvet00aj34c.feishu.cn/docx/CltUdiVfMoaSkXxGaTvcpAyWnWh?from=from_copylink) @蔡鋆捷

7. [YOLOv6详解](https://wvet00aj34c.feishu.cn/docx/Clvbd8PDAoLD4Jx1Asdc6Afon0d?from=from_copylink) @陈国威

8. [YOLOv7详解](https://wvet00aj34c.feishu.cn/docx/K5eCdF7fSohwvfxVpeIcF0ZLnK9?from=from_copylink) @蔡鋆捷

9. [YOLOv8详解](https://ycnosmsebbdf.feishu.cn/docx/EqtRdOuy2oPnAkxkIE6cNhBsnwc?from=from_copylink) @蔡鋆捷 @程宏

10. [YOLOv9详解](https://sxwqtaijh4.feishu.cn/docx/FRJ6dPhALoqyC7xhVP6cwgSVn4e?from=from_copylink) @陈国威

11. [YOLOv10详解](https://wvet00aj34c.feishu.cn/docx/VagAdssMbo7a3exoagOcXr8BnAh?from=from_copylink) @陈国威

12. [YOLO11详解](https://wvet00aj34c.feishu.cn/docx/ZUQ9d4LnmoYjv3xlBFTcprctnMg?from=from_copylink) @彭彩平

13. [YOLO系列模型的研究者/开发者趣闻和谣言](https://sxwqtaijh4.feishu.cn/docx/Yc40ddMGIo7nOyxSXVZc6KztnYd#UAHidbJDkoPaiaxDwBGcq7qInsg) @ 张小白
    
    - YOLO之父

    - YOLO的版本之争

    - YOLO系列模型时间线

### 第二部分 YOLO 全系列教程 ###

1. [YOLO系列算法的数据集制作与整理](https://wvet00aj34c.feishu.cn/docx/Tdv4d2ZpmoWX4vxPPhfcvEIQnLh?from=from_copylink) @程宏

2. YOLO系列实操教程 
   1. YOLO系列入门教程 @程宏 @余霆嵩
      1. [教程文档](https://wvet00aj34c.feishu.cn/docx/Ojcfd0ZF5olk4Yxwt9ZcjgSenUD?from=from_copylink)  
      2. [教程代码](./Hands_on_YoLo_with_ultralytics\0-dog-breed-detection)

   2. YOLO系列进阶教程 @程宏 @余霆嵩
      1. [教程文档](https://wvet00aj34c.feishu.cn/docx/U8STd5txXod1R5xhrrmcZh9fnTf?from=from_copylink)
      2. [教程代码](./Hands_on_YoLo_with_ultralytics\1-DOTA-obb)

   3. YOLO系列魔改教程 @白雪城 @谢彩承 @胡博毓
      1. [教程文档](hhttps://wvet00aj34c.feishu.cn/docx/RXJKdo5ZJoT5QPxiV3vcpGPwnzX?from=from_copylink)
      2. [教程代码](./Hacking_YoLo)

3. [YOLO系列模型从零开始实现教程](./Pytorch_YoLo_From_Scratch)  @ 

4. YOLO系列模型的硬件部署 @张小白

## 贡献者名单

| 姓名 | 职责 | 简介 |
| :----| :---- | :---- |
| [程宏](https://github.com/chg0901) | 项目负责人、发起者 | DataWhale意向成员 |
| [余霆嵩](https://github.com/TingsongYu)| 项目负责人 | DataWhale意向成员 |
| [蔡鋆捷](https://github.com/xinala-781) | 项目负责人 | DataWhale意向成员 |
| [白雪城](https://github.com/JackBaixue) | 项目负责人、发起者 | DataWhale成员 |
| [徐韵婉](https://github.com/) | 项目负责人、发起者 | DataWhale成员 |
| [刘伟鸿](https://github.com/Weihong-Liu) |V1详解，V1 Scratch | DataWhale成员 |
| [胡博毓](https://github.com/HuBoyu021124) | 魔改教程 | DataWhale成员 |
| [谢彩承](https://github.com/YoungBossX) | 魔改教程 |DataWhale意向成员 |
| [张小白](https://www.zhihu.com/people/zhanghui_china) | YOLO系列模型趣闻和谣言， 硬件部署（香橙派AIPro实现YOLOv5）|DataWhale意向成员  |
| [陈国威](https://github.com/gomevie) |  |DataWhale意向成员  |
| [彭彩平](https://github.com/caipingpeng) | | |
| [全政宇](https://github.com/EdQinHUST) | | DataWhale意向成员  |


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
