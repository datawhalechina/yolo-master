#!/bin/bash

# 检查是否提供了参数，如果没有提供，则打印脚本的使用说明并终止执行
if (( $# != 1 )); then
  echo "Usage: ./organize_voc.sh <dir_path>"
  echo  "在指定目录下创建一个新的 'VOC_Detection/' 目录，并按照以下方式组织训练集和测试集的图像及标注文件:"

  echo
  echo  "VOC_Detection/"
  echo  "|-- train/"
  echo  "|   |-- images/   # 存放训练集图片"
  echo  "|   \`-- targets/  # 存放训练集标注"
  echo  "\`-- test/"
  echo  "|   |-- images/   # 存放测试集图片"
  echo  "|   |-- targets/  # 存放测试集标注"

  echo
  echo "数据集的划分如下:"
  echo "  - 训练集: VOC 2007 训练集 + VOC 2007 验证集 + VOC 2012 训练集 + VOC 2012 验证集"
  echo "  - 测试集: VOC 2007 测试集"

  echo
  echo "旧的 'VOCdevkit/' 目录将被删除。"

  echo "参数说明:"
  echo -e "\t<dir_path>: 指定包含 PASCAL VOC 数据（特别是 'VOCdevkit/' 目录）的路径"
  exit 1  # 退出脚本，返回错误码 1
fi


# 创建新的目录结构，以便分类存放训练集和测试集的图片及标注文件
for dataset_part_dir in 'train' 'test'; do
    for xy_part_dir in 'images' 'targets'; do
        mkdir -p $(realpath -m $1/VOC_Detection/$dataset_part_dir/$xy_part_dir)
    done
done


# 移动 VOC 2007 和 VOC 2012 的训练集和验证集到 'VOC_Detection/train/' 目录
for year in 2007 2012; do
    old_dir=$(realpath -m $1/VOCdevkit/VOC$year)   # 旧的数据集目录
    new_dir_train=$(realpath -m $1/VOC_Detection/train)  # 训练集目标目录

    # 读取 trainval.txt（包含训练集和验证集的文件名），并移动图片和标注
    for img in $(cat $old_dir/ImageSets/Main/trainval.txt); do
        mv $old_dir/JPEGImages/$img.jpg $new_dir_train/images/    # 移动图片
        mv $old_dir/Annotations/$img.xml $new_dir_train/targets/  # 移动标注文件
    done
done


# 移动 VOC 2007 的测试集到 'VOC_Detection/test/' 目录
old_dir=$(realpath -m $1/VOCdevkit/VOC2007)  # 旧的 VOC 2007 目录
new_dir_test=$(realpath -m $1/VOC_Detection/test)  # 测试集目标目录

# 读取 test.txt（包含测试集文件名），并移动图片和标注
for img in $(cat $old_dir/ImageSets/Main/test.txt); do
    mv $old_dir/JPEGImages/$img.jpg $new_dir_test/images/    # 移动测试集图片
    mv $old_dir/Annotations/$img.xml $new_dir_test/targets/  # 移动测试集标注文件
done

# 删除旧的 VOCdevkit/ 目录
rm -rf $(realpath $1/VOCdevkit)

# 执行成功，返回 0 作为成功退出码
exit 0
