#!/bin/bash

# 检查是否提供了参数，如果没有提供，则打印脚本的使用说明并终止执行
if (( $# != 1 )); then
  echo "Usage: ./download_voc.sh <dir_path>"
  echo -e "Download the PASCAL VOC 2007 (train + val + test) and 2012 (train + val) datasets.\n"
  echo "Arguments:"
  echo -e "\t<dir_path>: The path of the directory that will be created to store the PASCAL VOC data"
  exit 1  # 退出脚本，返回错误码 1
fi

# 创建存储 PASCAL VOC 数据集的目录（如果目录不存在，则创建）
mkdir -p $1

# 下载 PASCAL VOC 2007 和 2012 的数据集（如果文件已存在，则不会重新下载）
# VOC数据集国内镜像 https://pjreddie.com/projects/pascal-voc-dataset-mirror/
wget -nc -P $1 http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar \
               http://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar \
               http://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
# 说明：
# - `wget` 是一个用于下载文件的工具。
# - `-nc` (no-clobber) 选项表示如果文件已存在，则不会重新下载。
# - `-P $1` 选项表示将下载的文件存放到 `$1` 目录（即用户提供的目标目录）。

# 解压已下载的 .tar 文件（不打印解压出的文件名）
tar -xvf $1/VOCtrainval_11-May-2012.tar -C $1 > /dev/null
tar -xvf $1/VOCtrainval_06-Nov-2007.tar -C $1 > /dev/null
tar -xvf $1/VOCtest_06-Nov-2007.tar -C $1 > /dev/null
# 说明：
# - `tar -xvf` 命令用于解压 .tar 文件。
# - `-x` 代表解压，`-v` 代表显示详细信息，`-f` 指定文件名。
# - `-C $1` 选项表示将文件解压到 `$1` 目录。
# - `> /dev/null` 用于屏蔽解压过程中打印的文件名。

# 执行成功，返回 0 作为成功退出码
exit 0
