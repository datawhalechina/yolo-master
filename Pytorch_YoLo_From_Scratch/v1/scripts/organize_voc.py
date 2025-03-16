#!/usr/bin/env python3
import os
import sys
import shutil
import argparse

def print_directory_structure():
    print("VOC_Detection/")
    print("|-- train/")
    print("|   |-- images/   # 存放训练集图片")
    print("|   `-- targets/  # 存放训练集标注")
    print("`-- test/")
    print("    |-- images/   # 存放测试集图片")
    print("    `-- targets/  # 存放测试集标注")
    print("\n数据集的划分如下:")
    print("  - 训练集: VOC 2007 训练集 + VOC 2007 验证集 + VOC 2012 训练集 + VOC 2012 验证集")
    print("  - 测试集: VOC 2007 测试集")
    print("\n旧的 'VOCdevkit/' 目录将被删除。")

def organize_voc_dataset(dir_path):
    # 创建目录结构
    dirs_to_create = [
        ('train', 'images'),
        ('train', 'targets'),
        ('test', 'images'),
        ('test', 'targets')
    ]
    
    for dataset_part, xy_part in dirs_to_create:
        full_path = os.path.realpath(os.path.join(dir_path, 'VOC_Detection', dataset_part, xy_part))
        os.makedirs(full_path, exist_ok=True)
        print(f"创建目录: {full_path}")

    # 处理训练数据
    for year in [2007, 2012]:
        voc_dir = os.path.realpath(os.path.join(dir_path, 'VOCdevkit', f'VOC{year}'))
        if not os.path.exists(voc_dir):
            raise FileNotFoundError(f"VOC{year}目录不存在: {voc_dir}")

        list_file = os.path.join(voc_dir, 'ImageSets', 'Main', 'trainval.txt')
        try:
            with open(list_file, 'r') as f:
                file_ids = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"[错误] 文件列表不存在: {list_file}")
            sys.exit(1)

        # 移动文件
        moved_count = 0
        for file_id in file_ids:
            src_jpg = os.path.join(voc_dir, 'JPEGImages', f"{file_id}.jpg")
            src_xml = os.path.join(voc_dir, 'Annotations', f"{file_id}.xml")
            
            dest_jpg = os.path.join(dir_path, 'VOC_Detection', 'train', 'images', f"{file_id}.jpg")
            dest_xml = os.path.join(dir_path, 'VOC_Detection', 'train', 'targets', f"{file_id}.xml")

            if os.path.exists(src_jpg):
                shutil.move(src_jpg, dest_jpg)
                moved_count += 1
            if os.path.exists(src_xml):
                shutil.move(src_xml, dest_xml)

        print(f"移动 VOC{year} 训练数据: {moved_count} 张图片及标注")

    # 处理测试数据 (仅2007)
    voc2007_dir = os.path.realpath(os.path.join(dir_path, 'VOCdevkit', 'VOC2007'))
    list_file = os.path.join(voc2007_dir, 'ImageSets', 'Main', 'test.txt')
    try:
        with open(list_file, 'r') as f:
            file_ids = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print(f"[错误] 测试列表不存在: {list_file}")
        sys.exit(1)

    # 移动测试文件
    moved_count = 0
    for file_id in file_ids:
        src_jpg = os.path.join(voc2007_dir, 'JPEGImages', f"{file_id}.jpg")
        src_xml = os.path.join(voc2007_dir, 'Annotations', f"{file_id}.xml")
        
        dest_jpg = os.path.join(dir_path, 'VOC_Detection', 'test', 'images', f"{file_id}.jpg")
        dest_xml = os.path.join(dir_path, 'VOC_Detection', 'test', 'targets', f"{file_id}.xml")

        if os.path.exists(src_jpg):
            shutil.move(src_jpg, dest_jpg)
            moved_count += 1
        if os.path.exists(src_xml):
            shutil.move(src_xml, dest_xml)
    
    print(f"移动 VOC2007 测试数据: {moved_count} 张图片及标注")

    # 删除旧目录
    vocdevkit_path = os.path.realpath(os.path.join(dir_path, 'VOCdevkit'))
    if os.path.exists(vocdevkit_path):
        print(f"删除旧目录: {vocdevkit_path}")
        shutil.rmtree(vocdevkit_path)

def main():
    parser = argparse.ArgumentParser(
        description='组织PASCAL VOC数据集目录结构',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('dir_path', 
                      help='包含VOCdevkit目录的路径')
    
    # 自定义帮助信息
    if len(sys.argv) == 1:
        parser.print_help()
        print_directory_structure()
        sys.exit(1)
    
    args = parser.parse_args()

    try:
        # 验证输入路径
        if not os.path.exists(args.dir_path):
            raise FileNotFoundError(f"指定路径不存在: {args.dir_path}")
            
        if not os.path.exists(os.path.join(args.dir_path, 'VOCdevkit')):
            raise FileNotFoundError("目标路径中未找到VOCdevkit目录")

        organize_voc_dataset(args.dir_path)
        print("\n数据集组织完成！")
        print("最终目录结构：")
        print_directory_structure()
        
    except Exception as e:
        print(f"\n[错误] 处理失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()