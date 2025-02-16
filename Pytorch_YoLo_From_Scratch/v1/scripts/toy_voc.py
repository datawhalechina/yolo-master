import os
import shutil
import pandas as pd
import argparse
from tqdm import tqdm

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Filter and copy VOC dataset based on selected classes.")
    parser.add_argument("--source_voc", type=str, required=True, help="Path to the original VOC dataset.")
    parser.add_argument("--target_voc", type=str, required=True, help="Path to the target small VOC dataset.")
    parser.add_argument("--classes", type=str, required=True, help="Comma-separated list of selected object classes.")
    return parser.parse_args()

# 确保目标目录结构存在
def ensure_dirs(target_voc_dir, splits):
    for split in splits:
        os.makedirs(os.path.join(target_voc_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(target_voc_dir, split, "targets"), exist_ok=True)

# 复制筛选后的图片和CSV文件
def filter_and_copy(source_voc_dir, target_voc_dir, selected_classes, splits):
    for split in splits:
        images_dir = os.path.join(source_voc_dir, split, "images")
        targets_dir = os.path.join(source_voc_dir, split, "targets")
        
        target_images_dir = os.path.join(target_voc_dir, split, "images")
        target_targets_dir = os.path.join(target_voc_dir, split, "targets")
        
        # 遍历所有标注文件
        for csv_file in tqdm(os.listdir(targets_dir), desc=f"Processing {split}"):
            csv_path = os.path.join(targets_dir, csv_file)
            df = pd.read_csv(csv_path)
            
            # 筛选目标类别
            filtered_df = df[df['object'].isin(selected_classes)]
            
            if not filtered_df.empty:
                # 复制图片
                image_file = csv_file.replace(".csv", ".jpg")
                src_img_path = os.path.join(images_dir, image_file)
                dst_img_path = os.path.join(target_images_dir, image_file)
                if os.path.exists(src_img_path):
                    shutil.copy(src_img_path, dst_img_path)
                
                # 复制筛选后的CSV
                dst_csv_path = os.path.join(target_targets_dir, csv_file)
                filtered_df.to_csv(dst_csv_path, index=False)

if __name__ == "__main__":
    args = parse_args()
    source_voc_dir = args.source_voc
    target_voc_dir = args.target_voc
    selected_classes = set(args.classes.split(","))
    splits = ["train", "test"]
    
    ensure_dirs(target_voc_dir, splits)
    filter_and_copy(source_voc_dir, target_voc_dir, selected_classes, splits)
    print("Small VOC dataset created successfully!")
