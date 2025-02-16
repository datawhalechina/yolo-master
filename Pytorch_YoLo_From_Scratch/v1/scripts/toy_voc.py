import os
import shutil
import pandas as pd
from tqdm import tqdm

# 配置参数
SOURCE_VOC_DIR = "data/voc/VOC_Detection"  # VOC数据集的原始路径
TARGET_VOC_DIR = "data/voc/small_voc"  # 目标小型VOC数据集的路径
SELECTED_CLASSES = {"car", "bus", "person"}  # 需要筛选的目标类别
SPLITS = ["train", "test"]  # 数据集划分

# 确保目标目录结构存在
def ensure_dirs():
    for split in SPLITS:
        os.makedirs(os.path.join(TARGET_VOC_DIR, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(TARGET_VOC_DIR, split, "targets"), exist_ok=True)

# 复制筛选后的图片和CSV文件
def filter_and_copy():
    for split in SPLITS:
        images_dir = os.path.join(SOURCE_VOC_DIR, split, "images")
        targets_dir = os.path.join(SOURCE_VOC_DIR, split, "targets")
        
        target_images_dir = os.path.join(TARGET_VOC_DIR, split, "images")
        target_targets_dir = os.path.join(TARGET_VOC_DIR, split, "targets")
        
        # 遍历所有标注文件
        for csv_file in tqdm(os.listdir(targets_dir), desc=f"Processing {split}"):
            csv_path = os.path.join(targets_dir, csv_file)
            df = pd.read_csv(csv_path)
            
            # 筛选目标类别
            filtered_df = df[df['object'].isin(SELECTED_CLASSES)]
            
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
    ensure_dirs()
    filter_and_copy()
    print("Small VOC dataset created successfully!")