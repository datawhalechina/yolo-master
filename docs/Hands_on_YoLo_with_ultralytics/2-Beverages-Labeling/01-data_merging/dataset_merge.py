import os
import shutil
import random

# 设置随机种子
random.seed(100)  # 为了方便代码复现，我这里将随机种子设置为100

drink_dataset_path = "./Drink_284_Detection_Labelme"  # 填写你的DRINK_284_DETECTION_LABELME目录的根路径
smart_goods_path = "./YOLO_datasets"  # 填写你的鱼眼镜头_智能销售数据集(YOLO格式)的根路径
new_dataset_path = "./merged_dataset"  # 填写你的新数据集的根路径,这里以merged_dataset为例

# 读取饮料数据集的类别列表
with open(os.path.join(drink_dataset_path, 'classes.txt'), 'r') as f:
    drink_classes = [line.strip() for line in f if line.strip()]

# 读取鱼眼镜头_智能销售数据集的类别列表
with open(os.path.join(smart_goods_path, 'labels.txt'), 'r') as f:
    smart_goods_classes = [line.strip() for line in f if line.strip()]

# 创建鱼眼镜头_智能销售数据集的类别索引到类别名称的映射
smart_goods_index_to_class = {str(idx): cls for idx, cls in enumerate(smart_goods_classes)}

# 尽可能找出重合的类别
common_classes = set(drink_classes).intersection(set(smart_goods_classes))  # 利用set的交集找出重合的类别
print("重合的类别为", common_classes)

# 创建饮料数据集的类别索引到类别名称的映射
drink_index_to_class = {str(idx): cls for idx, cls in enumerate(drink_classes)}  # 创建一个字典，将类别索引映射到类别名称

# 排序新的数据集的类别
new_datasets_classes = sorted(set(drink_classes + smart_goods_classes)) 
print("新数据集的所有类别:", new_datasets_classes)

# 创建类别名称到新索引的映射
class_to_new_index = {cls: idx for idx, cls in enumerate(new_datasets_classes)}  # 创建一个字典，将类别名称映射到新的索引
drink_index_to_new_index = {str(idx): class_to_new_index[cls] for idx, cls in enumerate(drink_classes)}  
smart_goods_index_to_new_index = {str(idx): class_to_new_index[cls] for idx, cls in enumerate(smart_goods_classes)}

# 检查标签文件是否包含重合类别
def contains_common_class(label_file, common_classes, index_to_class):
    if not os.path.exists(label_file):
        return False
    with open(label_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        new_lines_parts = line.strip().split() # 获取标签文件的每一行
        if new_lines_parts:
            index = new_lines_parts[0]
            cls = index_to_class.get(index)
            if cls in common_classes:
                return True
    return False

# 抽取样本
def extract_samples(smart_goods_subset, extract_count, common_classes, smart_goods_index_to_class):
    # 获取样本路径
    images_path = os.path.join(smart_goods_path, 'images', smart_goods_subset)
    labels_path = os.path.join(smart_goods_path, 'labels', smart_goods_subset)
    images = os.listdir(images_path)
    
    # 优先抽取包含重合类别的图像
    priority_images = []  # 包含重合类别的图像
    other_images = []  # 不包含重合类别的图像
    for img in images: 
        label_file = os.path.join(labels_path, img.replace('.jpg', '.txt').replace('.png', '.txt'))  # 获取标签文件路径
        if contains_common_class(label_file, common_classes, smart_goods_index_to_class):
            priority_images.append(img)  # 将包含重合类别的图像添加到优先图像列表中
        else:
            other_images.append(img)  # 将不包含重合类别的图像添加到其他图像列表中
    
    # 优先抽取包含重合类别的图像
    if len(priority_images) >= extract_count:
        extracted = random.sample(priority_images, extract_count)
    else:
        extracted = priority_images
        remaining_count = extract_count - len(priority_images)  # 计算剩余需要抽取的样本数
        if remaining_count > 0:  # 如果剩余需要抽取的样本数大于0
            # 只是保证尽可能重合，所以从其他图像中随机抽取剩余的样本
            extracted += random.sample(other_images, min(remaining_count, len(other_images)))  # 从其他图像中随机抽取剩余的样本
    # 返回抽取的样本
    return extracted

# 更新标签
def update_label_file(label_file, index_mapping):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        new_lines_parts = line.strip().split()
        if new_lines_parts:  # 确保行不为空
            old_index = new_lines_parts[0]
            new_index = index_mapping.get(old_index, old_index)  # 如果找不到映射，保持原样
            new_lines_parts[0] = str(new_index)
            new_lines.append(' '.join(new_lines_parts) + '\n')
    with open(label_file, 'w') as f:
        f.writelines(new_lines)

# 统计饮料数据集的train和val数量
drink_train_images = os.listdir(os.path.join(drink_dataset_path, 'images', 'train'))
drink_val_images = os.listdir(os.path.join(drink_dataset_path, 'images', 'val'))
drink_train_count = len(drink_train_images)
drink_val_count = len(drink_val_images)
print(f"饮料数据集的train数量:{drink_train_count},val数量:{drink_val_count}")

# 抽取两倍于饮料数据集的样本
smart_goods_train_extract_count = 2 * drink_train_count
smart_goods_val_extract_count = 2 * drink_val_count
print(f"需要从鱼眼镜头_智能销售数据集抽取的train数量:{smart_goods_train_extract_count}, val数量:{smart_goods_val_extract_count}")

# 抽取鱼眼镜头_智能销售数据集的train和val样本
smart_goods_train_extracted = extract_samples('train', smart_goods_train_extract_count, common_classes, smart_goods_index_to_class)
smart_goods_val_extracted = extract_samples('val', smart_goods_val_extract_count, common_classes, smart_goods_index_to_class)
print(f"抽取的鱼眼镜头_智能销售数据集的train样本数:{len(smart_goods_train_extracted)}, val样本数: {len(smart_goods_val_extracted)}")

# 为新数据集创建目录结构
for subset in ['train', 'val']:
    for folder in ['images', 'labels']:
        os.makedirs(os.path.join(new_dataset_path, folder, subset), exist_ok=True)

# 处理数据集，将图片和标签文件复制到新数据集，然后更新标签文件
def process_dataset(source_path, target_path, images_list, index_map, subset):
    for img in images_list:
        # 源图片和目标图片路径
        source_img = os.path.join(source_path, 'images', subset, img)
        target_img = os.path.join(target_path, 'images', subset, img)
        shutil.copy(source_img, target_img)
        
        # 处理标签文件
        base_name = os.path.splitext(img)[0] # 获取图片名称
        source_label = os.path.join(source_path, 'labels', subset, f"{base_name}.txt") # 获取标签文件路径
        target_label = os.path.join(target_path, 'labels', subset, f"{base_name}.txt") # 获取新数据集的标签文件路径
        
        # 如果标签文件存在，则复制标签文件到新数据集，更新标签文件
        if os.path.exists(source_label):
            shutil.copy(source_label, target_label)
            update_label_file(target_label, index_map)

# 处理饮料数据集
process_dataset(drink_dataset_path, new_dataset_path, drink_train_images, 
               drink_index_to_new_index, 'train')
process_dataset(drink_dataset_path, new_dataset_path, drink_val_images,
               drink_index_to_new_index, 'val')

# 处理智能商品数据集
process_dataset(smart_goods_path, new_dataset_path, smart_goods_train_extracted,
               smart_goods_index_to_new_index, 'train')
process_dataset(smart_goods_path, new_dataset_path, smart_goods_val_extracted,
               smart_goods_index_to_new_index, 'val')

# 写入新的类别文件
with open(os.path.join(new_dataset_path, 'classes.txt'), 'w') as f:
    for cls in new_datasets_classes:
        f.write(cls + '\n')

print(f"新数据集路径: {new_dataset_path}")
print(f"总类别数: {len(new_datasets_classes)}")
