import os  
import glob  

def main():
    drink_dataset_path = "./Drink_284_Detection_Labelme"  # 填写你的DRINK_284_DETECTION_LABELME目录的根路径
    drink_classes_path = "./Drink_284_Detection_Labelme/classes.txt"  # 填写你的 DRINK_284_DETECTION_LABELME 的 classes.txt 路径
    smart_goods_labels_path =  "./YOLO_datasets/labels.txt"  # 填写你的鱼眼镜头_智能销售数据集(YOLO格式)的 labels.txt 路径
    
    # 读取drink_dataset的类别列表
    with open(drink_classes_path, 'r', encoding='utf-8') as f:
        drink_classes = [line.strip() for line in f.readlines() if line.strip()] # 读取饮料数据集的类别列表
    
    # 读取鱼眼镜头_智能销售数据集的饮料数据集的类别列表
    with open(smart_goods_labels_path, 'r', encoding='utf-8') as f:
        smart_goods_classes = [line.strip() for line in f.readlines() if line.strip()]
    
    
    # 创建索引映射字典
    idx_map = {}
    for old_idx, class_name in enumerate(drink_classes):
        if class_name in smart_goods_classes:  # 如果类别在智能零售结算系统的饮料数据集的labels.txt中存在
            new_idx = smart_goods_classes.index(class_name)
            idx_map[old_idx] = new_idx  # 将旧索引映射到新索引
        else:
            print(f"类别'{class_name}'目标labels中未找到")
    
    # 打印索引映射
    print("索引开始更新\n")
    for old_idx, new_idx in idx_map.items():
        class_name = drink_classes[old_idx]
        print(f"类别'{class_name}'由源索引{old_idx}->现索引{new_idx}")
    
    
    # 获取train和val文件夹中的标签文件
    train_labels_path = os.path.join(drink_dataset_path, "labels", "train")  # 获取饮料数据集的训练标签文件路径
    val_labels_path = os.path.join(drink_dataset_path, "labels", "val")  # 获取饮料数据集的验证标签文件路径
    train_label_files = glob.glob(os.path.join(train_labels_path, "*.txt"))  # 获取训练标签文件列表
    val_label_files = glob.glob(os.path.join(val_labels_path, "*.txt"))  # 获取验证标签文件列表
    label_files = train_label_files + val_label_files 
    
    
    # 处理每个标签文件
    for label_file in label_files:
        with open(label_file, 'r', encoding='utf-8') as f:  # 读取标签文件
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            new_lines_parts = line.strip().split() # 把标签文件中的每一行拆分成一个列表
            # 更新索引
            old_idx = int(new_lines_parts[0])
            if old_idx in idx_map:
                new_idx = idx_map[old_idx]
                new_line = " ".join([str(new_idx)] + new_lines_parts[1:]) + "\n"  # 将更新后的索引和其余部分重新组合成新行
                new_lines.append(new_line)
            else:
                print(f"文件{label_file}中的索引{old_idx}未在映射中找到")
            
        
        # 将更新后的内容写回文件
        with open(label_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)  
    
    print("所有ID已映射完成")

if __name__ == "__main__":
    main()