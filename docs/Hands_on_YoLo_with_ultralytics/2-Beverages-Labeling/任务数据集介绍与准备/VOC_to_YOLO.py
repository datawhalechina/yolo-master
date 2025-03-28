import os
import xml.etree.ElementTree as ET
import shutil  

# 定义路径
voc_images_dir = "./VOC/JPEGImages"    # 改为自己的VOC数据集图像文件所在路径
voc_annotations_dir = "./VOC/Annotations"   # 改为自己的VOC数据集标签文件所在路径
yolo_labels_dir = "./YOLO_datasets/labels"   # 改为自己的YOLO数据集标签文件所在路径
yolo_images_dir = "./YOLO_datasets/images"   # 改为自己的YOLO数据集图像文件所在路径
labels_txt_path = "./VOC/labels.txt"  # 改为自己的VOC数据集类别文件所在路径

# 数据集划分文件
train_txt_path = "./VOC/train_list.txt"  # 改为自己的VOC数据集中训练集路径
val_txt_path = "./VOC/val_list.txt"  # 改为自己的VOC数据集中验证集路径
test_txt_path = "./VOC/test_list.txt"  # 改为自己的VOC数据集中测试集路径

# 创建目标目录
os.makedirs(os.path.join(yolo_images_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(yolo_images_dir, "val"), exist_ok=True)
os.makedirs(os.path.join(yolo_labels_dir, "train"), exist_ok=True)
os.makedirs(os.path.join(yolo_labels_dir, "val"), exist_ok=True)

if os.path.exists(test_txt_path):
    os.makedirs(os.path.join(yolo_images_dir, "test"), exist_ok=True)
    os.makedirs(os.path.join(yolo_labels_dir, "test"), exist_ok=True)

# 记录成功和失败的文件数量
success_count = 0
error_count = 0

# 读取类别列表
with open(labels_txt_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 处理文件并移动图片
def process_and_move_datasets_files(file_list, split_name):
    global success_count, error_count
    for file in file_list:
        file = file.strip()
        if not file:
            continue
        
        parts = file.split()
        
        # 分离图片和XML路径，并取出文件名
        img_path_part, xml_path_part = parts
        img_file = os.path.basename(img_path_part)
        xml_file = os.path.basename(xml_path_part)
        
        # 构造XML文件完整路径
        xml_path = os.path.join(voc_annotations_dir, xml_file)
    
        # 解析XML文件
        try:
            tree = ET.parse(xml_path)  # 解析XML文件
            root = tree.getroot()  # 获取根元素
        except Exception as e:
            print(f"处理XML文件 {xml_path} 失败: {str(e)}")
            error_count += 1
            break

        # 获取图像尺寸
        size = root.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        # 生成对应的YOLO格式txt文件
        txt_filename = os.path.splitext(img_file)[0] + ".txt"
        txt_path = os.path.join(yolo_labels_dir, split_name, txt_filename)

        with open(txt_path, "w") as f_txt:
            for obj in root.findall("object"):
                cls_name = obj.find("name").text
                if cls_name not in classes:
                    continue  # 如果类别不在 classes 中，则跳过
                cls_id = classes.index(cls_name)

                # 提取边界框坐标
                bbox = obj.find("bndbox")
                xmin = int(bbox.find("xmin").text)
                ymin = int(bbox.find("ymin").text)
                xmax = int(bbox.find("xmax").text)
                ymax = int(bbox.find("ymax").text)

                # 转换为YOLO格式，对坐标进行归一化
                x_center = (xmin + xmax) / 2 / width
                y_center = (ymin + ymax) / 2 / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height

                f_txt.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        # 移动图片文件
        src_img_path = os.path.join(voc_images_dir, img_file)
        dst_img_path = os.path.join(yolo_images_dir, split_name, img_file)

        try:
            if os.path.exists(src_img_path):
                shutil.copy2(src_img_path, dst_img_path)
                print(f"复制文件{src_img_path}成功")
            success_count += 1
        except Exception as e:
            print(f"复制文件{src_img_path}错误|{str(e)}")
            error_count += 1

# 处理训练集
with open(train_txt_path, "r") as f:
    train_files = f.readlines()
process_and_move_datasets_files(train_files, "train")

# 处理验证集
with open(val_txt_path, "r") as f:
    val_files = f.readlines()
process_and_move_datasets_files(val_files, "val")

# 处理测试集（选用）
if os.path.exists(test_txt_path):
    with open(test_txt_path, "r") as f:
        test_files = f.readlines()
    process_and_move_datasets_files(test_files, "test")

print("所有文件处理完成")
print(f"成功处理文件: {success_count} 个")
print(f"失败处理文件: {error_count} 个")