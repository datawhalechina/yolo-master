from ultralytics import YOLO

# 模型配置文件地址
model_yaml_path = ""

# 数据集配置文件地址
data_yaml_path = ""

if __name__ == '__main__':
    # 创建模型
    model = YOLO(model_yaml_path)
    # 验证模型
    results = model.val(
        data=data_yaml_path,
        split='val',
        imgsz=640,
        batch=4,
        project='runs/val',
        name='exp',
    )