from ultralytics import YOLO

# 模型配置文件地址
model_yaml_path = ""

# 数据集配置文件地址
data_yaml_path = ""

if __name__ == '__main__':
    # 创建模型
    model = YOLO(model_yaml_path)
    # 预测模型
    results = model.predict(
        source="",
        project="",
        name="",
        save=True,
        imgsz=640,
        conf=0.5,
        iou=0.7,
    )