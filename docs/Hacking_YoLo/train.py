from ultralytics import YOLO

# 模型配置文件地址
model_yaml_path = ''

# 数据集配置文件地址
dataset_yaml_path = ''

if __name__ == '__main__':
    # 创建模型
    model = YOLO(model_yaml_path)
    # 训练模型
    results = model.train(
        data=dataset_yaml_path,
        imgsz=640,
        epochs=100,
        batch=16,
        workers=0,
        amp=False,      # 如果出现训练损失为 Nan ，则关闭 amp
        project='runs/train',
        name='exp',
    )