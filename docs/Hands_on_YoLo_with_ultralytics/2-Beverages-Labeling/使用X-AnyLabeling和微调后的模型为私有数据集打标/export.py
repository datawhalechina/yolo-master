from ultralytics import YOLO

# 加载模型
model = YOLO('./output/my_model/weights/best.pt')  # 改为自己生成的best.pt的路径

# 导出为 ONNX 格式
model.export(format='onnx')