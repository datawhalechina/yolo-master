from ultralytics import YOLO

# 加载模型
model = YOLO('./bext.pt')  # 改为自己生成的bext.pt的路径

# 导出为 ONNX 格式
model.export(format='onnx')