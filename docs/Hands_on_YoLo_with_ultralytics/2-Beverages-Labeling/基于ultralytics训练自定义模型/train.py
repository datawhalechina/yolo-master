from ultralytics import YOLO


model = YOLO('./yolov8m.pt')  #　改为自己的预训练权重yolov8m.pt的路径


results = model.train(data='./data.yaml',  # 改为自己搭数据集的配置文件的路径
                      epochs=100,  # 训练轮数
                      imgsz=640,  # 训练图像的尺寸
                      batch=32, # 批次大小
                      device='0',  # GPU的ID号
                      workers=16,  # 数据加载器的工作线程数
                      project='./output',  # 改为自己想保存模型的路径
                      name='my_model',  # 改为自己想要设置的模型的名称
                      )