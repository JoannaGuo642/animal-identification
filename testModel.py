from ultralytics import  YOLO

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


model = YOLO("yolov8n.pt") #加载预训练模型


model.train(data="D:/pycharm_text/fruitSorting0624/data.yaml",epochs=25)