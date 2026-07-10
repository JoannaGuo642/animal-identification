from ultralytics import YOLO
import cv2
import os

# 加载动物预训练模型，替换为你的动物数据集yaml
MODEL_PATH = "best.pt"
DATA_YAML = "D:/Pycharm_text/animalDetect/data.yaml"
model = YOLO(MODEL_PATH)

# 1. 模型训练函数
def train_your_model(yaml_path, epochs=3):
    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=640,
        device=0
    )

# 2. 图像分类（原有保留）
def yolo_classify(img_path):
    res = model(img_path, task="classify")
    cls_name = res[0].names[int(res[0].probs.top1)]
    conf = float(res[0].probs.top1conf)
    return f"{cls_name} 置信度：{conf:.2f}"

# 3. 目标检测+绘制检测框（新增核心）
def yolo_detect_draw(img_path, conf_thres=0.5):
    img = cv2.imread(img_path)
    results = model(img_path, conf=conf_thres)
    animal_count = {}
    for box in results[0].boxes:
        x1,y1,x2,y2 = map(int,box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id]
        # 绘制框和文字
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(img,f"{cls_name} {conf:.2f}",(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        # 统计计数
        if cls_name in animal_count:
            animal_count[cls_name] += 1
        else:
            animal_count[cls_name] = 1
    # 保存临时绘图图片
    temp_save = "temp_detect.jpg"
    cv2.imwrite(temp_save, img)
    return temp_save, animal_count

# 4. 摄像头实时检测帧
def camera_detect_frame(frame, conf_thres=0.5):
    results = model(frame, conf=conf_thres)
    for box in results[0].boxes:
        x1,y1,x2,y2 = map(int,box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id]
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.putText(frame,f"{cls_name}{conf:.2f}",(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
    return frame

# 5. 批量导出结果txt
def export_result_txt(save_dir, all_result_list):
    save_path = os.path.join(save_dir, "动物识别报表.txt")
    with open(save_path,"w",encoding="utf-8") as f:
        f.write("====动物识别批量检测报表====\n")
        for line in all_result_list:
            f.write(line + "\n")
    return save_path