import os
from ultralytics import YOLO
from PIL import Image
import cv2
import time
import itertools


img_path = "latest.jpg"


# 類別名稱
class_names = [
    'down_left', 'down_middle', 'down_right',
    'middle_left', 'middle_middle', 'middle_right',
    'tel_left', 'tel_right', 'ten',
    'up_left', 'up_middle', 'up_right'
]

# 拍照函式：直到成功為止
def capture_image(output_path=img_path):
    cap = cv2.VideoCapture(0)  # 啟用攝影機
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite(output_path, frame)
            print(f"成功儲存圖片：{output_path}")
            break
        else:
            print("拍照失敗，1 秒後重試...")
            time.sleep(1)
    cap.release()

# YOLO 偵測並回傳實際座標
def run_yolo_and_get_coords(img_path):
    model = YOLO("yolov8.two/runs/detect/train/weights/best.pt")  # 匯入訓練好的模型
    results = model.predict(
        source=img_path,
        save_txt=True,
        save=True,
        imgsz=1280  # 用高解析度偵測
    )

    # 開圖取得尺寸
    img = Image.open(img_path)
    image_width, image_height = img.size

    # 自動找最新一筆預測的 labels 路徑
    predict_dir = "runs/detect/predict/labels"  # 預設初始資料夾
    for i in itertools.count(start=2):
        try_path = f"runs/detect/predict{i}/labels"
        if os.path.exists(try_path):
            predict_dir = try_path
        else:
            break  # 找不到下一個 predict 資料夾就停止

    txt_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
    txt_path = os.path.join(predict_dir, txt_name)

    # 讀取並轉換 YOLO 預測結果為實際像素座標
    with open(txt_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            x_pixel = x_center * image_width
            y_pixel = y_center * image_height
            box_width = width * image_width
            box_height = height * image_height

            x_min = x_pixel - box_width / 2
            y_min = y_pixel - box_height / 2
            x_max = x_pixel + box_width / 2
            y_max = y_pixel + box_height / 2

            class_name = class_names[class_id]
            print(f"{class_name}: 中心=({x_pixel:.1f}, {y_pixel:.1f})，框=({x_min:.1f}, {y_min:.1f}) ~ ({x_max:.1f}, {y_max:.1f})")


capture_image(img_path)
run_yolo_and_get_coords(img_path)
