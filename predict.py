from matplotlib import image
from ultralytics import YOLO

# ⭐ 用你的新模型
model = YOLO(r"D:\forest_detection_system\runs\detect\runs\train\forest_final_v2\weights\best.pt")
results = model.predict(
    source=image,
    conf=0.12,
    iou=0.7,
    imgsz=960,
    save=False,
    verbose=False
)