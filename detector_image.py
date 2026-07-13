import cv2
import os
from datetime import datetime
from ultralytics import YOLO
import platform

# =========================
# 模型路径（不改）
# =========================
MODEL_PATH = r"D:\forest_detection_system\runs\detect\runs\detect\forest_best_85plus\weights\best.pt"
model = YOLO(MODEL_PATH)

print(f"已加载模型: {MODEL_PATH}")

# =========================
# 保存路径
# =========================
SAVE_DIR = r"D:\forest_detection_system\results_video"
os.makedirs(SAVE_DIR, exist_ok=True)


def detect_image(file_path, save=True, for_ui=False):

    img = cv2.imread(file_path)

    if img is None:
        return None, []

    results = model.predict(
        source=img,
        conf=0.1,
        iou=0.20,
        imgsz=960,
        verbose=False
    )[0]

    detections = []

    person = axe = chainsaw = 0

    for box in results.boxes:

        cls = int(box.cls[0])
        name = model.names[cls]
        score = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        detections.append({
            "label": name,
            "score": score,
            "box": [x1, y1, x2, y2]
        })

        if name == "person":
            person += 1
        elif name == "axe":
            axe += 1
        elif name == "chainsaw":
            chainsaw += 1

    # ================= 风险判断（不变） =================
    if person and chainsaw:
        text, color = "HIGH RISK", (0, 0, 255)
    elif person and axe:
        text, color = "MEDIUM RISK", (0, 165, 255)
    else:
        text, color = "SAFE", (0, 255, 0)

    result_img = results.plot()

    cv2.putText(result_img, text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

    # ================= 保存 =================
    save_path = None

    if save:

        save_path = os.path.join(
            SAVE_DIR,
            f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        )

        cv2.imwrite(save_path, result_img)
        print("saved:", save_path)

        if platform.system() == "Windows":
            os.startfile(save_path)

    return result_img, detections