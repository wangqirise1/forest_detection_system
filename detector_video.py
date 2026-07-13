import cv2
from ultralytics import YOLO

model = YOLO(
    r"D:\forest_detection_system\runs\detect\runs\detect\forest_best_85plus\weights\best.pt"
)

print("模型加载成功")


def detect_video(video_path, save=False, for_ui=False):

    cap = cv2.VideoCapture(video_path)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame)

        annotated_frame = results[0].plot()

        # UI模式
        if for_ui:

            yield annotated_frame, results

        # 普通模式
        else:

            cv2.imshow(
                "Forest Detection",
                annotated_frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()

    if not for_ui:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


# ==================================================
# 本地测试
# ==================================================
if __name__ == "__main__":

    test_video = "test.mp4"

    for _frame, _ in detect_video(
            test_video,
            for_ui=False
    ):
        pass