from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('yolov8s.pt')

    model.train(
        data='data.yaml',

        # ===== 训练周期（不拉满200）=====
        epochs=120,              # ✔ 足够收敛 + 防过拟合
        patience=30,             # ✔ 提前停止避免震荡

        # ===== 输入与显存优化 =====
        imgsz=960,               # ✔ 保持细节（森林小目标重要）
        batch=6,                 # ✔ 6GB显存极限平衡
        device=0,
        workers=2,
        cache=True,

        # ===== 优化器（保留AdamW）=====
        optimizer='AdamW',

        # ===== 学习率（重点优化）=====
        lr0=0.0004,              # ✔ 比你原来更稳（关键提升点）
        lrf=0.01,
        warmup_epochs=3,
        weight_decay=0.0005,

        # ===== 数据增强（关键优化）=====
        mosaic=0.7,              # ↓ 降低震荡
        close_mosaic=10,         # ✔ 最后10轮关闭mosaic（非常重要）

        mixup=0.05,              # ↓ 减少噪声
        copy_paste=0.1,          # ↓ 防止过拟合假样本

        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,

        fliplr=0.5,

        # ===== 稳定训练 =====
        amp=True,
        pretrained=True,

        # ===== 输出 =====
        project='runs/detect',
        name='forest_best_85plus',
        exist_ok=True
    )