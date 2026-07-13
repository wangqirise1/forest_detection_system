import os
import cv2
import random
import numpy as np

# ----------------------------
# 配置
# ----------------------------
DATASET_ROOT = r"E:\数据集最终版本\axe chainsaw.v1i.yolov8"
SUBSET = "valid"  # 先检查验证集

ROOT = os.path.join(DATASET_ROOT, SUBSET)
IMG_DIR = os.path.join(ROOT, "images")

# 自动检测使用哪个labels目录
LABELS_DIR = os.path.join(ROOT, "labels_merged")
if not os.path.exists(LABELS_DIR):
    LABELS_DIR = os.path.join(ROOT, "labels")
    print(f"📁 使用labels目录: {LABELS_DIR}")
else:
    print(f"📁 使用labels_merged目录: {LABELS_DIR}")

# 检查目录是否存在
if not os.path.exists(LABELS_DIR):
    print(f"❌ 错误：找不到labels目录")
    print(f"   检查路径: {LABELS_DIR}")
    exit(1)

if not os.path.exists(IMG_DIR):
    print(f"❌ 错误：找不到images目录")
    print(f"   检查路径: {IMG_DIR}")
    exit(1)

VIS_DIR = os.path.join(ROOT, "visualization_check")
os.makedirs(VIS_DIR, exist_ok=True)

# 类别颜色和名称
CLASSES = {
    0: {"name": "axe", "color": (0, 255, 0)},  # 绿色
    1: {"name": "chainsaw", "color": (255, 0, 0)},  # 蓝色
    2: {"name": "person", "color": (0, 0, 255)}  # 红色
}

# ----------------------------
# 随机抽取图片检查
# ----------------------------
txt_files = [f for f in os.listdir(LABELS_DIR) if f.endswith('.txt')]

if len(txt_files) == 0:
    print(f"❌ 在 {LABELS_DIR} 中没有找到txt文件")
    exit(1)

sample_size = min(10, len(txt_files))
sample_files = random.sample(txt_files, sample_size)

print(f"\n随机抽取 {sample_size}/{len(txt_files)} 张图片进行可视化检查")
print("=" * 50)

total_person = 0
problem_person = 0
total_boxes = 0

for txt_name in sample_files:

    img_name = os.path.splitext(txt_name)[0]

    # 尝试不同的图片扩展名
    img_path = None
    for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']:
        temp_path = os.path.join(IMG_DIR, img_name + ext)
        if os.path.exists(temp_path):
            img_path = temp_path
            break

    if img_path is None:
        print(f"⚠ 找不到图片: {img_name}，跳过")
        continue

    # 读取图片和标注
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠ 无法读取图片: {img_path}")
        continue

    h, w = img.shape[:2]

    merged_path = os.path.join(LABELS_DIR, txt_name)

    with open(merged_path, 'r') as f:
        lines = f.readlines()

    # 统计各类别数量
    class_count = {0: 0, 1: 0, 2: 0}
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            class_id = int(parts[0])
            if class_id in class_count:
                class_count[class_id] += 1

    print(f"\n📷 {img_name}: "
          f"斧头×{class_count[0]}, 电锯×{class_count[1]}, 人×{class_count[2]}")

    person_boxes = []

    # 绘制所有框
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:])

        # 转换为像素坐标
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        # 边界检查
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        class_info = CLASSES.get(class_id, {"name": "unknown", "color": (128, 128, 128)})

        # 绘制边界框
        cv2.rectangle(img, (x1, y1), (x2, y2), class_info["color"], 2)

        # 添加标签
        label = f"{class_info['name']}"
        # 计算文本大小以添加背景
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img, (x1, y1 - text_h - 5), (x1 + text_w, y1), class_info["color"], -1)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        total_boxes += 1

        # 统计person框
        if class_id == 2:
            person_boxes.append((x1, y1, x2, y2))
            total_person += 1

            # 检查问题框
            box_w = x2 - x1
            box_h = y2 - y1
            box_area = box_w * box_h
            img_area = w * h

            # 检查边界框质量
            issues = []

            # 太大或太小的框可能有问题
            area_ratio = box_area / img_area
            if area_ratio > 0.8:
                issues.append(f"框过大({area_ratio * 100:.1f}%面积)")
            elif area_ratio < 0.001:
                issues.append(f"框过小({box_area}像素)")

            # 检查宽高比（人体通常0.3-2.0）
            aspect_ratio = box_w / max(box_h, 1)
            if aspect_ratio > 3.0:
                issues.append(f"过宽(宽高比{aspect_ratio:.1f})")
            elif aspect_ratio < 0.2:
                issues.append(f"过窄(宽高比{aspect_ratio:.1f})")

            # 检查是否在边缘（可能是截断的人）
            edge_margin = 0.05
            if x1 / w < edge_margin or y1 / h < edge_margin or x2 / w > 1 - edge_margin or y2 / h > 1 - edge_margin:
                issues.append("边缘截断")

            if issues:
                problem_person += 1
                print(f"  ⚠ Person框问题: {', '.join(issues)}")

    # 保存可视化图片
    vis_path = os.path.join(VIS_DIR, f"check_{img_name}.jpg")
    cv2.imwrite(vis_path, img)
    print(f"  💾 保存: check_{img_name}.jpg")

# ----------------------------
# 统计报告
# ----------------------------
print(f"\n{'=' * 50}")
print("📊 精度检查报告")
print(f"{'=' * 50}")
print(f"检查图片数: {len(sample_files)}")
print(f"总标注框数: {total_boxes}")
print(f"Person框总数: {total_person}")
print(f"可能问题框: {problem_person}")
if total_person > 0:
    print(f"问题率: {problem_person / total_person * 100:.1f}%")
    if problem_person / total_person < 0.1:
        print("✅ 精度良好！可以直接使用")
    elif problem_person / total_person < 0.2:
        print("⚠️  精度尚可，建议抽查确认")
    else:
        print("❌ 问题较多，建议优化检测或人工复核")

print(f"\n💡 可视化结果保存在: {VIS_DIR}")
print("🔍 请打开查看，手动确认标注质量")
print("\n图例:")
print("  🟢 绿色框 = axe (斧头)")
print("  🔵 蓝色框 = chainsaw (电锯)")
print("  🔴 红色框 = person (人)")