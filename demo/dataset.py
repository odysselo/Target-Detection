# -*- coding: utf-8 -*-
"""
demo/dataset.py
=========================================================================
合成数据集（几何图形检测）

实验要求"跑通目标检测任务"，但完整 VOC/COCO 在 CPU 上无法在合理时间内训练。
因此这里构造一个可控的合成数据集：在随机背景上画圆、方、三角三种目标，
用于验证 Faster R-CNN / YOLOv1 的
    数据流 -> 损失 -> 反向传播 -> 解码 -> NMS -> 可视化
整条链路是否真正跑通（损失能下降、框能收敛到目标上）。

样本由 seed 唯一确定，训练集/验证集使用不同 seed 段，保证互不重叠。
=========================================================================
"""
import os
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_NAMES = ["__background__", "circle", "square", "triangle"]   # 0 为背景（Faster R-CNN 需要）
NUM_CLASSES = len(CLASS_NAMES)

# 三种图形的绘制颜色（BGR）
COLORS = {"circle": (60, 60, 220), "square": (60, 180, 75), "triangle": (220, 140, 40)}


def _rand_color(rng: random.Random):
    """每种图形一个高对比度颜色（BGR），避免与灰色背景混淆"""
    palette = [(60, 60, 220), (60, 200, 90), (230, 150, 40), (200, 60, 200), (40, 200, 220)]
    return palette[rng.randrange(len(palette))]


def make_sample(seed: int, img_size: int = 320, max_objects: int = 1,
                min_size: int = 80, max_size: int = 110):
    """
    生成一张图 + 若干目标（默认 1 个，尺寸 80~110 px）

    设计说明：
        数据集刻意做得"简单且可控"——纯色背景 + 单个居中区域内的规则图形。
        目的是让 CPU 上的训练能在几分钟内真正收敛（验证数据流、损失、解码、NMS
        整条链路是对的），而不是追求 mAP。背景加了轻微噪声，避免网络只靠颜色作弊。

    Returns:
        img   : (H, W, 3) uint8 BGR
        boxes : (M, 4) float32 xyxy（像素）
        labels: (M,) int64，1=circle, 2=square, 3=triangle
    """
    rng = random.Random(seed)
    img = np.full((img_size, img_size, 3), 60, dtype=np.uint8)
    img = (img.astype(np.int32) +
           np.random.RandomState(seed).randint(-6, 6, img.shape)).clip(0, 255).astype(np.uint8)

    boxes, labels = [], []
    n_obj = rng.randint(1, max_objects)
    for _ in range(n_obj):
        cls = rng.choice([1, 2, 3])
        w = rng.randint(min_size, max_size)
        h = w
        x1 = rng.randint(10, max(11, img_size - w - 10))
        y1 = rng.randint(10, max(11, img_size - h - 10))
        color = _rand_color(rng)
        if cls == 1:
            cv2.circle(img, (int(x1 + w / 2), int(y1 + h / 2)), int(w / 2), color, -1)
        elif cls == 2:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, -1)
        else:
            pts = np.array([[x1 + w / 2, y1], [x1, y1 + h], [x1 + w, y1 + h]], dtype=np.int32)
            cv2.fillPoly(img, [pts], color)
        boxes.append(np.array([x1, y1, x1 + w, y1 + h], dtype=np.float32))
        labels.append(cls)

    return img, np.stack(boxes), np.array(labels, dtype=np.int64)


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / (union + 1e-6)


class ShapesDataset(Dataset):
    """返回 (image tensor[3,H,W] 0~1, boxes[M,4], labels[M])"""

    def __init__(self, length: int = 64, img_size: int = 320, seed_offset: int = 0):
        self.length = length
        self.img_size = img_size
        self.seed_offset = seed_offset

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img, boxes, labels = make_sample(self.seed_offset + idx, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, torch.from_numpy(boxes), torch.from_numpy(labels)


def fasterrcnn_collate(batch):
    """Faster R-CNN 需要"图像列表 + 标注字典列表"，不能简单 stack"""
    images = [b[0] for b in batch]
    targets = [{"boxes": b[1], "labels": b[2]} for b in batch]
    return images, targets


def to_yolov1_target(boxes: np.ndarray, labels: np.ndarray, S: int = 7, B: int = 2,
                     C: int = 3, img_size: int = 224) -> torch.Tensor:
    """
    把 (M,4) xyxy 像素框编码成 YOLOv1 的 (S*S*(B*5+C),) 目标向量

    编码规则（与 YOLOv1 论文一致）：
        1. 目标中心落在哪个网格，就由哪个网格负责
        2. 网格内的 x, y = 中心点相对该网格左上角的偏移（0~1）
        3. w, h = 框宽高 / 整图宽高（0~1）
        4. 该网格的两个框都填成 GT（训练时由 loss 选 IoU 更大的那个负责）
        5. conf = 1，对应类别位置 = 1
    """
    target = np.zeros((S * S * (B * 5 + C),), dtype=np.float32)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box / img_size
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        gi, gj = min(int(cy * S), S - 1), min(int(cx * S), S - 1)     # 行、列
        tx, ty = cx * S - gj, cy * S - gi

        cell = np.zeros((B * 5 + C,), dtype=np.float32)
        for b in range(B):
            cell[b * 5:b * 5 + 5] = [tx, ty, w, h, 1.0]
        cell[B * 5 + int(label) - 1] = 1.0                            # label 从 1 开始
        target.reshape(S, S, -1)[gi, gj] = cell
    return torch.from_numpy(target)


def yolo_collate(batch):
    """YOLO 的标注需要现场编码，因此不做默认 collate，原样返回列表"""
    return batch


def to_yolov1_target_batch(batch, S=7, B=2, C=3, img_size=224):
    """把一个 batch 的 (img, boxes, labels) 转成 (images, targets)"""
    images = torch.stack([b[0] for b in batch])
    targets = torch.stack([
        to_yolov1_target(b[1].numpy(), b[2].numpy(), S, B, C, img_size) for b in batch])
    return images, targets


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """
    OpenCV 的 imwrite 在 Windows 上不支持中文路径（会静默失败），
    这里用 imencode + tofile 绕过。
    """
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(path)
    return True


def draw_boxes(img_bgr: np.ndarray, boxes, labels, scores=None, class_names=None,
               thickness: int = 2) -> np.ndarray:
    """在图上画框（BGR），返回新图"""
    out = img_bgr.copy()
    class_names = class_names or CLASS_NAMES
    boxes = np.asarray(boxes, dtype=np.float32)
    labels = np.asarray(labels)
    scores = np.asarray(scores) if scores is not None else None
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = [int(round(float(v))) for v in box[:4]]
        name = class_names[int(label)] if int(label) < len(class_names) else str(int(label))
        color = COLORS.get(name, (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        text = f"{name} {float(scores[i]):.2f}" if scores is not None else name
        cv2.putText(out, text, (x1 + 2, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 1, cv2.LINE_AA)
    return out


if __name__ == "__main__":
    img, boxes, labels = make_sample(3, 320)
    print("image:", img.shape, "boxes:", boxes.shape, "labels:", labels.tolist())
    t = to_yolov1_target(boxes, labels, 7, 2, 3, 320)
    print("yolov1 target:", tuple(t.shape), "非零元素:", int((t != 0).sum()))
