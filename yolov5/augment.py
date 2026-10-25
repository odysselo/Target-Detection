# -*- coding: utf-8 -*-
"""
yolov5/augment.py
=========================================================================
YOLOv5 的数据增强 —— 对应实验任务 B.2 第 2 条之（8）Mosaic / letterbox

1) letterbox：保持长宽比缩放 + 灰边填充到正方形（推理时必用）
   为什么不用直接 resize？直接拉伸会让目标形变，破坏先验的 anchor 形状。

2) Mosaic：把 4 张图随机缩放、随机拼接到一张画布上
   优点：
     - 一张图里有 4 倍的目标，batch 统计更稳定
     - 强迫模型学习小目标、不完整目标（被裁切的目标）
     - 变相增大 batch size，BN 更稳
   这是 YOLOv4/v5 相比 YOLOv3 涨点最明显的增强之一。
=========================================================================
"""
import random
from typing import Tuple

import numpy as np


def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (640, 640),
              color: Tuple[int, int, int] = (114, 114, 114)):
    """
    等比缩放 + 居中填充
    Args:
        img: HWC, BGR, uint8
    Returns:
        (填充后的图像, 缩放比例 r, (左右填充, 上下填充))
    """
    import cv2
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2                      # 均分到两侧

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def mosaic(imgs, labels, out_size: int = 640):
    """
    4 图拼接（简化版，但拼接逻辑与 ultralytics 一致）
    Args:
        imgs  : 4 张 HWC BGR 图
        labels: 4 个 (M, 5) 数组 [cls, xc, yc, w, h]（归一化）
    Returns:
        (拼接图, 合并后的标签)
    """
    import cv2
    s = out_size
    # 随机拼接中心点，落在画布中间区域
    yc, xc = int(random.uniform(s * 0.4, s * 0.6)), int(random.uniform(s * 0.4, s * 0.6))
    canvas = np.full((s, s, 3), 114, dtype=np.uint8)
    out_labels = []

    for i, (img, lab) in enumerate(zip(imgs, labels)):
        h, w = img.shape[:2]
        scale = min(s / h, s / w) * random.uniform(0.6, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        img = cv2.resize(img, (nw, nh))

        # 4 个象限的放置位置
        if i == 0:      # 左上
            x1a, y1a, x2a, y2a = max(xc - nw, 0), max(yc - nh, 0), xc, yc
            x1b, y1b = x1a - max(xc - nw, 0), y1a - max(yc - nh, 0)
        elif i == 1:    # 右上
            x1a, y1a, x2a, y2a = xc, max(yc - nh, 0), min(xc + nw, s), yc
            x1b, y1b = 0, y1a - max(yc - nh, 0)
        elif i == 2:    # 左下
            x1a, y1a, x2a, y2a = max(xc - nw, 0), yc, xc, min(yc + nh, s)
            x1b, y1b = x1a - max(xc - nw, 0), 0
        else:           # 右下
            x1a, y1a, x2a, y2a = xc, yc, min(xc + nw, s), min(yc + nh, s)
            x1b, y1b = 0, 0

        canvas[y1a:y2a, x1a:x2a] = img[y1b:y2a - y1a + y1b, x1b:x2a - x1a + x1b]

        # 标签同步平移（归一化 -> 像素 -> 平移 -> 归一化）
        if len(lab):
            lab = lab.copy()
            lab[:, 1] = (lab[:, 1] * nw + x1a - x1b) / s
            lab[:, 2] = (lab[:, 2] * nh + y1a - y1b) / s
            lab[:, 3] = lab[:, 3] * nw / s
            lab[:, 4] = lab[:, 4] * nh / s
            # 过滤掉被裁到画布外的框
            keep = (lab[:, 1] > 0) & (lab[:, 1] < 1) & (lab[:, 2] > 0) & (lab[:, 2] < 1)
            out_labels.append(lab[keep])

    return canvas, np.concatenate(out_labels, 0) if out_labels else np.zeros((0, 5))


if __name__ == "__main__":
    import numpy as np
    imgs = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(4)]
    labs = [np.array([[0, 0.5, 0.5, 0.2, 0.3]]) for _ in range(4)]
    img, lab = mosaic(imgs, labs, 640)
    print("Mosaic 输出:", img.shape, "标签:", lab.shape)
    img2, r, pad = letterbox(imgs[0], (640, 640))
    print("letterbox:", img2.shape, "scale:", round(r, 3), "pad:", pad)
