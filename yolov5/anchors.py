# -*- coding: utf-8 -*-
"""
yolov5/anchors.py
=========================================================================
AutoAnchor：自适应 anchor —— 对应实验任务 B.2 第 2 条之（7）

YOLOv2/v3 的 anchor 是人工在 COCO 上用 k-means 聚出来的，
换数据集（比如医学影像、遥感）就严重失配。
YOLOv5 的做法：
    1. 训练前用 k-means（距离 = 1 - IoU）在自己的数据集上重新聚类出 9 个 anchor
    2. 用"最佳可能召回率"（BPR, Best Possible Recall）评估当前 anchor 是否够用
       BPR < 0.98 时自动重算 anchor
    3. 同时用 4 种不同尺度的 anchor（P3/P4/P5 各 3 个）覆盖小中大目标
=========================================================================
"""
import random
from typing import List, Tuple

import numpy as np
import torch


def kmeans_anchors(wh: np.ndarray, n: int = 9, iterations: int = 300, seed: int = 0):
    """
    对 GT 的 (w, h) 做 k-means 聚类，距离定义为 1 - IoU
    Args:
        wh: (M, 2) 所有 GT 的宽高（像素）
    Returns:
        (n, 2) 的 anchor，按面积升序排列
    """
    rng = np.random.RandomState(seed)
    n = min(n, len(wh))
    # 用 k-means++ 的思路初始化：随机挑 n 个点
    centers = wh[rng.choice(len(wh), n, replace=False)].copy()

    for _ in range(iterations):
        # 1 - IoU 作为距离
        iou = _iou_matrix(wh, centers)             # (M, n)
        assign = iou.argmax(axis=1)                # 每个 GT 归到最相似的 anchor
        new_centers = centers.copy()
        for k in range(n):
            members = wh[assign == k]
            if len(members):
                # 用中位数而不是均值，对离群点更鲁棒（与 yolov5 一致）
                new_centers[k] = np.median(members, axis=0)
        if np.allclose(new_centers, centers):
            break
        centers = new_centers

    return centers[np.argsort(centers.prod(axis=1))]     # 按面积升序


def _iou_matrix(boxes: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """以 (0,0) 为共同左上角计算 IoU（只比较宽高）"""
    b_w, b_h = boxes[:, 0][:, None], boxes[:, 1][:, None]
    c_w, c_h = clusters[:, 0][None, :], clusters[:, 1][None, :]
    inter = np.minimum(b_w, c_w) * np.minimum(b_h, c_h)
    union = b_w * b_h + c_w * c_h - inter
    return inter / (union + 1e-9)


def best_possible_recall(wh: np.ndarray, anchors: np.ndarray, thr: float = 0.20) -> float:
    """
    BPR = 有多少比例的 GT 能被"至少一个 anchor"以 IoU > thr 覆盖。
    yolov5 的阈值：BPR < 0.98 就重新聚类 anchor。
    """
    iou = _iou_matrix(wh, anchors)
    return float((iou.max(axis=1) > thr).mean())


def mean_best_iou(wh: np.ndarray, anchors: np.ndarray) -> float:
    """所有 GT 与"最匹配 anchor"的 IoU 平均值，用来衡量 anchor 与数据的贴合程度"""
    iou = _iou_matrix(wh, anchors)
    return float(iou.max(axis=1).mean())


def check_anchors(dataset_wh: np.ndarray, anchors: np.ndarray, thr: float = 0.20):
    """返回 (BPR, 建议的 anchor 或 None)"""
    bpr = best_possible_recall(dataset_wh, anchors, thr)
    if bpr < 0.98:
        new = kmeans_anchors(dataset_wh, n=len(anchors))
        return bpr, new
    return bpr, None


def anchors_from_clusters(clusters: np.ndarray) -> List[Tuple[int, int]]:
    """把 (9,2) 的聚类结果整理成 yolov5 yaml 里的 [[...],[...],[...]] 形式（每层 3 个）"""
    c = clusters.round().astype(int)
    return [(int(c[i][0]), int(c[i][1])) for i in range(len(c))]


if __name__ == "__main__":
    rng = np.random.RandomState(1)
    # 模拟一个"目标普遍很小"的数据集（例如遥感/航拍场景）
    wh = np.concatenate([rng.normal([11, 14], 3, (500, 2)),
                         rng.normal([17, 21], 4, (300, 2))]).clip(5, None)
    coco_anchors = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                             [59, 119], [116, 90], [156, 198], [373, 326]], dtype=float)
    new = kmeans_anchors(wh, 9)
    print(f"COCO anchor : BPR = {best_possible_recall(wh, coco_anchors):.3f}, "
          f"平均最佳 IoU = {mean_best_iou(wh, coco_anchors):.3f}")
    print(f"重新聚类后 : BPR = {best_possible_recall(wh, new):.3f}, "
          f"平均最佳 IoU = {mean_best_iou(wh, new):.3f}")
    print("新 anchor:", anchors_from_clusters(new))
