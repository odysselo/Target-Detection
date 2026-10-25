# -*- coding: utf-8 -*-
"""
demo/compare_speed.py
=========================================================================
实验任务 C "推理耗时"一栏的实测脚本

在同一台机器（CPU）上测量三种模型的：
    参数量 / 单张推理耗时 / FPS（不含预处理）

被测模型：
    1. Faster R-CNN (VGG16, 800x1333)      —— 论文配置
    2. Faster R-CNN (Light, 320x320)        —— 本实验实际训练的配置
    3. YOLOv1 (448x448, 论文配置)
    4. YOLOv5Lite (320x320)                 —— 三尺度检测头
=========================================================================
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from faster_rcnn import build_faster_rcnn
from yolov1 import YOLOv1
from yolov5 import build_yolov5_lite


def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def bench(model, size, warmup=2, runs=5, is_detector=True):
    x = torch.zeros(3, size, size)
    for _ in range(warmup):
        model([x] if is_detector else x[None])
    t0 = time.time()
    for _ in range(runs):
        model([x] if is_detector else x[None])
    dt = (time.time() - t0) / runs
    return dt


def main():
    rows = []

    m1 = build_faster_rcnn("vgg16", num_classes=21, min_size=800, max_size=1333).eval()
    rows.append(("Faster R-CNN (VGG16, 800x1333)", count_params(m1), bench(m1, 800)))

    m2 = build_faster_rcnn("light", num_classes=4, min_size=320, max_size=320).eval()
    rows.append(("Faster R-CNN (Light, 320x320)", count_params(m2), bench(m2, 320)))

    m3 = YOLOv1(S=7, B=2, C=20, img_size=448).eval()
    rows.append(("YOLOv1 (448x448)", count_params(m3), bench(m3, 448, is_detector=False)))

    m4 = build_yolov5_lite(nc=20).eval()
    rows.append(("YOLOv5Lite (320x320)", count_params(m4), bench(m4, 320, is_detector=False)))

    print(f"{'模型':34s}{'参数量(M)':>12s}{'耗时(ms)':>12s}{'FPS':>8s}")
    print("-" * 66)
    for name, p, dt in rows:
        print(f"{name:34s}{p:12.1f}{dt * 1000:12.1f}{1 / dt:8.1f}")


if __name__ == "__main__":
    main()
