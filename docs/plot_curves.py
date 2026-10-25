# -*- coding: utf-8 -*-
"""
docs/plot_curves.py
=========================================================================
把训练日志解析成损失曲线（实验报告"跑通"证据之一）

用法：
    python docs/plot_curves.py
输出：
    outputs/fasterrcnn_train.png
    outputs/yolov1_train.png
=========================================================================
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

# 匹配 "iter  120/600  total=6.339  objectness=0.147  ..."
FRCNN_RE = re.compile(r"iter\s+(\d+)/\d+\s+total=([\d.]+)\s+objectness=([\d.]+)\s+"
                      r"rpn_box_reg=([\d.]+)\s+classifier=([\d.]+)\s+box_reg=([\d.]+)")
YOLO_RE = re.compile(r"iter\s+(\d+)/\d+\s+loss=([\d.]+)")


def read(path):
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return f.read().decode("utf-8", errors="ignore").replace("\x00", "")


def plot_frcnn(log_path):
    text = read(log_path)
    rows = [[float(x) for x in m] for m in FRCNN_RE.findall(text)]
    if not rows:
        print(f"[skip] 未在 {log_path} 中找到 Faster R-CNN 日志")
        return
    iters = [r[0] for r in rows]
    total = [r[1] for r in rows]
    obj = [r[2] for r in rows]
    cls = [r[4] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(iters, total, color="#2C6FB5", lw=1.6)
    axes[0].set_title("Faster R-CNN 总损失")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(iters, obj, color="#E08A4A", lw=1.4, label="RPN objectness")
    axes[1].plot(iters, cls, color="#6A3FA0", lw=1.4, label="RoI classifier")
    axes[1].set_title("RPN 前景/背景损失 与 RoI 分类损失")
    axes[1].set_xlabel("iteration")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, "fasterrcnn_train.png")
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print("saved:", path, f"({len(rows)} 个记录点)")


def plot_yolov1(log_path):
    text = read(log_path)
    rows = [[float(x) for x in m] for m in YOLO_RE.findall(text)]
    if not rows:
        print(f"[skip] 未在 {log_path} 中找到 YOLOv1 日志")
        return
    iters = [r[0] for r in rows]
    loss = [r[1] for r in rows]

    plt.figure(figsize=(7, 4))
    plt.plot(iters, loss, color="#C96A20", lw=1.6)
    plt.title("YOLOv1 训练损失（坐标 + 置信度 + 分类）")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT, "yolov1_train.png")
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print("saved:", path, f"({len(rows)} 个记录点)")


if __name__ == "__main__":
    import sys
    frcnn_log = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "logs", "fasterrcnn.log")
    yolo_log = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "logs", "yolov1.log")
    plot_frcnn(frcnn_log)
    plot_yolov1(yolo_log)
