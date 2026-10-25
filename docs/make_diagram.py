# -*- coding: utf-8 -*-
"""
docs/make_diagram.py
=========================================================================
生成实验报告所需的结构图（实验任务 A 第 2 题、任务 B.1 第 0 题）

    figures/faster_rcnn_arch.png    Faster R-CNN 四阶段结构图（含张量形状）
    figures/yolov1_arch.png         YOLOv1 Backbone + Detection Head 结构图
    figures/compare_pipeline.png    Faster R-CNN 与 YOLO 检测流程对比

用法： python docs/make_diagram.py
=========================================================================
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# 配色
C_INPUT = "#E8F1FB"
C_BACKBONE = "#D6E4F7"
C_RPN = "#FDE2D0"
C_ROI = "#D9F2E0"
C_HEAD = "#EDE0F7"
C_OUT = "#FFF3C4"
C_EDGE = "#3B4A5A"


def box(ax, x, y, w, h, text, color, fontsize=10, bold=False, edge=C_EDGE, lw=1.4, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                       linewidth=lw, edgecolor=edge, facecolor=color, linestyle=ls, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold" if bold else "normal", zorder=3, linespacing=1.5)
    return p


def arrow(ax, xy1, xy2, style="-|>", ls="-", color=C_EDGE, lw=1.4, rad=0.0):
    a = FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=14, linewidth=lw,
                        color=color, linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=1)
    ax.add_patch(a)


# =========================================================================
# 图 1：Faster R-CNN 结构图
# =========================================================================
def draw_faster_rcnn():
    fig, ax = plt.subplots(figsize=(11.5, 10.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.985, "Faster R-CNN 结构图（数据流 + 各阶段张量形状）",
            ha="center", fontsize=16, fontweight="bold")

    cx, w = 0.615, 0.40         # 主链中心与宽度

    # ---------- 主链：自顶向下 ----------
    box(ax, cx - w / 2, 0.895, w, 0.062, "输入图像   3 × H × W", C_INPUT, fontsize=11, bold=True)
    box(ax, cx - w / 2, 0.795, w, 0.075,
        "Backbone：VGG16 conv1_1 ~ conv5_3\n13 个 3×3 卷积 + 4 个 2×2 池化（下采样 16 倍）",
        C_BACKBONE, fontsize=10, bold=True)
    box(ax, cx - w / 2, 0.700, w, 0.065,
        "特征图 conv5_3   512 × H/16 × W/16", C_BACKBONE, fontsize=10.5, bold=True)

    arrow(ax, (cx, 0.895), (cx, 0.870))
    arrow(ax, (cx, 0.795), (cx, 0.765))

    # ---------- 左：RPN ----------
    ax.add_patch(FancyBboxPatch((0.015, 0.415), 0.36, 0.24,
                                boxstyle="round,pad=0.010,rounding_size=0.02",
                                linewidth=1.6, edgecolor="#E08A4A", facecolor="#FFF8F3",
                                linestyle="--", zorder=1))
    ax.text(0.03, 0.635, "RPN（候选框生成网络）", fontsize=10.5,
            fontweight="bold", color="#C96A20")
    box(ax, 0.03, 0.545, 0.33, 0.062, "Anchor 生成：每位置 3 尺度 × 3 比例 = 9 个", C_RPN, fontsize=9)
    box(ax, 0.03, 0.470, 0.33, 0.062, "3×3 conv → 1×1 cls(9) + 1×1 reg(36)", C_RPN, fontsize=9)
    ax.text(0.195, 0.443, "解码 → 裁剪 → 去小框 → Top-6000 → NMS(0.7) → Top-1000",
            fontsize=8.3, ha="center", color="#C96A20")

    arrow(ax, (cx - 0.10, 0.700), (0.195, 0.655), rad=0.12)

    # ---------- 主链：RoI Align 及之后 ----------
    box(ax, cx - w / 2, 0.560, w, 0.062,
        "RoI Align   每个 proposal → 7 × 7 × 512", C_ROI, fontsize=10.5, bold=True)
    arrow(ax, (0.375, 0.550), (cx - w / 2, 0.588), rad=-0.12)
    ax.text(0.392, 0.512, "proposals", fontsize=8.2, ha="center", color="#C96A20",
            rotation=-38)

    box(ax, cx - w / 2, 0.455, w, 0.062, "FC6 + FC7   1024 维特征向量", C_HEAD, fontsize=10.5, bold=True)
    arrow(ax, (cx, 0.560), (cx, 0.517))

    # ---------- 双分支 ----------
    box(ax, 0.335, 0.315, 0.27, 0.070, "分类分支  cls_score\n(K+1) 维  softmax", C_HEAD, fontsize=9.5)
    box(ax, 0.645, 0.315, 0.27, 0.070, "回归分支  bbox_pred\n4×(K+1) 维（逐类偏移）", C_HEAD, fontsize=9.5)
    arrow(ax, (cx - 0.06, 0.455), (0.47, 0.385), rad=0.1)
    arrow(ax, (cx + 0.06, 0.455), (0.78, 0.385), rad=-0.1)

    # ---------- 后处理与输出 ----------
    box(ax, 0.20, 0.170, 0.62, 0.075,
        "后处理：逐类解码偏移 → 置信度过滤 → 逐类 NMS → 保留 Top-100", C_OUT, fontsize=10, bold=True)
    arrow(ax, (0.47, 0.315), (0.40, 0.245), rad=0.1)
    arrow(ax, (0.78, 0.315), (0.62, 0.245), rad=-0.1)

    box(ax, 0.30, 0.060, 0.42, 0.062, "输出检测框：[x1, y1, x2, y2, 类别, 分数]",
        C_OUT, fontsize=10, bold=True)
    arrow(ax, (0.51, 0.170), (0.51, 0.122))

    # ---------- 损失标注 ----------
    ax.text(0.015, 0.370, "RPN 损失 = objectness(BCE) + box_reg(smooth L1)",
            fontsize=9.5, color="#C96A20")
    ax.text(0.015, 0.335, "RoI 损失 = 分类(交叉熵) + box_reg(smooth L1)",
            fontsize=9.5, color="#6A3FA0")


    plt.tight_layout()
    path = os.path.join(FIG_DIR, "faster_rcnn_arch.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print("saved:", path)


# =========================================================================
# 图 2：YOLOv1 结构图
# =========================================================================
def draw_yolov1():
    fig, ax = plt.subplots(figsize=(15.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.955, "YOLOv1 结构图（24 层卷积 + 2 层全连接，无 Neck）",
            ha="center", fontsize=17, fontweight="bold")

    y, h = 0.60, 0.20
    stages = [
        ("输入图像\n448×448×3", C_INPUT, 0.015, 0.10),
        ("7×7×64 s2\n+ MaxPool\n224×224×64", C_BACKBONE, 0.135, 0.13),
        ("3×3×192\n+ MaxPool\n112×112×192", C_BACKBONE, 0.285, 0.13),
        ("1×1×128 / 3×3×256\n1×1×256 / 3×3×512\n+ MaxPool\n56×56×512", C_BACKBONE, 0.435, 0.15),
        ("[1×1×256 / 3×3×512]×4\n1×1×512 / 3×3×1024\n+ MaxPool\n28×28×1024", C_BACKBONE, 0.605, 0.155),
        ("[1×1×512 / 3×3×1024]×2\n14×14×1024", C_BACKBONE, 0.78, 0.13),
    ]
    for text, color, x, w in stages:
        box(ax, x, y, w, h, text, color, fontsize=8.6)
    for i in range(len(stages) - 1):
        x1 = stages[i][2] + stages[i][3]
        arrow(ax, (x1, y + h / 2), (stages[i + 1][2], y + h / 2))

    # 检测头
    y2, h2 = 0.24, 0.20
    box(ax, 0.60, y2, 0.15, h2, "4 × 3×3×1024 conv\n(第 2 个 s=2)\n7×7×1024", C_HEAD, fontsize=9)
    box(ax, 0.775, y2, 0.10, h2, "Flatten\n50176", C_HEAD, fontsize=9)
    box(ax, 0.885, y2, 0.10, h2, "FC 4096\n+Dropout\n+LeakyReLU", C_HEAD, fontsize=8.6)
    arrow(ax, (0.845, y), (0.675, y2 + h2), rad=0.1)
    arrow(ax, (0.75, y2 + h2 / 2), (0.775, y2 + h2 / 2))
    arrow(ax, (0.875, y2 + h2 / 2), (0.885, y2 + h2 / 2))

    box(ax, 0.885, 0.02, 0.10, 0.16, "FC 1470\n→ reshape\n7×7×30", C_OUT, fontsize=9, bold=True)
    arrow(ax, (0.935, y2), (0.935, 0.18))

    # 7x7x30 说明
    ax.text(0.06, 0.40, "输出张量 7 × 7 × 30 的含义：", fontsize=11, fontweight="bold")
    ax.text(0.06, 0.30, "7 × 7   ：图像被划分成 7×7 个网格，目标中心落在哪个网格，就由哪个网格负责\n"
                       "30 = B×5 + C = 2×5 + 20：每个网格预测 2 个框，每框 5 个值 (x, y, w, h, conf)，\n"
                       "                                        再加 20 个类别概率（Pascal VOC）",
            fontsize=10, va="top", linespacing=1.9)

    ax.text(0.60, 0.50, "无 Neck / FPN / RPN：\n特征直接送进全连接层输出预测",
            fontsize=9.5, color="#888888", ha="center")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "yolov1_arch.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print("saved:", path)


# =========================================================================
# 图 3：两阶段 vs 单阶段流程对比
# =========================================================================
def draw_compare():
    fig, ax = plt.subplots(figsize=(14, 5.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.95, "Faster R-CNN（两阶段）与 YOLOv1（单阶段）流程对比",
            ha="center", fontsize=16, fontweight="bold")

    # 上：Faster R-CNN
    ax.text(0.02, 0.80, "Faster R-CNN", fontsize=12, fontweight="bold", color="#2C6FB5")
    xs = [(0.02, 0.12, "图像"), (0.17, 0.13, "CNN 特征"), (0.33, 0.14, "RPN 生成\n~1000 候选"),
          (0.50, 0.14, "RoI Align\n逐框提特征"), (0.67, 0.14, "分类 + 回归\n(每框)"), (0.84, 0.14, "NMS\n输出")]
    for x, w, t in xs:
        box(ax, x, 0.62, w, 0.13, t, C_RPN, fontsize=9.5)
    for i in range(len(xs) - 1):
        arrow(ax, (xs[i][0] + xs[i][1], 0.685), (xs[i + 1][0], 0.685))

    # 下：YOLOv1
    ax.text(0.02, 0.44, "YOLOv1", fontsize=12, fontweight="bold", color="#C96A20")
    ys = [(0.02, 0.12, "图像"), (0.17, 0.13, "CNN 特征"), (0.33, 0.14, "FC 直接输出\n7×7×30"),
          (0.50, 0.14, "解码成\n98 个框"), (0.67, 0.14, "置信度过滤"), (0.84, 0.14, "NMS\n输出")]
    for x, w, t in ys:
        box(ax, x, 0.26, w, 0.13, t, C_ROI, fontsize=9.5)
    for i in range(len(ys) - 1):
        arrow(ax, (ys[i][0] + ys[i][1], 0.325), (ys[i + 1][0], 0.325))

    ax.text(0.02, 0.13, "关键差别：Faster R-CNN 先「粗筛」再「逐框精算」（两阶段）；"
                        "YOLO 一次前向直接给出全部预测，只在输出端过滤（单阶段）",
            fontsize=10.5, color="#444444")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "compare_pipeline.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print("saved:", path)


if __name__ == "__main__":
    draw_faster_rcnn()
    draw_yolov1()
    draw_compare()
