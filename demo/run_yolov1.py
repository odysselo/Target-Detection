# -*- coding: utf-8 -*-
"""
demo/run_yolov1.py
=========================================================================
YOLOv1 推理演示（实验任务 B.1 第 4 题的"跑通"环节）

流程（与源码一一对应）：
    model(image)                      -> (1, 1470)                 model.py / head.py
    reshape(1, 7, 7, 30)              -> 网格化预测
    decode_predictions()              -> 98 个候选框（7*7*2）        postprocess.py
    置信度过滤 + 逐类 NMS             -> 最终检测                     postprocess.py

输出：
    outputs/yolov1_result.png    三栏：真值 / 解码后全部 98 个框 / NMS 后结果
=========================================================================
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.dataset import CLASS_NAMES, draw_boxes, imwrite_unicode, make_sample
from yolov1 import YOLOv1, decode_predictions, postprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=os.path.join(ROOT, "weights", "yolov1_demo.pth"))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=1001)
    # YOLOv1 在本实验只训练了 900 步（原论文 135 epoch），置信度整体偏低，
    # 因此演示用 0.10 的阈值；用 --conf 0.25 可以看到它几乎没有输出。
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    S, B, C = 7, 2, 3
    model = YOLOv1(S=S, B=B, C=C, img_size=args.img_size).eval()

    if os.path.exists(args.weights):
        ckpt = torch.load(args.weights, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"已加载权重: {args.weights}")
    else:
        print("未加载权重（随机初始化，仅用于验证数据流）")

    img_bgr, gt_boxes, gt_labels = make_sample(args.seed, args.img_size)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

    # ---------- 1. 前向 + reshape ----------
    with torch.no_grad():
        raw = model(tensor[None])
    print(f"\n[1] 模型输出: {tuple(raw.shape)}  (S*S*(B*5+C) = 7*7*(2*5+3) = {S*S*(B*5+C)})")
    grid = raw.reshape(-1, S, S, B * 5 + C)
    print(f"[2] reshape 后: {tuple(grid.shape)}  每个网格 {B*5+C} 维 = "
          f"[x,y,w,h,conf] x {B} + {C} 类概率")

    # ---------- 2. 解码（网格偏移 -> 整图坐标）----------
    decoded = decode_predictions(raw, S, B, C)[0]           # (98, 6)
    print(f"[3] 解码得到候选框: {tuple(decoded.shape)}  (7*7*2 = 98 个)")

    # ---------- 3. 置信度过滤 + NMS ----------
    results = postprocess(raw, S, B, C, conf_threshold=args.conf,
                          iou_threshold=args.iou,
                          image_size=(args.img_size, args.img_size))[0]
    print(f"[4] 置信度过滤(>{args.conf}) + NMS(IoU<{args.iou}) 后: {tuple(results.shape)} 个框")

    # ---------- 4. 可视化 ----------
    low = decoded[decoded[:, 4] > 0.05].numpy()             # 未做 NMS 的原始预测
    if low.shape[0]:
        low_boxes = low[:, :4] * args.img_size
        low_scores, low_labels = low[:, 4], low[:, 5].astype(int) + 1
        panel_raw = draw_boxes(img_bgr, low_boxes, low_labels, low_scores,
                               class_names=CLASS_NAMES, thickness=1)
    else:
        panel_raw = img_bgr.copy()

    panel_gt = draw_boxes(img_bgr, gt_boxes, gt_labels, thickness=2)
    if results.shape[0]:
        panel_det = draw_boxes(img_bgr, results[:, :4], results[:, 5].numpy().astype(int) + 1,
                               results[:, 4], thickness=2)
    else:
        panel_det = img_bgr.copy()

    for panel, title in [(panel_gt, "GT"),
                         (panel_raw, "98 boxes"),
                         (panel_det, "after NMS")]:
        cv2.putText(panel, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    canvas = np.concatenate([panel_gt, panel_raw, panel_det], axis=1)
    path = os.path.join(OUT_DIR, "yolov1_result.png")
    imwrite_unicode(path, canvas)
    print(f"\n已保存可视化: {path}")

    # ---------- 5. 打印网格置信度热力图（文本版）----------
    conf_map = grid[0, ..., 4].numpy()
    print("\n[5] 7x7 网格的 objectness 热力图（每格取两个框中较大的 conf）：")
    conf2 = np.maximum(grid[0, ..., 4].numpy(), grid[0, ..., 9].numpy())
    for r in range(S):
        print("   " + " ".join(f"{conf2[r, c]:.2f}" for c in range(S)))


if __name__ == "__main__":
    main()
