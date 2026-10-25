# -*- coding: utf-8 -*-
"""
demo/run_faster_rcnn.py
=========================================================================
Faster R-CNN 推理演示 + 数据流追踪（实验任务 A 的"跑通"环节）

输出：
    outputs/faster_rcnn_result.png   三栏对比图：
        (1) 原图 + 真值框
        (2) RPN 的 proposal（Top-100，按前景分数着色）
        (3) 最终检测结果（RoI Heads 分类 + 回归 + NMS 后）
    outputs/faster_rcnn_trace.txt    每个阶段的张量形状（用于写实验报告）

用法：
    python demo/run_faster_rcnn.py                    # 用训练好的权重
    python demo/run_faster_rcnn.py --weights none     # 随机初始化（只看数据流）
    python demo/run_faster_rcnn.py --backbone vgg16   # 换成论文主干
=========================================================================
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.dataset import CLASS_NAMES, NUM_CLASSES, draw_boxes, imwrite_unicode, make_sample
from faster_rcnn import build_faster_rcnn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")


def load_model(weights: str, backbone: str, img_size: int):
    anchors = ((64, 96, 128),) if backbone == "light" else ((64, 128, 256),)
    model = build_faster_rcnn(backbone, num_classes=NUM_CLASSES,
                              min_size=img_size, max_size=img_size,
                              anchor_sizes=anchors,
                              rpn_fg_iou_thresh=0.5,
                              pre_nms_top_n=(600, 300), post_nms_top_n=(200, 150))
    if weights and weights.lower() != "none" and os.path.exists(weights):
        ckpt = torch.load(weights, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        print(f"已加载权重: {weights}")
    else:
        print("未加载权重（随机初始化，仅用于验证数据流）")
    return model.eval()


def draw_proposals(img, proposals, scores):
    """把 proposal 画成半透明细框，分数越高越红"""
    out = img.copy()
    overlay = out.copy()
    for box, s in zip(proposals, scores):
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        strength = float(min(max(s, 0.0), 1.0))
        color = (int(255 * (1 - strength)), int(120 * strength), int(255 * strength))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
    return cv2.addWeighted(overlay, 0.75, out, 0.25, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=os.path.join(ROOT, "weights", "fasterrcnn_demo.pth"))
    parser.add_argument("--backbone", default="light", choices=["light", "vgg16"])
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--score-thresh", type=float, default=0.3)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    model = load_model(args.weights, args.backbone, args.img_size)

    img_bgr, gt_boxes, gt_labels = make_sample(args.seed, args.img_size)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

    # ---------- 1. 数据流追踪 ----------
    print("\n[数据流] 每个阶段的张量形状")
    print("-" * 62)
    trace = model.trace(tensor)
    for k, v in trace.items():
        print(f"  {k:34s}{v}")
    with open(os.path.join(OUT_DIR, "faster_rcnn_trace.txt"), "w", encoding="utf-8") as f:
        for k, v in trace.items():
            f.write(f"{k:34s}{v}\n")

    # ---------- 2. 单独取 RPN 的 proposal（用于可视化）----------
    with torch.no_grad():
        image_list = model.transform([tensor])
        features = model.backbone(image_list.tensors)
        proposals, _ = model.rpn(image_list, features)
        detections = model([tensor])[0]

    props = proposals[0][:100].numpy()
    prop_scores = proposals[0][:100, 4].numpy()

    # ---------- 3. 过滤最终结果 ----------
    keep = detections["scores"] > args.score_thresh
    det_boxes = detections["boxes"][keep].numpy()
    det_scores = detections["scores"][keep].numpy()
    det_labels = detections["labels"][keep].numpy()

    print(f"\n[结果] GT {len(gt_boxes)} 个 | proposal {len(proposals[0])} 个"
          f" | 最终检测 {len(det_boxes)} 个 (score>{args.score_thresh})")
    for b, s, l in zip(det_boxes, det_scores, det_labels):
        print(f"   {CLASS_NAMES[l]:>9s}  score={s:.3f}  box=[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]")

    # ---------- 4. 画图 ----------
    panel_gt = draw_boxes(img_bgr, gt_boxes, gt_labels)
    panel_prop = draw_proposals(img_bgr, props, prop_scores)
    panel_det = draw_boxes(img_bgr, det_boxes, det_labels, det_scores)
    for panel, title in [(panel_gt, "Ground Truth"),
                         (panel_prop, "RPN proposals (Top-100)"),
                         (panel_det, "Final detections")]:
        cv2.putText(panel, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    canvas = np.concatenate([panel_gt, panel_prop, panel_det], axis=1)
    path = os.path.join(OUT_DIR, "faster_rcnn_result.png")
    imwrite_unicode(path, canvas)
    print(f"\n已保存可视化: {path}")


if __name__ == "__main__":
    main()
