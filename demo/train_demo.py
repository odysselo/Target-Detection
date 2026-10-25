# -*- coding: utf-8 -*-
"""
demo/train_demo.py
=========================================================================
在合成数据集上"跑通"两个检测器的训练流程（CPU 可运行）

用法：
    python demo/train_demo.py --model fasterrcnn --iters 300
    python demo/train_demo.py --model yolov1 --iters 200
    python demo/train_demo.py --model both

说明：
    * Faster R-CNN 使用 LightBackbone（与 VGG16 同构、通道缩减），
      因为 VGG16 在 CPU 上训练太慢；把 --backbone vgg16 可切换成论文主干。
    * YOLOv1 使用论文原始结构（268M 参数），输入 224x224 让特征图为 7x7。
=========================================================================
"""
import argparse
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.dataset import (NUM_CLASSES, ShapesDataset, fasterrcnn_collate,
                          to_yolov1_target_batch, yolo_collate)
from faster_rcnn import build_faster_rcnn
from yolov1 import YOLOv1, YOLOv1Loss

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")


@torch.no_grad()
def quick_eval(model, img, gt_boxes, score_thresh: float = 0.3):
    """在验证图上快速评估：proposal 数量 + 最高分检测框与 GT 的最佳 IoU"""
    was_training = model.training
    model.eval()
    image_list = model.transform([img])
    features = model.backbone(image_list.tensors)
    proposals, _ = model.rpn(image_list, features)
    det = model([img])[0]
    keep = det["scores"] > score_thresh
    boxes = det["boxes"][keep]

    best_iou = 0.0
    if len(boxes) and len(gt_boxes):
        def iou(a, b):
            x1, y1 = max(a[0], b[0]), max(a[1], b[1])
            x2, y2 = min(a[2], b[2]), min(a[3], b[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
            return inter / (ua + 1e-6)
        best_iou = max(iou(boxes[i].tolist(), gt_boxes[j].tolist())
                       for i in range(len(boxes)) for j in range(len(gt_boxes)))
    if was_training:
        model.train()
    return f"val: proposals={len(proposals[0])}, dets={len(boxes)}, bestIoU={best_iou:.2f}"


def train_faster_rcnn(iters: int = 300, img_size: int = 320, batch_size: int = 2,
                      lr: float = 1e-3, backbone: str = "light", device: str = "cpu"):
    print("=" * 70)
    print(f"训练 Faster R-CNN（backbone={backbone}, {img_size}x{img_size}, iters={iters}）")
    print("=" * 70)
    # 合成数据集的目标尺寸约 40~110 px，因此把 anchor 尺度调小（论文配置为 128/256/512）
    # 合成目标尺寸 80~110 px，因此 anchor 取 64/96/128 三档（论文配置为 128/256/512）
    anchors = ((64, 96, 128),) if backbone == "light" else ((64, 128, 256),)
    model = build_faster_rcnn(backbone, num_classes=NUM_CLASSES,
                              min_size=img_size, max_size=img_size,
                              anchor_sizes=anchors,
                              # 目标较小、anchor 密度有限，正样本 IoU 阈值从 0.7 放宽到 0.5
                              rpn_fg_iou_thresh=0.5,
                              pre_nms_top_n=(600, 300), post_nms_top_n=(200, 150),
                              pretrained_backbone=False).to(device)
    model.train()

    ds = ShapesDataset(length=256, img_size=img_size, seed_offset=0)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=fasterrcnn_collate, drop_last=True)

    params = [p for p in model.parameters() if p.requires_grad]
    # 小数据 + 小 batch 下 Adam 比 SGD 收敛快得多（实测 SGD 1200 步仍不收敛，Adam 800 步即可）
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters)

    # 验证图（不在训练集内），用于观察收敛情况
    val_img, val_boxes, val_labels = ShapesDataset(1, img_size, seed_offset=5000)[0]
    t0 = time.time()
    it = 0
    while it < iters:
        for images, targets in loader:
            images = [im.to(device) for im in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            total = sum(loss_dict.values())

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)
            optimizer.step()

            it += 1
            if it % 10 == 0 or it == 1:
                detail = "  ".join(f"{k.replace('loss_', '')}={v.item():.3f}" for k, v in loss_dict.items())
                extra = ""
                if it % 100 == 0 or it == 1:
                    extra = "  |  " + quick_eval(model, val_img, val_boxes)
                print(f"  iter {it:4d}/{iters}  total={total.item():.3f}  {detail}"
                      f"  ({time.time() - t0:.0f}s){extra}")
            if it >= iters:
                break
        scheduler.step()

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    path = os.path.join(WEIGHTS_DIR, "fasterrcnn_demo.pth")
    torch.save({"model": model.state_dict(), "backbone": backbone,
                "img_size": img_size, "num_classes": NUM_CLASSES}, path)
    print(f"已保存: {path}   用时 {time.time() - t0:.0f}s")
    return path


def train_yolov1(iters: int = 200, img_size: int = 224, batch_size: int = 2,
                 lr: float = 3e-4, device: str = "cpu"):
    print("=" * 70)
    print(f"训练 YOLOv1（{img_size}x{img_size}, S=7, B=2, C=3, iters={iters}）")
    print("=" * 70)
    S, B, C = 7, 2, 3
    model = YOLOv1(S=S, B=B, C=C, img_size=img_size, width=1.0).to(device)
    model.train()
    criterion = YOLOv1Loss(S=S, B=B, C=C, L_coord=5.0, L_noobj=0.5)

    ds = ShapesDataset(length=256, img_size=img_size, seed_offset=0)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        collate_fn=yolo_collate, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    t0 = time.time()
    it = 0
    while it < iters:
        for batch in loader:
            images, targets = to_yolov1_target_batch(batch, S, B, C, img_size)
            images, targets = images.to(device), targets.to(device)

            pred = model(images)
            loss = criterion(pred, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            it += 1
            if it % 10 == 0 or it == 1:
                print(f"  iter {it:4d}/{iters}  loss={loss.item():.3f}  ({time.time() - t0:.0f}s)")
            if it >= iters:
                break

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    path = os.path.join(WEIGHTS_DIR, "yolov1_demo.pth")
    torch.save({"model": model.state_dict(), "S": S, "B": B, "C": C,
                "img_size": img_size, "width": 1.0}, path)
    print(f"已保存: {path}   用时 {time.time() - t0:.0f}s")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="both", choices=["fasterrcnn", "yolov1", "both"])
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--img-size", type=int, default=0, help="0 表示用默认值")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0)
    parser.add_argument("--backbone", default="light", choices=["light", "vgg16"])
    args = parser.parse_args()

    torch.manual_seed(0)
    if args.model in ("fasterrcnn", "both"):
        train_faster_rcnn(iters=args.iters,
                          img_size=args.img_size or 320,
                          batch_size=args.batch,
                          lr=args.lr or 1e-3,
                          backbone=args.backbone)
    if args.model in ("yolov1", "both"):
        train_yolov1(iters=args.iters,
                     img_size=args.img_size or 224,
                     batch_size=args.batch,
                     lr=args.lr or 3e-4)
