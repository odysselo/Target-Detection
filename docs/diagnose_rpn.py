# -*- coding: utf-8 -*-
"""
docs/diagnose_rpn.py
=========================================================================
RPN 单图过拟合诊断（排查"框回归不收敛"用）

把同一张图训练 200 步。如果代码正确，RPN 应该能把这一张图"背下来"：
    loss_rpn_box_reg → 接近 0
    解码后的框宽 → 接近 GT 的宽
    proposal 数量 → 几百个（而不是 1 个）

用法： python docs/diagnose_rpn.py
=========================================================================
"""
import os
import sys

import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.dataset import NUM_CLASSES, make_sample
from faster_rcnn import build_faster_rcnn


def main(iters: int = 200):
    torch.manual_seed(0)
    img, gb, gl = make_sample(5, 320)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    targets = [{"boxes": torch.from_numpy(gb), "labels": torch.from_numpy(gl)}]

    model = build_faster_rcnn("light", num_classes=NUM_CLASSES, min_size=320, max_size=320,
                              anchor_sizes=((32, 64, 128),), rpn_fg_iou_thresh=0.5,
                              pre_nms_top_n=(1200, 600), post_nms_top_n=(400, 300))
    model.train()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

    for it in range(1, iters + 1):
        losses = model([t], targets)
        total = sum(losses.values())
        opt.zero_grad()
        total.backward()
        opt.step()

        if it % 25 == 0:
            model.eval()
            with torch.no_grad():
                il = model.transform([t])
                f = model.backbone(il.tensors)
                obj, d = model.rpn.head(list(f.values()))
                a = model.rpn.anchor_generator(list(f.values()), il.image_sizes)[0]
                dd = model.rpn._cat_levels(d)[0].view(-1, 4)
                b = model.rpn.box_coder.decode(dd, a)
                props, _ = model.rpn(il, f)
            print(f"iter {it:4d}  rpn_box_reg={losses['loss_rpn_box_reg']:.4f}  "
                  f"objectness={losses['loss_objectness']:.3f}  "
                  f"dw_mean={dd[:, 2].mean():+.2f}  decoded_w_mean={(b[:, 2] - b[:, 0]).mean():.0f}  "
                  f"proposals={len(props[0])}")
            model.train()

    print("GT boxes:", gb.round().tolist())
    print("GT 宽高:", (gb[:, 2] - gb[:, 0]).round().tolist())


if __name__ == "__main__":
    main()
