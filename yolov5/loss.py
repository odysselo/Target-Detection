# -*- coding: utf-8 -*-
"""
yolov5/loss.py
=========================================================================
YOLOv5 损失函数 —— 对应实验任务 B.2 第 2 条
"找到具体措施的实现代码（分条）"之（9）CIoU 损失 + 目标置信度 BCE

与 YOLOv1 的三大差异：
    YOLOv1：坐标用 MSE，且 w/h 开根号；分类用 MSE
    YOLOv5：坐标用 CIoU（直接优化 IoU，且考虑中心距离与长宽比）
            分类/置信度用 BCEWithLogits（多标签、可加 focal）
            obj 的回归目标 = 预测框与 GT 的 CIoU（软标签）

CIoU 公式：
    CIoU = IoU - ρ²(b, b_gt)/c² - α·v
    v    = (4/π²)(arctan(w_gt/h_gt) - arctan(w/h))²
    α    = v / (1 - IoU + v)
    Loss = 1 - CIoU
=========================================================================
"""
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------------
# 1. CIoU（也提供 GIoU / DIoU，便于对比实验）
# -------------------------------------------------------------------------
def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """box1: (N,4)   box2: (N,4)（逐元素比较）"""
    if xywh:      # (x, y, w, h) -> (x1, y1, x2, y2)
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    # 交集面积
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)   # 最小外接框宽
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps                          # 对角线平方
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4     # 中心点距离平方
            if CIoU:
                v = (4 / math.pi ** 2) * (torch.atan(w2 / (h2 + eps)) -
                                          torch.atan(w1 / (h1 + eps))) ** 2
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)              # CIoU
            return iou - rho2 / c2                                 # DIoU
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area                     # GIoU
    return iou


# -------------------------------------------------------------------------
# 2. 目标分配（build_targets）：把 GT 分配到"最匹配的 anchor + 网格"
# -------------------------------------------------------------------------
def build_targets(preds: List[torch.Tensor], targets: torch.Tensor, anchors: torch.Tensor):
    """
    preds  : 3 个 (N, na, ny, nx, no)
    targets: (M, 6) [img_idx, cls, x, y, w, h]，x/y/w/h 为归一化值
    anchors: (nl, na, 2)，单位为像素
    返回与 yolov5 一致的 ti/tj 形式：每层一个 (img, anchor, gy, gx, tx, ty, tw, th, cls)
    """
    device = preds[0].device
    nl = len(preds)
    out = [[] for _ in range(nl)]
    if targets.numel() == 0:
        return [torch.zeros((0, 9), device=device) for _ in range(nl)]

    img_size = 640.0            # 假设输入边长（归一化坐标 -> 像素）
    for gt in targets:
        img_i, cls, x, y, w, h = gt.tolist()
        gt_wh = torch.tensor([w, h], device=device) * img_size

        # (a) 选形状最匹配的 anchor：min(gt/anchor, anchor/gt) 越大越好（yolov5 的 shape ratio）
        ratio = gt_wh[None] / anchors.view(-1, 2)                 # (nl*na, 2)
        ratio = torch.max(ratio, 1 / ratio).max(dim=1)[0]
        best = int(torch.argmin(ratio))                            # 比例越接近 1 越好
        layer, a = best // anchors.shape[1], best % anchors.shape[1]

        ny, nx = preds[layer].shape[2:4]
        gx, gy = x * nx, y * ny                                    # 连续网格坐标
        gi, gj = int(gx), int(gy)                                  # 负责该 GT 的网格
        gi, gj = min(gi, nx - 1), min(gj, ny - 1)

        # (b) 回归目标：相对网格的偏移 tx,ty；相对 anchor 的尺度比 tw,th = log(gt/anchor)
        tx, ty = gx - gi, gy - gj
        aw, ah = anchors[layer, a]
        tw = math.log(gt_wh[0].item() / aw.item() + 1e-9)
        th = math.log(gt_wh[1].item() / ah.item() + 1e-9)
        out[layer].append([img_i, a, gj, gi, tx, ty, tw, th, cls])

    return [torch.tensor(t, device=device, dtype=torch.float32).view(-1, 9) if len(t)
            else torch.zeros((0, 9), device=device) for t in out]


# -------------------------------------------------------------------------
# 3. 总损失
# -------------------------------------------------------------------------
class ComputeLoss(nn.Module):
    def __init__(self, model, box_gain=0.05, cls_gain=0.5, obj_gain=1.0):
        super().__init__()
        self.box_gain, self.cls_gain, self.obj_gain = box_gain, cls_gain, obj_gain
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.anchors = model.detect.anchors
        self.nc = model.detect.nc

    def forward(self, preds, targets):
        device = preds[0].device
        nb = preds[0].shape[0]
        t = build_targets(preds, targets, self.anchors)

        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        lcls = torch.zeros(1, device=device)

        for i, pi in enumerate(preds):                     # 逐层
            # 构造该层的目标张量 (N, na, ny, nx, 5+nc)
            tobj = torch.zeros_like(pi[..., 0])
            tcls = torch.zeros_like(pi[..., 5:])
            if t[i].shape[0]:
                ti = t[i].long()
                b, a, gj, gi = ti[:, 0], ti[:, 1], ti[:, 2], ti[:, 3]
                # ---- 框回归（CIoU）----
                ps = pi[b, a, gj, gi]                      # 取负责该 GT 的预测
                pxy = ps[:, :2].sigmoid() * 2 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * self.anchors[i, a]
                pbox = torch.cat([pxy, pwh], 1)
                # 目标框（网格坐标 -> 像素）
                gxy = torch.stack([t[i][:, 4] + ti[:, 3], t[i][:, 5] + ti[:, 2]], 1)
                gwh = torch.exp(t[i][:, 6:8]) * self.anchors[i, a]
                tbox = torch.cat([gxy, gwh], 1)
                iou = bbox_iou(pbox, tbox, CIoU=True).squeeze()
                lbox += (1.0 - iou).mean()
                # ---- obj 用软标签（IoU），cls 用 one-hot ----
                tobj[b, a, gj, gi] = iou.detach().clamp(0).type(tobj.dtype)
                tcls[b, a, gj, gi, ti[:, 8]] = 1.0

            # ---- 置信度 / 分类 BCE ----
            lobj += self.bce(pi[..., 4], tobj).mean()
            if tcls.sum() > 0:
                lcls += self.bce(pi[..., 5:], tcls).mean()

        bs = nb
        lbox *= self.box_gain
        lobj *= self.obj_gain
        lcls *= self.cls_gain
        return (lbox + lobj + lcls) * bs, torch.cat([lbox, lobj, lcls]).detach()


if __name__ == "__main__":
    from .modules import build_yolov5_lite
    model = build_yolov5_lite(nc=20)
    model.train()
    x = torch.zeros(2, 3, 320, 320)
    preds = model(x)                     # 训练模式返回 3 个原始输出
    targets = torch.tensor([[0, 3, 0.5, 0.5, 0.2, 0.3],
                            [1, 7, 0.3, 0.4, 0.1, 0.15]])
    loss, items = ComputeLoss(model)(preds, targets)
    print("yolov5 loss:", float(loss), "| box/obj/cls:", [round(float(v), 4) for v in items])
