# -*- coding: utf-8 -*-
"""
yolov1/postprocess.py
=========================================================================
YOLOv1 后处理 —— 对应实验任务 B.1 第 4 题
"找到模型预测、边界框解码、置信度过滤和 NMS 的代码位置"

完整链路（4 步）：
    1. 模型预测：model(image) -> (N, 1470)，reshape 成 (N, 7, 7, 30)
    2. 边界框解码：网格偏移 + 归一化 -> 整图坐标 (xmin, ymin, xmax, ymax)
    3. 置信度过滤：score = conf * P(class)，低于阈值直接丢弃
    4. NMS：同一目标只保留得分最高的框

与 Faster R-CNN 的本质差别：
    YOLO 没有 proposal 阶段，7*7*2 = 98 个框"一次性"全部产生，
    过滤全部发生在输出端（置信度阈值 + NMS），没有 RPN 的 Top-N 预筛。
=========================================================================
"""
from typing import List, Tuple

import torch

from .utils import bbox_centers_to_corners


# -------------------------------------------------------------------------
# 3'. NMS（贪心抑制，与 faster_rcnn/nms.py 同一算法，独立实现保持包解耦）
# -------------------------------------------------------------------------
def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    """返回保留下来的下标"""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        rest = order[1:]
        xx1 = torch.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = torch.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = torch.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = torch.minimum(boxes[i, 3], boxes[rest, 3])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter).clamp(min=1e-6)
        order = rest[iou <= iou_threshold]
    return torch.stack(keep)


# -------------------------------------------------------------------------
# 2'. 解码：把 (N, S, S, 30) 变成候选框集合
# -------------------------------------------------------------------------
def decode_predictions(pred: torch.Tensor, S: int = 7, B: int = 2, C: int = 20) -> torch.Tensor:
    """
    输入 pred: (N, 1470) 或 (N, S, S, 30)
    输出: (N, S*S*B, 6)，最后一维 = [xmin, ymin, xmax, ymax, conf, class_idx]（归一化坐标）
    """
    if pred.dim() == 2:
        pred = pred.reshape(-1, S, S, B * 5 + C)
    n = pred.shape[0]

    all_boxes = []
    for b in range(B):
        offset = b * 5
        corners = bbox_centers_to_corners(pred[..., offset:offset + 4], S=S)   # 解码
        conf = pred[..., offset + 4:offset + 5]
        # 类别概率 (N, S, S, C) -> 取最大值作为该框的类别
        class_probs = pred[..., B * 5:]
        class_conf, class_idx = class_probs.max(dim=-1, keepdim=True)
        # 每个框的最终得分 = objectness * 类别概率
        score = conf * class_conf
        all_boxes.append(torch.cat([corners, score, class_idx.float()], dim=-1))

    # (N, B, S, S, 6) -> (N, S*S*B, 6)
    out = torch.stack(all_boxes, dim=1)
    out = out.permute(0, 2, 3, 1, 4).reshape(n, S * S * B, 6)
    return out


# -------------------------------------------------------------------------
# 3' + 4'. 置信度过滤 + 逐类 NMS
# -------------------------------------------------------------------------
def postprocess(pred: torch.Tensor, S: int = 7, B: int = 2, C: int = 20,
                conf_threshold: float = 0.25, iou_threshold: float = 0.5,
                image_size: Tuple[int, int] = None) -> List[torch.Tensor]:
    """
    Returns:
        List[Tensor]，每张图一个 (M, 6) 张量 [x1, y1, x2, y2, score, class]
        若给了 image_size=(H, W)，坐标会换算成像素值
    """
    decoded = decode_predictions(pred, S, B, C)          # (N, S*S*B, 6)
    results = []
    for b in range(decoded.shape[0]):
        boxes = decoded[b]
        # ---- 置信度过滤 ----
        mask = boxes[:, 4] > conf_threshold
        boxes = boxes[mask]
        if boxes.numel() == 0:
            results.append(boxes.reshape(0, 6))
            continue

        # ---- 逐类 NMS：不同类别的框互不抑制 ----
        keep_all = []
        for cls in boxes[:, 5].unique():
            idx = torch.where(boxes[:, 5] == cls)[0]
            keep = nms(boxes[idx, :4], boxes[idx, 4], iou_threshold)
            keep_all.append(idx[keep])
        boxes = boxes[torch.cat(keep_all)]
        boxes = boxes[boxes[:, 4].argsort(descending=True)]     # 按得分降序

        if image_size is not None:
            h, w = image_size
            boxes = boxes.clone()
            boxes[:, [0, 2]] *= w
            boxes[:, [1, 3]] *= h
        results.append(boxes)
    return results


if __name__ == "__main__":
    torch.manual_seed(0)
    pred = torch.rand(1, 1470) * 0.5
    out = postprocess(pred, conf_threshold=0.05)
    print("过滤后框数:", out[0].shape)
