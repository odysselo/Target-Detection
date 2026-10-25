# -*- coding: utf-8 -*-
"""
faster_rcnn/nms.py
=========================================================================
NMS（非极大值抑制）—— 对应实验任务 A 第 4 题之"NMS"

为什么需要 NMS：
    一个真实目标周围往往有几十个高分候选框（anchor 是稠密铺设的），
    它们互相高度重叠。NMS 的作用是"每个目标只留一个最好的框"。

算法（贪心）：
    1. 按得分降序排列所有框
    2. 取出得分最高的框 A 放入保留列表
    3. 删除与 A 的 IoU 大于阈值 iou_threshold 的其余框（认为在描述同一物体）
    4. 重复 2~3 直到没有框剩余

Faster R-CNN 中 NMS 出现两次：
    - RPN 阶段：对约 1.4 万个 anchor 的得分框做 NMS，只留 ~2000 个 proposal
    - 最终阶段：对每个类别的检测框做 NMS，得到最终输出
两次 NMS 的输入规模与作用完全不同，见实验报告任务 A 第 4 题的分析。
=========================================================================
"""
from typing import List

import torch
from torch import Tensor


def nms(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    """
    单类别 NMS
    boxes : (N, 4) xyxy
    scores: (N,)
    return: 保留框的下标，按得分降序排列
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    # 按得分降序，得到排序后的下标
    order = scores.argsort(descending=True)
    keep: List[Tensor] = []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    while order.numel() > 0:
        i = order[0]                     # 当前得分最高的框
        keep.append(i)
        if order.numel() == 1:
            break

        rest = order[1:]
        # 计算 i 与其余框的交集
        xx1 = torch.maximum(x1[i], x1[rest])
        yy1 = torch.maximum(y1[i], y1[rest])
        xx2 = torch.minimum(x2[i], x2[rest])
        yy2 = torch.minimum(y2[i], y2[rest])

        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        iou = inter / (areas[i] + areas[rest] - inter).clamp(min=1e-6)

        # 只保留 IoU 小于阈值的框，继续下一轮
        order = rest[iou <= iou_threshold]

    return torch.stack(keep)


def batched_nms(boxes: Tensor, scores: Tensor, idxs: Tensor, iou_threshold: float) -> Tensor:
    """
    多类别 NMS：不同类别的框不互相抑制（猫和狗可以重叠）。

    实现技巧（torchvision 同款）：给每个类别的框坐标加一个足够大的偏移，
    使不同类别的框在空间上互不相交，从而可以一次性调用单类别 NMS，
    避免 O(K) 次 Python 循环。
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    max_coordinate = boxes.max()
    offsets = idxs.to(boxes.dtype) * (max_coordinate + torch.tensor(1.0, device=boxes.device))
    boxes_for_nms = boxes + offsets[:, None]
    return nms(boxes_for_nms, scores, iou_threshold)


def soft_nms(boxes: Tensor, scores: Tensor, sigma: float = 0.5,
             score_threshold: float = 1e-3) -> Tensor:
    """
    Soft-NMS（选读）：不直接删除重叠框，而是按重叠程度衰减其得分，
    对密集目标（如人群）更友好。返回 (boxes, scores)。
    """
    boxes = boxes.clone()
    scores = scores.clone()
    N = boxes.shape[0]
    for i in range(N):
        max_pos = i + int(scores[i:].argmax())
        if max_pos != i:
            boxes[[i, max_pos]] = boxes[[max_pos, i]]
            scores[[i, max_pos]] = scores[[max_pos, i]]
        if i + 1 >= N:
            break
        xx1 = torch.maximum(boxes[i, 0], boxes[i + 1:, 0])
        yy1 = torch.maximum(boxes[i, 1], boxes[i + 1:, 1])
        xx2 = torch.minimum(boxes[i, 2], boxes[i + 1:, 2])
        yy2 = torch.minimum(boxes[i, 3], boxes[i + 1:, 3])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[i + 1:, 2] - boxes[i + 1:, 0]) * (boxes[i + 1:, 3] - boxes[i + 1:, 1])
        iou = inter / (area_i + area_r - inter).clamp(min=1e-6)
        # 高斯衰减：重叠越多，得分掉得越狠
        scores[i + 1:] *= torch.exp(-(iou ** 2) / sigma)
    keep = scores > score_threshold
    return boxes[keep], scores[keep]


if __name__ == "__main__":
    b = torch.tensor([[0., 0., 10., 10.],
                      [1., 1., 11., 11.],
                      [30., 30., 40., 40.],
                      [32., 32., 42., 42.]])
    s = torch.tensor([0.9, 0.85, 0.8, 0.75])
    print("NMS 保留下标:", nms(b, s, 0.5).tolist())   # 期望 [0, 2, 3]（第 2、3 个框 IoU=0.47<0.5，都保留）
