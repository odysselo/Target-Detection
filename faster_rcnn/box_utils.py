# -*- coding: utf-8 -*-
"""
faster_rcnn/box_utils.py
=========================================================================
边界框工具函数（对应实验任务 A 第 4 题："最终框回归"的数学核心）

本文件集中实现 Faster R-CNN 中所有"框"的坐标变换：
    1. 框的两种表示：中心宽高 (cx, cy, w, h) <-> 左上右下 (x1, y1, x2, y2)
    2. 边界框编码 encode_boxes   : (GT, Anchor) -> 回归目标 t*
    3. 边界框解码 decode_boxes   : (预测偏移 t, Anchor) -> 实际框坐标
    4. 裁剪 / 面积 / IoU / 过滤小框

论文公式（Faster R-CNN, Girshick 2015, 式 1/2）：

    t_x = (x  - x_a) / w_a          t_x* = (x*  - x_a) / w_a
    t_y = (y  - y_a) / h_a          t_y* = (y*  - y_a) / h_a
    t_w = log(w / w_a)              t_w* = log(w* / w_a)
    t_h = log(h / h_a)              t_h* = log(h* / h_a)

其中 (x, y, w, h) 为预测框，(x_a, y_a, w_a, h_a) 为 anchor，
带 * 的为真值框。网络学习的正是 4 维偏移量 t，而不是坐标本身。
=========================================================================
"""
import math
from typing import Tuple

import torch
from torch import Tensor


# -----------------------------------------------------------------------
# 一、框表示转换
# -----------------------------------------------------------------------
def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """(cx, cy, w, h) -> (x1, y1, x2, y2)"""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """(x1, y1, x2, y2) -> (cx, cy, w, h)"""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], dim=-1)


def box_area(boxes: Tensor) -> Tensor:
    """计算框面积，输入 (N, 4) 的 xyxy 形式"""
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    """
    成对 IoU 计算（向量化），返回 (iou 矩阵, union)
    boxes1: (N, 4)   boxes2: (M, 4)   ->   iou: (N, M)
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # 广播得到 (N, M) 的左上角/右下角
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])   # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])   # (N, M, 2)

    wh = (rb - lt).clamp(min=0)                                # 无交集时截断为 0
    inter = wh[:, :, 0] * wh[:, :, 1]                          # (N, M)

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)
    return iou, union


def box_iou_single(box: Tensor, boxes: Tensor) -> Tensor:
    """一个框 vs 一组框的 IoU，返回 (M,)"""
    return box_iou(box.view(1, 4), boxes)[0].view(-1)


# -----------------------------------------------------------------------
# 二、框编码 / 解码 —— 这就是"框回归"的全部内容
# -----------------------------------------------------------------------
class BoxCoder:
    """
    边界框编码解码器。

    weights = (wx, wy, ww, wh)：Faster R-CNN 原论文取 (10, 10, 5, 5)，
    用于放大平移量、缩小尺度量的量纲，使二者对损失的贡献相当。
    """

    def __init__(self, weights=(10.0, 10.0, 5.0, 5.0), bbox_xform_clip=math.log(1000.0 / 16)):
        self.weights = tuple(weights)
        self.bbox_xform_clip = bbox_xform_clip

    # ---- 编码：训练时把 GT 转成网络要回归的目标 t* ----
    def encode(self, reference_boxes: Tensor, proposals: Tensor) -> Tensor:
        """
        reference_boxes : GT 框 (N, 4) xyxy
        proposals       : 与之匹配的 anchor/proposal (N, 4) xyxy
        return          : 回归目标 (N, 4) = (tx*, ty*, tw*, th*)
        """
        wx, wy, ww, wh = self.weights

        # 1) 都转成中心点 + 宽高
        ex_widths = proposals[:, 2] - proposals[:, 0]
        ex_heights = proposals[:, 3] - proposals[:, 1]
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0]
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1]
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

        # 2) 公式：位移用"除以宽高"归一化，尺度用 log 比值
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)

        return torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)

    # ---- 解码：推理时把网络输出的 t 还原成真实坐标 ----
    def decode(self, rel_codes: Tensor, boxes: Tensor) -> Tensor:
        """
        rel_codes : 网络预测的偏移 (N, 4) 或 (N, 4*K) 展平后的 (N, 4)
        boxes     : 参考框（anchor 或 proposal）(N, 4) xyxy
        return    : 解码后的框 (N, 4) xyxy
        """
        boxes = box_xyxy_to_cxcywh(boxes)
        rel_codes = rel_codes.reshape(boxes.shape[0], -1, 4)   # (N, K, 4)
        wx, wy, ww, wh = self.weights

        # 反归一化
        dx = rel_codes[:, :, 0] / wx
        dy = rel_codes[:, :, 1] / wy
        dw = rel_codes[:, :, 2] / ww
        dh = rel_codes[:, :, 3] / wh

        # 关键：对 dw/dh 做上限裁剪，防止 exp 溢出导致框"爆炸"（训练早期常见）
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        # 还原中心与宽高
        pred_ctr_x = dx * boxes[:, None, 2] + boxes[:, None, 0]
        pred_ctr_y = dy * boxes[:, None, 3] + boxes[:, None, 1]
        pred_w = torch.exp(dw) * boxes[:, None, 2]
        pred_h = torch.exp(dh) * boxes[:, None, 3]

        pred_boxes = torch.stack(
            [pred_ctr_x - 0.5 * pred_w,
             pred_ctr_y - 0.5 * pred_h,
             pred_ctr_x + 0.5 * pred_w,
             pred_ctr_y + 0.5 * pred_h], dim=-1)

        return pred_boxes.reshape(-1, 4)


# -----------------------------------------------------------------------
# 三、后处理用的几何操作
# -----------------------------------------------------------------------
def clip_boxes_to_image(boxes: Tensor, size: Tuple[int, int]) -> Tensor:
    """把框裁剪到图像范围内，size = (height, width)"""
    dim = boxes.dim()
    boxes_x = boxes[..., 0::2]
    boxes_y = boxes[..., 1::2]
    height, width = size

    boxes_x = boxes_x.clamp(min=0, max=width)
    boxes_y = boxes_y.clamp(min=0, max=height)

    clipped = torch.stack((boxes_x, boxes_y), dim=dim)
    return clipped.reshape(boxes.shape)


def remove_small_boxes(boxes: Tensor, min_size: float) -> Tensor:
    """
    过滤掉宽或高小于 min_size 的框，返回保留框的索引。
    RPN 阶段用它剔除"细长条"这类无意义的候选框。
    """
    ws, hs = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    keep = (ws >= min_size) & (hs >= min_size)
    return keep.nonzero(as_tuple=False).squeeze(1)
