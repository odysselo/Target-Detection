# -*- coding: utf-8 -*-
"""
yolov1/utils.py
=========================================================================
YOLOv1 的几何工具：网格解码 + IoU

YOLOv1 的坐标约定（与 Faster R-CNN 完全不同！）：
    * 没有 anchor，坐标是"绝对预测"，不是相对参考框的偏移
    * 每个网格单元负责预测"中心点落在该格内"的目标
    * 网络输出 (x, y, w, h) 都是 0~1 的归一化值：
        x, y : 相对"所在网格左上角"的偏移
        w, h : 相对"整张图"的宽高比例
    * 还原到整图归一化坐标：X = (x + 列号) / S ,  Y = (y + 行号) / S
=========================================================================
"""
import torch


def bbox_centers_to_corners(bbox_data: torch.Tensor, S: int) -> torch.Tensor:
    """
    把 (x, y, w, h) 转成 (xmin, ymin, xmax, ymax)，并加上网格偏移。
    Args:
        bbox_data: (..., 4)，最后一维是 (x_c, y_c, w, h)
        S: 网格数
    Returns:
        corners: (..., 4)，归一化到 [0, 1] 的整图坐标
    """
    x_c = bbox_data[..., 0].unsqueeze(-1)
    y_c = bbox_data[..., 1].unsqueeze(-1)
    bbox_w = bbox_data[..., 2].unsqueeze(-1)
    bbox_h = bbox_data[..., 3].unsqueeze(-1)

    # 关键一步：加上网格坐标（列号 j 加到 x，行号 i 加到 y），再除以 S 归一化
    grid = torch.arange(S, device=bbox_data.device)
    i_grid = grid.view((1, S, 1, 1))      # 行
    j_grid = grid.view((1, 1, S, 1))      # 列
    x_c_img = (x_c + j_grid) / S
    y_c_img = (y_c + i_grid) / S

    xmin = x_c_img - bbox_w / 2
    ymin = y_c_img - bbox_h / 2
    xmax = x_c_img + bbox_w / 2
    ymax = y_c_img + bbox_h / 2

    return torch.cat((xmin, ymin, xmax, ymax), dim=-1)


def get_iou(corners_one: torch.Tensor, corners_target: torch.Tensor) -> torch.Tensor:
    """向量化 IoU，支持广播：corners_one (..., 4) vs corners_target (..., 4)"""
    xmin = torch.max(corners_one[..., 0], corners_target[..., 0])
    ymin = torch.max(corners_one[..., 1], corners_target[..., 1])
    xmax = torch.min(corners_one[..., 2], corners_target[..., 2])
    ymax = torch.min(corners_one[..., 3], corners_target[..., 3])

    area_one = ((corners_one[..., 2] - corners_one[..., 0]) *
                (corners_one[..., 3] - corners_one[..., 1])).abs()
    area_target = ((corners_target[..., 2] - corners_target[..., 0]) *
                   (corners_target[..., 3] - corners_target[..., 1])).abs()

    intersection = torch.clamp(xmax - xmin, min=0) * torch.clamp(ymax - ymin, min=0)
    union = area_one + area_target - intersection + 1e-9
    return intersection / union


def xywhn_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """归一化中心宽高 -> 归一化左上右下"""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
