# -*- coding: utf-8 -*-
"""
faster_rcnn/anchors.py
=========================================================================
Anchor 生成器 —— 对应实验任务 A 第 4 题之"Anchor 生成"

核心思想：
    特征图上的每一个位置 (i, j) 对应原图上的一个"感受野中心"，
    在该中心预先铺设 K 个不同"尺度 x 长宽比"的参考框，即 anchor。
    RPN 只负责预测"这个 anchor 里有没有物体"和"它需要怎么挪/缩放"，
    而不需要从零回归坐标 —— 这就是 anchor-based 方法的本质。

典型配置（Faster R-CNN 论文）：
    scales  = [128, 256, 512]          # 尺度（按面积定义）
    ratios  = [0.5, 1.0, 2.0]          # 长宽比 w/h
    -> 每个位置 3 x 3 = 9 个 anchor
    在 stride=16 的 conv5_3 特征图上，640x640 输入约 40x40 个位置
    -> 约 40*40*9 = 14400 个 anchor
=========================================================================
"""
import math
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn


class AnchorGenerator(nn.Module):
    """在特征图上生成 anchor（支持多尺度，本实验只用单层 conv5_3）"""

    def __init__(
        self,
        sizes: Tuple[Tuple[int, ...], ...] = ((128, 256, 512),),
        aspect_ratios: Tuple[Tuple[float, ...], ...] = ((0.5, 1.0, 2.0),),
    ):
        super().__init__()
        if len(sizes) != len(aspect_ratios):
            raise ValueError("sizes 与 aspect_ratios 的层数必须一致")
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios

        # 每个位置上的 anchor 数量（单层时 = 3 * 3 = 9）
        self.num_anchors_per_location = [len(s) * len(a) for s, a in zip(sizes, aspect_ratios)]

        self.cell_anchors: Optional[List[Tensor]] = None   # 以原点为中心的模板
        self.grid_cache = {}                               # 平移后的 anchor 缓存

    # ------------------------------------------------------------------
    # 1) 以原点为中心生成基础 anchor 模板
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_cell_anchors(sizes, aspect_ratios) -> Tensor:
        cell_anchors = []
        for size in sizes:
            area = size ** 2                       # 尺度按"面积"定义
            for ratio in aspect_ratios:
                # 由 w*h = area, w/h = ratio 解出 w、h
                w = math.sqrt(area / ratio)
                h = w * ratio
                cell_anchors.append([-w / 2, -h / 2, w / 2, h / 2])
        return torch.as_tensor(cell_anchors, dtype=torch.float32)   # (A, 4)

    def set_cell_anchors(self, dtype, device):
        if self.cell_anchors is not None and self.cell_anchors[0].dtype == dtype:
            return
        self.cell_anchors = [
            self._generate_cell_anchors(s, r).to(dtype=dtype, device=device)
            for s, r in zip(self.sizes, self.aspect_ratios)
        ]

    # ------------------------------------------------------------------
    # 2) 把模板平移到特征图的每个格点上
    # ------------------------------------------------------------------
    def _grid_anchors(self, grid_sizes: List[Tuple[int, int]], strides: List[Tuple[int, int]]):
        anchors = []
        for size, stride, base in zip(grid_sizes, strides, self.cell_anchors):
            grid_height, grid_width = size
            stride_height, stride_width = stride
            device, dtype = base.device, base.dtype

            # 每个格点中心在"原图坐标系"下的 x / y
            shifts_x = torch.arange(0, grid_width, dtype=dtype, device=device) * stride_width
            shifts_y = torch.arange(0, grid_height, dtype=dtype, device=device) * stride_height
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            # (H*W, 4) 的平移量  +  (A, 4) 的模板  ->  (H*W, A, 4) -> (H*W*A, 4)
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
            anchors.append((shifts.view(-1, 1, 4) + base.view(1, -1, 4)).reshape(-1, 4))
        return anchors

    def forward(self, feature_maps: List[Tensor], image_sizes: List[Tuple[int, int]]):
        """
        feature_maps : 每层的特征图 (N, C, H, W)
        image_sizes  : 每张图预处理后的 (H, W)，用于反推 stride
        return       : List[Tensor]，第 i 张图的所有 anchor，形状 (num_anchors, 4)
        """
        grid_sizes = [tuple(f.shape[-2:]) for f in feature_maps]
        # stride = 原图尺寸 / 特征图尺寸（本实验恒为 16）
        strides = [
            (int(round(img_h / g[0])), int(round(img_w / g[1])))
            for (img_h, img_w), g in zip(image_sizes, grid_sizes)
        ]
        dtype, device = feature_maps[0].dtype, feature_maps[0].device
        self.set_cell_anchors(dtype, device)

        key = (tuple(grid_sizes), tuple(strides), dtype, str(device))
        if key not in self.grid_cache:
            self.grid_cache[key] = self._grid_anchors(grid_sizes, strides)
        anchors_per_level = self.grid_cache[key]

        # 每张图复用同一组 anchor（anchor 只依赖特征图尺寸，与图像内容无关）
        num_images = feature_maps[0].shape[0]
        return [torch.cat(anchors_per_level, dim=0) for _ in range(num_images)]


if __name__ == "__main__":
    # 小自测：打印 anchor 数量与尺寸分布
    gen = AnchorGenerator()
    feat = torch.zeros(1, 512, 40, 40)          # 640x640 输入 -> 40x40 特征图
    anchors = gen([feat], [(640, 640)])[0]
    print(f"特征图 40x40, 每位置 9 个 anchor -> 总数 {anchors.shape[0]}")
    print("前 9 个 anchor（中心在 (0,0) 的模板）:\n", anchors[:9].round())
