# -*- coding: utf-8 -*-
"""
yolov1/backbone.py
=========================================================================
YOLOv1 Backbone（DarkNet 的前身）—— 对应实验任务 B.1 第 1 题"找到 Backbone"

结构与 YOLOv1 论文 Figure 3 的 24 层卷积完全一致：
    7x7x64-s2 -> maxpool 2x2-s2 -> 3x3x192 -> maxpool -> 1x1x128/3x3x256/1x1x256/3x3x512
    -> maxpool -> [1x1x256/3x3x512]x4 -> 1x1x512/3x3x1024 -> maxpool
    -> [1x1x512/3x3x1024]x2
输出：1024 x 14 x 14（输入 448x448，总下采样 32 倍；再经检测头的 stride-2 卷积才变成 7x7）

工程约定（与 RickyPyeet/yolov1-pytorch 保持一致）：
    * 每个卷积都是 Conv2d(bias=False) + BatchNorm2d + LeakyReLU(0.1)
    * padding = ceil((k - s) / 2)，保证 stride=2 时尺寸恰好减半
    * 权重用 kaiming_normal_(a=0.1, mode='fan_out', nonlinearity='leaky_relu')
=========================================================================
"""
import math
from typing import List, Union

import torch
from torch import nn


class YOLOv1BackboneBlock(nn.Module):
    """
    YOLOv1 的 24 层卷积主干（不含最后的检测头）

    Args:
        width: 通道缩放系数。1.0 = 论文原版（1024 通道输出）；
               小于 1 用于在 CPU 上快速跑通训练演示（demo 用 0.125）。
    """

    def __init__(self, width: float = 1.0):
        super().__init__()
        self.width = width
        # 用简写描述网络：(类型, 卷积核, 输出通道, 步长)；列表表示"重复 N 次"
        self.backbone_structure = [
            ("c", 7, 64, 2), ("mp", 2, 2),
            ("c", 3, 192, 1), ("mp", 2, 2),
            ("c", 1, 128, 1), ("c", 3, 256, 1), ("c", 1, 256, 1), ("c", 3, 512, 1), ("mp", 2, 2),
            [("c", 1, 256, 1), ("c", 3, 512, 1), 4], ("c", 1, 512, 1), ("c", 3, 1024, 1), ("mp", 2, 2),
            [("c", 1, 512, 1), ("c", 3, 1024, 1), 2],
        ]
        self.backbone = nn.Sequential(*self._parse_create_structure(self.backbone_structure, in_channels=3))
        self.out_channels = self._scaled(1024)

    def _scaled(self, c: int) -> int:
        return max(int(round(c * self.width)), 8)

    # ------------------------------------------------------------------
    def _parse_create_structure(self, structure, in_channels: int) -> List[nn.Module]:
        """把上面的简写配置展开成 nn.Module 列表，并自动串联通道数"""
        _structure: List[nn.Module] = []
        for substruct in structure:
            if isinstance(substruct, tuple):
                if substruct[0] == "c":
                    kernel_size, out_channels, stride = substruct[1:]
                    out_channels = self._scaled(out_channels)
                    padding = math.ceil((kernel_size - stride) / 2)
                    _structure.append(self._create_block(in_channels, out_channels,
                                                         kernel_size, stride, padding))
                    in_channels = out_channels
                elif substruct[0] == "mp":
                    kernel_size, stride = substruct[1:]
                    _structure.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride))
            elif isinstance(substruct, list):
                n_repeats = substruct[-1]                     # 最后一维是重复次数
                for _ in range(n_repeats):
                    for item in substruct[:-1]:
                        kernel_size, out_channels, stride = item[1:]
                        out_channels = self._scaled(out_channels)
                        padding = math.ceil((kernel_size - stride) / 2)
                        _structure.append(self._create_block(in_channels, out_channels,
                                                             kernel_size, stride, padding))
                        in_channels = out_channels
        return _structure

    @staticmethod
    def _create_block(in_channels, out_channels, kernel_size, stride, padding,
                      leaky_rate: float = 0.1) -> nn.Sequential:
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=leaky_rate),
        )
        nn.init.kaiming_normal_(block[0].weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu")
        return block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


if __name__ == "__main__":
    x = torch.zeros(1, 3, 448, 448)
    feat = YOLOv1BackboneBlock()(x)
    print("backbone 输出:", tuple(feat.shape))       # (1, 1024, 7, 7)
