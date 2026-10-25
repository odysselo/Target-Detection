# -*- coding: utf-8 -*-
"""
yolov1/head.py
=========================================================================
YOLOv1 Detection Head —— 对应实验任务 B.1 第 2 题
"追踪 YOLOv1DetectionBlock，寻找最后的全连接层、输出维度和 reshape 操作"

结构（论文 Figure 3 后半段）：
    conv 3x3x1024 (s=1)   -> 14x14
    conv 3x3x1024 (s=2)   -> 特征图 14x14 -> 7x7
    conv 3x3x1024 (s=1)   -> 7x7
    conv 3x3x1024 (s=1)   -> 7x7
    Flatten               -> 7*7*1024 = 50176 维向量
    FC 4096               -> 全连接层 1（50176 x 4096 ≈ 205.5 M 参数，全模型最大的一块）
    Dropout(0.5)
    LeakyReLU(0.1)
    FC 7*7*(B*5+C)        -> 输出维度 S*S*(B*5+C) = 7*7*30 = 1470
    reshape               -> (N, 7, 7, 30)

为什么最后是 1470 = 7*7*30？
    7x7 个网格；每个网格预测 B=2 个框，每框 5 个数 (x, y, w, h, conf)，
    再加 C=20 个类别概率 -> 每格 2*5+20 = 30 维。
    (x, y) 是相对网格左上角的偏移（0~1），(w, h) 是相对整图的比例（0~1）。
=========================================================================
"""
import math
from typing import List

import torch
from torch import nn


class YOLOv1ClassificationBlock(nn.Module):
    """预训练用的分类头（论文先在 ImageNet 上预训练 backbone + 这一层）"""

    def __init__(self, num_classes: int = 200, in_channels: int = 1024):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, num_classes),
        )
        nn.init.kaiming_normal_(self.classifier[2].weight)
        nn.init.constant_(self.classifier[2].bias, 0)

    def forward(self, x):
        return self.classifier(x)


class YOLOv1DetectionBlock(nn.Module):
    """
    Args:
        S: 网格数（论文 7）
        B: 每格预测的框数（论文 2）
        C: 类别数（VOC 20）
        img_size: 输入图像边长，用于自动推断 Flatten 后的维度
    """

    def __init__(self, S: int = 7, B: int = 2, C: int = 20, img_size: int = 448,
                 in_channels: int = 1024):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.img_size = img_size
        self.in_channels = in_channels

        # ---- 4 个 3x3x1024 卷积 ----
        self.detection_structure = [("c", 3, in_channels, 1), ("c", 3, in_channels, 2),
                                    ("c", 3, in_channels, 1), ("c", 3, in_channels, 1)]
        self.detection_start = nn.Sequential(
            *self._parse_create_structure(self.detection_structure, in_channels=in_channels))

        # ---- 推断 Flatten 维度：448 -> backbone 14x14 -> 这里 stride2 -> 7x7 ----
        feat = img_size // 32                      # backbone 总 stride 32
        feat = (feat + 2 - 3) // 2 + 1             # 3x3, stride=2, padding=1
        self.flatten_dim = feat * feat * in_channels

        # ---- 两层全连接 + Dropout ----
        self.detection_end = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 4096),          # 全连接层 1
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, self.S * self.S * (self.B * 5 + self.C)),   # 全连接层 2 -> 1470
        )
        nn.init.kaiming_normal_(self.detection_end[1].weight)
        nn.init.constant_(self.detection_end[1].bias, 0)
        nn.init.kaiming_normal_(self.detection_end[4].weight)
        nn.init.constant_(self.detection_end[4].bias, 0)

    # ------------------------------------------------------------------
    def _parse_create_structure(self, structure, in_channels: int) -> List[nn.Module]:
        _structure = []
        for substruct in structure:
            kernel_size, out_channels, stride = substruct[1:]
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
            # 原仓库此处用 GroupNorm 替代 BatchNorm2d（小 batch 训练更稳）
            nn.GroupNorm(num_groups=math.gcd(32, out_channels), num_channels=out_channels),
            nn.LeakyReLU(negative_slope=leaky_rate),
        )
        nn.init.kaiming_normal_(block[0].weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu")
        return block

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输出 (N, S*S*(B*5+C))，例如 (N, 1470)"""
        return self.detection_end(self.detection_start(x))


if __name__ == "__main__":
    head = YOLOv1DetectionBlock(S=7, B=2, C=20, img_size=448)
    x = torch.zeros(1, 1024, 14, 14)      # backbone 的输出：14x14
    out = head(x)
    print("检测头输出:", tuple(out.shape))                     # (1, 1470)
    print("reshape 后:", tuple(out.reshape(-1, 7, 7, 30).shape))  # (1, 7, 7, 30)
