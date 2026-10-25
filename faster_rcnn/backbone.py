# -*- coding: utf-8 -*-
"""
faster_rcnn/backbone.py
=========================================================================
Backbone（特征提取主干）—— 对应实验任务 A 第 3 题之"Backbone 部分"

Faster R-CNN 原论文用的是 ZF-Net / VGG16，这里实现 VGG16：
    13 个 3x3 卷积 + 4 个 2x2 最大池化，输出 conv5_3 特征图。
    输入 H x W  ->  输出 512 x (H/16) x (W/16)
    （4 次 stride=2 的池化，总下采样 16 倍）

关键点：
    * 只用到 conv5_3 为止，后面的 fc6/fc7 被丢弃（参数量太大且破坏空间结构）
    * RPN 和 RoI Heads 共享这一份特征，这就是 "RPN 与检测网络共享卷积特征"
    * stride=16 决定了 anchor 的稠密程度：特征图上 1 个点 = 原图 16x16 的区域
=========================================================================
"""
from collections import OrderedDict
from typing import Dict, List

import torch
from torch import Tensor, nn


class VGG16Backbone(nn.Module):
    """VGG16 主干，输出 conv5_3（512 通道，stride 16）"""

    # VGG16 的卷积配置，'M' 表示 MaxPool
    CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
           512, 512, 512, "M", 512, 512, 512]

    def __init__(self, pretrained: bool = False):
        super().__init__()
        self.out_channels = 512
        self.features = self._make_layers(self.CFG)

        # 5 个 stage 的结束位置，用于数据流追踪（conv1_2 / conv2_2 / conv3_3 / conv4_3 / conv5_3）
        self.stage_ends = [4, 9, 16, 23, len(self.features)]
        self.stage_names = ["conv1_2", "conv2_2", "conv3_3", "conv4_3", "conv5_3"]

        if pretrained:
            self._load_torchvision_weights()

    # ------------------------------------------------------------------
    def _make_layers(self, cfg: List) -> nn.Sequential:
        layers: List[nn.Module] = []
        in_ch = 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_ch, v, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_ch = v
        return nn.Sequential(*layers)

    def _load_torchvision_weights(self) -> None:
        """加载 torchvision 的 ImageNet 预训练 VGG16 卷积层权重（需要联网）"""
        try:
            import torchvision
            tv = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
            src = tv.features.state_dict()
            dst = self.features.state_dict()
            matched = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
            self.features.load_state_dict(matched, strict=False)
            print(f"[VGG16Backbone] 已加载 {len(matched)} 个预训练参数张量")
        except Exception as e:      # 离线环境则退化为随机初始化
            print(f"[VGG16Backbone] 预训练权重加载失败（{type(e).__name__}），使用随机初始化")

    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        返回 OrderedDict，键 "0" 对应 conv5_3。
        用 dict 是为了和 FPN 等多层特征的结构保持一致（本实验只有一层）。
        """
        return OrderedDict([("0", self.features(x))])

    # ------------------------------------------------------------------
    def forward_intermediates(self, x: Tensor) -> "OrderedDict[str, Tensor]":
        """
        逐 stage 输出特征图，用于"数据流追踪"演示：
            conv1_2 -> conv2_2 -> conv3_3 -> conv4_3 -> conv5_3
        """
        out, start = OrderedDict(), 0
        for end, name in zip(self.stage_ends, self.stage_names):
            out[name] = self.features[start:end](out[list(out)[-1]] if out else x)
            start = end
        return out


class LightBackbone(nn.Module):
    """
    轻量主干（4 个 stage，输出 256 通道，stride 16）。
    结构与 VGG16 同构（3x3 卷积堆叠 + 池化下采样），只是通道数大幅缩减，
    用于在 CPU 上快速跑通训练/推理演示（demo/train_demo.py）。

    注意：这里用 GroupNorm 而不是 BatchNorm。demo 的 batch 只有 2，
    BN 的滑动统计量会严重偏离真实分布，导致"训练时正常、eval 时全部预测成背景"。
    GroupNorm 与 batch size 无关，小 batch 下稳定得多。
    """

    def __init__(self, width: int = 32):
        super().__init__()
        self.out_channels = width * 8
        self.features = nn.Sequential(
            # stage1: H/2
            nn.Conv2d(3, width, 3, padding=1), nn.GroupNorm(8, width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.GroupNorm(8, width), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # stage2: H/4
            nn.Conv2d(width, width * 2, 3, padding=1), nn.GroupNorm(8, width * 2), nn.ReLU(inplace=True),
            nn.Conv2d(width * 2, width * 2, 3, padding=1), nn.GroupNorm(8, width * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # stage3: H/8
            nn.Conv2d(width * 2, width * 4, 3, padding=1), nn.GroupNorm(8, width * 4), nn.ReLU(inplace=True),
            nn.Conv2d(width * 4, width * 4, 3, padding=1), nn.GroupNorm(8, width * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # stage4: H/16  <- 输出层
            nn.Conv2d(width * 4, width * 8, 3, padding=1), nn.GroupNorm(8, width * 8), nn.ReLU(inplace=True),
            nn.Conv2d(width * 8, width * 8, 3, padding=1), nn.GroupNorm(8, width * 8), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.stage_ends = [7, 14, 21, 28]
        self.stage_names = ["s1", "s2", "s3", "s4"]

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return OrderedDict([("0", self.features(x))])

    def forward_intermediates(self, x: Tensor) -> "OrderedDict[str, Tensor]":
        out, start = OrderedDict(), 0
        for end, name in zip(self.stage_ends, self.stage_names):
            out[name] = self.features[start:end](out[list(out)[-1]] if out else x)
            start = end
        return out


if __name__ == "__main__":
    x = torch.zeros(1, 3, 640, 640)
    for bb in (VGG16Backbone(pretrained=False), LightBackbone()):
        feats = bb.forward_intermediates(x)
        print(type(bb).__name__)
        for k, v in feats.items():
            print(f"   {k:8s} -> {tuple(v.shape)}")
