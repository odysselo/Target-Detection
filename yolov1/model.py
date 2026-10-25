# -*- coding: utf-8 -*-
"""
yolov1/model.py
=========================================================================
YOLOv1 整体模型 —— 对应实验任务 B.1 第 0 题"代码架构"

    image (3, 448, 448)
      └─[YOLOv1BackboneBlock]─> (1024, 7, 7)        24 层卷积，下采样 64 倍
      └─[YOLOv1DetectionBlock]─> (1470,)            4 层卷积 + FC4096 + FC1470
      └─[reshape]─> (7, 7, 30)                      7x7 网格，每格 30 维

任务分工：
    Backbone  -> 提取特征（在 ImageNet 上预训练）
    Detect Head -> 直接输出"每个网格的框 + 置信度 + 类别"，没有 Neck / FPN
=========================================================================
"""
import torch
from torch import nn

from .backbone import YOLOv1BackboneBlock
from .head import YOLOv1ClassificationBlock, YOLOv1DetectionBlock


class YOLOv1(nn.Module):
    def __init__(self, S: int = 7, B: int = 2, C: int = 20,
                 mode: str = "detection", classific_num_classes: int = 200,
                 img_size: int = 448, width: float = 1.0):
        """
        Args:
            S: 网格数，默认 7
            B: 每格框数，默认 2
            C: 类别数，VOC = 20
            mode: "detection"（检测）/ "classification"（ImageNet 预训练）
            img_size: 输入边长，用于推断全连接层维度
            width: 通道缩放系数，1.0 为论文原版；demo 用 0.125 在 CPU 上快速训练
        """
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.mode = mode
        self.img_size = img_size

        self.backbone = YOLOv1BackboneBlock(width=width)
        if mode == "detection":
            self.detect = YOLOv1DetectionBlock(S=S, B=B, C=C, img_size=img_size,
                                               in_channels=self.backbone.out_channels)
        elif mode == "classification":
            self.classifier = YOLOv1ClassificationBlock(
                num_classes=classific_num_classes, in_channels=self.backbone.out_channels)
        else:
            raise ValueError("mode 只能是 'detection' 或 'classification'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        if self.mode == "detection":
            return self.detect(feat)        # (N, S*S*(B*5+C))
        return self.classifier(feat)


def create_yolo_detection(backbone_weights_path: str = None, seed: int = None, **kwargs) -> YOLOv1:
    """创建检测模型，可选加载预训练 backbone 权重（对应仓库的 create_yolo_detection）"""
    if seed is not None:
        torch.manual_seed(seed)
    model = YOLOv1(**kwargs)
    if backbone_weights_path:
        model.backbone.load_state_dict(torch.load(backbone_weights_path, map_location="cpu"))
        print(f"[YOLOv1] 已加载 backbone 权重: {backbone_weights_path}")
    return model


if __name__ == "__main__":
    m = YOLOv1(S=7, B=2, C=20, img_size=448).eval()
    x = torch.zeros(1, 3, 448, 448)
    y = m(x)
    print("输出:", tuple(y.shape), "->", tuple(y.reshape(-1, 7, 7, 30).shape))
