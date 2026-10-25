# -*- coding: utf-8 -*-
"""
faster_rcnn/model.py
=========================================================================
Faster R-CNN 整体组装 —— 对应实验任务 A 第 1、2 题（代码架构 + 结构图）

数据流（一条主线，四段）：
    image (3, H, W)
      └─[预处理]─> 归一化 + 短边缩放到 min_size
      └─[Backbone]─> conv5_3 特征 (512, H/16, W/16)          —— backbone.py
      └─[RPN]─> 约 1.4 万 anchor -> 1000 个 proposal (含分数)  —— rpn.py
      └─[RoI Align]─> 每个 proposal 变成 7x7x512 特征
      └─[RoI Heads]─> 分类 (K+1) + 逐类框回归
      └─[后处理]─> 置信度过滤 + 逐类 NMS -> 最终检测框

损失 = RPN 损失（objectness + box_reg） + RoI 损失（classifier + box_reg）
=========================================================================
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backbone import LightBackbone, VGG16Backbone
from .roi_heads import RoIHeads
from .rpn import RPN


# =========================================================================
# 1. 图像预处理容器
# =========================================================================
class ImageList:
    """
    把 batch 里的图像堆叠成 (N, 3, H, W)，并记录每张图预处理后的 (H, W)。
    RPN 需要 image_sizes 来裁剪框、反推 stride。
    """

    def __init__(self, tensors: Tensor, image_sizes: List[Tuple[int, int]]):
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, device):
        return ImageList(self.tensors.to(device), self.image_sizes)

    def __len__(self):
        return len(self.image_sizes)


class GeneralTransform(nn.Module):
    """
    与 torchvision detection 的 GeneralizedRCNNTransform 等价（简化版）：
        ToTensor -> Normalize(ImageNet 均值方差) -> Resize(短边=min_size)
    """

    def __init__(self, min_size: int = 800, max_size: int = 1333,
                 image_mean=(0.485, 0.456, 0.406), image_std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.register_buffer("image_mean", torch.tensor(image_mean).view(3, 1, 1))
        self.register_buffer("image_std", torch.tensor(image_std).view(3, 1, 1))

    def resize(self, image: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        _, h, w = image.shape
        scale = self.min_size / min(h, w)
        if max(h, w) * scale > self.max_size:              # 限制长边，防止显存爆炸
            scale = self.max_size / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        if (new_h, new_w) != (h, w):
            image = F.interpolate(image[None], size=(new_h, new_w),
                                  mode="bilinear", align_corners=False)[0]
        return image, (new_h, new_w)

    def forward(self, images: List[Tensor]) -> ImageList:
        resized, sizes = [], []
        for img in images:
            img = (img - self.image_mean) / self.image_std
            img, size = self.resize(img)
            resized.append(img)
            sizes.append(size)
        if len({tuple(r.shape) for r in resized}) != 1:
            raise ValueError("一个 batch 内图像尺寸必须一致（demo 中已固定输入尺寸）")
        return ImageList(torch.stack(resized), sizes)


# =========================================================================
# 2. Faster R-CNN
# =========================================================================
class FasterRCNN(nn.Module):
    def __init__(self, backbone: nn.Module, rpn: RPN, roi_heads: RoIHeads,
                 transform: GeneralTransform):
        super().__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(self, images: List[Tensor],
                targets: Optional[List[Dict[str, Tensor]]] = None):
        """
        images : List[(3, H, W)]，像素值 0~1
        targets: 训练时必须提供 [{"boxes": (M,4) xyxy, "labels": (M,)}]
        return : 训练 -> 损失字典；推理 -> 每张图的 {"boxes","scores","labels"}
        """
        original_sizes = [(img.shape[-2], img.shape[-1]) for img in images]
        image_list = self.transform(images)

        features = self.backbone(image_list.tensors)          # conv5_3
        proposals, rpn_losses = self.rpn(image_list, features, targets)
        detections, roi_losses = self.roi_heads(
            features, proposals, image_list.image_sizes, targets)

        if self.training and targets is not None:
            losses = {}
            losses.update(rpn_losses)
            losses.update(roi_losses)
            return losses

        # 把框还原到"原始输入尺寸"（因为预处理时做过缩放）
        results = []
        for det, (orig_h, orig_w), (new_h, new_w) in zip(detections, original_sizes, image_list.image_sizes):
            sx, sy = orig_w / new_w, orig_h / new_h
            boxes = det["boxes"].clone()
            boxes[:, 0::2] *= sx
            boxes[:, 1::2] *= sy
            results.append({"boxes": boxes, "scores": det["scores"], "labels": det["labels"]})
        return results

    # ------------------------------------------------------------------
    @torch.no_grad()
    def trace(self, image: Tensor) -> Dict[str, tuple]:
        """
        【数据流追踪】打印每个阶段的张量形状，用于实验报告"根据数据流定位源码"。
        """
        self.eval()
        info = {}
        info["input"] = tuple(image.shape)

        # 1) backbone 逐 stage
        intermediates = self.backbone.forward_intermediates(image[None])
        for name, feat in intermediates.items():
            info[f"backbone/{name}"] = tuple(feat.shape)

        image_list = self.transform([image])
        features = self.backbone(image_list.tensors)
        info["conv5_3"] = tuple(features["0"].shape)

        # 2) RPN：anchor 数量
        anchors = self.rpn.anchor_generator(list(features.values()), image_list.image_sizes)
        info["rpn/anchors"] = tuple(anchors[0].shape)

        objectness, deltas = self.rpn.head(list(features.values()))
        info["rpn/objectness_map"] = tuple(objectness[0].shape)
        info["rpn/bbox_pred_map"] = tuple(deltas[0].shape)

        proposals, _ = self.rpn(image_list, features)
        info["rpn/proposals(after filter)"] = tuple(proposals[0].shape)

        # 3) RoI Align
        rois = self.roi_heads.box_roi_pool(features, [proposals[0][:, :4]])
        info["roi_align/features"] = tuple(rois.shape)

        box_feat = self.roi_heads.box_head(rois)
        info["roi_heads/fc7"] = tuple(box_feat.shape)
        cls_logits, box_reg = self.roi_heads.box_predictor(box_feat)
        info["roi_heads/cls_score"] = tuple(cls_logits.shape)
        info["roi_heads/bbox_pred"] = tuple(box_reg.shape)

        det, _ = self.roi_heads(features, proposals, image_list.image_sizes)
        info["final/detections"] = tuple(det[0]["boxes"].shape)
        return info


# =========================================================================
# 3. 工厂函数
# =========================================================================
def build_faster_rcnn(
    backbone_name: str = "vgg16",
    num_classes: int = 21,
    min_size: int = 800,
    max_size: int = 1333,
    pretrained_backbone: bool = False,
    anchor_sizes=((128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),),
    rpn_fg_iou_thresh: float = 0.7,
    rpn_bg_iou_thresh: float = 0.3,
    pre_nms_top_n=(12000, 6000),
    post_nms_top_n=(2000, 1000),
    **kwargs,
) -> FasterRCNN:
    """
    backbone_name: "vgg16"（论文配置） / "light"（CPU 快速实验用）
    num_classes  : 包含背景的类别数，Pascal VOC = 20 + 1 = 21
    """
    if backbone_name == "vgg16":
        backbone = VGG16Backbone(pretrained=pretrained_backbone)
    elif backbone_name == "light":
        backbone = LightBackbone(width=kwargs.pop("width", 32))
    else:
        raise ValueError(f"未知 backbone: {backbone_name}")

    rpn = RPN(in_channels=backbone.out_channels, anchor_sizes=anchor_sizes,
              aspect_ratios=aspect_ratios,
              fg_iou_thresh=rpn_fg_iou_thresh, bg_iou_thresh=rpn_bg_iou_thresh,
              pre_nms_top_n=pre_nms_top_n, post_nms_top_n=post_nms_top_n)
    roi_heads = RoIHeads(in_channels=backbone.out_channels, num_classes=num_classes)
    transform = GeneralTransform(min_size=min_size, max_size=max_size)
    return FasterRCNN(backbone, rpn, roi_heads, transform)


if __name__ == "__main__":
    model = build_faster_rcnn("light", num_classes=4, min_size=320, max_size=320).eval()
    img = torch.rand(3, 320, 320)
    for k, v in model.trace(img).items():
        print(f"{k:32s} {v}")
