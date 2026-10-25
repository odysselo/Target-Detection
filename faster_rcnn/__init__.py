# -*- coding: utf-8 -*-
"""Faster R-CNN 从零实现（实验任务 A 的源码分析对象）"""

from .backbone import VGG16Backbone, LightBackbone
from .anchors import AnchorGenerator
from .box_utils import BoxCoder, box_iou, clip_boxes_to_image, remove_small_boxes
from .nms import nms, batched_nms, soft_nms
from .rpn import RPN, RPNHead, assign_targets_to_anchors
from .roi_heads import RoIHeads, RoIAlign, TwoMLPHead, FastRCNNPredictor, roi_align_reference
from .model import FasterRCNN, build_faster_rcnn, GeneralTransform, ImageList

__all__ = [
    "VGG16Backbone", "LightBackbone", "AnchorGenerator", "BoxCoder", "box_iou",
    "clip_boxes_to_image", "remove_small_boxes", "nms", "batched_nms", "soft_nms",
    "RPN", "RPNHead", "assign_targets_to_anchors", "RoIHeads", "RoIAlign",
    "TwoMLPHead", "FastRCNNPredictor", "roi_align_reference", "FasterRCNN",
    "build_faster_rcnn", "GeneralTransform", "ImageList",
]
