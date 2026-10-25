# -*- coding: utf-8 -*-
"""
faster_rcnn/roi_heads.py
=========================================================================
RoI Heads（检测头）—— 对应实验任务 A 第 3 题之"RoI Heads 部分"
                       + 第 4 题之"RoI 特征提取"与"最终框回归"

RPN 给出的是"可能含有物体的框"（proposal，不分类别）；
RoI Heads 要做的是：
    1. RoI Align：把每个 proposal 在特征图上"抠"成固定 7x7 的特征块
    2. 过两个 1024 维全连接层（fc6、fc7）
    3. 两条输出分支：
         - 分类分支 (K+1 类，含背景)
         - 回归分支 (每个类别 4 个偏移，逐类精修框)
    4. 后处理：逐类解码 -> 置信度过滤 -> 逐类 NMS -> 取 top-k

注意回归分支输出的是 4*(K+1) 维，即"每个类别一套偏移"，
推理时只取预测类别的 4 个值，这就是"逐类框回归"（class-specific regression）。
=========================================================================
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops import roi_align as tv_roi_align

from .box_utils import BoxCoder, clip_boxes_to_image, box_iou
from .nms import batched_nms


# =========================================================================
# 1. RoI 特征提取：RoI Align
# =========================================================================
def roi_align_reference(feature: Tensor, box: Tensor, output_size: int = 7,
                        spatial_scale: float = 1 / 16, sampling_ratio: int = 2) -> Tensor:
    """
    RoI Align 的教学版实现（单张特征图、单个 RoI），帮助理解双线性插值过程。

    与 RoI Pooling 的区别（Mask R-CNN 论文的核心贡献之一）：
        RoI Pooling 把浮点坐标量化成整数（两次取整），导致特征与框错位；
        RoI Align 保留浮点坐标，在每个 bin 内均匀采样 4 个点做双线性插值，
        不做任何量化，mask 精度提升 10%~50%。
    """
    C, H, W = feature.shape
    # 1) 把原图坐标换算到特征图坐标（除以 stride）
    x1, y1, x2, y2 = (box * spatial_scale).tolist()
    roi_w, roi_h = x2 - x1, y2 - y1
    bin_w, bin_h = roi_w / output_size, roi_h / output_size

    out = torch.zeros(C, output_size, output_size, device=feature.device)
    for i in range(output_size):
        for j in range(output_size):
            # 2) 每个 bin 内均匀采样 sampling_ratio^2 个点
            ys = y1 + (i + (torch.arange(sampling_ratio, device=feature.device) + 0.5) / sampling_ratio) * bin_h
            xs = x1 + (j + (torch.arange(sampling_ratio, device=feature.device) + 0.5) / sampling_ratio) * bin_w
            gy, gx = torch.meshgrid(ys, xs, indexing="ij")
            # 3) 双线性插值（grid_sample 要求归一化到 [-1, 1]）
            grid = torch.stack([
                2 * gx / max(W - 1, 1) - 1,
                2 * gy / max(H - 1, 1) - 1,
            ], dim=-1).unsqueeze(0)
            sampled = F.grid_sample(feature.unsqueeze(0), grid, mode="bilinear",
                                    align_corners=True)[0]          # (C, sr, sr)
            out[:, i, j] = sampled.mean(dim=(-1, -2))               # 4) bin 内取平均
    return out


class RoIAlign(nn.Module):
    """
    生产版 RoI Align：直接调用 torchvision 的 CUDA/C++ 算子（与 Detectron 数值一致），
    输入 boxes 为 List[Tensor]（每张图一组 (P,4) 框），输出 (sum(P), C, 7, 7)。
    """

    def __init__(self, output_size: int = 7, spatial_scale: float = 1 / 16,
                 sampling_ratio: int = 2, aligned: bool = True):
        super().__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, features: Dict[str, Tensor], boxes: List[Tensor]) -> Tensor:
        feats = list(features.values())
        # 把每张图的框加上 batch 索引，拼成一个 (sum(P), 5) 的张量
        rois = []
        for img_id, boxes_per_image in enumerate(boxes):
            idx = torch.full((boxes_per_image.shape[0], 1), img_id,
                             dtype=boxes_per_image.dtype, device=boxes_per_image.device)
            rois.append(torch.cat([idx, boxes_per_image], dim=1))
        rois = torch.cat(rois, dim=0)

        # 多层特征时按面积分配 RoI 到不同层（FPN）；本实验只有 conv5_3 一层
        return tv_roi_align(feats[0], rois, output_size=self.output_size,
                            spatial_scale=self.spatial_scale,
                            sampling_ratio=self.sampling_ratio, aligned=self.aligned)


# =========================================================================
# 2. 检测头：fc6 + fc7 + 双分支
# =========================================================================
class TwoMLPHead(nn.Module):
    """7x7x512 -> flatten(25088) -> fc6(1024) -> fc7(1024)"""

    def __init__(self, in_channels: int = 512, representation_size: int = 1024):
        super().__init__()
        self.fc6 = nn.Linear(in_channels * 7 * 7, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(start_dim=1)          # (N, 25088)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class FastRCNNPredictor(nn.Module):
    """
    最终的两条输出分支：
        cls_score : 1024 -> num_classes        (背景为第 0 类)
        bbox_pred : 1024 -> 4 * num_classes    (每个类别 4 个偏移)
    """

    def __init__(self, in_channels: int = 1024, num_classes: int = 21):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.cls_score(x), self.bbox_pred(x)


# =========================================================================
# 3. RoI Heads 主体
# =========================================================================
class RoIHeads(nn.Module):
    """
    proposal -> RoI 特征 -> 分类 + 回归 -> 后处理
    """

    def __init__(
        self,
        in_channels: int = 512,
        num_classes: int = 21,
        representation_size: int = 1024,
        fg_iou_thresh: float = 0.5,
        bg_iou_thresh: float = 0.5,
        batch_size_per_image: int = 512,
        positive_fraction: float = 0.25,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        detections_per_img: int = 100,
        box_weights=(10.0, 10.0, 5.0, 5.0),
    ):
        super().__init__()
        self.box_roi_pool = RoIAlign(output_size=7, spatial_scale=1 / 16)
        self.box_head = TwoMLPHead(in_channels, representation_size)
        self.box_predictor = FastRCNNPredictor(representation_size, num_classes)
        self.box_coder = BoxCoder(weights=box_weights)

        self.num_classes = num_classes
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_img = detections_per_img

    # ------------------------------------------------------------------
    def forward(self, features: Dict[str, Tensor], proposals: List[Tensor],
                image_sizes: List[Tuple[int, int]],
                targets: Optional[List[Dict[str, Tensor]]] = None):
        """
        proposals: List[(P, 5)]，前 4 列是框，第 5 列是 RPN 给的分数
        """
        boxes = [p[:, :4] for p in proposals]

        # ---- 训练：先采样 512 个 proposal（正样本 25%），再提特征 ----
        losses = {}
        if self.training and targets is not None:
            labels, matched_gt_boxes, sampled_idxs = self._sample_proposals(
                proposals, targets, image_sizes)
            boxes = [boxes[i][sampled_idxs[i]] for i in range(len(boxes))]

        # ---- RoI 特征提取：这是 RPN 与检测头之间唯一的"接口" ----
        box_features = self.box_roi_pool(features, boxes)        # (sum(P), 512, 7, 7)
        box_features = self.box_head(box_features)               # (sum(P), 1024)
        class_logits, box_regression = self.box_predictor(box_features)

        if self.training and targets is not None:
            losses = self._compute_loss(class_logits, box_regression, boxes, labels, matched_gt_boxes)
            return [], losses

        # ---- 推理：逐类解码 + 后处理 ----
        num_boxes_per_image = [b.shape[0] for b in boxes]
        logits_list = class_logits.split(num_boxes_per_image, dim=0)
        reg_list = box_regression.split(num_boxes_per_image, dim=0)
        detections = [
            self.postprocess_detections(lg, rg, bx, size)
            for lg, rg, bx, size in zip(logits_list, reg_list, boxes, image_sizes)
        ]
        return detections, losses

    # ------------------------------------------------------------------
    def _sample_proposals(self, proposals, targets, image_sizes):
        """
        【最终阶段的样本筛选】—— 与 RPN 阶段的目的不同：
            RPN 筛的是"有没有物体"（二分类，IoU 0.7/0.3）
            这里筛的是"是哪个类别"（多分类，IoU 0.5/0.5），
            并且要求正负比 1:3（512 个里 128 个正样本）。
        """
        labels, matched_gt_boxes, sampled_idxs = [], [], []
        for proposals_per_image, targets_per_image, image_size in zip(proposals, targets, image_sizes):
            gt_boxes = targets_per_image["boxes"]
            gt_labels = targets_per_image["labels"]
            boxes = proposals_per_image[:, :4]
            boxes = clip_boxes_to_image(boxes, image_size)

            if gt_boxes.numel() == 0:
                labels.append(torch.zeros(boxes.shape[0], dtype=torch.int64, device=boxes.device))
                matched_gt_boxes.append(torch.zeros((boxes.shape[0], 4), device=boxes.device))
                sampled_idxs.append(torch.arange(boxes.shape[0], device=boxes.device))
                continue

            iou = box_iou(gt_boxes, boxes)[0]                 # (num_gt, num_proposals)
            gt_max_iou, gt_argmax = iou.max(dim=0)

            labels_per_image = gt_labels[gt_argmax]           # 先按"最像的 GT"给类别
            labels_per_image = labels_per_image.clone()
            # IoU 不够的降为背景（第 0 类）
            labels_per_image[gt_max_iou < self.bg_iou_thresh] = 0
            # 每个 GT 至少有一个 proposal 被标成正样本（保证不会漏掉任何目标）
            best_prop = iou.max(dim=1)[1]                     # 每个 GT 最匹配的 proposal 下标
            labels_per_image[best_prop] = gt_labels
            matched_boxes = gt_boxes[gt_argmax]

            # 采样 512 个：正样本 25%
            fg_idx = torch.where(labels_per_image > 0)[0]
            bg_idx = torch.where(labels_per_image == 0)[0]
            num_fg = min(int(self.batch_size_per_image * self.positive_fraction), fg_idx.numel())
            num_bg = min(self.batch_size_per_image - num_fg, bg_idx.numel())
            idx = torch.cat([
                fg_idx[torch.randperm(fg_idx.numel(), device=fg_idx.device)[:num_fg]],
                bg_idx[torch.randperm(bg_idx.numel(), device=bg_idx.device)[:num_bg]],
            ])

            labels.append(labels_per_image[idx])
            matched_gt_boxes.append(matched_boxes[idx])
            sampled_idxs.append(idx)
        return labels, matched_gt_boxes, sampled_idxs

    # ------------------------------------------------------------------
    def _compute_loss(self, class_logits, box_regression, boxes, labels, matched_gt_boxes):
        """分类用交叉熵；回归用 smooth L1，且只对正样本、只取对应类别的 4 个偏移"""
        labels_cat = torch.cat(labels, dim=0)
        cls_loss = F.cross_entropy(class_logits, labels_cat)

        box_regression = box_regression.reshape(-1, self.num_classes, 4)
        # 按每张图的 proposal 数切开，才能与 labels/boxes 对齐
        reg_per_image = torch.split(box_regression, [len(l) for l in labels], dim=0)

        box_loss = torch.tensor(0., device=class_logits.device)
        num_pos = 0
        for labels_per_image, boxes_per_image, gt_boxes, reg in zip(
                labels, boxes, matched_gt_boxes, reg_per_image):
            pos = torch.where(labels_per_image > 0)[0]
            if pos.numel() == 0:
                continue
            regression_targets = self.box_coder.encode(gt_boxes[pos], boxes_per_image[pos])
            pred = reg[pos, labels_per_image[pos]]      # 只取"预测类别"那一套偏移
            box_loss = box_loss + F.smooth_l1_loss(pred, regression_targets, beta=1.0 / 9, reduction="sum")
            num_pos += pos.numel()
        if num_pos > 0:
            box_loss = box_loss / num_pos
        return {"loss_classifier": cls_loss, "loss_box_reg": box_loss}

    # ------------------------------------------------------------------
    def postprocess_detections(self, class_logits, box_regression, boxes, image_size):
        """
        【最终检测阶段的后处理】—— 实验任务 A 第 4 题重点
            1. softmax 得到每类概率，丢掉背景类
            2. 逐类解码偏移 -> 真实框
            3. 裁剪到图像内
            4. 低分框过滤（score_thresh）
            5. 逐类 NMS（IoU 0.5）
            6. 每张图最多保留 100 个检测
        """
        pred_scores = F.softmax(class_logits, dim=-1)[:, 1:]                 # 1
        box_regression = box_regression.reshape(-1, self.num_classes, 4)[:, 1:]  # 去掉背景那一套
        num_classes = pred_scores.shape[1]
        boxes = boxes[:, None, :].expand(-1, num_classes, -1).reshape(-1, 4)  # 每类复制一份 proposal
        pred_boxes = self.box_coder.decode(box_regression.reshape(-1, 4), boxes)  # 2
        pred_boxes = clip_boxes_to_image(pred_boxes, image_size)             # 3

        scores = pred_scores.reshape(-1)
        # 类别下标 1..K（0 保留给背景），与 torchvision / CLASS_NAMES 的约定一致
        labels = torch.arange(1, num_classes + 1, device=boxes.device).repeat(
            boxes.shape[0] // num_classes)
        keep = torch.where(scores > self.score_thresh)[0]                     # 4
        scores, labels, pred_boxes = scores[keep], labels[keep], pred_boxes[keep]

        keep = batched_nms(pred_boxes, scores, labels, self.nms_thresh)       # 5
        keep = keep[: self.detections_per_img]                               # 6
        return {"boxes": pred_boxes[keep], "scores": scores[keep], "labels": labels[keep]}


if __name__ == "__main__":
    # 自测：随机特征图 + 随机 proposal -> 检测结果
    heads = RoIHeads(in_channels=512, num_classes=21).eval()
    feat = {"0": torch.randn(1, 512, 40, 40)}
    props = [torch.cat([torch.rand(50, 4) * 600, torch.rand(50, 1)], dim=1)]
    with torch.no_grad():
        det, _ = heads(feat, props, [(640, 640)])
    print("检测输出:", {k: tuple(v.shape) for k, v in det[0].items()})
