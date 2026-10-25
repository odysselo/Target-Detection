# -*- coding: utf-8 -*-
"""
faster_rcnn/rpn.py
=========================================================================
RPN（Region Proposal Network）—— 对应实验任务 A 第 3 题之"RPN 部分"
                               + 第 4 题之"Proposal 筛选"

RPN 的输入：conv5_3 特征图 (N, 512, H/16, W/16)
RPN 的输出：约 2000 个"可能是物体"的候选框（proposal），以及分类/回归损失

结构（3 个卷积层，极轻量）：
    3x3 conv(512->512)  ──┬──> 1x1 conv(512->9)     前景/背景二分类（每个 anchor 一个分数）
                          └──> 1x1 conv(512->9*4)   每个 anchor 的 4 维偏移

损失（与原论文一致）：
    L_rpn = (1/N_cls) * Σ BCE(cls)  +  (1/N_reg) * Σ smoothL1(reg)
    * 只对采样出的 256 个 anchor 算分类损失，正负比 1:1
    * 只对正样本 anchor 算回归损失
=========================================================================
"""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .anchors import AnchorGenerator
from .box_utils import BoxCoder, clip_boxes_to_image, remove_small_boxes, box_iou
from .nms import batched_nms, nms


# =========================================================================
# 1. RPN 头部：3x3 共享卷积 + 两条 1x1 分支
# =========================================================================
class RPNHead(nn.Module):
    """
    RPN 的"检测头"。注意它完全卷积化，因此可以接受任意尺寸输入，
    这也是 Faster R-CNN 能端到端训练、对输入尺寸不敏感的原因。
    """

    def __init__(self, in_channels: int = 512, num_anchors: int = 9):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

        for layer in (self.conv, self.cls_logits, self.bbox_pred):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.constant_(layer.bias, 0)

    def forward(self, features: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        logits, bbox_reg = [], []
        for f in features:
            t = F.relu(self.conv(f))
            logits.append(self.cls_logits(t))     # (N, A, H, W)
            bbox_reg.append(self.bbox_pred(t))    # (N, A*4, H, W)
        return logits, bbox_reg


# =========================================================================
# 2. 锚框匹配：给每个 anchor 分配"正/负样本"标签和回归目标
# =========================================================================
def assign_targets_to_anchors(anchors: List[Tensor], targets: List[Dict[str, Tensor]],
                              fg_iou_thresh: float = 0.7, bg_iou_thresh: float = 0.3):
    """
    匹配规则（与 RPN 论文一致）：
        IoU > 0.7                     -> 正样本（label=1）
        IoU < 0.3                     -> 负样本（label=0）
        0.3 <= IoU <= 0.7             -> 忽略（label=-1，不计损失）
        另外：每个 GT 至少分配给它 IoU 最大的那个 anchor（防止漏检）
    """
    labels, matched_gt_boxes = [], []
    for anchors_per_image, targets_per_image in zip(anchors, targets):
        gt_boxes = targets_per_image["boxes"]
        if gt_boxes.numel() == 0:
            device = anchors_per_image.device
            labels.append(torch.zeros(anchors_per_image.shape[0], dtype=torch.float32, device=device))
            matched_gt_boxes.append(torch.zeros((0, 4), dtype=torch.float32, device=device))
            continue

        match_quality = box_iou(gt_boxes, anchors_per_image)[0]      # (num_gt, num_anchors)

        # (a) 每个 anchor 取 IoU 最大的 GT
        gt_max_quality, gt_argmax = match_quality.max(dim=0)         # 沿 GT 维度取最大
        # (b) 每个 GT 取 IoU 最大的 anchor（保证每个 GT 都有人负责）
        _, anchor_argmax = match_quality.max(dim=1)

        labels_per_image = torch.full_like(gt_max_quality, -1)
        labels_per_image[gt_max_quality < bg_iou_thresh] = 0          # 背景
        labels_per_image[gt_max_quality >= fg_iou_thresh] = 1         # 前景
        labels_per_image[anchor_argmax] = 1                           # 强制分配

        matched_gt_boxes.append(gt_boxes[gt_argmax])
        labels.append(labels_per_image)

    return labels, matched_gt_boxes


def sample_anchors(labels: List[Tensor], batch_size_per_image: int = 256,
                   positive_fraction: float = 0.5):
    """
    由于负样本 anchor 数量远多于正样本（1:100 以上），直接算损失会被背景淹没。
    因此每张图采样 256 个 anchor，其中正样本占 50%。
    返回每张图被采样 anchor 的下标。
    """
    sampled = []
    for labels_per_image in labels:
        pos_idx = torch.where(labels_per_image == 1)[0]
        neg_idx = torch.where(labels_per_image == 0)[0]

        num_pos = int(batch_size_per_image * positive_fraction)
        num_pos = min(pos_idx.numel(), num_pos)
        num_neg = min(neg_idx.numel(), batch_size_per_image - num_pos)

        # 随机无放回采样
        perm1 = torch.randperm(pos_idx.numel(), device=pos_idx.device)[:num_pos]
        perm2 = torch.randperm(neg_idx.numel(), device=neg_idx.device)[:num_neg]
        sampled.append(torch.cat([pos_idx[perm1], neg_idx[perm2]]))
    return sampled


# =========================================================================
# 3. RPN 主体
# =========================================================================
class RPN(nn.Module):
    """
    负责：生成 anchor -> 预测分数/偏移 -> 解码 -> 过滤 -> 输出 proposal
    """

    def __init__(
        self,
        in_channels: int = 512,
        anchor_sizes=((128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),),
        pre_nms_top_n: Tuple[int, int] = (12000, 6000),   # (train, test)
        post_nms_top_n: Tuple[int, int] = (2000, 1000),
        nms_thresh: float = 0.7,
        fg_iou_thresh: float = 0.7,
        bg_iou_thresh: float = 0.3,
        batch_size_per_image: int = 256,
        positive_fraction: float = 0.5,
        min_size: float = 1.0,
    ):
        super().__init__()
        self.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        self.head = RPNHead(in_channels, self.anchor_generator.num_anchors_per_location[0])
        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))   # RPN 用 1:1:1:1

        self.pre_nms_top_n = {"training": pre_nms_top_n[0], "testing": pre_nms_top_n[1]}
        self.post_nms_top_n = {"training": post_nms_top_n[0], "testing": post_nms_top_n[1]}
        self.nms_thresh = nms_thresh
        self.fg_iou_thresh = fg_iou_thresh
        self.bg_iou_thresh = bg_iou_thresh
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.min_size = min_size

    # ------------------------------------------------------------------
    def forward(self, images: Tensor, features: Dict[str, Tensor],
                targets: Optional[List[Dict[str, Tensor]]] = None):
        feature_list = list(features.values())
        objectness, pred_bbox_deltas = self.head(feature_list)     # [(N,A,H,W)], [(N,A*4,H,W)]
        anchors = self.anchor_generator(feature_list, images.image_sizes)

        # ---- 训练分支：计算 RPN 损失 ----
        losses = {}
        if self.training and targets is not None:
            labels, matched_gt_boxes = assign_targets_to_anchors(
                anchors, targets, self.fg_iou_thresh, self.bg_iou_thresh)
            sampled_idxs = sample_anchors(labels, self.batch_size_per_image, self.positive_fraction)
            losses = self._compute_loss(objectness, pred_bbox_deltas, anchors,
                                        labels, matched_gt_boxes, sampled_idxs)

        # ---- 推理分支：解码 + 筛选，得到 proposal ----
        num_images = len(anchors)
        # 把 (N, A, H, W) 展平成 (N, H*W*A)，与 anchors 的顺序严格对应
        objectness = self._cat_levels(objectness)              # (N, num_anchors)
        pred_bbox_deltas = self._cat_levels(pred_bbox_deltas)  # (N, num_anchors*4)

        proposals = []
        for i in range(num_images):
            boxes = self.box_coder.decode(pred_bbox_deltas[i], anchors[i])   # 解码
            boxes = boxes.view(-1, 4)
            scores = objectness[i]
            boxes, scores = self.filter_proposals(boxes, scores, images.image_sizes[i])
            proposals.append(torch.cat([boxes, scores[:, None]], dim=1))     # (P, 5) 前 4 列坐标 + 分数
        return proposals, losses

    # ------------------------------------------------------------------
    @staticmethod
    def _flatten(x: List[Tensor]) -> List[Tensor]:
        """
        把每层的 (N, A, H, W) 变成 (N, H*W*A)：
        先 permute 成 (N, H, W, A) 再 reshape，保证与 anchor 的生成顺序
        (先按 y、再按 x、最后按 anchor 模板) 一一对应。这是最容易出错的一步。
        """
        out = []
        for t in x:
            t = t.permute(0, 2, 3, 1).reshape(t.shape[0], -1)
            out.append(t)
        return out

    def _cat_levels(self, x: List[Tensor]) -> Tensor:
        """单层时直接返回 (N, num_anchors)；多层时按层拼接 (N, Σnum_anchors)"""
        return torch.cat(self._flatten(x), dim=1)

    # ------------------------------------------------------------------
    def filter_proposals(self, boxes: Tensor, scores: Tensor, image_size: Tuple[int, int]):
        """
        【Proposal 筛选】—— 实验任务 A 第 4 题重点
        五步流水线：
            1. 裁剪到图像内
            2. 删除过小的框
            3. 按前景分数取 Top-N（pre-NMS，默认 6000）
            4. NMS（IoU 0.7）去掉高度重叠的重复框
            5. 再取 Top-N（post-NMS，默认 1000）交给 RoI Heads
        这样把 1.4 万个 anchor 压到 1000 个高质量候选，后续 RoI 计算才可行。
        """
        boxes = clip_boxes_to_image(boxes, image_size)                 # 1
        keep = remove_small_boxes(boxes, self.min_size)                 # 2
        boxes, scores = boxes[keep], scores[keep]

        top_n = self.pre_nms_top_n["training" if self.training else "testing"]
        if top_n > 0 and scores.numel() > top_n:                       # 3
            scores, topk_idx = scores.topk(top_n, sorted=False)
            boxes = boxes[topk_idx]

        keep = nms(boxes, scores, self.nms_thresh)                     # 4
        boxes, scores = boxes[keep], scores[keep]

        post_n = self.post_nms_top_n["training" if self.training else "testing"]
        if post_n > 0 and scores.numel() > post_n:                     # 5
            scores, topk_idx = scores.topk(post_n, sorted=False)
            boxes = boxes[topk_idx]

        return boxes, scores

    # ------------------------------------------------------------------
    def _compute_loss(self, objectness, pred_bbox_deltas, anchors,
                      labels, matched_gt_boxes, sampled_idxs) -> Dict[str, Tensor]:
        """RPN 损失：objectness 用 BCEWithLogits，框回归用 smooth L1"""
        objectness = self._cat_levels(objectness)
        pred_bbox_deltas = self._cat_levels(pred_bbox_deltas)

        losses = {"loss_objectness": torch.tensor(0., device=objectness[0].device),
                  "loss_rpn_box_reg": torch.tensor(0., device=objectness[0].device)}

        num_samples = 0
        for i in range(len(anchors)):
            idx = sampled_idxs[i]
            if idx.numel() == 0:
                continue

            # ---- 分类损失 ----
            gt_labels = labels[i][idx]
            pred_scores = objectness[i][idx]
            losses["loss_objectness"] = losses["loss_objectness"] + \
                F.binary_cross_entropy_with_logits(pred_scores, gt_labels, reduction="sum")

            # ---- 回归损失：只用正样本 ----
            pos = torch.where(gt_labels == 1)[0]
            if pos.numel() > 0:
                idx_pos = idx[pos]
                # 目标偏移 t* = encode(GT, anchor)
                regression_targets = self.box_coder.encode(matched_gt_boxes[i][idx_pos], anchors[i][idx_pos])
                # 预测偏移 t，先把 (num_anchors*4,) reshape 成 (num_anchors, 4) 再按 anchor 下标取
                pred_deltas = pred_bbox_deltas[i].reshape(-1, 4)[idx_pos]
                losses["loss_rpn_box_reg"] = losses["loss_rpn_box_reg"] + \
                    F.smooth_l1_loss(pred_deltas, regression_targets, beta=1.0 / 9, reduction="sum")
            num_samples += idx.numel()

        # 与论文一致：除以参与计算的正负样本总数（等价于原论文的 N_cls）
        n = max(num_samples, 1)
        return {k: v / n for k, v in losses.items()}


if __name__ == "__main__":
    # 自测：随机特征图 -> RPN -> proposal 数量与形状
    rpn = RPN(in_channels=512).eval()
    feat = {"0": torch.randn(1, 512, 40, 40)}

    class _Img:                       # 简易 images 容器，模拟 torchvision 的 ImageList
        image_sizes = [(640, 640)]
    proposals, _ = rpn(_Img(), feat)
    print("proposal 形状:", tuple(proposals[0].shape))   # (<=1000, 5)
