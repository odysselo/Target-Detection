# -*- coding: utf-8 -*-
"""
yolov1/loss.py
=========================================================================
YOLOv1 Loss —— 对应实验任务 B.1 第 3 题
"坐标损失、置信度损失、分类损失；找到负责预测框的匹配方式"

总损失（论文式 3）：

    L = λ_coord * Σ_i^obj Σ_j [ (x-x̂)² + (y-ŷ)² ]
      + λ_coord * Σ_i^obj Σ_j [ (√w-√ŵ)² + (√h-√ĥ)² ]
      +          Σ_i^obj Σ_j (C - Ĉ)²
      + λ_noobj * Σ_i^noobj Σ_j (C - Ĉ)²
      +          Σ_i^obj          Σ_c (p(c) - p̂(c))²

其中 λ_coord = 5，λ_noobj = 0.5，均为 MSE。

三个关键设计（也是 YOLO 最容易被问到的点）：
  1) 【负责预测框的匹配方式】"responsible predictor"：
     一个 GT 只由"中心点所在的那个网格"负责；该网格内的 B 个框中，
     取与 GT 的 IoU 最大的那个框负责回归（其余框不参与坐标损失）。
     实现：把 B 个框的 IoU 拼起来 argmax，得到 responsible_mask。
  2) 【坐标损失权重 λ_coord=5】定位误差比分类误差更"难"，加大权重。
  3) 【置信度损失分 obj / noobj】一张图里绝大多数网格没有物体，
     若不加权，模型会倾向于把所有置信度预测成 0。故 noobj 项乘 0.5。
  4) 【w/h 用 sqrt】小框的绝对误差与大框相同时，对小框更严重；
     开根号后误差尺度被拉平（论文的经典技巧）。
=========================================================================
"""
import torch
from torch import nn

from .utils import bbox_centers_to_corners, get_iou


class YOLOv1Loss(nn.Module):
    def __init__(self, S: int = 7, B: int = 2, C: int = 20,
                 L_coord: float = 5.0, L_noobj: float = 0.5):
        super().__init__()
        self.S, self.B, self.C = S, B, C
        self.L_coord = L_coord
        self.L_noobj = L_noobj
        # reduction="sum"：论文的损失是"对所有样本求和再除以 batch"
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, y_pred: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        # ---- 0. 从 (N, 1470) reshape 回 (N, 7, 7, 30) ----
        y_pred = y_pred.reshape(-1, self.S, self.S, self.C + self.B * 5)
        y_target = y_target.reshape(-1, self.S, self.S, self.C + self.B * 5)

        # ---- 1. 拆出两个预测框、目标框、物体掩码 ----
        target_bbox = y_target[..., :4]          # GT 框 (x, y, w, h)
        bbox_one = y_pred[..., :4]               # 第 1 个预测框
        bbox_two = y_pred[..., 5:9]              # 第 2 个预测框
        obj = y_target[..., 4].unsqueeze(-1)     # 1_{i}^{obj}：该格是否含物体

        # ---- 2. 【匹配】选出"负责"的框：与 GT 的 IoU 更大的那个 ----
        corners_one = bbox_centers_to_corners(bbox_one, S=self.S)
        corners_two = bbox_centers_to_corners(bbox_two, S=self.S)
        target_corners = bbox_centers_to_corners(target_bbox, S=self.S)

        iou_one = get_iou(corners_one, target_corners).unsqueeze(-1)
        iou_two = get_iou(corners_two, target_corners).unsqueeze(-1)
        best_iou, best_iou_idx = torch.max(torch.cat((iou_one, iou_two), dim=-1), dim=-1)
        best_iou = best_iou.unsqueeze(-1)
        responsible_mask = best_iou_idx.unsqueeze(-1)     # 0 -> 框1，1 -> 框2

        # 用 mask 选出负责框（1_{ij}^{obj} 的载体）
        responsible_bbox = (1 - responsible_mask) * bbox_one + responsible_mask * bbox_two

        target_bbox = target_bbox * obj                    # 只保留有物体的格
        masked_bbox = obj * responsible_bbox               # 只保留负责框

        # ---- 3. 坐标损失：中心点 + 宽高（宽高开根号） ----
        target_hw = torch.sqrt(target_bbox[..., 2:4])
        # 预测的 w/h 可能为负，先取绝对值再开根号，最后补回符号（保证梯度存在）
        responsible_hw = (torch.sign(masked_bbox[..., 2:4]) *
                          torch.sqrt(torch.abs(masked_bbox[..., 2:4]) + 1e-6))
        bbox_center_loss = self.mse(target_bbox[..., :2], masked_bbox[..., :2])
        bbox_hw_loss = self.mse(target_hw, responsible_hw)

        # ---- 4. 置信度损失（有物体）----
        target_conf = y_target[..., 4].unsqueeze(-1)
        conf_one = y_pred[..., 4].unsqueeze(-1)
        conf_two = y_pred[..., 9].unsqueeze(-1)
        responsible_conf = (1 - responsible_mask) * conf_one + responsible_mask * conf_two
        masked_responsible = obj * responsible_conf
        exist_target_conf = obj * target_conf
        # 训练目标不是常数 1，而是"预测框与 GT 的 IoU"（论文 4.3 节），
        # .detach() 防止 IoU 项把梯度传回坐标分支
        obj_conf_loss = self.mse(exist_target_conf * best_iou.detach(), masked_responsible)

        # ---- 5. 置信度损失（无物体）：两个框都要被压到 0 ----
        no_obj = 1 - obj
        no_obj_conf_loss = self.mse(no_obj * target_conf, no_obj * conf_one)
        no_obj_conf_loss += self.mse(no_obj * target_conf, no_obj * conf_two)

        # ---- 6. 分类损失：只对有物体的格计算 ----
        target_class = obj * y_target[..., self.B * 5:]
        pred_class = obj * y_pred[..., self.B * 5:]
        class_loss = self.mse(target_class, pred_class)

        # ---- 7. 加权求和 ----
        total_loss = (self.L_coord * (bbox_center_loss + bbox_hw_loss)
                      + obj_conf_loss
                      + self.L_noobj * no_obj_conf_loss
                      + class_loss)
        return total_loss / y_pred.shape[0]      # 除以 batch size


if __name__ == "__main__":
    S, B, C = 7, 2, 20
    crit = YOLOv1Loss(S, B, C)
    pred = torch.rand(2, S * S * (B * 5 + C))
    target = torch.zeros(2, S * S * (B * 5 + C))
    target[:, 4] = 1.0                  # 假装第一个格有物体
    print("loss =", crit(pred, target).item())
