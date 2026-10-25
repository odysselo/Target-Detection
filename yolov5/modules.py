# -*- coding: utf-8 -*-
"""
yolov5/modules.py
=========================================================================
YOLOv5 关键模块实现 —— 对应实验任务 B.2 "找到具体措施的实现代码（分条）"

逐条对应（每一条都在下面有同名类/函数）：
    (1) Focus 切片下采样            -> class Focus        （v6.0 后改为 6x6 Conv）
    (2) CSPNet 跨阶段局部网络        -> class C3 / Bottleneck
    (3) SPPF 空间金字塔池化（快速版） -> class SPPF
    (4) SiLU 激活 + BN 融合          -> class Conv
    (5) FPN + PAN 双向特征融合        -> build_yolov5_lite() 中的 neck
    (6) 多尺度 anchor 检测头 + 解耦解码 -> class Detect
    (7) 自适应 anchor（AutoAnchor）   -> yolov5/anchors.py
    (8) Mosaic / letterbox 数据增强   -> yolov5/augment.py
    (9) CIoU 损失 + 目标置信度 BCE    -> yolov5/loss.py
=========================================================================
"""
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------------
# (4) Conv = Conv2d + BatchNorm + SiLU（YOLOv5 的基础积木，替换了 LeakyReLU）
# -------------------------------------------------------------------------
def autopad(k, p=None, d=1):
    """same padding：保证 stride=1 时尺寸不变"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """标准卷积块：Conv2d(bias=False) + BN + SiLU"""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(Conv):
    """深度可分离卷积（YOLOv5 的 Bottleneck 中用于轻量化）"""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


# -------------------------------------------------------------------------
# (1) Focus：把 HxW 图像按 2x2 切片成 4 份再 concat，无信息丢失地下采样 2 倍
#     YOLOv5 v6.0 起被 Conv(k=6, s=2) 取代（等价但更快）
# -------------------------------------------------------------------------
class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s)

    def forward(self, x):
        # x: (N, C, H, W) -> 4 个 (N, C, H/2, W/2) -> concat 成 (N, 4C, H/2, W/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2],
                                    x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


# -------------------------------------------------------------------------
# (2) CSPNet：Bottleneck + C3（跨阶段部分连接）
# -------------------------------------------------------------------------
class Bottleneck(nn.Module):
    """标准瓶颈块：1x1 降维 -> 3x3 卷积 -> 残差"""

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """
    CSP 结构（v5 的 C3）：把输入分成两半，
    一半直接短路到输出，另一半经过 n 个 Bottleneck，
    最后 concat + 1x1 融合。好处：减少计算量、增强梯度回传。
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


# -------------------------------------------------------------------------
# (3) SPPF：串行 3 个 5x5 最大池化，等效 SPP 的并行池化但更快
# -------------------------------------------------------------------------
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)              # 感受野 5
        y2 = self.m(y1)             # 感受野 9（级联等效并行池化）
        y3 = self.m(y2)             # 感受野 13
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


# -------------------------------------------------------------------------
# (6) Detect：三个尺度的解耦检测头 + 解码
# -------------------------------------------------------------------------
class Detect(nn.Module):
    """
    输出格式：(N, na * (5 + nc), H, W)
        na = 每层 anchor 数（3）
        前 na*2 通道: xy 偏移；接着 na*2: wh；接着 na: obj；最后 na*nc: 类别
    """

    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)          # 检测层数（P3/P4/P5）
        self.na = len(anchors[0]) // 2  # 每层 anchor 数
        self.grid = [torch.zeros(1)] * self.nl
        self.stride = torch.zeros(self.nl)
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)                          # (nl, na, 2)
        self.register_buffer("anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

    def forward(self, x):
        """训练时直接返回原始输出；推理时返回解码后的 (N, num_anchors, 5+nc)"""
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                # 解码：xy 在网格内偏移（*2-0.5 允许跨越到相邻格），wh 用 (2σ)² 缩放 anchor
                y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))
        return x if self.training else torch.cat(z, 1)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing="ij")
        return torch.stack((xv, yv), 2).view(1, 1, ny, nx, 2).float()


# -------------------------------------------------------------------------
# (5) 组装一个 YOLOv5s 结构的小模型：Backbone(CSP) + FPN + PAN + 三尺度头
# -------------------------------------------------------------------------
class YOLOv5Lite(nn.Module):
    """
    结构（与 yolov5s 的拓扑一致，通道数缩小便于 CPU 运行）：
        P1: Conv(s2)                    -> H/2
        P2: Conv(s2) + C3               -> H/4
        P3: Conv(s2) + C3               -> H/8    ┐ 检测层 1（小目标）
        P4: Conv(s2) + C3               -> H/16   ┤ 检测层 2（中目标）
        P5: Conv(s2) + C3 + SPPF        -> H/32   ┘ 检测层 3（大目标）
        Neck: FPN 自顶向下 + PAN 自底向上（双向融合）
    """

    def __init__(self, nc=20, width=0.5, depth=1):
        super().__init__()
        w = lambda c: max(int(c * width), 8)
        self.stem = Conv(3, w(32), 3, 2)                 # P1/2
        self.dark2 = nn.Sequential(Conv(w(32), w(64), 3, 2), C3(w(64), w(64), n=depth))
        self.dark3 = nn.Sequential(Conv(w(64), w(128), 3, 2), C3(w(128), w(128), n=depth * 3))
        self.dark4 = nn.Sequential(Conv(w(128), w(256), 3, 2), C3(w(256), w(256), n=depth * 3))
        self.dark5 = nn.Sequential(Conv(w(256), w(512), 3, 2), C3(w(512), w(512), n=depth),
                                   SPPF(w(512), w(512)))

        # ---- FPN（自顶向下）----
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.lat4 = Conv(w(512), w(256), 1, 1)
        self.c3_n4 = C3(w(512), w(256), n=depth)
        self.lat3 = Conv(w(256), w(128), 1, 1)
        self.c3_n3 = C3(w(256), w(128), n=depth)

        # ---- PAN（自底向上）----
        self.down4 = Conv(w(128), w(128), 3, 2)
        self.c3_o4 = C3(w(384), w(256), n=depth)
        self.down5 = Conv(w(256), w(256), 3, 2)
        self.c3_o5 = C3(w(768), w(512), n=depth)

        self.detect = Detect(nc, anchors=[[10, 13, 16, 30, 33, 23],
                                         [30, 61, 62, 45, 59, 119],
                                         [116, 90, 156, 198, 373, 326]],
                             ch=(w(128), w(256), w(512)))
        self.detect.stride = torch.tensor([8., 16., 32.])

    def forward(self, x):
        p1 = self.stem(x)
        p2 = self.dark2(p1)
        p3 = self.dark3(p2)
        p4 = self.dark4(p3)
        p5 = self.dark5(p4)

        n4 = self.c3_n4(torch.cat([self.up(self.lat4(p5)), p4], 1))   # FPN: P5 -> P4
        n3 = self.c3_n3(torch.cat([self.up(self.lat3(n4)), p3], 1))   # FPN: P4 -> P3

        o4 = self.c3_o4(torch.cat([self.down4(n3), n4], 1))           # PAN: P3 -> P4
        o5 = self.c3_o5(torch.cat([self.down5(o4), p5], 1))           # PAN: P4 -> P5
        return self.detect([n3, o4, o5])


def build_yolov5_lite(nc: int = 20, width: float = 0.5, depth: int = 1) -> YOLOv5Lite:
    return YOLOv5Lite(nc=nc, width=width, depth=depth)


if __name__ == "__main__":
    model = build_yolov5_lite(nc=20).eval()
    x = torch.zeros(1, 3, 320, 320)
    with torch.no_grad():
        out = model(x)
    print("推理输出（已解码）:", tuple(out.shape), "= 3 层 anchor 合计 (40²+20²+10²)×3 个预测")
