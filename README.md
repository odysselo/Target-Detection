# Target Detection：Faster R-CNN 与 YOLO 源码分析与实现

本仓库是《实验三：目标检测》的完整代码与报告，包含 **Faster R-CNN 从零实现**、**YOLOv1 实现（对齐开源仓库）**、**YOLOv5 改进点对照实现**，以及可一键跑通的训练 / 推理 / 耗时对比脚本。

> 仓库地址：<https://github.com/odysselo/Target-Detection>
> 实验报告：[`docs/实验报告.md`](docs/实验报告.md) ｜ Word 版 [`docs/实验报告.docx`](docs/实验报告.docx)
>
> 注：训练好的权重（`weights/*.pth`，约 580 MB）未入库，执行一次 `demo/train_demo.py` 即可重新生成（CPU 约 16 分钟）。

## 目录结构

```text
Target Detection/
├── faster_rcnn/          # 任务 A：Faster R-CNN（逐段中文注释）
│   ├── backbone.py       #   VGG16 / LightBackbone
│   ├── anchors.py        #   Anchor 生成（模板 + 平移）
│   ├── rpn.py            #   RPNHead、正负样本匹配、Proposal 筛选
│   ├── roi_heads.py      #   RoI Align、FC6/FC7、分类+回归双分支、后处理
│   ├── box_utils.py      #   框编解码（BoxCoder）、裁剪、IoU
│   ├── nms.py            #   NMS / batched NMS / Soft-NMS
│   └── model.py          #   整体组装 + trace() 数据流追踪
├── yolov1/               # 任务 B.1：YOLOv1
│   ├── backbone.py       #   24 层卷积主干（DarkNet 前身）
│   ├── head.py           #   YOLOv1DetectionBlock（FC4096 → FC1470）
│   ├── loss.py           #   坐标 / 置信度 / 分类损失 + 负责框匹配
│   ├── postprocess.py    #   解码 + 置信度过滤 + 逐类 NMS
│   └── model.py
├── yolov5/               # 任务 B.2：YOLOv5 改进点
│   ├── modules.py        #   Conv / C3 / SPPF / Focus / Detect / FPN+PAN
│   ├── loss.py           #   CIoU + 目标分配 + BCE
│   ├── anchors.py        #   AutoAnchor（k-means + BPR）
│   └── augment.py        #   Mosaic / letterbox
├── demo/                 # 跑通脚本
│   ├── dataset.py        #   合成数据集（圆 / 方 / 三角）
│   ├── train_demo.py     #   训练 Faster R-CNN / YOLOv1
│   ├── run_faster_rcnn.py#   推理 + 数据流追踪 + 三栏可视化
│   ├── run_yolov1.py     #   推理 + 解码/NMS 可视化 + 网格热力图
│   ├── compare_speed.py  #   推理耗时实测（任务 C）
│   └── run_yolo26.py     #   选做：YOLO26 推理
├── docs/
│   ├── 实验报告.md        # ★ 完整实验报告（对应实验要求逐条作答）
│   ├── 实验报告.docx      #   同内容的 Word 版
│   ├── make_diagram.py   #   生成结构图
│   └── plot_curves.py    #   由训练日志绘制损失曲线
├── figures/              # 结构图（Faster R-CNN / YOLOv1 / 流程对比）
├── logs/                 # 训练日志（供 plot_curves.py 画曲线）
├── weights/              # 训练好的权重
└── outputs/              # 推理可视化结果 + 损失曲线
```

## 快速开始

```bash
pip install -r requirements.txt

# 1) 训练（CPU 约 16 分钟；只想看结构可跳过）
python demo/train_demo.py --model fasterrcnn --iters 600 --lr 0.001
python demo/train_demo.py --model yolov1 --iters 900 --lr 0.001

# 2) 推理 + 可视化
python demo/run_faster_rcnn.py      # -> outputs/faster_rcnn_result.png
python demo/run_yolov1.py           # -> outputs/yolov1_result.png

# 3) 推理耗时对比
python demo/compare_speed.py

# 4) 结构图
python docs/make_diagram.py

# 5) 选做：YOLO26
pip install ultralytics && python demo/run_yolo26.py
```

## 每个模块单独自测

```bash
python -m faster_rcnn.anchors      # 打印 anchor 数量
python -m faster_rcnn.nms          # NMS 小例子
python -m faster_rcnn.model        # 打印完整数据流形状
python -m yolov1.head              # 检测头输出维度 (1,1470)
python -m yolov1.loss              # 损失数值
python -m yolov5.modules           # 三尺度输出形状
python -m yolov5.anchors           # AutoAnchor 的 BPR 对比
```

## 实测结果摘要

| 模型 | 训练步数 | 损失变化 | 验证 bestIoU | 单张耗时(CPU) |
| --- | --- | --- | --- | --- |
| Faster R-CNN (Light, 320²) | 600 | 5.70 → 0.68 | **0.96** | 176.8 ms |
| YOLOv1 (448²) | 900 | 205 → 3.0 | 0.3~0.4 | 114.6 ms |
| YOLOv5Lite (320²) | 未训练 | — | — | 20.7 ms |

> Faster R-CNN 有 anchor 先验，只需回归小偏移，因此收敛快；YOLOv1 从零回归绝对坐标，
> 且每张图只有 1 个正样本，900 步只能学到"大致位置"。这一差异本身就是任务 C 的重要观察。

## 关键设计说明

| 主题 | 说明 |
| --- | --- |
| 为什么有两个 backbone | `VGG16Backbone` 是论文配置（CPU 训练太慢）；`LightBackbone` 与 VGG16 **同构**（3×3 卷积堆叠 + 池化，stride 16），只缩减通道，用于 CPU 上真正跑通训练 |
| 为什么用合成数据集 | CPU 无法在合理时间内训练 VOC/COCO；合成数据（单目标、80~110 px 的圆/方/三角、纯色背景）可控、可复现，足以验证"数据流 → 损失 → 反传 → 解码 → NMS"整条链路。目标数、尺寸刻意做得简单，是为了让 CPU 上的训练能在几分钟内真正收敛 |
| anchor 尺度 | 默认 `(128, 256, 512)` 是论文/COCO 配置；合成目标只有 40~110 px，因此 demo 改用 `(32, 64, 128)`，否则 IoU 全 < 0.3、没有正样本 |
| 标签约定 | 0 = 背景，1..K = 前景类别（与 torchvision、`CLASS_NAMES` 一致） |
| YOLOv1 输出维度 | `S*S*(B*5+C)`；VOC 为 7×7×30=1470，本实验合成数据 C=3 时为 7×7×13=637 |

## 参考

- Faster R-CNN: *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*, NIPS 2015
- YOLOv1: *You Only Look Once: Unified, Real-Time Object Detection*, CVPR 2016
- YOLOv1 参考实现：<https://github.com/RickyPyeet/yolov1-pytorch>
- YOLOv5: <https://github.com/ultralytics/yolov5>
- YOLO26: <https://docs.ultralytics.com/models/yolo26>
