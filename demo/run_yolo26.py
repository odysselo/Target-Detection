# -*- coding: utf-8 -*-
"""
demo/run_yolo26.py
=========================================================================
选做部分：用 Ultralytics 的 YOLO26（或任意 YOLOv8/v11 权重）做推理

前置：
    pip install ultralytics

用法：
    python demo/run_yolo26.py                        # 自动下载 yolo26n.pt 并推理示例图
    python demo/run_yolo26.py --source path/to.jpg --model yolo26s.pt

说明：
    YOLO26 是 Ultralytics 2025 年发布的新一代模型，主打：
        * 端到端 NMS-free（去掉后处理 NMS）
        * DFL 移除、更简洁的回归头
        * 更适合边缘部署
    本脚本只负责调用官方接口做推理与可视化，模型结构与训练逻辑以官方仓库为准：
        https://github.com/ultralytics/ultralytics
=========================================================================
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--source", default=None, help="图片路径；缺省则用合成图")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("未安装 ultralytics，请先执行：pip install ultralytics")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    # 没有指定图片时，用本项目的合成数据生成一张
    if args.source is None:
        import cv2
        from demo.dataset import make_sample
        img, _, _ = make_sample(7, 640)
        args.source = os.path.join(OUT_DIR, "yolo26_input.png")
        cv2.imwrite(args.source, img)
        print(f"已生成测试图: {args.source}")

    model = YOLO(args.model)                  # 首次运行会自动下载权重
    results = model.predict(args.source, conf=args.conf, imgsz=args.imgsz, verbose=False)

    for r in results:
        print(f"检测到 {len(r.boxes)} 个目标")
        for box, cls, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            print(f"  {model.names[int(cls)]:>12s}  {conf:.3f}  {[round(v, 1) for v in box]}")
        r.save(filename=os.path.join(OUT_DIR, "yolo26_result.png"))
    print(f"结果已保存到 {os.path.join(OUT_DIR, 'yolo26_result.png')}")


if __name__ == "__main__":
    main()
