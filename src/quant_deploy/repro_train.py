#!/usr/bin/env python3
"""全量从零复现训练: 复现 qr_ft_E 水平。
数据 = 全量 WIDER FACE(12880 bbox) + 300W(600 关键点) = 13480 张
骨干 = mobilenetv3(v3_small, ImageNet 预训练, 1.193M)  输入 320
配方 = lr 5e-4 / wd 5e-4 / batch 8 / epochs 20
用法: python3 repro_train.py cuda
"""
import sys

from common.scripts.train_cli import train

device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
argv = [
    "--data_root", "common/data/facebench_wider",
    "--input_size", "320",
    "--epochs", "20",
    "--lr", "5e-4",
    "--wd", "5e-4",
    "--batch", "8",
    "--width", "1.0",
    "--extra_ann", "common/data/facebench_wider/annotations/landmark_300w.json",
    "--extra_img", "/mnt/data3/300W/300W",
    "--out_dir", "repro_mobilenetv3",
    "--device", device,
]
train("mobilenetv3", argv)
