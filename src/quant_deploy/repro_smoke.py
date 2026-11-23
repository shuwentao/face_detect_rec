#!/usr/bin/env python3
"""小型冒烟训练: 验证 复现流水线 能跑通(少数据/少 epoch)。
用法:
    python repro_smoke.py cuda            # 用 GPU + ImageNet 预训练骨干
    python repro_smoke.py cuda --no_pretrained   # 随机初始化(无网络依赖)
    python repro_smoke.py cpu             # 无 GPU 时
"""
import sys

from common.scripts.train_cli import train

device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
no_pre = "--no_pretrained" in sys.argv

argv = [
    "--data_root", "common/data/facebench_wider",
    "--input_size", "320",
    "--epochs", "2",
    "--lr", "5e-4",
    "--wd", "5e-4",
    "--batch", "8",
    "--width", "1.0",
    "--max_images", "400",          # 仅取前 400 张 WIDER(300W 仍全量混入)做小样本验证
    "--extra_ann", "common/data/facebench_wider/annotations/landmark_300w.json",
    "--extra_img", "/mnt/data3/300W/300W",
    "--out_dir", "repro_smoke",
    "--device", device,
]
if no_pre:
    argv.append("--no_pretrained")

train("mobilenetv3", argv)
