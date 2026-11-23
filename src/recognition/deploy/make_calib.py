"""从训练分片里抽一批真实图，做成量化校准集 calib.npy。

为什么必须用真实图（而不是随机噪声）
--------------------------------------------------------------------------
Vitis AI 的 INT8 校准是「按激活值分布挑量化阈值」。用噪声算出来的阈值
和真实人脸的分布差很远，量化后精度会掉一大截。
校准集不需要标签，通常 100~1000 张就够（这里默认 1000）。

产出的 calib.npy 形状 (N, 3, 112, 112) float32，值域 [-1, 1]，
和 ONNX / DPU 的输入约定完全一致，容器里直接 np.load 就能用。

    python deploy/make_calib.py --shards shards --out deploy/calib.npy --num 1000
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import load_shard, ValAug  # noqa: E402

cv2.setNumThreads(0)


def main():
    ap = argparse.ArgumentParser("生成 INT8 量化校准集")
    ap.add_argument("--shards", default="shards", help="分片目录")
    ap.add_argument("--out", default="deploy/calib.npy")
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--size", type=int, default=112)
    args = ap.parse_args()

    import json
    meta = json.load(open(os.path.join(args.shards, "meta.json")))
    aug = ValAug(args.size)
    xs = []
    for name in meta["shards"]:
        if len(xs) >= args.num:
            break
        items = load_shard(os.path.join(args.shards, name))
        for _, jpg in items:
            img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            xs.append(aug(img))
            if len(xs) >= args.num:
                break
    x = np.stack(xs[:args.num])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, x)
    print(f"[done] 校准集 {x.shape} 值域[{x.min():.2f},{x.max():.2f}] -> {args.out}")


if __name__ == "__main__":
    main()
