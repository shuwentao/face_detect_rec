"""自检脚本：不依赖真实数据集，验证模型 / 损失 / 增强 / 分片读写是否正常。

    python test_smoke.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402
from models.ghostfacenet import build_backbone  # noqa: E402
from models.heads import build_head  # noqa: E402
from data.dataset import FaceAlignAug, ValAug, load_shard  # noqa: E402
from data.build_shard import MAGIC as SHARD_MAGIC  # noqa: E402
from eval_bin import _best_threshold, eval_10fold  # noqa: E402


def t_backbone():
    print("== 骨干 ==")
    for arch, w in (("ghostfacenet", 1.0), ("mobilefacenet", 1.0)):
        m = build_backbone(arch, 512, w).eval()
        n = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            y = m(torch.randn(2, 3, 112, 112))
        assert y.shape == (2, 512), y.shape
        print(f"  {arch:<14} w={w}  参数={n/1e6:.2f}M  输出={tuple(y.shape)}")


def t_head():
    print("== 分类头 / loss ==")
    for loss in ("adaface", "arcface", "cosface"):
        for sr in (1.0, 0.05):
            head = build_head(512, 1000, loss, sample_rate=sr, sub_centers=1)
            feats = torch.randn(16, 512, requires_grad=True)
            labels = torch.randint(0, 1000, (16,))
            loss_v, acc = head(feats, labels)
            loss_v.backward()
            assert torch.isfinite(loss_v), loss_v
            assert feats.grad is not None and torch.isfinite(feats.grad).all()
            print(f"  {loss:<8s} sample_rate={sr:<4} loss={loss_v.item():.4f} "
                  f"acc={acc.item():.3f}")

    # AdaFace 的自适应 margin 必须真的随范数变化
    head = build_head(512, 1000, "adaface", sample_rate=1.0)
    head.train()
    for _ in range(30):
        f = torch.randn(64, 512) * torch.rand(64, 1) * 5
        head(f, torch.randint(0, 1000, (64,)))
    n = torch.tensor([1.0, 30.0])
    h = ((n - head.norm_mean) / (head.norm_std + 1e-3)).clamp(-1, 1)
    m = -head.m * h
    print(f"  自适应 margin: 低范数→{m[0]:+.3f}  高范数→{m[1]:+.3f}  "
          f"(norm_mean={head.norm_mean.item():.2f})")
    assert m[0] > m[1], "低质量样本应拿到更大的 margin"

    # 推理路径：labels=None 应返回全类余弦
    with torch.no_grad():
        cos = head(torch.randn(4, 512), None)
    assert cos.shape == (4, 1000)
    assert float(cos.min()) >= -1.001 and float(cos.max()) <= 1.001
    print(f"  推理返回余弦: {tuple(cos.shape)} 范围 "
          f"[{cos.min().item():.3f}, {cos.max():.3f}]")


def t_aug():
    print("== 数据增强 ==")
    img = (np.random.rand(112, 112, 3) * 255).astype(np.uint8)
    aug = FaceAlignAug(seed=0)
    outs = np.stack([aug(img) for _ in range(8)])
    assert outs.shape == (8, 3, 112, 112), outs.shape
    print(f"  训练增强: {outs.shape} 值域[{outs.min():.2f}, {outs.max():.2f}]")
    assert outs.min() >= -1.001 and outs.max() <= 1.001
    v = ValAug()(img)
    assert v.shape == (3, 112, 112)
    print(f"  验证预处理: {v.shape} 值域[{v.min():.2f}, {v.max():.2f}]")


def t_shard():
    print("== 分片读写 ==")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "shard_00000.bin")
        items = [(i % 7, cv2.imencode(".jpg", (np.random.rand(112, 112, 3) * 255)
                                      .astype(np.uint8))[1].tobytes())
                 for i in range(50)]
        with open(p, "wb") as f:
            f.write(SHARD_MAGIC)
            f.write(np.array([len(items)], dtype="<u4").tobytes())
            for lab, jpg in items:
                f.write(np.array([lab, len(jpg)], dtype="<i4").tobytes())
                f.write(jpg)
        back = load_shard(p)
        assert len(back) == len(items)
        assert all(a[0] == b[0] for a, b in zip(items, back))
        assert all(a[1] == b[1] for a, b in zip(items, back))
        print(f"  往返一致: {len(back)} 条")


def t_eval():
    print("== 验证集指标计算 ==")
    rng = np.random.default_rng(0)
    same = rng.random(2000) < 0.5
    sim = np.where(same, rng.normal(0.6, 0.2, 2000),
                   rng.normal(0.0, 0.2, 2000))
    thr, acc = _best_threshold(sim, same)
    acc10, thr10 = eval_10fold(sim, same)
    print(f"  单阈值 best: acc={acc:.4f} @thr={thr:.3f}")
    print(f"  10-fold    : acc={acc10:.4f} @thr={thr10:.3f}")
    assert acc > 0.9 and acc10 > 0.9

    # 完全可分的数据应该接近 100%
    sim2 = np.where(same, 0.9, -0.9)
    acc10b, _ = eval_10fold(sim2, same)
    print(f"  理想可分   : acc={acc10b:.4f}")
    assert acc10b > 0.999


if __name__ == "__main__":
    torch.manual_seed(0)
    t_backbone()
    t_head()
    t_aug()
    t_shard()
    t_eval()
    print("\n全部自检通过 ✓")
