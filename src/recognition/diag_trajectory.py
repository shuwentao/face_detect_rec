"""看特征有效维度随 epoch 的演化轨迹。

区分两种可能：
  A 能力不足 —— 从头到尾都没撑开过
  B 训练塌陷 —— 先撑开，后来被压回去
"""
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/tiny/tmp/face_rec")
from eval_bin import load_bin, extract_features          # noqa: E402
from models.ghostfacenet import build_backbone            # noqa: E402

LFW = "/mnt/data1/val_bins/lfw.bin"
NPair = 3000          # 取前 3000 对（6000 张），够做 PCA
N_IMG = NPair * 2


def eff_dim(f):
    fc = f - f.mean(0)
    ev = np.linalg.svd(fc, compute_uv=False) ** 2
    ev = ev / ev.sum()
    return 1.0 / (ev ** 2).sum(), ev[0]


def run(tag, ckpt_dir, arch, max_ep):
    bins, issame = load_bin(LFW)
    bins, issame = bins[:N_IMG], issame[:NPair]

    print(f"\n=== {tag} ({arch}) ===")
    print(f"{'epoch':>6}{'有效维':>9}{'PC1占比':>9}")
    rows = []
    for ep in range(max_ep + 1):
        ck = os.path.join(ckpt_dir, f"backbone_ep{ep}.pth")
        if not os.path.exists(ck):
            continue
        m = build_backbone(arch, 512, 1.0).cuda().eval()
        sd = torch.load(ck, map_location="cpu")
        m.load_state_dict(sd.get("backbone", sd), strict=False)
        m.cuda()
        f = extract_features(m, bins, batch_size=128)
        ed, pc1 = eff_dim(f)
        rows.append((ep, ed, pc1))
        print(f"{ep:>6}{ed:>9.1f}{pc1:>9.3f}")
        del m, f
        torch.cuda.empty_cache()
    return rows


if __name__ == "__main__":
    run("② GFN 直接训 136k", "checkpoints_gfn136_lr3", "ghostfacenet", 11)
    run("基线 MFNet 136k", "checkpoints_136k", "mobilefacenet", 14)
