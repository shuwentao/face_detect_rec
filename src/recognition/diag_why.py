"""诊断：为什么 GhostFaceNet 在 136k 上训不动。

对同一批 LFW 图像提特征，比较：
  1. 同人 / 不同人 余弦分布的分离度 d'
  2. 特征 PCA 谱 —— 前 k 个主成分解释的方差占比（维度塌陷检测）
"""
import sys
import numpy as np
import torch

sys.path.insert(0, "/home/tiny/tmp/face_rec")
from eval_bin import load_bin, extract_features          # noqa: E402
from models.ghostfacenet import build_backbone            # noqa: E402

LFW = "/mnt/data1/val_bins/lfw.bin"

EXPS = [
    ("基线 MFNet 136k   98.50%", "checkpoints_136k/backbone_ep11.pth",     "mobilefacenet"),
    ("① GFN 两阶段      97.53%", "checkpoints_gfn136_ft/backbone_ep9.pth", "ghostfacenet"),
    ("② GFN 直接训      85.20%", "checkpoints_gfn136_lr3/backbone_ep11.pth", "ghostfacenet"),
    ("   GFN 仅30k      96.40%", "checkpoints_gfn30k/backbone_ep14.pth",   "ghostfacenet"),
]


def main():
    bins, issame = load_bin(LFW)
    print(f"LFW: {len(bins)} 张 / {len(issame)} 对（同人 {issame.sum()}）\n")

    hdr = f"{'实验':<22}{'同人cos':>9}{'异人cos':>9}{'间距':>8}{'d-prime':>9}{'PC1':>8}{'前10':>8}{'有效维':>8}"
    print(hdr)
    print("-" * len(hdr))

    for name, ckpt, arch in EXPS:
        m = build_backbone(arch, 512, 1.0).cuda().eval()
        sd = torch.load(ckpt, map_location="cpu")
        m.load_state_dict(sd.get("backbone", sd), strict=False)
        m.cuda()

        f = extract_features(m, bins, batch_size=256)
        a, b = f[0::2], f[1::2]
        sim = (a * b).sum(1)

        s = sim[issame]
        d = sim[~issame]
        gap = s.mean() - d.mean()
        dprime = gap / np.sqrt((s.var() + d.var()) / 2)

        # PCA 谱
        fc = f - f.mean(0)
        ev = np.linalg.svd(fc, compute_uv=False) ** 2
        ev = ev / ev.sum()
        pc1 = ev[0]
        top10 = ev[:10].sum()
        # 参与率（有效维度）：越大越均匀
        eff = 1.0 / (ev ** 2).sum()

        print(f"{name:<22}{s.mean():>9.3f}{d.mean():>9.3f}{gap:>8.3f}"
              f"{dprime:>9.2f}{pc1:>8.3f}{top10:>8.3f}{eff:>8.1f}")
        del m, f
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
