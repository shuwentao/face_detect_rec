"""诊断：量化误差在 cls / box / lm 三类通道上的分布。

输出张量通道布局 = cls(1) + box(4) + lm(10) = 15。
"""
import argparse
import glob
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/workspace/mymodel/gate320")
from eval_dump import load_model, letterbox                    # noqa: E402

try:
    from eval_dump import MEAN, STD
except ImportError:
    MEAN, STD = 0.5, 0.5

N = 20
SIZE = 320
paths = sorted(glob.glob("/mnt/data1/face_bench/data/images/300w/*.png"))[:N]


def prep(ps):
    xs = []
    for p in ps:
        im = Image.open(p).convert("RGB")
        canvas, s, left, top = letterbox(im, SIZE)
        x = np.asarray(canvas, dtype=np.float32) / 255.0
        x = (x - MEAN) / STD
        xs.append(torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))))
    return torch.stack(xs)


X = prep(paths)
dev = torch.device("cpu")

models = {
    "float": load_model(argparse.Namespace(jit=None, ckpt="best_sd.pt"), dev),
    "int": load_model(
        argparse.Namespace(jit="qr_qat320_ft/FaceDetector_int.pt", ckpt=None), dev),
}


def outs_of(m):
    with torch.no_grad():
        outs = m(X)
    if isinstance(outs, (list, tuple)) and not outs[0].is_floating_point():
        outs = [o.float() / 16.0 for o in outs]
    return [o.numpy() for o in outs]


res = {}
for name, (m, kind) in models.items():
    res[name] = outs_of(m)
    print("[%s] %s  level 形状=%s dtype=%s"
          % (name, kind, [tuple(o.shape) for o in res[name]],
             res[name][0].dtype))

GROUPS = (("cls", 0, 1), ("box", 1, 5), ("lm", 5, 15))

print()
print("=" * 72)
print("%-6s %-5s %10s %10s %10s %10s"
      % ("level", "通道", "浮点std", "误差std", "误差|mean|", "相对误差"))
print("-" * 72)
for li in range(len(res["float"])):
    f = res["float"][li]
    q = res["int"][li]
    d = q - f
    for gname, a, b in GROUPS:
        fs = f[:, a:b].std()
        ds = d[:, a:b].std()
        dm = np.abs(d[:, a:b]).mean()
        rel = ds / fs if fs > 1e-9 else float("nan")
        print("%-6d %-5s %10.4f %10.4f %10.4f %9.1f%%"
              % (li, gname, fs, ds, dm, rel * 100))
print("=" * 72)

print()
print("=== lm 通道逐个看（level 0）===")
f = res["float"][0]
q = res["int"][0]
for c in range(5, 15):
    fv = f[:, c]
    dv = q[:, c] - fv
    print("  ch%2d  浮点 std %7.3f | 误差 std %7.3f | 误差 mean %+7.3f "
          "| 相对 %5.1f%%" % (c, fv.std(), dv.std(), dv.mean(),
                              dv.std() / max(fv.std(), 1e-9) * 100))
