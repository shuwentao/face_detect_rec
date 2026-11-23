"""检查量化模型 head 输出是否被 ±8 量程截断。

量化信息里输出层是 (8,4) 定点格式 -> 还原刻度 1/16 -> int8 量程 ±128/16 = ±8.0
若浮点输出超出该范围，量化后会削平，box/cls 的大值首当其冲。
"""
import json
import os

import numpy as np
import torch
from PIL import Image

import eval_dump as E

LIMIT = 8.0


def main():
    try:
        import pytorch_nndct.nn as n
        n.load_kernels()
    except Exception:
        pass
    m = torch.jit.load("qr_ft_lmA/FaceDetector_int.pt", map_location="cpu")
    m.eval()

    ann = json.load(open(E.DEFAULT_ANN))
    images = ann["images"][:30]

    acc = {"cls": [], "box": [], "lm": []}
    clip = {"cls": 0, "box": 0, "lm": 0}
    tot = 0
    for info in images:
        p = os.path.join(E.DEFAULT_IMG, info["file_name"])
        im = Image.open(p).convert("RGB")
        canvas, s, left, top = E.letterbox(im, 320)
        x = np.asarray(canvas, dtype=np.float32) / 255.0
        x = (x - E.MEAN) / E.STD
        inp = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0)
        with torch.no_grad():
            outs = m(inp)
        if isinstance(outs, torch.Tensor):
            outs = [outs]
        for o in outs:
            o = o[0]
            tot += 1
            for k, sl in (("cls", slice(0, 1)), ("box", slice(1, 5)), ("lm", slice(5, 15))):
                v = float(o[sl].abs().max())
                acc[k].append(v)
                if v >= LIMIT - 0.01:
                    clip[k] += 1

    print("量化模型 head 输出（30 张图 × %d 个尺度）" % tot)
    for k in ("cls", "box", "lm"):
        v = np.array(acc[k])
        print("  %3s : absmax 均值 %7.3f   最大 %7.3f   触及±8 边界的尺度数 %d/%d"
              % (k, v.mean(), v.max(), clip[k], tot))


if __name__ == "__main__":
    main()
