"""测 head 三个分支（cls/box/lm）输出的动态范围。

若三者在 concat 后共用 per-tensor 激活 scale，则范围最小的分支
（很可能是 lm）被范围最大的分支压榨量化精度，表现为关键点 NME 系统性变差。
"""
import json
import os

import numpy as np
import torch
from PIL import Image

import eval_dump as E


def absmax(t):
    return float(t.abs().max())


def main():
    model = E.build_detector("mobilenetv3", pretrained=False, width=1.0)
    sd = torch.load("best.pt", map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd, strict=True)
    model.eval()

    ann = json.load(open(E.DEFAULT_ANN))
    images = ann["images"][:30]

    acc = {"cls": [], "box": [], "lm": []}
    for info in images:
        p = os.path.join(E.DEFAULT_IMG, info["file_name"])
        im = Image.open(p).convert("RGB")
        canvas, s, left, top = E.letterbox(im, 320)
        x = np.asarray(canvas, dtype=np.float32) / 255.0
        x = (x - E.MEAN) / E.STD
        t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0)
        with torch.no_grad():
            outs = model(t)
        if isinstance(outs, torch.Tensor):
            outs = [outs]
        for o in outs:
            o = o[0]                      # (15, H, W)
            acc["cls"].append(absmax(o[0]))
            acc["box"].append(absmax(o[1:5]))
            acc["lm"].append(absmax(o[5:15]))

    print("head 输出动态范围（|max|，30 张图 × 各尺度）")
    for k in ("cls", "box", "lm"):
        v = np.array(acc[k])
        print("  %3s : 均值 %8.3f   中位 %8.3f   p90 %8.3f   max %8.3f"
              % (k, v.mean(), np.median(v), np.percentile(v, 90), v.max()))
    c = np.array(acc["cls"]).mean()
    l = np.array(acc["lm"]).mean()
    print("  → cls/lm 范围比 = %.2f  （lm 仅占 %.1f%% 的动态范围）"
          % (c / max(l, 1e-9), 100 * l / max(c, 1e-9)))


if __name__ == "__main__":
    main()
