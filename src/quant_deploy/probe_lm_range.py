"""统计浮点模型各分支输出的真实值域，判断量化量程是否够用。"""
import os
import sys

import numpy as np
import torch
from PIL import Image

BASE = "/workspace/mymodel/gate320"
sys.path.insert(0, BASE)
from common.models.detector import build_detector  # noqa: E402
from eval_dump import letterbox, MEAN, STD  # noqa: E402

IMG = "/mnt/data1/face_bench/data/images/300w"

m = build_detector("mobilenetv3", pretrained=False, width=1.0)
sd = torch.load(BASE + "/best_sd.pt", map_location="cpu")
if isinstance(sd, dict) and "model" in sd:
    sd = sd["model"]
m.load_state_dict(sd, strict=True)
m.eval()

files = sorted(os.listdir(IMG))[:24]
det_mx, lm_mx = 0.0, 0.0
cls_mx, box_mx = 0.0, 0.0
with torch.no_grad():
    for fn in files:
        im = Image.open(os.path.join(IMG, fn)).convert("RGB")
        canvas, s, left, top = letterbox(im, 320)
        x = np.asarray(canvas, dtype=np.float32) / 255.0
        x = (x - MEAN) / STD
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]
        for o in m(x):                       # (1,15,H,W)
            p = o[0]
            cls_mx = max(cls_mx, p[0].abs().max().item())
            box_mx = max(box_mx, p[1:5].abs().max().item())
            det_mx = max(det_mx, p[:5].abs().max().item())
            lm_mx = max(lm_mx, p[5:15].abs().max().item())

print("浮点模型真实值域（24 张图）:")
print("  cls  最大绝对值: %.3f" % cls_mx)
print("  box  最大绝对值: %.3f" % box_mx)
print("  det(cls+box)   : %.3f" % det_mx)
print("  lm             : %.3f" % lm_mx)
print()
for fp, rng in [(4, 7.94), (5, 3.97), (6, 1.98), (7, 0.99)]:
    print("  fix %d -> 量程 ±%.2f | cls %s | lm %s"
          % (fp, rng,
             "够用" if cls_mx <= rng else "削平",
             "够用" if lm_mx <= rng else "削平"))
