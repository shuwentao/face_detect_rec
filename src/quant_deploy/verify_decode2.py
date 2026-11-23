"""严格交叉校验：按 IoU 把 bench 与 eval_dump 的框配对后再比数值。"""
import json
import os

import numpy as np

BENCH = "/tmp/vidtest.jsonl"
REF = "/workspace/mymodel/gate320/eval_E.jsonl"


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    a1 = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    a2 = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (a1 + a2 - inter)


bench = {}
with open(BENCH) as f:
    for line in f:
        d = json.loads(line)
        bench[os.path.basename(d["name"])] = d

ref = {}
with open(REF) as f:
    for line in f:
        d = json.loads(line)
        ref[os.path.basename(d["image"])] = d["det"]

box_d, lm_d, score_d, ious, cnt_b, cnt_r = [], [], [], [], [], []

for n in [x for x in bench if x in ref]:
    b, r = bench[n], ref[n]
    cnt_b.append(len(b["box"]))
    cnt_r.append(len(r))
    for rb in r:                                  # 遍历 ref 的每个框
        if not b["box"]:
            continue
        cand = list(range(len(b["box"])))
        j = max(cand, key=lambda k: iou(b["box"][k], rb["box"]))
        ov = iou(b["box"][j], rb["box"])
        if ov < 0.9:
            continue                              # 没配上，跳过
        ious.append(ov)
        box_d.append(float(np.abs(np.array(b["box"][j]) - np.array(rb["box"])).max()))
        lm_d.append(float(np.abs(np.array(b["lm"][j]) - np.array(rb["lm"])).mean()))
        score_d.append(abs(b["score"][j] - rb["score"]))

print("配对成功 %d 个目标（IoU>0.9）" % len(box_d))
print("两边框数量：bench 平均 %.1f | ref 平均 %.1f"
      % (np.mean(cnt_b), np.mean(cnt_r)))
print("=" * 60)
if box_d:
    print("平均 IoU      : %.5f" % np.mean(ious))
    print("框坐标最大差  : %.3f px" % np.mean(box_d))
    print("关键点平均差  : %.3f px" % np.mean(lm_d))
    print("score 平均差  : %.5f" % np.mean(score_d))
    print()
    print("判据：IoU 应 >0.999，坐标差应 <1px")
    print("     （残余差异只应来自 cv2 与 PIL 的缩放插值不同）")
    for lo in (0.99, 0.95, 0.9):
        hi = [i for i, v in enumerate(ious) if v > lo]
        if not hi:
            continue
        print("  IoU>%.2f 的子集(%d 个): 框差 %.3f px | 点差 %.3f px"
              % (lo, len(hi),
                 np.mean([box_d[i] for i in hi]),
                 np.mean([lm_d[i] for i in hi])))
else:
    print("没有配上任何框！说明不是精度问题，而是坐标口径不一致。")
    n0 = [x for x in bench if x in ref][0]
    print("\n样例 %s:" % n0)
    print("  bench box[0]:", [round(v, 1) for v in bench[n0]["box"][0]])
    print("  ref   box[0]:", [round(v, 1) for v in ref[n0][0]["box"]])
    print("  bench lm[0][:2]:", [[round(v, 1) for v in p] for p in bench[n0]["lm"][0][:2]])
    print("  ref   lm[0][:2]:", [[round(v, 1) for v in p] for p in ref[n0][0]["lm"][:2]])
