"""交叉校验：bench 脚本的解码结果 vs eval_dump.py 官方口径。"""
import json
import os

import numpy as np

BENCH = "/tmp/vidtest.jsonl"
REF = "/workspace/mymodel/gate320/eval_E.jsonl"

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

print("bench 帧数 %d | eval_E 图片数 %d" % (len(bench), len(ref)))
print("=" * 62)

common = [n for n in bench if n in ref][:6]
if not common:
    print("没有同名图片，检查字段")
    print("bench 样例名:", list(bench)[:3])
    print("ref   样例名:", list(ref)[:3])

box_d, score_d, lm_d = [], [], []
for n in common:
    b, r = bench[n], ref[n]
    if not b["box"] or not r:
        print("%-28s 一方为空 (bench %d / ref %d)" % (n, len(b["box"]), len(r)))
        continue
    # 取 score 最高的框做对比
    ib = int(np.argmax(b["score"]))
    ir = int(np.argmax([x["score"] for x in r]))
    bb = np.array(b["box"][ib], np.float32)
    br = np.array(r[ir]["box"], np.float32)
    lb = np.array(b["lm"][ib], np.float32)
    lr = np.array(r[ir]["lm"], np.float32)
    sb, sr = b["score"][ib], r[ir]["score"]

    bd = float(np.abs(bb - br).max())
    ld = float(np.abs(lb - lr).mean())
    box_d.append(bd)
    score_d.append(abs(sb - sr))
    lm_d.append(ld)
    print("%-30s 框差 %6.2f px | score %.4f vs %.4f (%+.4f) | 点均差 %5.2f px"
          % (n[:30], bd, sb, sr, sb - sr, ld))

if box_d:
    print("=" * 62)
    print("平均：框差 %.2f px | score 差 %.5f | 关键点差 %.2f px"
          % (np.mean(box_d), np.mean(score_d), np.mean(lm_d)))
    print("判定：框差应 < 2px、关键点差 < 2px（差异只应来自 cv2 与 PIL 缩放插值不同）")
