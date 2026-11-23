"""估算「视频多帧择优」对关键点的收益。

场景：检测器每帧都跑（得到 score），关键点每秒只在 2~3 帧上输出。
若这 2~3 帧是从连续 k 帧里挑出来的（按 score 取最优），质量能提升多少？

做法：bootstrap。把 600 个样本的经验 (score, NME) 联合分布当作
「同一张脸在不同帧上的质量分布」，从中抽 k 次模拟 k 帧，取 score 最高那帧。
注意：这是乐观上界——真实视频里相邻帧质量相关，择优收益会小于此估计。
"""
import json
import sys

import numpy as np

BASE = "/workspace/mymodel/gate320"
sys.path.insert(0, BASE)
from common.eval_api import iou_matrix  # noqa: E402

ANN = "/mnt/data1/face_bench/data/annotations/300w.json"
DUMP = sys.argv[1] if len(sys.argv) > 1 else BASE + "/eval_E.jsonl"


def nme_one(pred_lm, gt_lm, gt_bbox):
    g = np.asarray(gt_lm, dtype=np.float32)
    if g.shape[0] < 5 or np.any(g[:, 2] < 0):
        return None
    gt_pts = g[:, :2]
    d_box = float(np.sqrt(float(gt_bbox[2]) * float(gt_bbox[3])))
    err = float(np.linalg.norm(pred_lm - gt_pts, axis=1).mean())
    if d_box <= 0:
        return None
    return err / d_box


ann = json.load(open(ANN))
img_by_name = {im["file_name"]: im["id"] for im in ann["images"]}
by_img = {}
for a in ann["annotations"]:
    by_img.setdefault(a["image_id"], []).append(a)

sc, nme = [], []
with open(DUMP) as f:
    for line in f:
        d = json.loads(line)
        dets = d["det"]
        for g in by_img.get(img_by_name.get(d["image"]), []):
            if not g.get("has_lm") or not dets:
                continue
            b = np.array([x["box"] for x in dets], np.float32)
            gtb = g["bbox"]
            gt1 = np.array([[gtb[0], gtb[1], gtb[0] + gtb[2],
                             gtb[1] + gtb[3]]], np.float32)
            j = int(np.argmax(iou_matrix(b, gt1)[:, 0]))
            r = nme_one(np.array(dets[j]["lm"], np.float32), g["landmarks"], gtb)
            if r is None:
                continue
            sc.append(float(dets[j]["score"]))
            nme.append(r)

sc = np.array(sc)
nme = np.array(nme)
n = len(nme)
rng = np.random.default_rng(0)
N = 4000

print("样本 %d，单帧基线 NME %.4f / 失效率 %.1f%%"
      % (n, nme.mean(), 100.0 * (nme > 0.1).mean()))
print("浮点上限 NME 0.0749 / 失效率 19.7%")
print()
print("%10s %12s %12s %12s" % ("候选帧数 k", "择优NME", "失效率", "随机取帧NME"))
for k in [1, 2, 3, 5, 8, 10, 15, 20]:
    pick, rnd = [], []
    for _ in range(N):
        idx = rng.integers(0, n, k)
        pick.append(nme[idx[np.argmax(sc[idx])]])     # 按 score 择优
        rnd.append(nme[idx[rng.integers(0, k)]])      # 随机取一帧
    pick = np.array(pick)
    rnd = np.array(rnd)
    print("%10d %12.4f %11.1f%% %12.4f"
          % (k, pick.mean(), 100.0 * (pick > 0.1).mean(), rnd.mean()))

# 择优 + 额外 score 阈值
print()
print("再叠加 score 门控（先按阈值过滤，再在通过者里择优）:")
print("%10s %8s %12s %12s" % ("k", "阈值", "NME", "可用率"))
for k in [5, 10, 15]:
    for t in [0.3, 0.5]:
        got, used = [], []
        for _ in range(N):
            idx = rng.integers(0, n, k)
            ok = idx[sc[idx] >= t]
            if len(ok) == 0:
                continue          # 这一秒没找到合格帧 -> 不输出点
            used.append(1)
            got.append(nme[ok[np.argmax(sc[ok])]])
        if not got:
            continue
        got = np.array(got)
        print("%10d %8.2f %12.4f %11.1f%%"
              % (k, t, got.mean(), 100.0 * np.mean(used)))
