"""门控策略分析：用预测框的 score 做阈值，看关键点质量的取舍。

判定方式与 eval_dump.py 一致：每个 GT 取 IoU 最大的预测框。
阈值 t 之下：该 GT 的最佳框若 score < t 则「拒绝输出关键点」。
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
    d_eye = float(np.linalg.norm(gt_pts[0] - gt_pts[1]))
    d_box = float(np.sqrt(float(gt_bbox[2]) * float(gt_bbox[3])))
    err = float(np.linalg.norm(pred_lm - gt_pts, axis=1).mean())
    if d_eye <= 0 or d_box <= 0:
        return None
    return err / d_eye, err / d_box


ann = json.load(open(ANN))
img_by_name = {im["file_name"]: im["id"] for im in ann["images"]}
by_img = {}
for a in ann["annotations"]:
    by_img.setdefault(a["image_id"], []).append(a)

recs = []          # 每个 GT 一条：最佳框的 score / iou / nme
nmiss = 0
with open(DUMP) as f:
    for line in f:
        d = json.loads(line)
        dets = d["det"]
        for g in by_img.get(img_by_name.get(d["image"]), []):
            if not g.get("has_lm"):
                continue
            if not dets:
                nmiss += 1
                continue
            b = np.array([x["box"] for x in dets], np.float32)
            gtb = g["bbox"]
            gt1 = np.array([[gtb[0], gtb[1], gtb[0] + gtb[2],
                             gtb[1] + gtb[3]]], np.float32)
            ious = iou_matrix(b, gt1)[:, 0]
            j = int(np.argmax(ious))
            r = nme_one(np.array(dets[j]["lm"], np.float32),
                        g["landmarks"], gtb)
            if r is None:
                continue
            recs.append({"score": float(dets[j]["score"]),
                         "iou": float(ious[j]), "nme": r[1]})

n = len(recs)
sc = np.array([r["score"] for r in recs])
iou = np.array([r["iou"] for r in recs])
nme = np.array([r["nme"] for r in recs])

print("dump: %s   GT 样本 %d（漏检 %d）" % (DUMP, n, nmiss))
print("score 与 IoU 的 Pearson 相关: %.3f" % np.corrcoef(sc, iou)[0, 1])
print("score 与 NME 的 Pearson 相关: %.3f" % np.corrcoef(sc, nme)[0, 1])
print()
print("%8s %8s %9s %11s %11s %10s"
      % ("阈值", "保留数", "覆盖率", "NME", "失效率>0.1", "平均IoU"))
for t in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    m = sc >= t
    if m.sum() == 0:
        continue
    print("%8.2f %8d %8.1f%% %11.4f %10.1f%% %10.3f"
          % (t, m.sum(), 100.0 * m.sum() / n, nme[m].mean(),
             100.0 * (nme[m] > 0.1).mean(), iou[m].mean()))
print()
print("基线（全输出）: NME %.4f  失效率 %.1f%%"
      % (nme.mean(), 100.0 * (nme > 0.1).mean()))
print("浮点上限参考  : NME 0.0749  失效率 19.7%")
