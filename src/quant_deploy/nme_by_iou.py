"""按「匹配框的 IoU」分层看 NME：判断关键点误差到底来自框不准还是点本身不准。

口径与 eval_dump.py 完全一致：每个 GT 取 IoU 最大的预测框，用它的 5 点算 NME。
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

bins = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]
stat = {b: [] for b in bins}
nmiss = 0

with open(DUMP) as f:
    for line in f:
        d = json.loads(line)
        dets = d["det"]
        gts = by_img.get(img_by_name.get(d["image"]), [])
        for g in gts:
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
            iou = float(ious[j])
            r = nme_one(np.array(dets[j]["lm"], np.float32),
                        g["landmarks"], gtb)
            if r is None:
                continue
            for (lo, hi) in bins:
                if lo <= iou < hi:
                    stat[(lo, hi)].append(r[1])
                    break

print("dump:", DUMP)
print("%-12s %8s %12s %12s" % ("IoU区间", "样本数", "NME(框归一)", "失效率>0.1"))
for (lo, hi) in bins:
    v = np.array(stat[(lo, hi)])
    if len(v) == 0:
        print("%-12s %8d %12s %12s" % ("%.2f-%.2f" % (lo, hi), 0, "-", "-"))
        continue
    print("%-12s %8d %12.4f %11.1f%%"
          % ("%.2f-%.2f" % (lo, hi), len(v), v.mean(), 100.0 * (v > 0.1).mean()))
allv = np.array([x for b in bins for x in stat[b]])
print("%-12s %8d %12.4f %11.1f%%"
      % ("合计", len(allv), allv.mean(), 100.0 * (allv > 0.1).mean()))
print("漏检(该GT无框) 数:", nmiss)
