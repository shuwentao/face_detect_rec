"""从板子 dump 的 jsonl 直接算 mAP（复用 eval_dump 的评测口径）。

用法： python score_board.py [最大图片数]
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/workspace/mymodel/gate320")
from common.eval_api import compute_ap, iou_matrix          # noqa: E402
from eval_dump import DEFAULT_ANN, nme_one                   # noqa: E402

PRED = "/workspace/mymodel/gate320/board_dump.jsonl"
MAXN = int(sys.argv[1]) if len(sys.argv) > 1 else None

ann = json.load(open(DEFAULT_ANN))
by_img = {}
for a in ann["annotations"]:
    by_img.setdefault(a["image_id"], []).append(a)

preds = {}
for line in open(PRED):
    r = json.loads(line)
    preds[r["image"]] = r["det"]

images = ann["images"][:MAXN] if MAXN else ann["images"]

aps50, aps_coco, nmes_eye, nmes_box, nfail, ndet = [], [], [], [], 0, 0
missing = 0
used = 0

for info in images:
    fn = info["file_name"]
    det = preds.get(fn)
    if det is None:
        missing += 1
        continue
    used += 1
    if len(det) == 0:
        b = np.zeros((0, 4), np.float32)
        sc = np.zeros((0,), np.float32)
        lm = np.zeros((0, 5, 2), np.float32)
    else:
        b = np.array([d["box"] for d in det], dtype=np.float32)
        sc = np.array([d["score"] for d in det], dtype=np.float32)
        lm = np.array([d["lm"] for d in det], dtype=np.float32)
    ndet += len(b)

    gts = by_img.get(info["id"], [])
    gt_boxes = np.array([[g["bbox"][0], g["bbox"][1],
                          g["bbox"][0] + g["bbox"][2],
                          g["bbox"][1] + g["bbox"][3]] for g in gts],
                        dtype=np.float32).reshape(-1, 4)
    if len(gt_boxes):
        ap50 = compute_ap(gt_boxes, b, sc, 0.5)
        if ap50 >= 0:
            aps50.append(ap50)
        cocos = [compute_ap(gt_boxes, b, sc, t)
                 for t in np.arange(0.5, 0.96, 0.05)]
        cocos = [c for c in cocos if c >= 0]
        if cocos:
            aps_coco.append(float(np.mean(cocos)))

    for g in gts:
        if not g.get("has_lm"):
            continue
        if len(b) == 0:
            nfail += 1
            continue
        gt1 = np.array([[g["bbox"][0], g["bbox"][1],
                         g["bbox"][0] + g["bbox"][2],
                         g["bbox"][1] + g["bbox"][3]]], np.float32)
        j = int(np.argmax(iou_matrix(b, gt1)[:, 0]))
        r = nme_one(lm[j], g["landmarks"], g["bbox"])
        if r is None:
            continue
        nmes_eye.append(r[0])
        nmes_box.append(r[1])
        if r[1] > 0.1:
            nfail += 1

n = max(used, 1)
print("=" * 58)
print("[board] %s" % PRED)
print("图片数        : %d  (缺预测 %d)" % (used, missing))
print("mAP@0.5       : %.4f" % (float(np.mean(aps50)) if aps50 else 0.0))
print("mAP@0.5:0.95  : %.4f" % (float(np.mean(aps_coco)) if aps_coco else 0.0))
if nmes_box:
    print("NME(框归一)    : %.4f   (样本 %d)  <- 与旧结果同一口径"
          % (np.mean(nmes_box), len(nmes_box)))
    print("NME(眼距归一)  : %.4f" % np.mean(nmes_eye))
    print("失效率(NME>.1) : %.1f%%" % (100.0 * nfail / len(nmes_box)))
print("平均检出框数   : %.1f" % (ndet / float(n)))
print("=" * 58)
