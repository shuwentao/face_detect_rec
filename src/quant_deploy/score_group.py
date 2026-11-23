"""按 indoor / outdoor 子集分别统计板上 dump 的精度。"""
import json
import sys

import numpy as np

sys.path.insert(0, "/workspace/mymodel/gate320")
from common.eval_api import compute_ap, iou_matrix          # noqa: E402
from eval_dump import DEFAULT_ANN, nme_one                   # noqa: E402

PRED = "/workspace/mymodel/gate320/board_dump.jsonl"

ann = json.load(open(DEFAULT_ANN))
by_img = {}
for a in ann["annotations"]:
    by_img.setdefault(a["image_id"], []).append(a)

preds = {}
for line in open(PRED):
    r = json.loads(line)
    preds[r["image"]] = r["det"]


def new_bucket():
    return dict(aps50=[], aps_coco=[], nme=[], nfail=0, ndet=0, used=0)


buckets = {"indoor": new_bucket(), "outdoor": new_bucket(), "ALL": new_bucket()}


def key_of(fn):
    if "indoor" in fn:
        return "indoor"
    if "outdoor" in fn:
        return "outdoor"
    return "other"


for info in ann["images"]:
    fn = info["file_name"]
    det = preds.get(fn)
    if det is None:
        continue
    k = key_of(fn)
    if k not in buckets:
        buckets[k] = new_bucket()
    B1 = buckets[k]
    B1["used"] += 1
    buckets["ALL"]["used"] += 1

    if len(det) == 0:
        b = np.zeros((0, 4), np.float32)
        sc = np.zeros((0,), np.float32)
        lm = np.zeros((0, 5, 2), np.float32)
    else:
        b = np.array([d["box"] for d in det], dtype=np.float32)
        sc = np.array([d["score"] for d in det], dtype=np.float32)
        lm = np.array([d["lm"] for d in det], dtype=np.float32)
    for tgt in (B1, buckets["ALL"]):
        tgt["ndet"] += len(b)

    gts = by_img.get(info["id"], [])
    gt_boxes = np.array([[g["bbox"][0], g["bbox"][1],
                          g["bbox"][0] + g["bbox"][2],
                          g["bbox"][1] + g["bbox"][3]] for g in gts],
                        dtype=np.float32).reshape(-1, 4)
    if len(gt_boxes):
        for tgt in (B1, buckets["ALL"]):
            ap50 = compute_ap(gt_boxes, b, sc, 0.5)
            if ap50 >= 0:
                tgt["aps50"].append(ap50)
            cocos = [compute_ap(gt_boxes, b, sc, t)
                     for t in np.arange(0.5, 0.96, 0.05)]
            cocos = [c for c in cocos if c >= 0]
            if cocos:
                tgt["aps_coco"].append(float(np.mean(cocos)))

    for g in gts:
        if not g.get("has_lm"):
            continue
        if len(b) == 0:
            for tgt in (B1, buckets["ALL"]):
                tgt["nfail"] += 1
            continue
        gt1 = np.array([[g["bbox"][0], g["bbox"][1],
                         g["bbox"][0] + g["bbox"][2],
                         g["bbox"][1] + g["bbox"][3]]], np.float32)
        j = int(np.argmax(iou_matrix(b, gt1)[:, 0]))
        r = nme_one(lm[j], g["landmarks"], g["bbox"])
        if r is None:
            continue
        for tgt in (B1, buckets["ALL"]):
            tgt["nme"].append(r[1])
            if r[1] > 0.1:
                tgt["nfail"] += 1

print("=" * 66)
print("%-9s %5s %9s %9s %8s %7s %8s"
      % ("子集", "图片", "mAP@.5", "mAP@.5:.95", "NME", "失效%", "框数/图"))
print("-" * 66)
for name in ("indoor", "outdoor", "ALL"):
    B1 = buckets[name]
    if not B1["used"]:
        continue
    n = float(B1["used"])
    print("%-9s %5d %9.4f %9.4f %8.4f %6.1f%% %8.1f"
          % (name, B1["used"],
             float(np.mean(B1["aps50"])) if B1["aps50"] else 0.0,
             float(np.mean(B1["aps_coco"])) if B1["aps_coco"] else 0.0,
             float(np.mean(B1["nme"])) if B1["nme"] else 0.0,
             100.0 * B1["nfail"] / max(len(B1["nme"]), 1),
             B1["ndet"] / n))
print("=" * 66)
