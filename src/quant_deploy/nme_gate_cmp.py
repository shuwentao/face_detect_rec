"""比较不同门控信号在「相同覆盖率」下的关键点质量。

信号候选：
  score     : 检测置信度
  area      : 框面积（大脸通常更准）
  score*area: 联合
  geo       : 关键点几何自检（无需 GT）：眼距占比 + 两眼水平度 + 点的合理次序
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


def geo_score(lm, box):
    """几何自检分，越大越可疑（1=正常，>1 偏异常）。无需 GT。"""
    x1, y1, x2, y2 = box
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    le, re, nose, lm_l, lm_r = lm[0], lm[1], lm[2], lm[3], lm[4]
    d_eye = float(np.linalg.norm(le - re)) + 1e-6
    bad = 0.0
    # 1) 眼距应占脸宽合理比例
    r = d_eye / w
    bad += max(0.0, 0.15 - r) * 6 + max(0.0, r - 0.75) * 2
    # 2) 两眼连线应大致水平
    bad += abs(float(np.arctan2(re[1] - le[1], re[0] - le[0]))) / 0.6
    # 3) 鼻子应在两眼中点下方附近
    mid = (le + re) / 2
    bad += max(0.0, (mid[1] - nose[1]) / h) * 3      # 鼻子跑到眼睛上方
    bad += max(0.0, (nose[1] - lm_l[1]) / h) * 3     # 鼻子跑到嘴角下方
    # 4) 左右嘴角应分居鼻子两侧
    bad += max(0.0, -(lm_l[0] - nose[0]) / w) + max(0.0, (lm_r[0] - nose[0]) / w)
    return bad


ann = json.load(open(ANN))
img_by_name = {im["file_name"]: im["id"] for im in ann["images"]}
by_img = {}
for a in ann["annotations"]:
    by_img.setdefault(a["image_id"], []).append(a)

recs = []
with open(DUMP) as f:
    for line in f:
        d = json.loads(line)
        dets = d["det"]
        for g in by_img.get(img_by_name.get(d["image"]), []):
            if not g.get("has_lm"):
                continue
            if not dets:
                continue
            b = np.array([x["box"] for x in dets], np.float32)
            gtb = g["bbox"]
            gt1 = np.array([[gtb[0], gtb[1], gtb[0] + gtb[2],
                             gtb[1] + gtb[3]]], np.float32)
            j = int(np.argmax(iou_matrix(b, gt1)[:, 0]))
            lm = np.array(dets[j]["lm"], np.float32)
            r = nme_one(lm, g["landmarks"], gtb)
            if r is None:
                continue
            box = b[j]
            area = max(box[2] - box[0], 1) * max(box[3] - box[1], 1)
            recs.append({"score": float(dets[j]["score"]), "area": area,
                         "geo": geo_score(lm, box), "nme": r})

n = len(recs)
nme = np.array([r["nme"] for r in recs])
sc = np.array([r["score"] for r in recs])
ar = np.array([r["area"] for r in recs])
geo = np.array([r["geo"] for r in recs])

print("样本 %d" % n)
print("相关性（与 NME）: score %.3f | log(area) %.3f | geo %.3f"
      % (np.corrcoef(sc, nme)[0, 1],
         np.corrcoef(np.log(ar + 1), nme)[0, 1],
         np.corrcoef(geo, nme)[0, 1]))
print()
sigs = {
    "score": sc,
    "area": ar,
    "score*area": sc * np.sqrt(ar),
    "geo(越小越好)": -geo,
    "score-geo": sc - 0.3 * geo,
}
for cov in [0.8, 0.6, 0.5, 0.4, 0.3]:
    k = int(round(n * cov))
    row = []
    for name, v in sigs.items():
        idx = np.argsort(-v)[:k]
        row.append("%s=%.4f" % (name, nme[idx].mean()))
    print("覆盖率 %3d%% (保留 %d): %s" % (cov * 100, k, "  ".join(row)))
print()
print("基线 NME %.4f | 浮点上限 0.0749" % nme.mean())
