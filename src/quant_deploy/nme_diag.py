"""NME 诊断：按「匹配框的 IoU」分桶统计关键点误差。

目的：区分
  (a) 关键点回归本身被量化打崩   -> 各 IoU 桶的 NME 都高
  (b) NME 被检测框偏移污染       -> 高 IoU 桶的 NME 明显更低
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

import eval_dump as E


def iou_matrix(a, b):
    """a:(N,4) b:(M,4) xyxy -> (N,M)"""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def collect(model, dev, images, by_img, size=320, batch=8, conf=0.05, nms=0.45):
    recs = []
    for i in range(0, len(images), batch):
        chunk = images[i:i + batch]
        xs, metas = [], []
        for info in chunk:
            p = os.path.join(E.DEFAULT_IMG, info["file_name"])
            im = Image.open(p).convert("RGB")
            canvas, s, left, top = E.letterbox(im, size)
            x = np.asarray(canvas, dtype=np.float32) / 255.0
            x = (x - E.MEAN) / E.STD
            xs.append(torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))))
            metas.append((s, left, top, info.get("width"), info.get("height")))

        with torch.no_grad():
            outs = model(torch.stack(xs).to(dev))
        if isinstance(outs, (list, tuple)) and not outs[0].is_floating_point():
            outs = [o.float() / 16.0 for o in outs]
        dec = E.decode_preds(outs, conf_thr=conf, nms_thr=nms)

        for (boxes, scores, lms), (s, left, top, w0, h0), info in zip(dec, metas, chunk):
            b = boxes.float().cpu().numpy()
            lm = lms.float().cpu().numpy()
            if len(b):
                b[:, [0, 2]] = (b[:, [0, 2]] - left) / s
                b[:, [1, 3]] = (b[:, [1, 3]] - top) / s
                lm[:, :, 0] = (lm[:, :, 0] - left) / s
                lm[:, :, 1] = (lm[:, :, 1] - top) / s
                if w0 and h0:
                    b[:, 0::2] = np.clip(b[:, 0::2], 0, w0)
                    b[:, 1::2] = np.clip(b[:, 1::2], 0, h0)

            for g in by_img.get(info["id"], []):
                if not g.get("has_lm"):
                    continue
                gt1 = np.array([[g["bbox"][0], g["bbox"][1],
                                 g["bbox"][0] + g["bbox"][2],
                                 g["bbox"][1] + g["bbox"][3]]], np.float32)
                if len(b) == 0:
                    recs.append((0.0, None))
                    continue
                ious = iou_matrix(b, gt1)[:, 0]
                j = int(np.argmax(ious))
                r = E.nme_one(lm[j], g["landmarks"], g["bbox"])
                recs.append((float(ious[j]), None if r is None else r[1]))
    return recs


def stat(recs, thr):
    v = np.array([n for iou, n in recs if n is not None and iou >= thr])
    if len(v) == 0:
        return None
    return len(v), float(v.mean()), float(np.percentile(v, 50)), float(np.percentile(v, 90))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jit")
    ap.add_argument("--ckpt")
    ap.add_argument("--max_images", type=int, default=600)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, kind = E.load_model(a, dev)
    print("[model] %s (%s)" % (a.jit or a.ckpt, kind), flush=True)

    ann = json.load(open(E.DEFAULT_ANN))
    images = ann["images"][:a.max_images]
    by_img = {}
    for x in ann["annotations"]:
        by_img.setdefault(x["image_id"], []).append(x)

    recs = collect(model, dev, images, by_img, batch=a.batch)
    miss = sum(1 for _, n in recs if n is None)
    print("GT 样本 %d，漏检/不可算 %d" % (len(recs), miss), flush=True)

    for thr in (0.0, 0.3, 0.5, 0.7, 0.9):
        s = stat(recs, thr)
        if s:
            print("IoU>=%.1f  n=%4d  NME均值 %.4f  p50 %.4f  p90 %.4f"
                  % (thr, s[0], s[1], s[2], s[3]), flush=True)

    iv = np.array([iou for iou, n in recs if n is not None])
    print("匹配IoU: 均值 %.3f  p10 %.3f  p50 %.3f" %
          (iv.mean(), np.percentile(iv, 10), np.percentile(iv, 50)), flush=True)
    print("IoU>=0.5 占比: %.1f%%" % (100 * (iv >= 0.5).mean()), flush=True)


if __name__ == "__main__":
    main()
