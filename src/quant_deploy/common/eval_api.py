"""评估：WIDER Face AP(easy/medium/hard) + 关键点 NME。

检测 AP 的 GT 用官方 wider_face_val 的 easy/medium/hard mat；
关键点 NME 用 lmval 子集(val 集无关键点标注)。
"""
import os
import json
import numpy as np
import cv2
import torch
import scipy.io as sio

from .dataset import letterbox, MEAN, STD
from .models.detector import decode_preds


# --------------------------------------------------------------------------- #
# 推理
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict_image(model, img_bgr, input_size=640, device="cuda",
                  conf_thr=0.05, nms_thr=0.45):
    """返回原图坐标下的 (boxes xyxy, scores, lms(5,2))。"""
    h0, w0 = img_bgr.shape[:2]
    canvas, r, left, top = letterbox(img_bgr, input_size, 1.0)
    x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).unsqueeze(0)
    t = t.to(device)
    preds = model(t)
    boxes, scores, lms = decode_preds(preds, conf_thr=conf_thr,
                                      nms_thr=nms_thr)[0]
    boxes = boxes.float().cpu().numpy()
    scores = scores.float().cpu().numpy()
    lms = lms.float().cpu().numpy()
    if len(boxes):
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - left) / r
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - top) / r
        lms[:, :, 0] = (lms[:, :, 0] - left) / r
        lms[:, :, 1] = (lms[:, :, 1] - top) / r
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w0)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h0)
    return boxes, scores, lms


def preprocess_batch(imgs_bgr, input_size=640):
    """批量预处理 -> tensor + 逆变换参数。"""
    tensors, metas = [], []
    for img in imgs_bgr:
        h0, w0 = img.shape[:2]
        canvas, r, left, top = letterbox(img, input_size, 1.0)
        x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = (x - MEAN) / STD
        tensors.append(torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))))
        metas.append((r, left, top, w0, h0))
    return torch.stack(tensors), metas


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #
def iou_matrix(a, b):
    """a:(N,4) b:(M,4) xyxy -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-6), 0.0).astype(np.float32)


def compute_ap(gt, boxes, scores, iou_thr=0.5):
    """all-point interpolated AP (VOC2010+)。gt/boxes 均 xyxy。"""
    n_gt = len(gt)
    if n_gt == 0:
        return -1.0
    if len(boxes) == 0:
        return 0.0
    order = np.argsort(-scores)
    boxes = boxes[order]
    ious = iou_matrix(boxes, gt)
    matched = np.zeros(n_gt, bool)
    tp = np.zeros(len(boxes))
    fp = np.zeros(len(boxes))
    for i in range(len(boxes)):
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= iou_thr and not matched[j]:
            tp[i] = 1.0
            matched[j] = True
        else:
            fp[i] = 1.0
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    rec = tp_cum / n_gt
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _gt_to_idx(x):
    """easy/medium/hard 的 gt_list 存的是「该图第几个框」的 1-based 索引。"""
    a = np.asarray(x)
    if a.dtype == object:
        parts = []
        for p in a.reshape(-1):
            pa = np.asarray(p)
            if pa.size:
                parts.append(pa.reshape(-1).astype(np.int64))
        return np.concatenate(parts) if parts else np.zeros(0, np.int64)
    return a.reshape(-1).astype(np.int64)


def _bbx_to_arr(x):
    """face_bbx_list 里每张图的全部框 -> (M,4) xywh。"""
    a = np.asarray(x)
    if a.dtype == object:
        for p in a.reshape(-1):
            pa = np.asarray(p)
            if pa.size:
                a = pa
                break
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a[:, :4] if a.size else np.zeros((0, 4), np.float32)


def _to_str(x):
    """mat 的 cell 可能是多层嵌套的 object array，逐层剥到字符串。"""
    v = np.asarray(x).reshape(-1)[0]
    for _ in range(8):
        if isinstance(v, (str, np.str_)):
            return str(v)
        v = np.asarray(v).reshape(-1)[0]
    return str(v)


def load_wider_gt(mat_dir):
    """返回 {level: {rel_path: gt_xyxy(M,4)}}，rel 形如 '0--Parade/xxx.jpg'。"""
    val = sio.loadmat(os.path.join(mat_dir, "wider_face_val.mat"))
    file_list, event_list = val["file_list"], val["event_list"]
    face_bbx_list = val["face_bbx_list"]
    out = {}
    for level in ("easy", "medium", "hard"):
        d = sio.loadmat(os.path.join(mat_dir, f"wider_{level}_val.mat"))
        gt_list = d["gt_list"]
        table = {}
        for e in range(len(event_list)):
            ev = _to_str(event_list[e][0])
            for i in range(len(file_list[e][0])):
                name = _to_str(file_list[e][0][i])
                rel = f"{ev}/{name}.jpg"
                all_boxes = _bbx_to_arr(face_bbx_list[e][0][i])
                idx = _gt_to_idx(gt_list[e][0][i])
                idx = idx[(idx >= 1) & (idx <= len(all_boxes))] - 1
                if len(idx) == 0:
                    table[rel] = np.zeros((0, 4), np.float32)
                    continue
                g = all_boxes[idx]
                x, y, w, h = g[:, 0], g[:, 1], g[:, 2], g[:, 3]
                table[rel] = np.stack([x - 1, y - 1, x - 1 + w, y - 1 + h], 1)
        out[level] = table
    return out


@torch.no_grad()
def eval_detection(model, ann_json, img_dir, mat_dir, device="cuda",
                   input_size=640, conf_thr=0.02, nms_thr=0.45,
                   max_images=None, batch_size=8):
    """在 WIDER val 子集上算 easy/medium/hard AP。"""
    model.eval()
    ann = json.load(open(ann_json))
    gts = load_wider_gt(mat_dir)
    images = ann["images"][:max_images] if max_images else ann["images"]
    preds_all = {}
    for i in range(0, len(images), batch_size):
        chunk = images[i:i + batch_size]
        imgs = [cv2.imread(os.path.join(img_dir, c["file_name"])) for c in chunk]
        imgs = [im if im is not None else np.zeros((640, 640, 3), np.uint8)
                for im in imgs]
        t, metas = preprocess_batch(imgs, input_size)
        p = model(t.to(device))
        dec = decode_preds(p, conf_thr=conf_thr, nms_thr=nms_thr)
        for (boxes, scores, lms), (r, left, top, w0, h0), info in zip(dec, metas, chunk):
            b = boxes.float().cpu().numpy()
            s = scores.float().cpu().numpy()
            lm = lms.float().cpu().numpy()
            if len(b):
                b[:, [0, 2]] = (b[:, [0, 2]] - left) / r
                b[:, [1, 3]] = (b[:, [1, 3]] - top) / r
                lm[:, :, 0] = (lm[:, :, 0] - left) / r
                lm[:, :, 1] = (lm[:, :, 1] - top) / r
                b[:, 0::2] = np.clip(b[:, 0::2], 0, w0)
                b[:, 1::2] = np.clip(b[:, 1::2], 0, h0)
            preds_all[info.get("wider_rel", info["file_name"])] = (b, s, lm)

    res = {}
    for level in ("easy", "medium", "hard"):
        table = gts[level]
        aps = []
        for rel, (b, s, _) in preds_all.items():
            g = table.get(rel)
            if g is None or len(g) == 0:
                continue
            aps.append(compute_ap(g, b, s))
        res[level] = float(np.mean(aps)) if aps else -1.0
        res[f"n_{level}"] = len(aps)
    return res


@torch.no_grad()
def eval_landmark(model, ann_json, img_dir, device="cuda", input_size=640,
                  conf_thr=0.05, nms_thr=0.45, max_images=None):
    """关键点 NME：按 sqrt(gt_w*gt_h) 归一化，统计 mean/median 与 FR@0.1。"""
    model.eval()
    ann = json.load(open(ann_json))
    by_image = {}
    for a in ann["annotations"]:
        if a["has_lm"]:
            by_image.setdefault(a["image_id"], []).append(a)
    images = {im["id"]: im for im in ann["images"]}
    items = [(iid, a) for iid, v in by_image.items() for a in v]
    if max_images:
        items = items[:max_images]

    errs, matched, total = [], 0, 0
    cache = {}
    for iid, a in items:
        info = images[iid]
        if iid not in cache:
            img = cv2.imread(os.path.join(img_dir, info["file_name"]))
            if img is None:
                continue
            boxes, scores, lms = predict_image(model, img, input_size, device,
                                               conf_thr, nms_thr)
            cache[iid] = (boxes, scores, lms)
        boxes, scores, lms = cache[iid]
        gt = np.array(a["bbox"], np.float32).copy()
        gt[2:] += gt[:2]
        gl = np.array(a["landmarks"], np.float32)[:, :2]
        total += 1
        if len(boxes) == 0:
            continue
        ious = iou_matrix(boxes, gt[None])[:, 0]
        order = np.argsort(-scores)
        hit = None
        for j in order:
            if ious[j] >= 0.5:
                hit = j
                break
        if hit is None:
            continue
        matched += 1
        norm = np.sqrt(max((gt[2] - gt[0]) * (gt[3] - gt[1]), 1.0))
        errs.append(np.linalg.norm(lms[hit] - gl, axis=1) / norm)
    if not errs:
        return {"nme_mean": -1.0, "nme_median": -1.0, "fr_0.1": -1.0,
                "match_rate": 0.0, "n_samples": 0, "n_gt": total}
    errs = np.concatenate(errs)
    return {"nme_mean": float(errs.mean()), "nme_median": float(np.median(errs)),
            "fr_0.1": float((errs > 0.1).mean()), "match_rate": matched / max(total, 1),
            "n_samples": int(errs.size), "n_gt": total}
