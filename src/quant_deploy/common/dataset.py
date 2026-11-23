"""统一人脸检测数据集：letterbox + 增强，输出 bbox(xyxy) 与 5 点关键点。

所有算法共用，保证「同一份数据、同一套预处理」。
关键点顺序：0 左眼 1 右眼 2 鼻尖 3 左嘴角 4 右嘴角（retinaface 标准）。
"""
import os
import json
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
# 水平翻转时关键点的对称关系
FLIP_PAIRS = [(0, 1), (3, 4)]
STRIDES = (8, 16, 32)
# WIDER Face 以小脸为主(中位边长约 23px)，阈值按 640 输入上的像素边长划分
SCALE_RANGES = ((0, 32), (32, 96), (96, 1e9))


def letterbox(img, size=640, scale=1.0, pad_value=114):
    """保持宽高比缩放(长边 -> size*scale)后居中对齐到 size x size。

    scale<=1: 缩小后 pad；scale>1: 放大后中心裁剪。
    返回 (canvas, ratio, pad_left, pad_top)，pad 为负表示裁剪偏移，
    逆变换一律用 (x - pad) / ratio。
    """
    h, w = img.shape[:2]
    target = int(round(size * scale))
    r = min(target / h, target / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if nh > size or nw > size:                      # 放大 -> 中心裁剪(再 pad 补齐)
        t0, l0 = max(0, (nh - size) // 2), max(0, (nw - size) // 2)
        crop = img[t0:t0 + min(nh, size), l0:l0 + min(nw, size)]
        canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
        top, left = (size - crop.shape[0]) // 2, (size - crop.shape[1]) // 2
        canvas[top:top + crop.shape[0], left:left + crop.shape[1]] = crop
        return canvas, r, left - l0, top - t0
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = img
    return canvas, r, left, top


def color_jitter(img, rng):
    """亮度/对比度/饱和度抖动，输入 BGR uint8。"""
    if rng.random() < 0.5:
        alpha = 1.0 + rng.uniform(-0.3, 0.3)
        img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        beta = rng.uniform(-30, 30)
        img = np.clip(img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + rng.uniform(-0.3, 0.3)), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img


def build_targets(boxes, lms, input_size=640, strides=STRIDES,
                  ranges=SCALE_RANGES):
    """FCOS 式中心点分配 + 分尺度(按框边长) + 小框优先覆盖。

    返回 list[stride] of (cls[H,W], box[H,W,4], lm[H,W,10], lm_mask[H,W])
    box 目标 = (l,t,r,b)/stride，lm 目标 = (dx,dy)/stride，均相对该点中心。
    """
    tgts = []
    for s, (lo, hi) in zip(strides, ranges):
        hw = input_size // s
        gy, gx = np.mgrid[0:hw, 0:hw]
        cx = (gx + 0.5) * s
        cy = (gy + 0.5) * s
        cls = np.zeros((hw, hw), np.float32)
        box = np.zeros((hw, hw, 4), np.float32)
        lm = np.zeros((hw, hw, 10), np.float32)
        lmm = np.zeros((hw, hw), np.float32)
        amap = np.full((hw, hw), np.inf, np.float32)   # 已分配框面积(小框优先)

        for (x1, y1, x2, y2), l5 in zip(boxes, lms):
            side = max(x2 - x1, y2 - y1)
            if side < lo or side >= hi:
                continue
            gx1, gx2 = max(0, int(x1 // s)), min(hw - 1, int(x2 // s))
            gy1, gy2 = max(0, int(y1 // s)), min(hw - 1, int(y2 // s))
            if gx2 < gx1 or gy2 < gy1:
                continue
            sx = cx[gy1:gy2 + 1, gx1:gx2 + 1]
            sy = cy[gy1:gy2 + 1, gx1:gx2 + 1]
            m = (sx >= x1) & (sx <= x2) & (sy >= y1) & (sy <= y2)
            area = (x2 - x1) * (y2 - y1)
            upd = m & (amap[gy1:gy2 + 1, gx1:gx2 + 1] > area)
            ys, xs = np.where(upd)
            if ys.size == 0:
                if m.any():
                    continue          # 该区域已被更小的框占用，保持小框优先
                # 极小脸可能没有任何格心落入框内：退化为框中心所在的那一格，
                # 保证每个 GT 至少有一个正样本
                ccx = min(hw - 1, max(0, int((x1 + x2) / 2 // s)))
                ccy = min(hw - 1, max(0, int((y1 + y2) / 2 // s)))
                if amap[ccy, ccx] <= area:
                    continue
                ys = np.array([ccy - gy1])
                xs = np.array([ccx - gx1])
                if ys[0] < 0 or xs[0] < 0 or \
                        ys[0] > (gy2 - gy1) or xs[0] > (gx2 - gx1):
                    continue
            gyy, gxx = ys + gy1, xs + gx1
            amap[gyy, gxx] = area
            cls[gyy, gxx] = 1.0
            box[gyy, gxx, 0] = (sx[ys, xs] - x1) / s
            box[gyy, gxx, 1] = (sy[ys, xs] - y1) / s
            box[gyy, gxx, 2] = (x2 - sx[ys, xs]) / s
            box[gyy, gxx, 3] = (y2 - sy[ys, xs]) / s
            if np.all(l5[:, 2] >= 0):
                lmm[gyy, gxx] = 1.0
                # (M,1,2) 中心 - (1,5,2) 关键点 -> 每个正样本点到 5 个点的偏移
                d = l5[None, :, :2] \
                    - np.stack([sx[ys, xs], sy[ys, xs]], axis=1)[:, None, :]
                lm[gyy, gxx, 0::2] = d[:, :, 0] / s
                lm[gyy, gxx, 1::2] = d[:, :, 1] / s
        tgts.append((cls, box, lm, lmm))
    return tgts


class FaceDataset(Dataset):
    def __init__(self, ann_json, img_dir, input_size=640, train=True,
                 jitter=0.5, flip=0.5, max_boxes=300, seed=42, scale_max=1.2):
        self.ann = json.load(open(ann_json))
        self.img_dir = img_dir
        self.input_size = input_size
        self.train = train
        self.jitter = jitter
        self.flip = flip
        self.max_boxes = max_boxes
        # 训练时随机缩放的上限；>1 表示会放大后中心裁剪(制造"大脸"样本)
        self.scale_max = scale_max
        self.rng = random.Random(seed)

        self.by_image = {}
        for a in self.ann["annotations"]:
            self.by_image.setdefault(a["image_id"], []).append(a)
        self.images = self.ann["images"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        info = self.images[idx]
        path = os.path.join(info.get("_img_dir", self.img_dir),
                            info["file_name"])
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((self.input_size, self.input_size, 3), np.uint8)
        h0, w0 = img.shape[:2]

        anns = self.by_image.get(info["id"], [])
        boxes = np.array([a["bbox"] for a in anns], dtype=np.float32).reshape(-1, 4)
        lms = np.array([a["landmarks"] for a in anns], dtype=np.float32).reshape(-1, 5, 3) \
            if anns else np.zeros((0, 5, 3), np.float32)
        boxes[:, 2:] += boxes[:, :2]      # xywh -> xyxy

        scale = 1.0
        if self.train:
            if self.rng.random() < self.jitter:
                scale = self.rng.uniform(0.5, self.scale_max)
            if self.rng.random() < self.flip:
                img = img[:, ::-1].copy()
                x1 = w0 - boxes[:, 2]
                x2 = w0 - boxes[:, 0]
                boxes[:, 0], boxes[:, 2] = x1, x2
                lms[:, :, 0] = w0 - lms[:, :, 0]
                for i, j in FLIP_PAIRS:
                    lms[:, [i, j], :] = lms[:, [j, i], :]
            img = color_jitter(img, self.rng)

        canvas, r, left, top = letterbox(img, self.input_size, scale)
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * r + left
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * r + top
        lms[:, :, 0] = lms[:, :, 0] * r + left
        lms[:, :, 1] = lms[:, :, 1] * r + top

        # 丢弃被裁掉/退化的框
        keep = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
        boxes, lms = boxes[keep], lms[keep]
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, self.input_size)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, self.input_size)
        # 放大裁剪后可能有框被切到只剩边缘，二次过滤
        keep = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
        boxes, lms = boxes[keep], lms[keep]
        if len(boxes) > self.max_boxes:
            boxes, lms = boxes[:self.max_boxes], lms[:self.max_boxes]

        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        canvas = (canvas - MEAN) / STD
        tensor = torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1)))

        meta = {
            "index": idx,
            "file_name": info["file_name"],
            "wider_rel": info.get("wider_rel", info["file_name"]),
            "orig_size": (h0, w0),
            "ratio": r,
            "pad": (left, top),
        }
        tgts = build_targets(boxes, lms, self.input_size)
        tgts = [tuple(torch.from_numpy(np.ascontiguousarray(t)) for t in tt)
                for tt in tgts]
        return tensor, tgts, meta


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    metas = [b[2] for b in batch]
    tgts = []
    for si in range(len(batch[0][1])):
        tgts.append(tuple(
            torch.stack([b[1][si][k] for b in batch]) for k in range(4)))
    return imgs, tgts, metas
