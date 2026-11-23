"""端到端验证：原图 + 5 个关键点 → 仿射对齐 → 特征提取 → 余弦比对。

这是「PC 浮点阶段」的验收脚本，复刻了板子上要跑的完整链路：
    ① 检测/给定 5 点
    ② 相似变换对齐到 112×112（ArcFace 标准模板）
    ③ 特征提取（onnxruntime，和 DPU 上同一个 ONNX 图）
    ④ L2 归一化 + 余弦相似度 → 同一人 / 不同人

三种输入方式
--------------------------------------------------------------------------
    1) 已经对齐好的 112×112 图（最省事，直接比对）
       python tools/verify_align.py --onnx face.onnx --img a.jpg --img b.jpg

    2) 原图 + 手写 5 点（顺序：左眼 右眼 鼻尖 左嘴角 右嘴角）
       python tools/verify_align.py --onnx face.onnx \
           --img "a.jpg:38,52,74,52,56,72,42,92,71,92" --img b.jpg

    3) 原图 + YuNet 自动检测（和 board 上一致）
       python tools/verify_align.py --onnx face.onnx \
           --detector src/detection/face_detection_yunet_2023mar.onnx \
           --img a.jpg --img b.jpg

输出每张图的对齐图（--save_aligned dir）方便肉眼确认对齐质量。
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import cv2
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cv2.setNumThreads(0)

# ArcFace 标准 112×112 5 点模板（左眼, 右眼, 鼻尖, 左嘴角, 右嘴角）
TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.6963],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.3655],
], dtype=np.float32)


def align_by_kps(img_bgr: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
    """5 点相似变换对齐。kps 顺序与 TEMPLATE 一致。"""
    kps = np.asarray(kps, dtype=np.float32).reshape(5, 2)
    if size != 112:
        tpl = TEMPLATE * (size / 112.0)
    else:
        tpl = TEMPLATE
    M, _ = cv2.estimateAffinePartial2D(kps, tpl, method=cv2.LMEDS)
    if M is None:
        M = cv2.estimateAffinePartial2D(kps, tpl, method=cv2.LMEDS)[0]
    return cv2.warpAffine(img_bgr, M, (size, size),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


class YuNet:
    """极简 YuNet 封装（与 face_detector_yunet/align.py 的解码保持一致）。"""

    def __init__(self, model_path: str, score_thr: float = 0.6):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(model_path, so,
                                         providers=["CPUExecutionProvider"])
        self.w = int(self.sess.get_inputs()[0].shape[3])
        self.h = int(self.sess.get_inputs()[0].shape[2])
        self.score_thr = score_thr

    def detect(self, img_bgr: np.ndarray):
        H, W = self.h, self.w
        blob = cv2.resize(img_bgr, (W, H)).astype(np.float32)
        blob = blob.transpose(2, 0, 1)[None]
        names = [o.name for o in self.sess.get_outputs()]
        raw = dict(zip(names, self.sess.run(names,
                                            {self.sess.get_inputs()[0].name: blob})))
        sx, sy = img_bgr.shape[1] / W, img_bgr.shape[0] / H
        dets = []
        for s in (8, 16, 32):
            cls = raw[f"cls_{s}"][0, :, 0]
            obj = raw[f"obj_{s}"][0, :, 0]
            bx, kp = raw[f"bbox_{s}"][0], raw[f"kps_{s}"][0]
            g = H // s
            for idx in range(g * g):
                score = float(np.sqrt(max(min(cls[idx], 1), 0)
                                      * max(min(obj[idx], 1), 0)))
                if score < self.score_thr:
                    continue
                gy, gx = idx // g, idx % g
                w = np.exp(bx[idx, 2]) * s
                h = np.exp(bx[idx, 3]) * s
                cx = (gx + bx[idx, 0]) * s
                cy = (gy + bx[idx, 1]) * s
                kps = np.array([[(gx + kp[idx, 2 * p]) * s,
                                 (gy + kp[idx, 2 * p + 1]) * s] for p in range(5)],
                               dtype=np.float32) * np.array([sx, sy])
                dets.append({"score": score, "kps": kps,
                             "bbox": ((cx - w / 2) * sx, (cy - h / 2) * sy,
                                      w * sx, h * sy)})
        if not dets:
            return None
        dets.sort(key=lambda d: d["score"], reverse=True)
        return dets[0]

    @staticmethod
    def to_template_order(kps: np.ndarray) -> np.ndarray:
        """YuNet 输出的 5 点顺序不定，按 x 重排成（左眼 右眼 鼻 左嘴角 右嘴角）。"""
        eyes = sorted(kps[:2], key=lambda p: p[0])
        mouth = sorted(kps[3:5], key=lambda p: p[0])
        return np.array([eyes[0], eyes[1], kps[2], mouth[0], mouth[1]],
                        dtype=np.float32)


class Recognizer:
    def __init__(self, onnx_path: str):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(onnx_path, so,
                                         providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        shp = self.sess.get_inputs()[0].shape
        self.size = shp[2] if isinstance(shp[2], int) else 112

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        if aligned_bgr.shape[:2] != (self.size, self.size):
            aligned_bgr = cv2.resize(aligned_bgr, (self.size, self.size))
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = ((rgb - 127.5) / 127.5).transpose(2, 0, 1)[None]
        y = self.sess.run(None, {self.in_name: x})[0][0]
        return y / max(float(np.linalg.norm(y)), 1e-12)      # CPU 端 L2 归一化


def parse_img_arg(v: str):
    """支持 `path` 或 `path:x1,y1,x2,y2,...`。"""
    if ":" in v and v.rsplit(":", 1)[1].count(",") >= 9:
        parts = v.rsplit(":", 1)
        path = parts[0]
        kps = np.array([float(t) for t in parts[1].split(",")],
                       dtype=np.float32).reshape(5, 2)
        return path, kps
    return v, None


def main():
    ap = argparse.ArgumentParser("端到端对齐 + 比对验证")
    ap.add_argument("--onnx", required=True, help="识别模型 ONNX")
    ap.add_argument("--img", action="append", required=True,
                    help="可重复；`路径` 或 `路径:10个坐标`")
    DEFAULT_DETECTOR = os.environ.get("FACE_DETECTOR_ONNX",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "detection", "face_detection_yunet_2023mar.onnx"))
    ap.add_argument("--detector", default=DEFAULT_DETECTOR, help="YuNet ONNX（可选，默认 src/detection 下）")
    ap.add_argument("--score", type=float, default=0.6, help="YuNet 阈值")
    ap.add_argument("--thr", type=float, default=0.30,
                    help="判为同一人的余弦阈值（1:1 场景典型 0.28~0.40）")
    ap.add_argument("--save_aligned", default=None)
    args = ap.parse_args()

    has_manual = any(parse_img_arg(v)[1] is not None for v in args.img)
    need_detect = (not has_manual) and all(
        parse_img_arg(v)[1] is None for v in args.img) and args.detector is None
    if need_detect:
        print("[info] 既没给关键点也没给 --detector："
              "按「输入已是 112×112 对齐图」处理")
    det = YuNet(args.detector, args.score) if args.detector else None
    rec = Recognizer(args.onnx)
    if args.save_aligned:
        os.makedirs(args.save_aligned, exist_ok=True)

    embs, names = [], []
    for v in args.img:
        path, kps = parse_img_arg(v)
        img = cv2.imread(path)
        if img is None:
            raise SystemExit(f"读不到图片: {path}")
        if kps is not None:
            aligned = align_by_kps(img, kps, rec.size)
            how = "手写5点对齐"
        elif det is not None:
            d = det.detect(img)
            if d is None:
                raise SystemExit(f"YuNet 未检测到人脸: {path}")
            kps = YuNet.to_template_order(d["kps"])
            aligned = align_by_kps(img, kps, rec.size)
            how = f"YuNet检测对齐(score={d['score']:.2f})"
        else:
            aligned = cv2.resize(img, (rec.size, rec.size))
            how = "视为已对齐，仅 resize"
        stem = os.path.splitext(os.path.basename(path))[0]
        if args.save_aligned:
            out = os.path.join(args.save_aligned, f"{stem}_aligned.jpg")
            cv2.imwrite(out, aligned)
        e = rec.embed(aligned)
        embs.append(e)
        names.append(f"{stem}[{how}]")
        print(f"[ok] {stem:<20s} {how}  特征维度={e.shape[0]}")

    embs = np.stack(embs)
    print(f"\n两两余弦相似度（阈值 {args.thr}）:")
    print("        " + "".join(f"{i:>9d}" for i in range(len(embs))))
    for i in range(len(embs)):
        row = "".join(f"{float(embs[i] @ embs[j]):9.3f}"
                      for j in range(len(embs)))
        print(f"  [{i}]   {row}")
    if len(embs) == 2:
        s = float(embs[0] @ embs[1])
        verdict = "同一人" if s > args.thr else "不同人"
        print(f"\n结论: cos={s:.3f} → {verdict}")
    print("\n注：阈值只是 1:1 场景的粗参考；真实业务要按 FAR/FRR 曲线定，"
          "可用 eval_bin.py 的 10-fold 结果作依据。")


if __name__ == "__main__":
    main()
