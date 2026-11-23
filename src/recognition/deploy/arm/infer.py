"""ARM CPU 上的人脸识别推理（ncnn / onnxruntime，纯 CPU，不依赖 DPU / Vitis AI）。

两件事
--------------------------------------------------------------------------
1. 单张图 → 512 维特征（已 L2 归一化）
2. 两张图 → 余弦相似度 + 是否同一人

输入怎么给
--------------------------------------------------------------------------
* 图已经是**对齐好的 112×112** → 直接给路径
* 图是**原图** → 用 `--kps` 给 5 点，顺序：左眼,右眼,鼻尖,左嘴角,右嘴角，
  脚本按 ArcFace 标准模板做相似变换对齐

用法
--------------------------------------------------------------------------
    # onnxruntime（板子已装 ort 时最省事，直接吃 onnx）
    python3 infer.py --model ../mobilefacenet.onnx --img a.jpg --img2 b.jpg

    # ncnn（需要先从 onnx 转换，ARM 上通常更快）
    python3 infer.py --model face_arm.ncnn.param --img a.jpg --img2 b.jpg

    # 原图 + 5 点（顺序：左眼,右眼,鼻尖,左嘴角,右嘴角）
    python3 infer.py --model ../mobilefacenet.onnx \
        --img "a.jpg:38,52,74,52,56,72,42,92,71,92" \
        --img2 "b.jpg:40,50,75,51,57,73,43,93,72,93"

    # 存特征
    python3 infer.py --model ../mobilefacenet.onnx --img a.jpg --save a.npy
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np

SIZE = 112
# ArcFace 标准 112×112 5 点模板（左眼, 右眼, 鼻尖, 左嘴角, 右嘴角）
TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.6963],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.3655],
], dtype=np.float32)

# 判定阈值：LFW 上最佳阈值附近。实际业务请按自己的 FAR 要求重标定。
THRESHOLD = 0.28


def align(img_rgb: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """5 点相似变换对齐到 112×112。"""
    import cv2
    M, _ = cv2.estimateAffinePartial2D(kps.reshape(5, 2),
                                       TEMPLATE, method=cv2.LMEDS)
    if M is None:
        raise RuntimeError("5 点仿射估计失败，检查关键点顺序/坐标")
    out = cv2.warpAffine(img_rgb, M, (SIZE, SIZE), borderValue=0.0)
    return out


def load_image(path: str, kps: str | None = None) -> np.ndarray:
    """读图 → RGB →（可选对齐）→ (x-127.5)/127.5 → CHW。"""
    try:
        import cv2
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except ImportError:
        from PIL import Image
        rgb = np.array(Image.open(path).convert("RGB"))

    if kps:
        pts = np.array([float(v) for v in kps.split(",")]).reshape(5, 2)
        rgb = align(rgb, pts.astype(np.float32))      # 这一步必须要 cv2
    elif rgb.shape[:2] != (SIZE, SIZE):
        print(f"[warn] {os.path.basename(path)} 尺寸 {rgb.shape[1]}x{rgb.shape[0]}，"
              f"期望已对齐的 {SIZE}x{SIZE}；没给 --kps 就直接 resize 了")
        try:
            import cv2
            rgb = cv2.resize(rgb, (SIZE, SIZE))
        except ImportError:                            # 没有 cv2 就用 PIL 兜底
            from PIL import Image
            rgb = np.array(Image.fromarray(rgb).resize((SIZE, SIZE)))

    x = rgb.astype(np.float32)
    x = (x - 127.5) / 127.5                      # -> [-1, 1]
    return np.ascontiguousarray(x.transpose(2, 0, 1))   # HWC -> CHW


class FaceRec:
    """推理封装：按扩展名自动选后端。

    * `*.param` → ncnn（ARM 上最快，但要先从 onnx 转换）
    * `*.onnx`  → onnxruntime（通用，板子上已装就能直接跑）

    两个后端输出的都是未归一化的 512 维，这里统一做 L2，
    所以返回值范数恒为 1，直接点乘就是余弦相似度。
    """

    def __init__(self, model: str, threads: int = 4):
        ext = os.path.splitext(model)[1].lower()
        self.backend = "ncnn" if ext == ".param" else "ort"
        if self.backend == "ncnn":
            import ncnn
            binf = os.path.splitext(model)[0] + ".bin"
            self.net = ncnn.Net()
            self.net.opt.num_threads = threads
            self.net.load_param(model)
            self.net.load_model(binf)
        else:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
            so.graph_optimization_level = \
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sess = ort.InferenceSession(
                model, so, providers=["CPUExecutionProvider"])
            self.in_name = self.sess.get_inputs()[0].name

    def __call__(self, chw: np.ndarray) -> np.ndarray:
        if self.backend == "ncnn":
            import ncnn
            # NCNN 的 Mat 从 numpy 构造时按 (c, h, w) 解释
            with self.net.create_extractor() as ex:
                ex.input("in0", ncnn.Mat(chw))
                _, out = ex.extract("out0")
            y = np.array(out).reshape(-1).astype(np.float32)
        else:
            # 导出的 onnx 固定 batch=1，CHW -> NCHW
            y = self.sess.run(None, {self.in_name: chw[None]})[0][0]
            y = np.asarray(y, dtype=np.float32).reshape(-1)
        return y / max(float(np.linalg.norm(y)), 1e-12)


def main():
    ap = argparse.ArgumentParser("ARM CPU 人脸识别推理")
    ap.add_argument("--model", default="face_arm.ncnn.param")
    ap.add_argument("--img", required=True, help="路径 或 路径:5点坐标")
    ap.add_argument("--img2", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--save", default=None, help="把第一张图的特征存成 .npy")
    args = ap.parse_args()

    def split(spec):
        if ":" in spec:
            p, k = spec.split(":", 1)
            return p, k
        return spec, None

    p1, k1 = split(args.img)
    rec = FaceRec(args.model, args.threads)
    f1 = rec(load_image(p1, k1))
    print(f"[emb ] {os.path.basename(p1)}  512-d  范数={np.linalg.norm(f1):.4f}  "
          f"前8维={np.round(f1[:8], 4).tolist()}")
    if args.save:
        np.save(args.save, f1)
        print(f"[save] -> {args.save}")

    if args.img2:
        p2, k2 = split(args.img2)
        f2 = rec(load_image(p2, k2))
        print(f"[emb ] {os.path.basename(p2)}  512-d  范数={np.linalg.norm(f2):.4f}")
        sim = float(np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2)))
        same = sim >= args.threshold
        print(f"\n[结果] 余弦相似度 = {sim:.4f}   阈值 = {args.threshold}")
        print(f"       → {'同一人 ✅' if same else '不同人 ❌'}")


if __name__ == "__main__":
    sys.exit(main())
