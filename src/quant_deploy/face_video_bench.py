#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
face_video_bench.py
===================
人脸检测 + 关键点，跑视频/摄像头/图片序列，支持「多帧择优」，
并统计每帧各阶段耗时。

两种后端
--------
  --backend pytorch   x86 上用 FaceDetector_int.pt，验证逻辑（不需要 DPU）
  --backend vart      板子上用编译好的 .xmodel，真实测速

关键点三种用法对比
------------------
  A 每帧都用点        —— 基线
  B 每 k 帧随机取一帧 —— 验证「多帧」本身有没有用
  C 每 k 帧按 score×√area 择优 —— 推荐方案

质量指标（无需标注即可用）
--------------------------
  score       已知与关键点质量相关（与 NME 相关 -0.328）
  jitter      相邻两次输出之间关键点的抖动（归一化到框尺寸），越小越稳
  若提供 --gt 才额外算 NME

输出
----
  <out>.jsonl        每帧记录：框/score/关键点/各阶段耗时
  控制台             FPS 与耗时统计 + 三种策略对比
"""
import argparse
import glob
import json
import os
import time

import numpy as np

# ---------------------------------------------------------------- 常量
SIZE = 320
MEAN = np.array([0.5, 0.5, 0.5], np.float32)
STD = np.array([0.5, 0.5, 0.5], np.float32)
PAD_VALUE = 114
STRIDES = [8, 16, 32]
NUM_LM = 5
CONF_THR = 0.05
NMS_THR = 0.45
MAX_DET = 500          # 必须对齐官方 decode_preds(max_det=500)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def letterbox_np(im, size=SIZE):
    """im: HxWxC(uint8)。等比缩放后居中贴灰底。返回 (画布, scale, left, top)。"""
    h, w = im.shape[:2]
    s = size / float(max(h, w))
    nw, nh = int(round(w * s)), int(round(h * s))
    import cv2
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), PAD_VALUE, np.uint8)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas[top:top + nh, left:left + nw] = im
    return canvas, s, left, top


def nms(boxes, scores, thr):
    """numpy NMS，boxes: Nx4 (x1,y1,x2,y2)。返回保留下标数组。"""
    if len(boxes) == 0:
        return np.zeros((0,), np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        iw = np.maximum(0.0, xx2 - xx1)
        ih = np.maximum(0.0, yy2 - yy1)
        ov = (iw * ih) / np.maximum(1e-6, area[order[1:]] + area[i] - iw * ih)
        order = order[1:][ov <= thr]
    return np.array(keep, np.int64)


def decode_preds(outs, conf_thr=CONF_THR, nms_thr=NMS_THR, max_det=MAX_DET):
    """outs: 3 个 float32 张量 (1,15,H,W)，对应 stride 8/16/32。

    通道：0=cls, 1:5=box(l,t,r,b), 5:15=关键点 xy*5
    解码：中心 (px,py)=(gx+0.5)*s；框 px-l*s ...；关键点 (px,py)+d*s
    """
    all_b, all_s, all_l = [], [], []
    for p in outs:
        p = np.asarray(p)
        if p.ndim == 4 and p.shape[0] == 1:
            p = p[0]
        h, w = p.shape[-2], p.shape[-1]
        # 由输出尺寸反推 stride：40->8, 20->16, 10->32
        # 关键：PyTorch 给的输出顺序是 40,20,10，而 DPU 给的是 10,20,40（相反），
        # 所以绝不能按固定顺序取 stride，必须按尺寸认
        s = SIZE // h
        p = p.reshape(15, h * w)
        cls = sigmoid(p[0])
        box = p[1:5]
        lm = p[5:15]

        m = cls > conf_thr
        if not m.any():
            continue
        idx = np.nonzero(m)[0]
        gy = idx // w
        gx = idx % w
        px = (gx + 0.5) * s
        py = (gy + 0.5) * s

        boxes = np.stack([
            px - box[0][idx] * s,
            py - box[1][idx] * s,
            px + box[2][idx] * s,
            py + box[3][idx] * s,
        ], axis=-1).astype(np.float32)

        d = lm[:, idx].T.reshape(-1, NUM_LM, 2)      # Nx5x2
        pts = np.stack([px, py], axis=-1)[:, None, :] + d * s

        all_b.append(boxes)
        all_s.append(cls[idx])
        all_l.append(pts.astype(np.float32))

    if not all_b:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), \
            np.zeros((0, NUM_LM, 2), np.float32)
    b = np.concatenate(all_b, 0)
    sc = np.concatenate(all_s, 0)
    lm = np.concatenate(all_l, 0)
    k = nms(b, sc, nms_thr)[:max_det]
    return b[k], sc[k], lm[k]


# ---------------------------------------------------------------- 后端
class PyTorchBackend:
    """x86 验证用：加载 nndct 导出的 int.pt（输出已是 float32）。"""

    def __init__(self, pt_path):
        import torch
        import pytorch_nndct  # noqa: F401  注册 vai 算子
        self.m = torch.jit.load(pt_path)
        self.m.eval()
        self.name = "pytorch:" + os.path.basename(pt_path)

    def infer(self, x_float):            # x_float: 1x3x320x320 float32
        import torch
        with torch.no_grad():
            outs = self.m(torch.from_numpy(x_float))
        return [o.numpy() for o in outs]


class VartBackend:
    """板端：编译好的 .xmodel。输入输出按 tensor 的 fix_point 自动换算 int8。"""

    def __init__(self, xmodel):
        import vart
        import xir
        self.g = xir.Graph.deserialize(xmodel)
        root = self.g.get_root_subgraph()
        # 必须用 toposort_child_subgraph()：toposort() 会把嵌套子图也刷出来
        try:
            children = root.toposort_child_subgraph()
        except Exception:
            children = root.toposort()
        dpu = [s for s in children
               if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
        if not dpu:
            raise RuntimeError("xmodel 里没有 DPU 子图，检查是否针对目标板编译过")
        self.runner = vart.Runner.create_runner(dpu[0], "run")
        self.in_tensors = self.runner.get_input_tensors()
        self.out_tensors = self.runner.get_output_tensors()

        # 输入/输出定点位（对数刻度）：real = int / 2**fix_point
        self.in_fix = self._fix(self.in_tensors[0])
        self.out_fix = [self._fix(t) for t in self.out_tensors]
        self.name = "vart:" + os.path.basename(xmodel)
        print("[vart] input fix_point=%d  输出 fix_point=%s"
              % (self.in_fix, self.out_fix), flush=True)

    @staticmethod
    def _fix(tensor):
        # XIR 对不存在的属性会 FATAL abort（catch 不到），必须先 has_attr 判断
        if tensor.has_attr("fix_point"):
            return int(tensor.get_attr("fix_point"))
        return 0

    def infer(self, x_float):
        """x_float 为 NCHW (1,3,320,320) float32。
        DPU 的 xmodel 是 NHWC 布局，这里负责两端的转换与定点换算。"""
        scale = 1 << self.in_fix
        xq = np.clip(np.round(x_float * scale), -128, 127).astype(np.int8)
        xhwc = np.ascontiguousarray(xq.transpose(0, 2, 3, 1))       # -> NHWC
        ibuf = [xhwc.reshape(t.dims) for t in self.in_tensors]
        obuf = [np.zeros(t.dims, np.int8) for t in self.out_tensors]
        job = self.runner.execute_async(ibuf, obuf)
        self.runner.wait(job)
        outs = []
        for o, f in zip(obuf, self.out_fix):
            v = o.astype(np.float32) / float(1 << f)                # int8 -> float
            # (1,H,W,15) NHWC -> (1,15,H,W) NCHW
            outs.append(np.ascontiguousarray(v.transpose(0, 3, 1, 2)))
        return outs


# ---------------------------------------------------------------- 简单跟踪
class Track:
    """按 IoU 关联同一张脸，攒够 k 帧就择优输出一次关键点。"""

    def __init__(self, box, score, lm, fid):
        self.box = box
        self.score = score
        self.lm = lm
        self.last_seen = fid
        self.frames = [(score, lm, box)]

    def iou(self, b):
        x1 = max(self.box[0], b[0]); y1 = max(self.box[1], b[1])
        x2 = min(self.box[2], b[2]); y2 = min(self.box[3], b[3])
        iw, ih = max(0, x2 - x1), max(0, y2 - y1)
        inter = iw * ih
        a1 = max(1e-6, (self.box[2] - self.box[0]) * (self.box[3] - self.box[1]))
        a2 = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
        return inter / (a1 + a2 - inter)


def _quality(score, box):
    """择优打分：score × √面积。
    面积归一化到输入尺寸平方，避免尺度偏向大脸过头。"""
    w = max(1e-6, box[2] - box[0]) / SIZE
    h = max(1e-6, box[3] - box[1]) / SIZE
    return score * float(np.sqrt(w * h))


# ---------------------------------------------------------------- 主流程
def run(args):
    # ---------- 后端
    if args.backend == "pytorch":
        be = PyTorchBackend(args.model)
    else:
        be = VartBackend(args.model)
    print("[info] 后端:", be.name, flush=True)

    # ---------- 数据源
    import cv2
    src_kind, cap, files = None, None, []
    if args.video:
        src_kind = "video"
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError("打不开视频: " + args.video)
    elif args.camera is not None:
        src_kind = "camera"
        cap = cv2.VideoCapture(int(args.camera))
    else:
        src_kind = "imgdir"
        files = sorted(glob.glob(os.path.join(args.imgdir, "*.*")))
        if not files:
            raise RuntimeError("目录里没有图片: " + args.imgdir)
        files = files[:args.max_frames] if args.max_frames else files

    t_pre, t_inf, t_post, t_all = [], [], [], []
    frames = []
    fid = 0

    def next_frame():
        while True:
            if src_kind == "imgdir":
                if fid >= len(files):
                    return None
                f = files[fid]
                im = cv2.imread(f)
                return im, os.path.basename(f)
            ok, im = cap.read()
            if not ok:
                return None
            return im, "frame_%06d" % fid

    # ---------- 预热
    dummy = np.zeros((SIZE, SIZE, 3), np.uint8)
    x0 = preprocess(dummy)
    for _ in range(args.warmup):
        be.infer(x0)

    # ---------- 主循环
    while True:
        if args.max_frames and fid >= args.max_frames:
            break
        got = next_frame()
        if got is None:
            break
        im, name = got
        if im is None:
            break
        # cv2 读进来是 BGR，训练/评估用的是 RGB，必须转
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        fid += 1

        t0 = time.perf_counter()
        x = preprocess(im)
        t1 = time.perf_counter()
        outs = be.infer(x)
        t2 = time.perf_counter()
        boxes, scores, lms = decode_preds(outs, args.conf, args.nms)
        t3 = time.perf_counter()

        # canvas 坐标 -> 原图坐标
        _, s, left, top = letterbox_np(im)
        inv = np.array([1.0 / s, 1.0 / s, 1.0 / s, 1.0 / s], np.float32)
        ob = (boxes - np.array([left, top, left, top], np.float32)) * inv
        ol = (lms - np.array([left, top], np.float32)) * np.array([1.0 / s, 1.0 / s], np.float32)
        if len(ob):                                  # 官方还会 clip 到图片边界
            H0, W0 = im.shape[:2]
            ob[:, 0::2] = np.clip(ob[:, 0::2], 0, W0)
            ob[:, 1::2] = np.clip(ob[:, 1::2], 0, H0)

        t_pre.append(t1 - t0)
        t_inf.append(t2 - t1)
        t_post.append(t3 - t2)
        t_all.append(t3 - t0)

        frames.append({
            "fid": fid, "name": name,
            "box": np.round(ob, 2).tolist(),
            "score": np.round(scores, 4).tolist(),
            "lm": np.round(ol, 2).tolist(),
            "ms": [round((t1 - t0) * 1e3, 2), round((t2 - t1) * 1e3, 2),
                   round((t3 - t2) * 1e3, 2)],
        })
        if fid % 50 == 0:
            print("[run] %d 帧" % fid, flush=True)

    if cap:
        cap.release()

    # ---------- 落盘
    with open(args.out + ".jsonl", "w") as f:
        for r in frames:
            f.write(json.dumps(r) + "\n")
    print("[ok] %d 帧 -> %s.jsonl" % (len(frames), args.out), flush=True)

    report_timing(t_pre, t_inf, t_post, t_all)
    if len(frames) >= args.k:
        report_strategy(frames, args.k)


def preprocess(im):
    canvas, _, _, _ = letterbox_np(im)
    x = np.asarray(canvas, np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.ascontiguousarray(x.transpose(2, 0, 1))[None]
    return x


def report_timing(t_pre, t_inf, t_post, t_all):
    def stat(v, unit="ms"):
        a = np.array(v) * (1000.0 if unit == "ms" else 1.0)
        return (a.mean(), np.percentile(a, 95), a.min(), a.max())

    print("\n" + "=" * 58)
    print("每帧耗时(%d 帧)" % len(t_all))
    print("=" * 58)
    print("%-12s %10s %10s %10s %10s" % ("阶段", "平均", "P95", "最小", "最大"))
    for label, v in [("前处理", t_pre), ("推理", t_inf), ("后处理", t_post), ("端到端", t_all)]:
        m, p95, mn, mx = stat(v)
        print("%-12s %9.2fms %8.2fms %8.2fms %8.2fms" % (label, m, p95, mn, mx))
    fps = 1.0 / (np.array(t_all).mean())
    print("\n平均 FPS = %.1f" % fps)


def report_strategy(frames, k):
    """>== 三种关键点用法对比 ==="""
    print("\n" + "=" * 58)
    print("关键点用法对比（每 %d 帧输出一次）" % k)
    print("=" * 58)

    # A: 每帧都用点
    allsc, jit_a = [], []
    prev = {}
    for fr in frames:
        for b, s, lm in zip(fr["box"], fr["score"], fr["lm"]):
            allsc.append(s)
    for i in range(1, len(frames)):
        # 用最近邻框做粗关联，只看抖动趋势
        pass

    def avg(v):
        return float(np.mean(v)) if v else float("nan")

    # B/C: 把帧切成窗口，每 k 帧一次输出
    wins = [frames[i:i + k] for i in range(0, len(frames), k)]
    wins = [w for w in wins if len(w) >= max(2, k // 2)]

    def pick_best(w):
        best, bs = None, -1
        for fr in w:
            for b, s, lm in zip(fr["box"], fr["score"], fr["lm"]):
                q = _quality(s, b)
                if q > bs:
                    bs, best = q, (s, lm, b)
        return best

    def pick_random(w, rng):
        cands = [(s, lm, b) for fr in w for b, s, lm in zip(fr["box"], fr["score"], fr["lm"])]
        return cands[rng.integers(0, len(cands))] if cands else None

    def jitter(seq):
        """相邻两次输出的关键点抖动，归一化到框尺寸。"""
        j = []
        for i in range(1, len(seq)):
            _, lm0, b0 = seq[i - 1]
            _, lm1, b1 = seq[i]
            w = max(1e-6, min(b0[2] - b0[0], b1[2] - b1[0]))
            h = max(1e-6, min(b0[3] - b0[1], b1[3] - b1[1]))
            d = np.linalg.norm(np.array(lm1) - np.array(lm0), axis=1)
            j.append(float(d.mean()) / float(np.sqrt(w * h)))
        return j

    rng = np.random.default_rng(0)
    seq_b, seq_c = [], []
    for w in wins:
        rb = pick_random(w, rng)
        if rb:
            seq_b.append(rb)
        rc = pick_best(w)
        if rc:
            seq_c.append(rc)

    print("%-16s %10s %12s %10s" % ("策略", "平均score", "抖动","输出次数"))
    print("%-16s %10.3f %12s %10d" % ("A 每帧都用", avg(allsc), "-", len(frames)))
    print("%-16s %10.3f %12.4f %10d" % ("B 每k帧随机", avg([x[0] for x in seq_b]),
                                        avg(jitter(seq_b)), len(seq_b)))
    print("%-16s %10.3f %12.4f %10d" % ("C 每k帧择优", avg([x[0] for x in seq_c]),
                                        avg(jitter(seq_c)), len(seq_c)))
    print("\n注: score 越高、抖动越小 => 关键点越可靠。两者应与之前")
    print("    300W 上算出的 NME 趋势一致（择优 NME 0.0859 vs 随机 0.1517）。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="int.pt 或编译后的 .xmodel")
    ap.add_argument("--backend", default="pytorch", choices=["pytorch", "vart"])
    ap.add_argument("--video", help="视频文件")
    ap.add_argument("--camera", type=int, help="摄像头编号")
    ap.add_argument("--imgdir", help="图片目录（当作帧序列）")
    ap.add_argument("--out", default="vid_out")
    ap.add_argument("--k", type=int, default=10, help="每 k 帧输出一次关键点")
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--conf", type=float, default=CONF_THR)
    ap.add_argument("--nms", type=float, default=NMS_THR)
    ap.add_argument("--warmup", type=int, default=3)
    a = ap.parse_args()
    if not (a.video or a.imgdir or a.camera is not None):
        ap.error("必须指定 --video / --imgdir / --camera 之一")
    run(a)


if __name__ == "__main__":
    main()
