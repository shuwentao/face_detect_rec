#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐层诊断量化 ONNX：把每个 QuantizeLinear 输出的 int8 张量 dump 出来做统计。

判据：
  * sat  = |x| >= 120 的占比  -> 截顶（阈值定小了）
  * zero = x == 0 的占比      -> 阈值定大了，整层被抹平
  * 有效电平数                -> 远低于 256 说明没用满 8 bit

用法（宿主机即可，不需要 docker）：
  python3 diag_int_onnx.py quantize_result/FaceDetector_int.onnx \
      --image /mnt/data1/face_bench/data/images/300w/xxx.jpg
"""

import argparse
import collections
import os

import numpy as np
import onnx
from onnx import ValueInfoProto

PAD_VALUE = 114


def letterbox(path, size):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = size / float(max(h, w))
    nw, nh = int(round(w * s)), int(round(h * s))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (PAD_VALUE,) * 3)
    canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def preprocess(path, size):
    x = np.asarray(letterbox(path, size), dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return np.ascontiguousarray(x.transpose(2, 0, 1))[None]


def main():
    ap = argparse.ArgumentParser("量化 ONNX 逐层饱和诊断")
    ap.add_argument("model")
    ap.add_argument("--image", default=None, help="用真实图；不给则用 [-1,1] 随机")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    m = onnx.load(args.model)
    g = m.graph
    in_name = g.input[0].name
    shape = [(d.dim_value or d.dim_param) for d in g.input[0].type.tensor_type.shape.dim]
    print("[model] %s" % args.model)
    print("[model] input = %s %s" % (in_name, shape))

    # 把所有中间张量挂到 graph.output 上，才能从 ORT 取出来
    init = set(i.name for i in g.initializer)
    have = set(o.name for o in g.output) | set(in_name)
    q_nodes = [n for n in g.node if n.op_type == "QuantizeLinear"]
    wanted = []
    for n in g.node:
        for o in n.output:
            if o and o not in init and o not in have and o not in wanted:
                wanted.append(o)
    for o in wanted:
        g.output.append(ValueInfoProto(name=o))

    sess = onnxruntime_session(m)
    if args.image:
        x = preprocess(args.image, args.size)
    else:
        rng = np.random.default_rng(0)
        x = rng.uniform(-1, 1, [1, 3, args.size, args.size]).astype(np.float32)
    outs = sess.run(None, {in_name: x})
    values = dict(zip([o.name for o in g.output][len(have) - 1:], outs[1:]))
    # 保险起见：按名字重新对齐
    names = [o.name for o in g.output]
    values = dict(zip(names, outs))

    print("[run] 取到 %d 个张量，其中 QuantizeLinear 输出 %d 个\n"
          % (len(values), len(q_nodes)))

    rows = []
    for n in q_nodes:
        v = values.get(n.output[0])
        if v is None or not np.issubdtype(v.dtype, np.integer):
            continue
        a = np.abs(v.astype(np.int32))
        n_elem = a.size
        sat = float((a >= 120).sum()) / n_elem
        zero = float((v == 0).sum()) / n_elem
        lvl = int(np.unique(v).size)
        rows.append((n.output[0], tuple(v.shape), int(a.max()), sat, zero, lvl))

    print("=== 截顶最严重的 %d 个张量（sat 高 = 阈值定小了）===" % args.top)
    print("  %-58s %-18s %6s %8s %8s %6s"
          % ("tensor", "shape", "max|i|", "sat", "zero", "lvl"))
    for r in sorted(rows, key=lambda r: -r[3])[:args.top]:
        print("  %-58s %-18s %6d %8.4f %8.4f %6d"
              % (r[0][:58], str(r[1])[:18], r[2], r[3], r[4], r[5]))

    print("\n=== 被抹平最严重的 %d 个张量（zero 高 = 阈值定大了）===" % args.top)
    for r in sorted(rows, key=lambda r: -r[4])[:args.top]:
        print("  %-58s %-18s %6d %8.4f %8.4f %6d"
              % (r[0][:58], str(r[1])[:18], r[2], r[3], r[4], r[5]))

    sat_hi = [r for r in rows if r[3] > 0.01]
    zero_hi = [r for r in rows if r[4] > 0.9]
    print("\n[小结] 共 %d 个量化张量" % len(rows))
    print("  sat>1%% 的: %d" % len(sat_hi))
    print("  zero>90%% 的: %d" % len(zero_hi))
    c = collections.Counter(r[5] for r in rows)
    small = sum(v for k, v in c.items() if k <= 8)
    print("  有效电平数<=8 的: %d（这些层基本被压成了几档）" % small)

    # ---- 逐张量量化误差：Q 的输入(真值) vs DQ 的输出(量化重建) ----
    producer = {}
    for n in g.node:
        for o in n.output:
            producer.setdefault(o, n)
    init_map = {i.name: onnx.numpy_helper.to_array(i) for i in g.input if False}
    init_map = {}
    for i in g.initializer:
        try:
            init_map[i.name] = onnx.numpy_helper.to_array(i)
        except Exception:
            pass

    def fetch(name):
        if name in values:
            return values[name]
        return init_map.get(name)

    print("\n=== 逐张量量化误差（Q 前真值 vs DQ 后重建）===")
    print("  %-40s %-16s %8s %8s %8s"
          % ("tensor(Q 的输入)", "shape", "pos", "rel-L2", "cos"))
    errs = []
    for n in g.node:
        if n.op_type != "DequantizeLinear":
            continue
        int_name = n.input[0]
        q = producer.get(int_name)
        if q is None or q.op_type != "QuantizeLinear":
            continue
        a = fetch(q.input[0])          # 真值（float）
        b = fetch(n.output[0])         # 量化重建（float）
        if a is None or b is None:
            continue
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)
        na = np.linalg.norm(a)
        if na < 1e-30:
            continue
        rel = float(np.linalg.norm(a - b) / na)
        cos = float(a @ b / (na * np.linalg.norm(b) + 1e-30))
        sc = init_map.get(n.input[1])
        pos = -np.log2(float(np.asarray(sc).reshape(-1)[0])) if sc is not None else float("nan")
        errs.append((q.input[0], tuple(np.asarray(a).shape), pos, rel, cos))

    for r in sorted(errs, key=lambda r: -r[3])[:args.top]:
        print("  %-40s %-16s %8.2f %8.4f %8.4f"
              % (r[0][:40], str(r[1])[:16], r[2], r[3], r[4]))

    bad = [r for r in errs if r[3] > 0.2]
    print("\n[小结] %d 个张量可计算误差，rel-L2>0.2 的有 %d 个" % (len(errs), len(bad)))
    print("       单层 rel-L2 应 < 0.02；超过 0.1 就是这一层在毁模型")


def onnxruntime_session(m):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(m.SerializeToString(), so,
                                providers=["CPUExecutionProvider"])


if __name__ == "__main__":
    main()
