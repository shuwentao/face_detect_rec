"""在 CPU（x86 调试机 / ARM 板）上给人脸识别模型测速。

支持的后端
--------------------------------------------------------------------------
* `ncnn`  —— `*.ncnn.param`，ARM CPU 上最快，推荐
* `ort`   —— `*.onnx`，onnxruntime，通用、无需转换

测的是**纯推理**（batch=1，112×112×3 输入 → 512 维特征），
不含解码/对齐/归一化等前处理。

用法
--------------------------------------------------------------------------
    python deploy/arm/bench.py --model deploy/arm/face_arm.ncnn.param
    python deploy/arm/bench.py --model face.onnx --threads 1 2 4
    python deploy/arm/bench.py --model face_arm.ncnn.param --runs 300
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import platform
import numpy as np

SIZE = 112


def cpu_info():
    try:
        with open("/proc/cpuinfo") as f:
            txt = f.read()
        model = ""
        for line in txt.splitlines():
            if line.strip().startswith("Hardware") or \
               line.strip().startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
        cores = txt.count("processor\t:")
        return model, cores
    except Exception:
        return platform.machine(), os.cpu_count()


def make_ort(path, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    x = np.random.randn(1, 3, SIZE, SIZE).astype(np.float32)

    def run():
        return sess.run(None, {name: x})[0]
    return run


def make_ncnn(path, threads):
    import ncnn
    binf = os.path.splitext(path)[0] + ".bin"
    net = ncnn.Net()
    net.opt.num_threads = threads
    net.load_param(path)
    net.load_model(binf)
    x = np.random.randn(1, 3, SIZE, SIZE).astype(np.float32)
    # NCNN 的 Mat 从 numpy 构造时按 (c, h, w) 解释，直接给 CHW
    xh = np.ascontiguousarray(x[0])

    def run():
        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(xh))
            _, out = ex.extract("out0")
        return np.array(out).reshape(-1)
    return run


def bench(run, warmup, runs):
    for _ in range(warmup):
        run()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        run()
        ts.append((time.perf_counter() - t0) * 1000.0)
    a = np.array(ts)
    return a.mean(), np.percentile(a, 50), np.percentile(a, 95), a.min()


def main():
    ap = argparse.ArgumentParser("CPU 推理测速")
    ap.add_argument("--model", required=True)
    ap.add_argument("--threads", type=int, nargs="+", default=None,
                    help="线程数列表，默认 [1, 2, 4]（按实际核数裁剪）")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--runs", type=int, default=100)
    args = ap.parse_args()

    path = args.model
    ext = os.path.splitext(path)[1].lower()
    backend = "ncnn" if ext == ".param" or path.endswith(".param") else "ort"
    maker = make_ncnn if backend == "ncnn" else make_ort

    model, cores = cpu_info()
    n_max = cores or 4
    # 自动档只测 1/2/4 线程，外加一个「核数」档。核数很多时（36 线程那种）
    # ncnn 的线程池开销会反过来拖垮小模型，所以封顶 8。
    threads = args.threads or [t for t in (1, 2, 4) if t <= n_max] or [1]
    cap = min(n_max, 8)
    if max(threads) < cap:
        threads = sorted(set(threads + [cap]))

    size_mb = os.path.getsize(path) / 1e6
    if backend == "ncnn":
        b = os.path.splitext(path)[0] + ".bin"
        size_mb += os.path.getsize(b) / 1e6
    print(f"[cpu ] {model}  逻辑核数={cores}  {platform.machine()}")
    print(f"[load] 后端={backend}  模型={os.path.basename(path)}  {size_mb:.2f} MB")

    print(f"\n{'线程':>4}  {'均值 ms':>9}  {'p50 ms':>8}  {'p95 ms':>8}  "
          f"{'最快 ms':>8}  {'FPS':>7}")
    print("-" * 56)
    for t in threads:
        run = maker(path, t)
        mean, p50, p95, mn = bench(run, args.warmup, args.runs)
        print(f"{t:>4}  {mean:>9.2f}  {p50:>8.2f}  {p95:>8.2f}  "
              f"{mn:>8.2f}  {1000.0/mean:>7.1f}")
    print("\n注：纯推理耗时，不含图像解码/人脸对齐/前处理。")


if __name__ == "__main__":
    sys.exit(main())
