"""通用评估入口：检测 AP(easy/medium/hard) + 关键点 NME + 模型开销 + 报告。"""
import os
import sys
import json
import time
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from common.models.detector import build_detector
from common.eval_api import eval_detection, eval_landmark
from common.utils import (count_params, model_size_mb, FlopCounter,
                          measure_latency, save_json, load_json)
from common.report import plot_curves, save_vis, gen_algo_report

DEFAULT_ROOT = "/mnt/data1/face_bench"
DEFAULT_MAT = ("/home/tiny/github/Ultra-Light-Fast-Generic-Face-Detector-1MB"
               "/widerface_evaluate/ground_truth")


def evaluate(algo, argv=None):
    ap = argparse.ArgumentParser(f"eval {algo}")
    ap.add_argument("--data_root", default=os.path.join(DEFAULT_ROOT, "data"))
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--mat_dir", default=DEFAULT_MAT)
    ap.add_argument("--input_size", type=int, default=640)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--no_vis", action="store_true")
    ap.add_argument("--lm_ann", default=None,
                    help="关键点评估集 json，默认 annotations/lmval.json")
    ap.add_argument("--lm_img", default=None,
                    help="关键点评估集图片目录，默认 images/lmval")
    args = ap.parse_args(argv)

    out_dir = args.out_dir or os.path.join(DEFAULT_ROOT, algo)
    ckpt = args.ckpt or os.path.join(out_dir, "ckpt", "best.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(out_dir, "ckpt", "last.pt")
    model = build_detector(algo, pretrained=False, width=args.width)
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    device = args.device if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"[info] 加载权重 {ckpt}")

    t0 = time.time()
    det = eval_detection(
        model, os.path.join(args.data_root, "annotations", "val.json"),
        os.path.join(args.data_root, "images", "val"), args.mat_dir,
        device=device, input_size=args.input_size,
        max_images=args.max_images or None)
    print(f"[info] 检测评估 {det} ({time.time()-t0:.0f}s)")

    t0 = time.time()
    lm_ann = args.lm_ann or os.path.join(
        args.data_root, "annotations", "lmval.json")
    lm_img = args.lm_img or os.path.join(args.data_root, "images", "lmval")
    lm = eval_landmark(
        model, lm_ann, lm_img,
        device=device, input_size=args.input_size,
        max_images=args.max_images or None)
    print(f"[info] 关键点评估 {lm} ({time.time()-t0:.0f}s)")

    gflops = FlopCounter(model, args.input_size).count(device=device)
    gpu_ms, _ = measure_latency(model, args.input_size, device=device, repeat=30)
    cpu_ms, _ = measure_latency(model.cpu(), args.input_size, device="cpu", repeat=5)
    model.to(device)

    cfg = {}
    cfg_path = os.path.join(out_dir, "config.json")
    if os.path.exists(cfg_path):
        cfg = load_json(cfg_path)

    metrics = {
        "algo": algo, "backbone": algo,
        "params": count_params(model), "params_m": count_params(model) / 1e6,
        "size_mb": model_size_mb(model), "gflops": gflops,
        "latency_gpu_ms": gpu_ms, "latency_cpu_ms": cpu_ms,
        "ap_easy": det["easy"], "ap_medium": det["medium"], "ap_hard": det["hard"],
        "nme_mean": lm["nme_mean"], "nme_median": lm["nme_median"],
        "fr_0.1": lm["fr_0.1"], "match_rate": lm["match_rate"],
        "n_gt_landmark": lm["n_gt"], "lm_eval_set": os.path.basename(lm_ann),
        "input_size": args.input_size, "ckpt": ckpt,
        "epochs": cfg.get("epochs", -1), "batch": cfg.get("batch", -1),
        "train_minutes": cfg.get("train_minutes", -1),
        "eval_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    # 保留已导出的 ONNX 信息
    mpath = os.path.join(out_dir, "metrics.json")
    if os.path.exists(mpath):
        old = load_json(mpath)
        for k in ("onnx_size_mb", "onnx_max_diff", "onnx_path"):
            if k in old:
                metrics[k] = old[k]
    save_json(metrics, mpath)

    history = {}
    hp = os.path.join(out_dir, "history.json")
    if os.path.exists(hp):
        history = load_json(hp)
    curves = plot_curves(history, os.path.join(out_dir, "curves.png")) \
        if history else None
    vis = None
    if not args.no_vis:
        vis = save_vis(model, os.path.join(args.data_root, "annotations", "val.json"),
                       os.path.join(args.data_root, "images", "val"),
                       os.path.join(out_dir, "vis.png"), device=device,
                       input_size=args.input_size)
    gen_algo_report(out_dir, algo, cfg or vars(args), metrics, history,
                    curves, vis)
    print(f"[done] 指标写入 {mpath}，报告 {os.path.join(out_dir,'report.md')}")
    return metrics
