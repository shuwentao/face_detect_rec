"""导出 ONNX（仅 head 输出，NMS 在部署侧做）并用 onnxruntime 验证数值一致性。"""
import os
import sys
import json
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from common.models.detector import build_detector
from common.utils import save_json, load_json

DEFAULT_ROOT = "/mnt/data1/face_bench"


def export(algo, argv=None):
    ap = argparse.ArgumentParser(f"export {algo}")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--input_size", type=int, default=640)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--onnx_path", default=None)
    args = ap.parse_args(argv)

    out_dir = args.out_dir or os.path.join(DEFAULT_ROOT, algo)
    onnx_path = args.onnx_path or os.path.join(out_dir, f"{algo}.onnx")
    ckpt = args.ckpt or os.path.join(out_dir, "ckpt", "best.pt")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(out_dir, "ckpt", "last.pt")

    model = build_detector(algo, pretrained=False, width=args.width)
    sd = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(sd["model"] if "model" in sd else sd)
    model.eval().cpu()

    x = torch.zeros(1, 3, args.input_size, args.input_size)
    with torch.no_grad():
        ref = [p.numpy() for p in model(x)]
    kwargs = dict(
        input_names=["input"],
        output_names=[f"pred_stride_{s}" for s in (8, 16, 32)],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      **{f"pred_stride_{s}": {0: "batch"} for s in (8, 16, 32)}},
        opset_version=args.opset, do_constant_folding=True)
    try:
        # 旧(TorchScript)导出器：单文件、opset 可低，便于 OpenCV DNN 加载
        torch.onnx.export(model, x, onnx_path, dynamo=False, **kwargs)
    except Exception as e:                                   # pragma: no cover
        print(f"[warn] 旧导出器失败({e})，回退到 dynamo 导出器")
        torch.onnx.export(model, x, onnx_path, **kwargs)
    print(f"[done] ONNX 导出: {onnx_path} "
          f"({os.path.getsize(onnx_path)/1024/1024:.2f} MB)")

    max_diff = -1.0
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path,
                                    providers=["CPUExecutionProvider"])
        out = sess.run(None, {"input": x.numpy()})
        max_diff = max(float(np.abs(a - b).max()) for a, b in zip(ref, out))
        print(f"[info] onnxruntime 校验：输出层数 {len(out)}，"
              f"最大绝对误差 {max_diff:.2e}")
        print("[info] 输出形状: " + ", ".join(str(o.shape) for o in out))
    except Exception as e:
        print(f"[warn] onnxruntime 校验失败: {e}")

    mpath = os.path.join(out_dir, "metrics.json")
    metrics = load_json(mpath) if os.path.exists(mpath) else {"algo": algo}
    metrics.update({"onnx_path": onnx_path,
                    "onnx_size_mb": os.path.getsize(onnx_path) / 1024 / 1024,
                    "onnx_max_diff": max_diff,
                    "onnx_opset": args.opset})
    save_json(metrics, mpath)
    print(f"[done] metrics 更新 {mpath}")
    return onnx_path
