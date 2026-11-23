#!/usr/bin/env python3
"""用我们复现的 baseline (repro_mobilenetv3/ckpt/best.pt) 做单图/整目录人脸检测。

用法:
    # 单张图,结果画到 out.jpg
    python3 infer_baseline.py --img /path/to/face.jpg --out out.jpg

    # 一个目录下所有图片,结果写到 out_dir/
    python3 infer_baseline.py --img_dir /path/to/images --out_dir out/

可选:
    --ckpt 指定其它权重; --conf 置信度阈值(默认0.05); --input_size 默认320
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.models.detector import build_detector, decode_preds  # noqa: E402
from common.eval_api import preprocess_batch  # noqa: E402

DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "repro_mobilenetv3", "ckpt", "best.pt")


def load_model(ckpt, device="cuda"):
    model = build_detector("mobilenetv3", pretrained=False, width=1.0)
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    return model


def detect(model, img_bgr, input_size=320, conf=0.05, device="cuda"):
    xs, metas = preprocess_batch([img_bgr], input_size=input_size)
    with torch.no_grad():
        outs = model(xs.to(device))
    dec = decode_preds(outs, conf_thr=conf, nms_thr=0.45)
    boxes, scores, lms = dec[0]
    if len(boxes) == 0:
        return [], [], []
    b = boxes.float().cpu().numpy()
    lm = lms.float().cpu().numpy()
    s, left, top = metas[0][:3]
    b[:, [0, 2]] = (b[:, [0, 2]] - left) / s
    b[:, [1, 3]] = (b[:, [1, 3]] - top) / s
    lm[:, :, 0] = (lm[:, :, 0] - left) / s
    lm[:, :, 1] = (lm[:, :, 1] - top) / s
    return b, scores.float().cpu().numpy(), lm


def draw(img_bgr, boxes, scores, lms):
    out = img_bgr.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for i, (box, score, lm) in enumerate(zip(boxes, scores, lms)):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, "%.2f" % score, (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for k, (px, py) in enumerate(lm):
            cv2.circle(out, (int(px), int(py)), 2, colors[k], -1)
    return out


def main():
    ap = argparse.ArgumentParser("baseline 检测")
    ap.add_argument("--img", help="单张图片")
    ap.add_argument("--img_dir", help="图片目录(批量)")
    ap.add_argument("--out", help="单图输出路径")
    ap.add_argument("--out_dir", help="批量输出目录")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--input_size", type=int, default=320)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print("[model] 加载", args.ckpt, "->", device)
    model = load_model(args.ckpt, device)

    if args.img:
        img = cv2.imread(args.img)
        if img is None:
            print("[err] 读不到", args.img); return
        boxes, scores, lms = detect(model, img, args.input_size, args.conf, device)
        print("[det] 检出 %d 张脸, 分数: %s" % (len(boxes),
              [round(float(s), 3) for s in scores]))
        out = args.out or "infer_out.jpg"
        cv2.imwrite(out, draw(img, boxes, scores, lms))
        print("[out]", out)
    elif args.img_dir:
        os.makedirs(args.out_dir or "infer_out", exist_ok=True)
        paths = []
        for ext in ("*.jpg", "*.png", "*.jpeg", "*.bmp"):
            paths += glob.glob(os.path.join(args.img_dir, ext))
        print("[dir] %d 张图" % len(paths))
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            boxes, scores, lms = detect(model, img, args.input_size, args.conf, device)
            dst = os.path.join(args.out_dir or "infer_out", os.path.basename(p))
            cv2.imwrite(dst, draw(img, boxes, scores, lms))
        print("[out] ->", args.out_dir or "infer_out")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
