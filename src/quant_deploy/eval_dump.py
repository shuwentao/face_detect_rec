"""在 300W 上算 mAP / NME，并 dump 每张图的框（供与板端对拍）。

与训练、板端完全同口径的前处理：
    letterbox(320, 114 灰底居中) -> /255 -> (x-0.5)/0.5 -> CHW

用法（vitis-ai-pytorch 环境）：
    # 浮点基线（应得 mAP≈0.96）
    python eval_dump.py --ckpt best_sd.pt --max_images 100
    # 量化模型
    python eval_dump.py --jit qr_qat320a/FaceDetector_int.pt --max_images 100 \
                        --dump host_int.jsonl
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common.models.detector import build_detector, decode_preds   # noqa: E402
from common.eval_api import compute_ap, iou_matrix                # noqa: E402

MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
PAD_VALUE = 114.0

DEFAULT_ANN = "/mnt/data1/face_bench/data/annotations/300w.json"
DEFAULT_IMG = "/mnt/data1/face_bench/data/images/300w"


def letterbox(im, size):
    """等比缩放 + 居中贴 114 灰底。返回 (画布, scale, left, top)。"""
    w, h = im.size
    s = size / float(max(h, w))
    nw, nh = int(round(w * s)), int(round(h * s))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size),
                       (int(PAD_VALUE),) * 3)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas.paste(im, (left, top))
    return canvas, s, left, top


def load_model(args, dev):
    if args.jit:
        # TorchScript 里含 vai::fix_neuron 等自定义 op，必须先注册内核再 load
        try:
            import pytorch_nndct.nn as _nndct_nn
            _nndct_nn.load_kernels()
        except Exception as e:                       # pragma: no cover
            print("[warn] load_kernels 失败: %r" % (e,))
        m = torch.jit.load(args.jit, map_location=dev)
        m.eval()
        return m, "jit"
    model = build_detector("mobilenetv3", pretrained=False, width=1.0)
    sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd, strict=True)
    model.eval().to(dev)
    return model, "float"


def nme_one(pred_lm, gt_lm, gt_bbox):
    """返回 (眼距归一化 NME, 人脸框归一化 NME)；不可算返回 None。

    人脸框归一化用 sqrt(w*h)（gt_bbox 是 [x,y,w,h]），与既有结果同一口径。
    """
    g = np.asarray(gt_lm, dtype=np.float32)
    if g.shape[0] < 5 or np.any(g[:, 2] < 0):
        return None
    gt_pts = g[:, :2]
    d_eye = float(np.linalg.norm(gt_pts[0] - gt_pts[1]))
    d_box = float(np.sqrt(float(gt_bbox[2]) * float(gt_bbox[3])))
    err = float(np.linalg.norm(pred_lm - gt_pts, axis=1).mean())
    if d_eye <= 0 or d_box <= 0:
        return None
    return err / d_eye, err / d_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", help="浮点权重 .pt")
    ap.add_argument("--jit", help="TorchScript 模型（量化的 .pt）")
    ap.add_argument("--ann", default=DEFAULT_ANN)
    ap.add_argument("--img_dir", default=DEFAULT_IMG)
    ap.add_argument("--max_images", type=int, default=100)
    ap.add_argument("--input_size", type=int, default=320)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--dump", help="dump 每张贴的框到 jsonl")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, kind = load_model(args, dev)
    print("[model] %s (%s)" % (args.jit or args.ckpt, kind))

    ann = json.load(open(args.ann))
    images = ann["images"][:args.max_images]
    by_img = {}
    for a in ann["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    dump = open(args.dump, "w") if args.dump else None
    aps50, aps_coco, nmes_eye, nmes_box, nfail, ndet = [], [], [], [], 0, 0

    size = args.input_size
    for i in range(0, len(images), args.batch):
        chunk = images[i:i + args.batch]
        xs, metas = [], []
        for info in chunk:
            p = os.path.join(args.img_dir, info["file_name"])
            im = Image.open(p).convert("RGB")
            canvas, s, left, top = letterbox(im, size)
            x = np.asarray(canvas, dtype=np.float32) / 255.0
            x = (x - MEAN) / STD
            xs.append(torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))))
            metas.append((s, left, top, info.get("width"), info.get("height")))

        with torch.no_grad():
            outs = model(torch.stack(xs).to(dev))
        if isinstance(outs, (list, tuple)) and not outs[0].is_floating_point():
            outs = [o.float() / 16.0 for o in outs]      # int8 输出需反量化
        dec = decode_preds(outs, conf_thr=args.conf, nms_thr=args.nms)

        for (boxes, scores, lms), (s, left, top, w0, h0), info in zip(dec, metas, chunk):
            b = boxes.float().cpu().numpy()
            sc = scores.float().cpu().numpy()
            lm = lms.float().cpu().numpy()
            if len(b):
                b[:, [0, 2]] = (b[:, [0, 2]] - left) / s
                b[:, [1, 3]] = (b[:, [1, 3]] - top) / s
                lm[:, :, 0] = (lm[:, :, 0] - left) / s
                lm[:, :, 1] = (lm[:, :, 1] - top) / s
                if w0 and h0:
                    b[:, 0::2] = np.clip(b[:, 0::2], 0, w0)
                    b[:, 1::2] = np.clip(b[:, 1::2], 0, h0)
            ndet += len(b)

            if dump is not None:
                dump.write(json.dumps({
                    "image": info["file_name"],
                    "det": [{"score": float(sc[k]),
                             "box": [float(v) for v in b[k]],
                             "lm": [[float(v) for v in lm[k][j]] for j in range(5)]}
                            for k in range(len(b))]}) + "\n")

            gts = by_img.get(info["id"], [])
            gt_boxes = np.array([[g["bbox"][0], g["bbox"][1],
                                  g["bbox"][0] + g["bbox"][2],
                                  g["bbox"][1] + g["bbox"][3]]
                                 for g in gts], dtype=np.float32).reshape(-1, 4)
            if len(gt_boxes):
                ap50 = compute_ap(gt_boxes, b, sc, 0.5)
                if ap50 >= 0:
                    aps50.append(ap50)
                cocos = [compute_ap(gt_boxes, b, sc, t)
                         for t in np.arange(0.5, 0.96, 0.05)]
                cocos = [c for c in cocos if c >= 0]
                if cocos:
                    aps_coco.append(float(np.mean(cocos)))

            for g in gts:
                if not g.get("has_lm"):
                    continue
                if len(b) == 0:
                    nfail += 1
                    continue
                gt1 = np.array([[g["bbox"][0], g["bbox"][1],
                                 g["bbox"][0] + g["bbox"][2],
                                 g["bbox"][1] + g["bbox"][3]]], np.float32)
                # iou_matrix 返回 (N_pred, N_gt)，取第 0 列才是「所有预测框 vs 这个 GT」
                j = int(np.argmax(iou_matrix(b, gt1)[:, 0]))
                r = nme_one(lm[j], g["landmarks"], g["bbox"])
                if r is None:
                    continue
                nmes_eye.append(r[0])
                nmes_box.append(r[1])
                if r[1] > 0.1:          # 失效判据沿用框归一化口径
                    nfail += 1

    if dump:
        dump.close()

    n = max(len(aps50), 1)
    print("=" * 58)
    print("图片数        : %d" % len(images))
    print("mAP@0.5       : %.4f" % (float(np.mean(aps50)) if aps50 else 0.0))
    print("mAP@0.5:0.95  : %.4f" % (float(np.mean(aps_coco)) if aps_coco else 0.0))
    if nmes_eye:
        print("NME(框归一)    : %.4f   (样本 %d)  <- 与旧结果同一口径"
              % (np.mean(nmes_box), len(nmes_box)))
        print("NME(眼距归一)  : %.4f" % np.mean(nmes_eye))
        print("失效率(NME>.1) : %.1f%%" % (100.0 * nfail / len(nmes_box)))
    print("平均检出框数   : %.1f" % (ndet / max(len(images), 1)))
    print("=" * 58)


if __name__ == "__main__":
    main()
