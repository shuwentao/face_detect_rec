"""YuNet 量化（Vitis-AI 3.0 TF2 docker 内运行）。

关键：DPU 只跑 backbone+head（4 个原始输出），解码/NMS 在 CPU（见 models/yunet.decode_predictions）。
因此量化只需喂输入图做校准，输出无所谓。

为什么重建结构 + load_weights（与 quantize_face.py 同思路）：
  训练环境的 TF 与 docker 内 TF 2.10 版本可能不同，直接 load 整个 .h5 易因
  Keras 版本差异失败；这里在 docker 内用 models/yunet.build_yunet 重建同结构，
  再 load_weights 训练产出的 .weights.h5（纯权重）。

用法（在 Vitis-AI 3.0 docker 的 vitis-ai-tensorflow2 环境内）：
  python quantize_yunet.py \
      --weights ../checkpoints/yunet_ep1.weights.h5 \
      --input_size 320 --calib_dir /data/face_images --n_calib 200
"""
import os
import argparse
import numpy as np
import tensorflow as tf


IMG_SIZE = 320


def make_calib_from_images(calib_dir, n_calib, img_size=IMG_SIZE):
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    paths = []
    for root, _, files in os.walk(calib_dir):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    paths = sorted(paths)[:n_calib]

    def gen():
        for p in paths:
            raw = tf.io.read_file(p)
            img = tf.image.decode_jpeg(raw, channels=3)
            img = tf.image.resize(img, (img_size, img_size))
            img = tf.cast(img, tf.float32)[..., ::-1]  # BGR, [0,255]
            yield img[None]
    return tf.data.Dataset.from_generator(
        gen, output_types=tf.float32,
        output_shapes=(None, img_size, img_size, 3)).batch(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--input_size", type=int, default=IMG_SIZE)
    ap.add_argument("--calib_dir", required=True)
    ap.add_argument("--n_calib", type=int, default=200)
    ap.add_argument("--out", default="yunet_quantized.h5")
    args = ap.parse_args()

    # 1) 在 docker TF 内重建结构
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models.yunet import build_yunet
    model = build_yunet(args.input_size)
    model.load_weights(args.weights)
    print(f"[info] 已载入权重: {args.weights}")

    # 2) 校准集（图像目录）
    calib_ds = make_calib_from_images(args.calib_dir, args.n_calib, args.input_size)

    # 3) 量化
    try:
        from vitis_quantize import VitisQuantizer
    except Exception:
        from vai_q_tensorflow2 import VitisQuantizer
    quantizer = VitisQuantizer(model)
    qmodel = quantizer.quantize_model(calib_dataset=calib_ds, verbose=0)
    qmodel.save(args.out)
    print(f"[ok] 量化模型已存: {args.out}")


if __name__ == "__main__":
    main()
