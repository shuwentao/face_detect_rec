"""YuNet 检测器训练（WIDER FACE，bbox + score；关键点可选）。

用法（在训练用 TF 环境，非 Vitis docker）：
  python train.py --data_dir /data/WIDER --input_size 320 \
      --batch 16 --epochs 50 --out_dir checkpoints

快速冒烟：
  python train.py --data_dir /data/WIDER --max_records 200 --epochs 1

输出（每 epoch）：
  checkpoints/yunet_ep{N}.weights.h5   仅权重（续训）
  checkpoints/yunet_ep{N}.h5           完整浮点模型（供 quantize_yunet.py 量化）

注意：WIDER FACE 无关键点，landmark 头本脚本不训（mask 掉）。
要关键点请用官方预训练 YuNet ONNX 部署，或换带关键点的数据集。
"""
import os
import argparse
import numpy as np
import tensorflow as tf

from models.yunet import build_yunet, STRIDES, HEAD_CHANNELS, NUM_LANDMARKS


def compute_targets(bboxes, input_size, strides):
    """bboxes: (Nmax,4) 绝对像素 [x1,y1,x2,y2]。返回 list[4] 的 (H/s,W/s,15) 目标。
    采用 FCOS 式中心点分配：anchor 中心落在某 GT 框内 => 正样本，回归到该框。
    """
    B = bboxes.shape[0]
    targets = []
    for s in strides:
        h = w = input_size // s
        # 网格中心
        gy, gx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        cx = (gx + 0.5) * s
        cy = (gy + 0.5) * s
        cx = cx.astype(np.float32); cy = cy.astype(np.float32)
        tgt = np.zeros((B, h, w, HEAD_CHANNELS), np.float32)
        for b in range(B):
            boxes = bboxes[b]
            if boxes.ndim == 1:
                boxes = boxes[None]
            for (x1, y1, x2, y2) in boxes:
                if x2 <= x1 or y2 <= y1:
                    continue
                # 哪些 anchor 中心落在该框内
                m = (cx >= x1) & (cx <= x2) & (cy >= y1) & (cy <= y2)
                tgt[b, m, 0] = 1.0
                tgt[b, m, 1] = (cx[m] - x1) / s   # left  (网格单位)
                tgt[b, m, 2] = (cy[m] - y1) / s   # top
                tgt[b, m, 3] = (x2 - cx[m]) / s   # right
                tgt[b, m, 4] = (y2 - cy[m]) / s   # bottom
                # landmark 通道保持 0（无标注，mask）
        targets.append(tgt)
    return targets


def focal_bce(y_pred, y_true, alpha=0.25, gamma=2.0):
    y_pred = tf.clip_by_value(y_pred, 1e-6, 1 - 1e-6)
    bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    return alpha * tf.pow(1 - pt, gamma) * bce


def smooth_l1(y_pred, y_true):
    diff = tf.abs(y_pred - y_true)
    return tf.where(diff < 1.0, 0.5 * diff * diff, diff - 0.5)


def train_step(model, opt, imgs, bboxes, input_size, strides):
    targets = compute_targets(bboxes.numpy(), input_size, strides)
    with tf.GradientTape() as tape:
        preds = model(imgs, training=True)
        loss_obj = loss_box = 0.0
        for p, t in zip(preds, targets):
            p = tf.cast(p, tf.float32)
            obj = tf.nn.sigmoid(p[..., 0])
            pos = t[..., 0] > 0.5
            loss_obj += tf.reduce_mean(focal_bce(obj, t[..., 0]))
            if tf.reduce_sum(tf.cast(pos, tf.float32)) > 0:
                for c in range(1, 5):
                    loss_box += tf.reduce_mean(
                        smooth_l1(p[..., c][pos], t[..., c][pos]))
        loss = loss_obj + loss_box
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss, loss_obj, loss_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--input_size", type=int, default=320)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--max_records", type=int, default=0,
                    help=">0 时只取前 N 张做冒烟测试")
    args = ap.parse_args()

    from data.widerface import build_widerface_dataset
    ds, n = build_widerface_dataset(args.data_dir, args.input_size, args.batch,
                                    split="train")
    if args.max_records > 0:
        ds = ds.take((args.max_records + args.batch - 1) // args.batch)

    model = build_yunet(args.input_size, width=args.width)
    opt = tf.keras.optimizers.Adam(args.lr)
    os.makedirs(args.out_dir, exist_ok=True)

    for ep in range(args.epochs):
        tot = obj = box = 0.0; steps = 0
        for imgs, bboxes in ds:
            l, lo, lb = train_step(model, opt, imgs, bboxes,
                                  args.input_size, STRIDES)
            tot += float(l); obj += float(lo); box += float(lb); steps += 1
        print(f"[ep{ep+1}] loss={tot/steps:.3f} obj={obj/steps:.3f} "
              f"box={box/steps:.3f}")
        model.save(os.path.join(args.out_dir, f"yunet_ep{ep+1}.weights.h5"))
        # 完整浮点模型（供量化）
        model.save(os.path.join(args.out_dir, f"yunet_ep{ep+1}.h5"))


if __name__ == "__main__":
    main()
