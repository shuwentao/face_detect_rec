"""YuNet 风格人脸检测器（TF/Keras 实现，DPU 友好）。

======== 在 DPU 上部署的只有「骨干+neck+检测头(原始输出)」========
输入 : (N, H, W, 3)  float32，数值范围 [0,255]（BGR，不做归一化，
      与 OpenCV FaceDetectorYN 保持一致；量化校准也用真实图 [0,255]）
输出 : 4 个检测头的原始张量，每个 shape (N, H/s, W/s, 15)，其中 15 = 
       [obj(1) | bbox_distance(4: 左,上,右,下, 单位=输入像素)
        | landmarks(10: 5点 × (dx,dy), 单位=输入像素, 相对 anchor 中心)]
       解码(anchor->绝对框 + NMS)在 CPU 做，见 decode_predictions()。

设计约束（关键）：
  - 全部用 DPU 支持的算子：Conv2D / DepthwiseConv2D / BatchNormalization /
    ReLU6 / Add / Concatenate。不开注意力、不用 Transpose/Reshape/Gather/MatMul，
    这样 backbone+head 能被 vai_q + DPU compiler 直接吃。
  - 注意 OpenCV 官方 YuNet ONNX 里有个自定义 "Integrate" 解码 op（anchor->框），
    那个 **不是** DPU 算子；本实现把它从网络里拿掉，改由 CPU 端
    decode_predictions() 完成，DPU 只跑纯 CNN 部分。

为何 WIDER FACE 训不出关键点：WIDER FACE 只有 bbox 没有 5 点标注。
  - 若只要检测框：在 WIDER FACE 上训 obj+bbox 即可（train.py 默认）。
  - 若要关键点（对正人脸必需）：请用官方预训练 YuNet ONNX（已含关键点）直接部署，
    或在带关键点的数据集(如 WFLW/300W/ADI)上额外训 landmark 头。
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


# 4 个检测层的下采样倍率（与 YuNet 同量级；具体 anchor 数随输入尺寸而定，
# 从零训练时无需与官方 4420 个 anchor 完全一致）
STRIDES = [8, 16, 32, 64]
NUM_LANDMARKS = 5
HEAD_CHANNELS = 1 + 4 + NUM_LANDMARKS * 2  # = 15


def _conv_bn(x, filters, kernel, stride, name, act=True):
    x = layers.Conv2D(filters, kernel, strides=stride, padding="same",
                      use_bias=False, name=name + "_conv")(x)
    x = layers.BatchNormalization(name=name + "_bn")(x)
    if act:
        x = layers.ReLU(max_value=6.0, name=name + "_relu")(x)
    return x


def _sep_bn(x, filters, stride, name):
    x = layers.DepthwiseConv2D(3, strides=stride, padding="same",
                               use_bias=False, name=name + "_dw")(x)
    x = layers.BatchNormalization(name=name + "_dw_bn")(x)
    x = layers.ReLU(max_value=6.0, name=name + "_dw_relu")(x)
    x = layers.Conv2D(filters, 1, padding="same", use_bias=False,
                      name=name + "_pw")(x)
    x = layers.BatchNormalization(name=name + "_pw_bn")(x)
    x = layers.ReLU(max_value=6.0, name=name + "_pw_relu")(x)
    return x


def build_yunet(input_size=320, width=1.0, training=False):
    """构造 YuNet 检测器。返回 Model(inputs, [head8, head16, head32, head64])。

    training: 仅影响 Dropout 等（本轻量版无 dropout，保留参数以兼容调用）。
    width: 通道宽度缩放（0.5~1.0），用于更小的变体。
    """
    c = lambda n: max(8, int(n * width))
    inp = layers.Input(shape=(input_size, input_size, 3), name="input")

    # ---- 骨干（逐级下采样，同时记录 stride 8/16/32/64 的特征图）----
    x = _conv_bn(inp, c(16), 3, 2, "stem")          # /2  -> 160
    x = _sep_bn(x, c(24), 2, "s1")                  # /4  -> 80
    f8 = _sep_bn(x, c(32), 2, "s2")                # /8  -> 40   stride 8
    f16 = _sep_bn(f8, c(48), 2, "s3")              # /16 -> 20   stride 16
    f32 = _sep_bn(f16, c(64), 2, "s4")             # /32 -> 10   stride 32
    f64 = _sep_bn(f32, c(96), 2, "s5")             # /64 -> 5    stride 64

    # ---- 检测头：每个特征图接 1x1 卷积出 15 通道（纯 CNN，DPU 友好）----
    # 加一个轻量 3x3 提升头表达能力，再 1x1 出 15 通道。
    heads = []
    feats = [f8, f16, f32, f64]
    for s, f in zip(STRIDES, feats):
        h = _conv_bn(f, c(64), 3, 1, f"head{s}_a")
        h = layers.Conv2D(HEAD_CHANNELS, 1, padding="same", use_bias=True,
                          name=f"head{s}_out")(h)   # 原始输出，无激活
        heads.append(h)

    model = Model(inp, heads, name="yunet")
    return model


def decode_predictions(heads, input_size=320, score_thresh=0.5,
                       nms_thresh=0.4, strides=STRIDES):
    """CPU 端解码 + NMS（对应官方 ONNX 里被移除的 Integrate 算子）。

    heads: 长度 4 的 list，每个为 (H/s, W/s, 15) 的 numpy 数组（单张图）。
    返回: boxes (M,5)=(x1,y1,x2,y2,score), landmarks (M,5,2)
    """
    boxes, scores, lms = [], [], []
    for s, head in zip(strides, heads):
        h, w = head.shape[:2]
        # 网格中心（输入像素坐标）
        grid_y, grid_x = tf.meshgrid(tf.range(h), tf.range(w), indexing="ij")
        cx = (tf.cast(grid_x, tf.float32) + 0.5) * s
        cy = (tf.cast(grid_y, tf.float32) + 0.5) * s
        cx = cx.numpy().reshape(-1)
        cy = cy.numpy().reshape(-1)
        flat = head.reshape(-1, HEAD_CHANNELS)
        sc = 1.0 / (1.0 + tf.exp(-flat[:, 0]).numpy())      # sigmoid(obj)
        mask = sc >= score_thresh
        if not mask.any():
            continue
        sel = flat[mask]
        scs = sc[mask]
        cx = cx[mask]; cy = cy[mask]
        left, top, right, bot = sel[:, 1] * s, sel[:, 2] * s, \
            sel[:, 3] * s, sel[:, 4] * s
        x1 = cx - left; y1 = cy - top
        x2 = cx + right; y2 = cy + bot
        boxes.append(tf.stack([x1, y1, x2, y2], axis=-1).numpy())
        scores.append(scs)
        # 关键点：相对中心偏移
        lm = sel[:, 5:].reshape(-1, NUM_LANDMARKS, 2)
        lms.append(tf.stack([cx[:, None] + lm[..., 0],
                             cy[:, None] + lm[..., 1]], axis=-1).numpy())

    if not boxes:
        return (tf.zeros((0, 5)).numpy(), tf.zeros((0, NUM_LANDMARKS, 2)).numpy())

    boxes = tf.concat(boxes, axis=0).numpy()
    scores = tf.concat(scores, axis=0).numpy()
    lms = tf.concat(lms, axis=0).numpy()

    # 按分数排序后做 IoU NMS（CPU）
    order = scores.argsort()[::-1]
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ix1, iy1, ix2, iy2 = boxes[i]
        rx1, ry1, rx2, ry2 = boxes[rest].T
        xx1, yy1 = np.maximum(ix1, rx1), np.maximum(iy1, ry1)
        xx2, yy2 = np.minimum(ix2, rx2), np.minimum(iy2, ry2)
        iw, ih = np.maximum(0, xx2 - xx1), np.maximum(0, yy2 - yy1)
        inter = iw * ih
        area_i = (ix2 - ix1) * (iy2 - iy1)
        area_r = (rx2 - rx1) * (ry2 - ry1)
        iou = inter / (area_i + area_r - inter + 1e-6)
        order = rest[iou <= nms_thresh]
    keep = np.array(keep)
    return boxes[keep], lms[keep]


if __name__ == "__main__":
    import numpy as np
    m = build_yunet(320)
    m.summary()
    x = np.random.rand(1, 320, 320, 3).astype(np.float32) * 255.0
    outs = m.predict(x)
    print("head shapes:", [o.shape for o in outs])
