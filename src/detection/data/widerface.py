"""WIDER FACE 数据加载（仅 bbox；该数据集无 5 点关键点标注）。

数据集结构（官方）：
  $data_dir/
    WIDER_train/images/0--Parade/xxx.jpg ...
    WIDER_train/wider_face_train_bbx_gt.txt
    WIDER_val/images/...
    WIDER_val/wider_face_val_bbx_gt.txt

wider_face_train_bbx_gt.txt 格式：
  第一行: 图片相对路径（相对 WIDER_train/images/）
  第二行: 人脸数 n
  接着 n 行: x1 y1 w h blur expression illumination invalid occlusion pose
  （我们取 x1,y1,w,h；w<=0 或 h<=0 的框丢弃）

重要：WIDER FACE 没有关键点，所以本 loader 只产出 bbox。
若要训练关键点，请用带关键点的数据集或在 train.py 外补 landmark 标注。
"""
import os
import numpy as np
import tensorflow as tf


def _parse_gt(txt_path):
    """解析 WIDER FACE 标注，对个别缺框行/错行做容错恢复。

    用「路径含 '/'」识别图片行，下一行是 count；框行须有 >=4 个数字，
    否则视为缺框导致下一图路径串入，回退该行交还外层重新对齐。
    返回 list[(img_rel_path, np.ndarray(N,4) of [x1,y1,w,h])]。
    """
    samples = []
    with open(txt_path, "r") as f:
        lines = [l.strip() for l in f if l.strip() != ""]
    i, N = 0, len(lines)
    while i < N:
        line = lines[i]
        if "/" in line or line.lower().endswith((".jpg", ".png", ".jpeg")):
            img_rel = line
            i += 1
            n = 0
            if i < N:
                try:
                    n = int(lines[i])
                except ValueError:
                    n = 0
                i += 1
            boxes = []
            for _ in range(n):
                if i >= N:
                    break
                parts = lines[i].split()
                if len(parts) < 4:        # 非法框行 -> 回退，交还外层
                    i -= 1
                    break
                x1, y1, w, h = (float(parts[0]), float(parts[1]),
                                float(parts[2]), float(parts[3]))
                if w > 0 and h > 0:
                    boxes.append([x1, y1, w, h])
                i += 1
            samples.append((img_rel, np.array(boxes, dtype=np.float32)))
        else:
            i += 1                       # 跳过杂行（如缺 count）
    return samples


def _resolve_img_root(gt, data_dir, first_rel):
    """根据 gt 里首张图的相对路径，反推真正的 images 根目录。

    例：gt 在 wider_face_split/，但其内写的是 '0--Parade/xxx.jpg'（相对
    WIDER_train/images），所以 images 根应是 $data_dir/WIDER_train/images，
    而非 gt 同级的 images/。
    """
    bases = [
        os.path.join(data_dir, "WIDER_train", "images"),
        os.path.join(data_dir, "WIDER_val", "images"),
        os.path.join(os.path.dirname(gt), "images"),
        os.path.join(os.path.dirname(gt), "WIDER_train", "images"),
        os.path.join(data_dir, "images"),
        os.path.join(data_dir, "WIDER_train"),
        data_dir,
    ]
    for b in bases:
        if first_rel and os.path.exists(os.path.join(b, first_rel)):
            return b
    # 都没命中：返回最可能的猜测，交给 build 的早校验给出清晰报错
    return os.path.join(data_dir, "WIDER_train", "images")


def _find_gt(data_dir, split):
    """递归定位 gt 文件，并据首张图反推 images 根目录。

    返回 (gt_path, img_root)。
    """
    fname = {"train": "wider_face_train_bbx_gt.txt",
             "val": "wider_face_val_bbx_gt.txt"}[split]
    # 1) 快速候选
    quick = [
        os.path.join(data_dir, f"WIDER_{split}", fname),
        os.path.join(data_dir, fname),
        os.path.join(data_dir, "WIDER", fname),
    ]
    hits = [c for c in quick if os.path.exists(c)]
    # 2) 递归搜索（深度<=3，只读目录名，不读图片内容）
    if not hits:
        for root, dirs, files in os.walk(data_dir):
            if fname in files:
                hits.append(os.path.join(root, fname))
            depth = root[len(data_dir):].count(os.sep)
            if depth >= 3:
                dirs[:] = []
    if not hits:
        raise FileNotFoundError(f"找不到 {fname}（已递归搜索 {data_dir}）")
    gt = hits[0]
    # 读 gt 首张图相对路径，用来反推 images 根
    first_rel = ""
    with open(gt, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                first_rel = line
                break
    img_root = _resolve_img_root(gt, data_dir, first_rel)
    print(f"[info] 找到 gt: {gt}\n[info] images 根: {img_root}")
    return gt, img_root


def _load_and_encode(img_abs, input_size):
    raw = tf.io.read_file(img_abs)
    img = tf.image.decode_jpeg(raw, channels=3)
    h, w = tf.cast(tf.shape(img)[0], tf.float32), tf.cast(tf.shape(img)[1], tf.float32)
    img = tf.image.resize(img, (input_size, input_size))
    img = tf.cast(img, tf.float32)  # [0,255]，BGR? WIDER FACE 是 RGB;
    # 注意：OpenCV YuNet 用 BGR。训练时可统一转 BGR 以与部署一致：
    img = img[..., ::-1]
    scale = tf.constant(input_size, tf.float32) / tf.stack([h, w])
    return img, scale


def build_widerface_dataset(data_dir, input_size=320, batch=16,
                            split="train", max_boxes=200, shuffle=True):
    """返回 tf.data.Dataset -> (img[0,255] float32, bboxes[N,4] abs 像素)。"""
    gt, img_root = _find_gt(data_dir, split)
    samples = _parse_gt(gt)

    # 早校验：gt 引用的首张图是否真实存在，避免训练中途才崩
    if samples:
        first = os.path.join(img_root, samples[0][0])
        if not os.path.exists(first):
            raise FileNotFoundError(
                f"gt 引用的首张图不存在: {first}\n"
                f"说明 images 根推断有误。请确认 {gt} 中图片相对路径"
                f"是相对于哪个目录（通常是 gt 同级的 images/）。")

    paths = [os.path.join(img_root, r) for r, _ in samples]
    boxlist = [b for _, b in samples]

    def gen():
        for p, b in zip(paths, boxlist):
            img, scale = _load_and_encode(p, input_size)
            # bbox 坐标按短边缩放近似（简化：长宽同比例缩放到 input_size）
            if b.shape[0] == 0:
                b = np.zeros((1, 4), np.float32)
            # scale=[sh, sw]；x(宽)乘 sw，y(高)乘 sh
            fac = tf.stack([scale[1], scale[0], scale[1], scale[0]])
            bx = b * fac.numpy()  # x1,y1,w,h（b 已是 numpy，结果仍是 numpy）
            x1, y1, w, h = bx[:, 0], bx[:, 1], bx[:, 2], bx[:, 3]
            abs_boxes = np.stack([x1, y1, x1 + w, y1 + h], axis=-1)
            abs_boxes = np.clip(abs_boxes, 0, input_size)
            if abs_boxes.shape[0] > max_boxes:        # 截断，防止单图人脸过多撑爆
                abs_boxes = abs_boxes[:max_boxes]
            yield img.numpy(), abs_boxes.astype(np.float32)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec((input_size, input_size, 3), tf.float32),
            tf.TensorSpec((None, 4), tf.float32),
        ))
    if shuffle:
        ds = ds.shuffle(1000)
    #  padded batch（不同图框数不同；框维度按 batch 内最大值 pad，避免固定
    #  200 在单图人脸超 200 时报 "pad to smaller size" 错）
    ds = ds.padded_batch(batch, padded_shapes=((input_size, input_size, 3),
                                               (None, 4)))
    return ds, len(samples)
