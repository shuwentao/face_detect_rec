"""把 300W 数据集转成检测器训练用的 COCO 风格 JSON（仅提供关键点真值）。

背景
----
部署用的 PyTorch 检测器（quant_deploy/common）需要 5 点关键点真值来训关键点头，
而 WIDER FACE 没有关键点。300W（300 Faces In-the-Wild）是带 68 点高精度关键点的
基准数据集，通过 train_cli.py 的 --extra_ann / --extra_img 在训练时混入
（见 mix_data.merge_extra），与 WIDER FACE 的 bbox 一起训，从而复刻原始 face_bench。

本脚本只做「格式转换」，不下载数据：
    - 读 300W 各子集（afw / helen / lfpw / ibug）下的 *.pts（68 点）
    - 映射成检测器要的 5 点：0 左眼 1 右眼 2 鼻尖 3 左嘴角 4 右嘴角
      （顺序按 x 坐标自动判定，保证与 dataset.py 的 FLIP_PAIRS=(0,1),(3,4) 一致）
    - 关键点 vis=1（参与训练）；bbox 由 68 点外扩得到（或读 bounding_boxes_300W.txt）

输出 JSON 字段（与 FaceDataset + merge_extra 严格对齐）：
    images:      [{id, file_name(相对 --root), _img_dir, wider_rel}]
    annotations: [{id, image_id, bbox:[x,y,w,h], landmarks:5×[x,y,vis]}]
训练时：
    python scripts/train_cli.py --data_root <wider_json根> \
        --extra_ann <本脚本输出的 json> --extra_img <300W根目录>

用法
----
    python landmark_300w_to_json.py \
        --root /mnt/data3/300W \
        --out  /mnt/data3/cv_face_baseline/src/quant_deploy/common/data/facebench_wider/annotations/landmark_300w.json \
        --bbox_file /mnt/data3/300W/bounding_boxes_300W.txt   # 可选，更准的框
"""
import os
import json
import argparse
import glob


# 68 点 -> 5 点 的索引（0-based）。retinaface 标准：
#   左/右眼 = 各自两眼角中点；鼻尖 = 30；嘴角 = 48 / 54。
# 这里取眼角中点后，再按 x 坐标重排左右，确保「图像左侧=index0」，
# 与 dataset.py 水平翻转时交换 (0,1)/(3,4) 的逻辑一致。
LANDMARK_MAP = {
    "eye_r": (36, 39),   # 右眼（画面左侧，subject 的右眼）
    "eye_l": (42, 45),   # 左眼（画面右侧）
    "nose":  (30,),
    "mouth_l": (48,),    # 左嘴角（画面左侧）
    "mouth_r": (54,),
}


def parse_pts(path):
    """解析 300W / 各类 .pts 文件，返回 (N,2) 的 float 数组（通常 68 点）。

    格式：version: 1 / n_points: N / { / x y(每行) / }。
    注意 '{' 与 '}' 是分隔符，不是坐标；下面扫描时跳过非数值行，
    直到凑满 n 个点，避免把 '{' 行算作一个坐标导致少一个点。
    """
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    n = None
    idx = 0
    for i, l in enumerate(lines):
        if l.lower().startswith("n_points"):
            n = int(l.split(":")[1])
            idx = i + 1
            break
    if n is None:                      # 退化：没有 n_points 头，直接全当坐标
        idx = 0
        n = len(lines)
    pts = []
    while idx < len(lines) and len(pts) < n:
        l = lines[idx]
        idx += 1
        if l in ("{", "}"):
            continue
        parts = l.split()
        if len(parts) < 2:
            continue
        try:
            x, y = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        pts.append([x, y])
    return pts


def to_five(pts):
    """68 点 -> 5 点（按 x 自动定左右）。返回 5×[x,y]。"""
    pts = list(pts)
    eye0 = [(pts[36][0] + pts[39][0]) / 2, (pts[36][1] + pts[39][1]) / 2]
    eye1 = [(pts[42][0] + pts[45][0]) / 2, (pts[42][1] + pts[45][1]) / 2]
    nose = pts[30]
    m0, m1 = pts[48], pts[54]
    # 按 x 重排：x 小的是画面左侧
    left_eye, right_eye = (eye0, eye1) if eye0[0] <= eye1[0] else (eye1, eye0)
    left_mouth, right_mouth = (m0, m1) if m0[0] <= m1[0] else (m1, m0)
    return [left_eye, right_eye, nose, left_mouth, right_mouth]


def derive_bbox(pts, pad=0.1):
    """没有官方框时，用 68 点外接矩形外扩 pad 比例生成 bbox（xywh）。"""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    w, h = x2 - x1, y2 - y1
    x1 -= pad * w; y1 -= pad * h
    x2 += pad * w; y2 += pad * h
    return [x1, y1, (x2 - x1), (y2 - y1)]


def load_bbox_file(path):
    """解析 iBUG 提供的 bounding_boxes_300W.txt：每行 'image_name x y w h'。"""
    d = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            name = parts[0]
            # 有的格式带引号或后缀，统一去掉目录/扩展名做 key
            key = os.path.splitext(os.path.basename(name))[0]
            d[key] = [float(v) for v in parts[1:5]]
    return d


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    ap = argparse.ArgumentParser(description="300W .pts -> face_bench 关键点 JSON")
    ap.add_argument("--root", required=True,
                    help="300W 根目录（含 afw/helen/lfpw/ibug 子目录，也是 --extra_img 的值）")
    ap.add_argument("--out", required=True, help="输出 json 路径")
    ap.add_argument("--bbox_file", default=None,
                    help="可选：iBUG bounding_boxes_300W.txt（优先于从关键点推导的框）")
    ap.add_argument("--subsets", default="all",
                    help="要包含的 300W 子集，逗号分隔；默认 all=扫描 root 下所有 *.pts")
    args = ap.parse_args()

    # subsets="all" 时直接扫 root 下的所有 *.pts；否则只扫指定子目录
    if not args.subsets or args.subsets.lower() == "all":
        sub_dirs = [args.root]
    else:
        sub_dirs = [os.path.join(args.root, s) for s in args.subsets.split(",") if s.strip()]
    bbox_map = load_bbox_file(args.bbox_file) if args.bbox_file else {}

    images, annotations = [], []
    img_id, ann_id = 1, 1
    for sub_dir in sub_dirs:
        pts_files = sorted(glob.glob(os.path.join(sub_dir, "**", "*.pts"), recursive=True))
        if not pts_files:                  # 有些打包把 .pts 直接放子集根
            pts_files = sorted(glob.glob(os.path.join(sub_dir, "*.pts")))
        for pts_path in pts_files:
            base = os.path.splitext(os.path.basename(pts_path))[0]
            # 找对应图片
            img_path = None
            for ext in IMG_EXTS:
                cand = os.path.join(os.path.dirname(pts_path), base + ext)
                if os.path.exists(cand):
                    img_path = cand
                    break
            if img_path is None:
                print(f"  ! 跳过(找不到图): {pts_path}")
                continue
            pts = parse_pts(pts_path)
            if len(pts) != 68:
                print(f"  ! 跳过(非68点,有{len(pts)}): {pts_path}")
                continue
            five = to_five(pts)
            # 框：优先官方 bbox 文件，否则从关键点推导
            if base in bbox_map:
                bbox = bbox_map[base]
            else:
                bbox = derive_bbox(pts)
            rel = os.path.relpath(img_path, args.root)   # 相对 --root，供 --extra_img 拼接
            images.append({
                "id": img_id,
                "file_name": rel,
                "_img_dir": args.root,
                "wider_rel": rel,
            })
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "bbox": [round(v, 2) for v in bbox],
                # vis=1：build_targets 会据此置 lm_mask，关键点损失生效
                "landmarks": [[round(x, 2), round(y, 2), 1.0] for (x, y) in five],
            })
            img_id += 1
            ann_id += 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"images": images, "annotations": annotations}, open(args.out, "w"))
    print(f"完成: {len(images)} 图 / {len(annotations)} 标注 -> {args.out}")


if __name__ == "__main__":
    main()
