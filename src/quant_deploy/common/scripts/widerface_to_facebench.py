"""把 WIDER FACE 转成检测器训练用的 COCO 风格 JSON（face_bench 格式）。

背景
----
部署用的 PyTorch 检测器（quant_deploy/common）的训练数据 face_bench 是从
WIDER FACE 抽的子集。本脚本把 WIDER FACE 还原成 `FaceDataset` 需要的格式：
    annotations/train.json
    annotations/val.json

每条 annotation 字段（与 dataset.py 的 FaceDataset 严格对应）：
    { "id", "image_id",
      "bbox": [x, y, w, h],          # xywh，与 WIDER FACE 标注一致
      "landmarks": 5 × [x, y, vis] } # 5 点关键点；vis>=0 才参与训练

关键点说明
----------
WIDER FACE 只有 bbox、没有关键点，因此这里 landmarks 全部置 vis=-1，
`build_targets` 看到 vis<0 就不会给该样本置 lm_mask，关键点损失自然不生效。
若你有 300W 之类的高精度关键点数据集，用 train_cli.py 的 --extra_ann /
--extra_img 在训练时混进来即可（见 mix_data.merge_extra）。

图片不拷贝
----------
每条 image 写入 "_img_dir" 指向 WIDER FACE 原图目录，file_name 为相对路径；
FaceDataset.__getitem__ 会优先用 _img_dir，所以不需要把 12k+ 张图复制出来。

用法
----
    python widerface_to_facebench.py \
        --wider_root /mnt/data3/WideFace \
        --out_dir /mnt/data3/cv_face_baseline/src/quant_deploy/common/data/facebench_wider \
        --sample_rate 1.0        # <1 时按该比例随机抽子集，复刻"抽了一部分"
然后训练（train_cli.py 没有 __main__，用 -c 调它的 train(algo, argv)）：
    cd src/quant_deploy
    python -c "from common.scripts.train_cli import train; train('mobilenetv3', [
        '--data_root', '<out_dir>', '--input_size', '320'])"
（train_cli 会从 <out_dir>/annotations/train.json 与 <out_dir>/images/... 读取；
 因为 _img_dir 已内嵌，images/ 目录不存在也没关系。完整的从零复现流程见上级目录 reproduce.sh。）
"""
import os
import json
import argparse
import random


def parse_wider(txt_path):
    """解析 WIDER FACE 标注文件，返回 [(rel_path, [(x,y,w,h), ...]), ...]。

    文件结构（逐行）：
        <相对路径>           例如 0--Parade/0_Parade_marchingband_1_849.jpg
        <人脸数 N>
        N 行: x y w h blur expression illumination invalid occlusion pose

    不严格依赖「N 行后恰好 N 个框」——WIDER FACE 个别条目 N 与实际行数偶有
    出入。改为按行类型自适应分类：含 '.jpg' 的当作图片路径（顺带 flush 上一张），
    否则若当前还没有读到 N 就当作 N 行，其余当作框行。这样即使某张图框行数
    对不上，也不至于整体错位。
    """
    samples = []
    cur_path, cur_n, cur_boxes = None, None, []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ".jpg" in line or line.endswith(".png"):   # 新图片路径
                if cur_path is not None:
                    samples.append((cur_path, cur_boxes))
                cur_path, cur_n, cur_boxes = line, None, []
            else:
                toks = line.split()
                if cur_n is None:                          # 该图的 N 行
                    cur_n = int(toks[0])
                else:                                     # 框行
                    x, y, w, h = map(int, toks[:4])        # 前 4 个是 bbox（xywh）
                    cur_boxes.append([x, y, w, h])
    if cur_path is not None:                              # flush 最后一张
        samples.append((cur_path, cur_boxes))
    return samples


def build_json(samples, img_dir, sample_rate=1.0, seed=42):
    """samples -> COCO 风格 dict（images / annotations）。"""
    rng = random.Random(seed)
    if 0.0 < sample_rate < 1.0:                 # 复刻"从 WIDER FACE 抽一部分"
        samples = [s for s in samples if rng.random() < sample_rate]

    images, annotations = [], []
    img_id, ann_id = 1, 1
    no_lm = [[0.0, 0.0, -1.0]] * 5              # vis=-1：无关键点真值
    for rel, boxes in samples:
        images.append({
            "id": img_id,
            "file_name": rel,                    # 相对 WIDER_train/images 的路径
            "_img_dir": img_dir,                 # 内嵌原图目录，免拷贝
            "wider_rel": rel,                    # 溯源字段（FaceDataset 会回传）
        })
        for (x, y, w, h) in boxes:
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "bbox": [x, y, w, h],            # xywh
                "landmarks": [list(p) for p in no_lm],
            })
            ann_id += 1
        img_id += 1
    return {"images": images, "annotations": annotations}


def main():
    ap = argparse.ArgumentParser(description="WIDER FACE -> face_bench COCO JSON")
    ap.add_argument("--wider_root", default="/mnt/data3/WideFace")
    ap.add_argument("--out_dir", required=True,
                    help="输出根目录，会生成 <out_dir>/annotations/{train,val}.json")
    ap.add_argument("--sample_rate", type=float, default=1.0,
                    help="训练集抽样比例，<1 复刻「抽了一部分」；默认 1.0 用全量")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # split -> (标注文件相对路径, 图片目录相对路径)
    splits = {
        "train": ("wider_face_split/wider_face_train_bbx_gt.txt", "WIDER_train/images"),
        "val":   ("wider_face_split/wider_face_val_bbx_gt.txt",   "WIDER_val/images"),
    }

    ann_dir = os.path.join(args.out_dir, "annotations")
    os.makedirs(ann_dir, exist_ok=True)

    for split, (txt_rel, img_rel) in splits.items():
        txt = os.path.join(args.wider_root, txt_rel)
        img_dir = os.path.join(args.wider_root, img_rel)
        samples = parse_wider(txt)
        data = build_json(samples, img_dir, args.sample_rate, args.seed)
        out = os.path.join(ann_dir, f"{split}.json")
        json.dump(data, open(out, "w"))
        n_img, n_ann = len(data["images"]), len(data["annotations"])
        print(f"[{split}] {n_img} 图 / {n_ann} 框 -> {out}")
        # 抽检前 200 张图片路径是否真的可访问
        miss = 0
        for im in data["images"][:200]:
            if not os.path.exists(os.path.join(im["_img_dir"], im["file_name"])):
                miss += 1
        if miss:
            print(f"   ! 前 200 张中有 {miss} 张图片不存在，请检查 --wider_root/img_rel")
        else:
            print(f"   ✓ 前 200 张图片路径校验通过 (img_dir={img_dir})")


if __name__ == "__main__":
    main()
