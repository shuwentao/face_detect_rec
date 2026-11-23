"""通用训练入口：python train.py [选项]（每个算法目录各自调用，ALGO 不同）。"""
import os
import sys
import time
import json
import argparse

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from common.dataset import FaceDataset, collate
from common.models.detector import build_detector
from common.loss import DetectionLoss
from common.engine import Trainer
from common.utils import seed_everything, save_json, ensure_dir, count_params

DEFAULT_ROOT = "/mnt/data1/face_bench"


def train(algo, argv=None):
    ap = argparse.ArgumentParser(f"train {algo}")
    ap.add_argument("--data_root", default=os.path.join(DEFAULT_ROOT, "data"))
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--input_size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--train_ann", default=None,
                    help="训练集标注(json)，默认 annotations/train.json")
    ap.add_argument("--scale_max", type=float, default=1.2,
                    help="随机缩放上限，>1 会放大后中心裁剪以制造大脸样本")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--max_images", type=int, default=0,
                    help=">0 时只用前 N 张(冒烟)")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--extra_ann", default=None,
                    help="额外训练集标注(json)，用于混入高精度关键点数据(如 300W)")
    ap.add_argument("--extra_img", default=None, help="额外训练集图片目录")
    ap.add_argument("--extra_repeat", type=int, default=1,
                    help="额外训练集重复次数(用于提升其采样权重)")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--val_batches", type=int, default=60)
    args = ap.parse_args(argv)

    seed_everything(args.seed)
    out_dir = args.out_dir or os.path.join(DEFAULT_ROOT, algo)
    ensure_dir(out_dir)

    train_ann = args.train_ann or os.path.join(
        args.data_root, "annotations", "train.json")
    train_ds = FaceDataset(train_ann,
                           os.path.join(args.data_root, "images", "train"),
                           input_size=args.input_size, train=True,
                           scale_max=args.scale_max)
    val_ds = FaceDataset(os.path.join(args.data_root, "annotations", "val.json"),
                         os.path.join(args.data_root, "images", "val"),
                         input_size=args.input_size, train=False)
    if args.max_images > 0:
        train_ds.images = train_ds.images[:args.max_images]
        val_ds.images = val_ds.images[:min(args.max_images, len(val_ds.images))]
    if args.extra_ann and args.extra_img:
        from common.mix_data import merge_extra
        ni, na = merge_extra(train_ds, args.extra_ann, args.extra_img,
                             args.extra_repeat)
        print(f"[混合训练] 追加 {ni} 张图 / {na} 条标注 <- {args.extra_ann}")
    tl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, collate_fn=collate,
                    pin_memory=True, drop_last=True,
                    persistent_workers=args.workers > 0)
    vl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                    num_workers=max(2, args.workers // 2), collate_fn=collate,
                    pin_memory=True)

    model = build_detector(algo, pretrained=not args.no_pretrained,
                           width=args.width)
    if args.resume and os.path.exists(args.resume):
        sd = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(sd["model"])
        print(f"[info] 从 {args.resume} 恢复权重")
    print(f"[info] {algo}: 参数量 {count_params(model)/1e6:.3f} M, "
          f"训练集 {len(train_ds)} 张, 验证集 {len(val_ds)} 张")

    trainer = Trainer(model, tl, vl, out_dir, epochs=args.epochs, lr=args.lr,
                      wd=args.wd, device=args.device, log_every=args.log_every,
                      val_batches=args.val_batches)
    t0 = time.time()
    history = trainer.fit(DetectionLoss())
    save_json({"algo": algo, **vars(args),
               "params": count_params(model),
               "train_minutes": (time.time() - t0) / 60,
               "out_dir": out_dir},
              os.path.join(out_dir, "config.json"))
    print(f"[done] {algo} 训练完成，用时 {(time.time()-t0)/60:.1f} min，"
          f"输出目录 {out_dir}")
    return out_dir
