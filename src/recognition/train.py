"""人脸识别训练入口（PC 浮点阶段）。

流程
--------------------------------------------------------------------------
    分片数据 → GhostFaceNet 骨干 → AdaFace(+PartialFC) 分类头 → CE loss
                                    ↑
                            360K 类只抽 10% 参与计算

设计取舍说明
--------------------------------------------------------------------------
* 优化器用 SGD + momentum：margin-based softmax 系列（ArcFace/AdaFace/…）
  在大规模人脸数据上的经验最优解，Adam 在 36 万类上容易过拟合且更吃显存。
* BN 的 weight decay 和 bias 置 0；分类头 weight decay 置 0。
* 混合精度（fp16）默认开启：GhostFaceNet 很小，开 AMP 后显存和速度都会明显受益。
* 学习率按「每步」走 cosine，并带 warmup —— IterableDataset 一个 epoch
  就是全量，按 step 调度比按 epoch 平滑得多。

示例
--------------------------------------------------------------------------
快速跑通（先建 4 万类的小分片，见 README）：

    python train.py --shards shards_quick --arch ghostfacenet \
        --loss adaface --batch_size 128 --epochs 5 --lr 0.05 \
        --num_workers 6 --amp

正式训练（36 万类）：

    python train.py --shards shards --arch ghostfacenet --width 1.0 \
        --loss adaface --sample_rate 0.08 --batch_size 192 --epochs 25 \
        --lr 0.05 --num_workers 8 --amp --out_dir checkpoints
"""
from __future__ import annotations

import os
import sys
import math
import time
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.dataset import (ShardDataset, RecDataset, ImageListDataset,  # noqa: E402
                          FaceAlignAug, ValAug, build_loader)
from models.ghostfacenet import build_backbone                        # noqa: E402
from models.topofr import TopoFRLoss                                  # noqa: E402
from models.heads import build_head                                   # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser("人脸识别训练")
    # 数据
    ap.add_argument("--shards", default=None,
                    help="分片目录（build_shard.py 产出，推荐）")
    ap.add_argument("--rec", default=None, help="或不分片，直接读 .rec（慢）")
    ap.add_argument("--index", default=None, help="配合 --rec 的索引 npz")
    ap.add_argument("--rec_limit", type=int, default=None,
                    help="--rec 模式下只读前 N 条（快速验证）")
    ap.add_argument("--list_file", default=None,
                    help="或使用图片清单（每行 `路径 标签`）")
    ap.add_argument("--img_root", default="")

    # 模型
    # README 第二节的算法选型定的是 ghostfacenet（IEEE Access 2023）。
    # 曾经因为「要上 DPU」把默认改成 mobilefacenet（它能整网编成 1 个 DPU 子图，
    # ghostfacenet 会被 Concat 切成 17 段，见 README 第六节）。
    # 现在部署走 CPU/ncnn，不再受 DPU 算子约束，默认改回 ghostfacenet。
    # 若确实要编 DPU，显式加 --arch mobilefacenet。
    ap.add_argument("--arch", default="ghostfacenet",
                    choices=["ghostfacenet", "mobilefacenet"])
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--embedding_size", type=int, default=512)
    ap.add_argument("--max_feat_norm", type=float, default=64.0,
                    help="特征范数硬上限，堵住 AdaFace 靠拉大范数降难度的捷径；0=关闭")
    # 默认 cosface：实测在 3 万类上 LFW 一路涨到 92%+，曲线单调。
    # adaface 是我们自己实现的「批内 z-score + 范数上限」变体，
    # 实测在同一数据上 LFW 卡在 55% 不涨（见 README 第六节），暂时别用。
    ap.add_argument("--loss", default="cosface",
                    choices=["adaface", "arcface", "cosface"])
    ap.add_argument("--m", type=float, default=0.4)
    ap.add_argument("--s", type=float, default=64.0)
    ap.add_argument("--sub_centers", type=int, default=1,
                    help=">1 启用 Sub-center ArcFace")
    ap.add_argument("--sample_rate", type=float, default=0.1,
                    help="Partial FC 每步抽多少比例的类（1.0=全类）")
    ap.add_argument("--label_smooth", type=float, default=0.0)
    # TopoFR（NeurIPS 2024）拓扑对齐正则。纯训练期，推理时骨干与比对方式不变。
    ap.add_argument("--topo_weight", type=float, default=0.0,
                    help="TopoFR 拓扑正则系数（论文用 0.1）；0=关闭")
    ap.add_argument("--topo_sample", type=int, default=0,
                    help="只在 batch 前 k 个样本上算持久同调，0=用满整个 batch")

    # 优化
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    # 学习率实测标定（3 万类 / 26 万张 / bs=128 / 单卡）：
    #   0.05  → warmup 一结束 loss 就从 3.5 反弹到 11，发散
    #   0.025 → epoch0 降一点，epoch1 起一路回升，仍然偏大
    #   0.005 → 稳定单调下降（慢，但方向对）
    # insightface 的 bs=512 / lr=0.1 是 8 卡多机配置，单卡小 batch 下梯度噪声大，
    # 线性缩放出来的 0.025 在这里仍然太激进。默认取 0.01（稳定与速度的折中）。
    # 想更稳就给 0.005，代价是要训更多 epoch。
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--min_lr", type=float, default=1e-5)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--warmup_epochs", type=float, default=1.0)
    ap.add_argument("--adaface_nonneg", action="store_true",
                    help="AdaFace 改用非负 margin∈[0,m]（更保守、更慢；"
                         "默认走论文原式 [-m,+m]，收敛快得多）")
    ap.add_argument("--num_workers", type=int, default=6)
    ap.add_argument("--amp", action="store_true", help="混合精度（推荐）")

    # 增强
    ap.add_argument("--aug_rot", type=float, default=8.0)
    ap.add_argument("--aug_scale", type=float, default=0.05)
    ap.add_argument("--aug_trans", type=float, default=0.04)
    ap.add_argument("--no_flip", action="store_true")

    # 运行
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every_epochs", type=int, default=1)
    ap.add_argument("--resume", default=None, help="续训 checkpoint")
    ap.add_argument("--finetune", default=None,
                    help="只加载 backbone 权重（不含 head/优化器状态）")
    ap.add_argument("--lfw_bin", default=None, help="可选：每 epoch 末跑一次 LFW")
    ap.add_argument("--bd", type=int, default=1, help="打印前几个 step 的调试信息")
    return ap.parse_args()


def build_dataset(args, aug, training: bool):
    if args.list_file:
        return ImageListDataset(args.list_file, aug, args.img_root)
    if args.shards:
        return ShardDataset(args.shards, aug, training=training, seed=args.seed)
    if args.rec:
        if not args.index:
            raise SystemExit("--rec 模式需要同时给 --index")
        return RecDataset(args.rec, args.index, aug,
                          limit=args.rec_limit, seed=args.seed, shuffle=training)
    raise SystemExit("需要指定 --shards 或 --rec/--index 或 --list_file")


class CosineScheduler:
    """warmup（线性）+ 余弦退火，按 step 走。"""

    def __init__(self, optimizer, base_lr, min_lr, warmup_steps, total_steps):
        self.opt = optimizer
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup = max(1, warmup_steps)
        self.total = max(self.warmup + 1, total_steps)
        self.step_i = 0

    def step(self):
        self.step_i += 1
        i = self.step_i
        if i <= self.warmup:
            lr = self.base_lr * i / self.warmup
        else:
            t = (i - self.warmup) / max(1, self.total - self.warmup)
            t = min(1.0, t)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * t))
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr


@torch.no_grad()
def update_partialfc_weight(head, lr: float, momentum: float):
    """手动动量 SGD：只更新本步采样到的类行，并把梯度清掉。

    注意 `w[idx]` 是高级索引 → 返回副本，所以必须用 index_copy_ 写回，
    直接 `w[idx].sub_(...)` 是无效的（这也是容易写错的地方）。
    """
    idx = head.last_indices
    if idx is None or head.weight.grad is None:
        return
    w = head.weight
    g = w.grad.index_select(0, idx)
    if not torch.isfinite(g).all():
        w.grad = None               # 本步梯度无效，跳过（scaler 也会跳过其他参数）
        return
    mom = head.weight_mom
    mom.index_copy_(0, idx, momentum * mom.index_select(0, idx) + g)
    w.index_copy_(0, idx, w.index_select(0, idx) - lr * mom.index_select(0, idx))
    w.grad = None       # 该参数不在 optimizer 里，必须手动清零，否则梯度会累积


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}  torch={torch.__version__}  amp={args.amp}")

    aug = FaceAlignAug(size=112, rot=args.aug_rot, scale=args.aug_scale,
                       trans=args.aug_trans, flip_p=0.0 if args.no_flip else 0.5,
                       seed=args.seed)
    train_ds = build_dataset(args, aug, training=True)
    num_classes = train_ds.num_classes
    n_samples = len(train_ds)
    steps_per_epoch = max(1, n_samples // args.batch_size)
    print(f"[data] {n_samples:,} 张 / {num_classes:,} 类  "
          f"每 epoch ≈ {steps_per_epoch:,} step (bs={args.batch_size})")

    backbone = build_backbone(args.arch, args.embedding_size, args.width).to(device)
    backbone.max_feat_norm = float(args.max_feat_norm)
    head = build_head(args.embedding_size, num_classes, args.loss,
                      m=args.m, s=args.s, sub_centers=args.sub_centers,
                      sample_rate=args.sample_rate,
                      label_smooth=args.label_smooth,
                      allow_negative_margin=not args.adaface_nonneg).to(device)
    n_bb = sum(p.numel() for p in backbone.parameters())
    n_hd = sum(p.numel() for p in head.parameters())
    print(f"[model] {args.arch}(w={args.width}) 骨干 {n_bb/1e6:.2f}M 参数, "
          f"分类头 {n_hd/1e6:.2f}M 参数, 输出 {args.embedding_size}-d")

    # TopoFR 拓扑正则（NeurIPS 2024）。纯训练期，不进 optimizer、不参与推理。
    topo = None
    if args.topo_weight > 0:
        topo = TopoFRLoss(weight=args.topo_weight,
                          topo_sample=args.topo_sample or None)
        print(f"[model] + TopoFR 拓扑正则 weight={args.topo_weight} "
              f"sample={args.topo_sample or 'full'}")

    # 参数分组：BN 的 γ/β、所有 bias、以及分类头权重都不做 weight decay
    # （margin-based softmax 的通用做法；分类头 36 万类，衰减会明显伤害收敛）
    decay, no_decay = [], []
    for name, p in backbone.named_parameters():
        if p.ndim <= 1:
            no_decay.append(p)
        else:
            decay.append(p)
    # 分类头权重：Partial FC 模式下每步只采样一部分类，必须走「只更新采样行」
    # 的手动动量 SGD（见 heads.py 里 weight_mom 的注释），否则未采样类会被
    # 动量缓冲推着漂移，几个 epoch 后训练 acc 会直接崩到 0。
    groups = [{"params": decay, "weight_decay": args.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    if not head.manual_update:          # 全类参与时交给普通 SGD 即可
        groups.append({"params": [head.weight], "weight_decay": 0.0})
    optimizer = torch.optim.SGD(groups, lr=args.lr, momentum=args.momentum,
                                nesterov=True)

    start_epoch = 0
    global_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        backbone.load_state_dict(ckpt["backbone"])
        head.load_state_dict(ckpt["head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        print(f"[resume] 从 {args.resume} 恢复，epoch={start_epoch}")
    elif args.finetune and os.path.exists(args.finetune):
        sd = torch.load(args.finetune, map_location=device)
        sd = sd.get("backbone", sd) if isinstance(sd, dict) else sd
        missing, unexpected = backbone.load_state_dict(sd, strict=False)
        print(f"[finetune] 加载骨干 {args.finetune}  "
              f"missing={len(missing)} unexpected={len(unexpected)}")

    total_steps = steps_per_epoch * args.epochs
    sched = CosineScheduler(optimizer, args.lr, args.min_lr,
                            warmup_steps=int(args.warmup_epochs * steps_per_epoch),
                            total_steps=total_steps)
    sched.step_i = global_step

    loader = build_loader(train_ds, args.batch_size, args.num_workers)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    amp_dtype = torch.float16

    history = []
    # ===== 训练主循环（每步 pipeline 总览）=====
    # 每个 epoch 遍历 DataLoader 产出的 (imgs, labels)，单步依次做：
    #   1) 骨干前向（autocast fp16）→ 512-d 特征（不归一化，保留范数给 head 算 margin）
    #   2) 分类头算 (loss, acc)（必须 fp32：margin 里有 sqrt(1-cos²)，fp16 会负→nan）；
    #      可选 TopoFR 拓扑正则（同样 fp32，距离矩阵有 sqrt 也怕 fp16）
    #   3) AMP 反向：scaler.scale(loss).backward() → unscale_ → 梯度裁剪
    #   4) 分类头若是 Partial FC（manual_update），需手动把 head.weight.grad 除回
    #      scale，再跑「只更新采样行」的动量 SGD（见 update_partialfc_weight）
    #   5) scaler.step / update；每个 epoch 末存 backbone（部署用）+ 完整 ckpt（续训用），
    #      可选在 --lfw_bin 上跑一次验证
    # 学习率按 step 走 cosine（CosineScheduler），含 warmup。
    for epoch in range(start_epoch, args.epochs):
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        backbone.train()
        head.train()
        t0 = time.time()
        run_loss, run_acc, n_seen = 0.0, 0.0, 0
        run_topo = 0.0
        for step, (imgs, labels) in enumerate(loader):
            lr = sched.step()
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()

            # 只有骨干走 fp16。分类头必须留在 fp32：margin 里有
            # sqrt(1-cos²)，fp16 下 1-cos² 会因精度损失变成负数 → sqrt 出 nan
            # → GradScaler 一路把 scale 降到底 → 训练从第二个 epoch 开始退化。
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=args.amp):
                feats = backbone(imgs)
            loss, acc = head(feats, labels)
            if topo is not None:
                # 拓扑正则只在 fp32 下算：距离矩阵里有 sqrt，fp16 会因
                # 精度损失产生负数 → nan，与分类头那边踩的是同一个坑。
                loss_topo = topo(imgs.float(), feats.float())
                loss = loss + args.topo_weight * loss_topo
                run_topo += loss_topo.item() * imgs.size(0)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # head.weight 已不在 optimizer 里，GradScaler 不会替它 unscale，
            # 必须手动除回 scale —— 否则梯度被放大上万倍，分类头直接炸掉
            # （表现为 loss 卡在 log(K) 附近、acc 恒为 0）。
            if head.manual_update and head.weight.grad is not None:
                head.weight.grad.div_(scaler.get_scale())
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(head.parameters()), 5.0)
            if args.bd and step % args.log_every == 0:
                with torch.no_grad():
                    gb = sum(p.grad.norm().item() ** 2
                             for p in backbone.parameters() if p.grad is not None) ** 0.5
                    gw = (float(head.weight.grad.norm())
                          if head.weight.grad is not None else 0.0)
                    fn = float(feats.detach().norm(dim=1).mean())
                print(f"  [diag] |g_bb|={gb:.3e} |g_head|={gw:.3e} "
                      f"feat_norm={fn:.2f} scale={scaler.get_scale():.0f}",
                      flush=True)
            scaler.step(optimizer)
            scaler.update()
            if head.manual_update:
                update_partialfc_weight(head, lr, args.momentum)

            bs = imgs.size(0)
            run_loss += loss.item() * bs
            run_acc += acc.item() * bs
            n_seen += bs
            global_step += 1

            if step % args.log_every == 0:
                el = time.time() - t0
                print(f"[ep{epoch} {step}/{steps_per_epoch}] "
                      f"loss={run_loss/max(n_seen,1):.4f} "
                      f"acc={run_acc/max(n_seen,1):.4f} "
                      f"lr={lr:.2e} {n_seen/el:.0f} img/s "
                      f"eta={(steps_per_epoch-step)*el/max(step+1,1)/60:.1f}min",
                      flush=True)
            if step >= steps_per_epoch:
                break

        ep_loss = run_loss / max(n_seen, 1)
        ep_acc = run_acc / max(n_seen, 1)
        print(f"[epoch {epoch}] loss={ep_loss:.4f} acc={ep_acc:.4f} "
              + (f"topo={run_topo / max(n_seen, 1) * args.topo_weight:.4f} "
                 if topo is not None else "")
              + f"用时 {(time.time()-t0)/60:.1f} min")

        # 每 epoch 存骨干（导出 ONNX / 部署用）与完整 checkpoint（续训用）
        ep_path = os.path.join(args.out_dir, f"backbone_ep{epoch}.pth")
        torch.save({"backbone": backbone.state_dict(),
                    "arch": args.arch, "width": args.width,
                    "embedding_size": args.embedding_size,
                    "num_classes": num_classes}, ep_path)
        last = os.path.join(args.out_dir, "last.pth")
        torch.save({"backbone": backbone.state_dict(),
                    "head": head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "global_step": global_step,
                    "args": vars(args)}, last)

        rec = {"epoch": epoch, "loss": ep_loss, "acc": ep_acc}
        if args.lfw_bin:
            try:
                from eval_bin import evaluate_bin
                acc_lfw, thr = evaluate_bin(backbone, args.lfw_bin,
                                            batch_size=256, device=device)
                rec["lfw"] = acc_lfw
                print(f"[eval] LFW acc={acc_lfw:.4f} @thr={thr:.3f}")
                backbone.train()
            except Exception as e:                          # noqa: BLE001
                print(f"[warn] LFW 评测失败: {e}")
        history.append(rec)
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print(f"[done] 训练结束，权重在 {args.out_dir}/")


if __name__ == "__main__":
    main()
