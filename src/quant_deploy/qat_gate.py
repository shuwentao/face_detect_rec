"""gate (MobileNetV3-small + FPN-lite) 的 QAT 微调脚本。

在 PTQ（maxmin/diffs/CLE/max_fix_position）与 AdaQuant 都无法把 rel-L2 压下来
之后走这条路：量化感知训练让网络自己去适配「per_tensor + power-of-two」的刻度，
典型的 PTQ 失效（折叠后 max/std 高达 6~33 的深度可分离卷积）都靠它解决。

与常规训练的区别：
  * 模型先经 QatProcessor 包成「带伪量化算子的可训练模型」
  * 优化器分两组：普通权重用小 lr，量化步进ჿ参（quantizer）用大 lr
  * 不用 AMP：伪量化算子与 GradScaler 不兼容
  * 训练完要 to_deployable() 才能导出给 DPU

用法（在 vitis-ai-pytorch 环境下）：
  冒烟：python qat_gate.py --input_size 640 --max_images 200 --epochs 1
  正式：python qat_gate.py --input_size 640 --epochs 3 --lr 2e-5 \\
            --batch 4 --calib_dir qr640 --out_dir qat640
"""
import argparse
import copy
import math
import os
import sys
import time


def _set_cpu_isa():
    """限制 oneDNN 指令集，绕开 "could not create a primitive descriptor"。

    必须在 import torch 之前设：某些 CPU 上 oneDNN 会挑 AVX512 的 conv kernel，
    而这些 kernel 不支持深度可分离卷积的常见形状（5x5 / 通道数非 16 倍数），
    backward 时就抛 "could not create a primitive descriptor iterator"。
    命令行可用 --dnnl_isa AVX2|SSE41|AVX512_CORE 覆盖。
    """
    isa = "AVX2"
    for i, a in enumerate(sys.argv):
        if a == "--dnnl_isa" and i + 1 < len(sys.argv):
            isa = sys.argv[i + 1]
        elif a.startswith("--dnnl_isa="):
            isa = a.split("=", 1)[1]
    for var in ("DNNL_MAX_CPU_ISA", "ONEDNN_MAX_CPU_ISA"):
        os.environ.setdefault(var, isa)
    return isa


_ISA = _set_cpu_isa()

import torch                                                  # noqa: E402
from torch.utils.data import DataLoader                       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.dataset import FaceDataset, collate
from common.loss import DetectionLoss
from common.models.detector import build_detector
from common.utils import seed_everything, ensure_dir, count_params

DEFAULT_ROOT = "/mnt/data1/face_bench"


def build_model(ckpt, device):
    model = build_detector("mobilenetv3", pretrained=False, width=1.0)
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    info = model.load_state_dict(sd, strict=True)
    print("[model] load_state_dict ->", info)
    return model.eval().to(device)


def make_loaders(args):
    """注意：QAT 模型是按固定 batch 被 trace 的，loader 必须 drop_last，
    否则最后一个不满批的 batch 会让 shape 对不上。"""
    tr_ds = FaceDataset(args.train_ann,
                        os.path.join(args.data_root, "images", "train"),
                        input_size=args.input_size, train=True,
                        scale_max=args.scale_max)
    va_ds = FaceDataset(os.path.join(args.data_root, "annotations", "val.json"),
                        os.path.join(args.data_root, "images", "val"),
                        input_size=args.input_size, train=False)
    if args.max_images > 0:
        tr_ds.images = tr_ds.images[:args.max_images]
        va_ds.images = va_ds.images[:min(args.max_images, len(va_ds.images))]
    tl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, collate_fn=collate,
                    pin_memory=True, drop_last=True,
                    persistent_workers=args.workers > 0)
    vl = DataLoader(va_ds, batch_size=args.batch, shuffle=False,
                    num_workers=max(2, args.workers // 2), collate_fn=collate,
                    pin_memory=True, drop_last=True)
    print("[data] 训练 %d 张 / 验证 %d 张, batch=%d, input=%d"
          % (len(tr_ds), len(va_ds), args.batch, args.input_size))
    return tl, vl


def lr_at(epoch, step, steps_per_epoch, args):
    """warmup + cosine（与普通训练一致，只是 QAT 用更小的 base lr）。"""
    if epoch < args.warmup_epochs:
        t = (step + epoch * steps_per_epoch) / max(
            1, args.warmup_epochs * steps_per_epoch)
        return args.lr * (0.1 + 0.9 * t)
    total = max(1, args.epochs - args.warmup_epochs)
    prog = min(((epoch - args.warmup_epochs) + step / max(1, steps_per_epoch))
               / total, 1.0)
    return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 *
                      (1 + math.cos(math.pi * prog)))


def move_targets(tgts, device):
    return [tuple(x.to(device, non_blocking=True) for x in t) for t in tgts]


@torch.no_grad()
def validate(model, loader, criterion, device, limit=None):
    model.eval()
    n = limit or max(1, int(0.3 * len(loader)))
    acc = {"loss": 0.0, "obj": 0.0, "box": 0.0, "lm": 0.0}
    cnt = 0
    for i, (imgs, tgts, _) in enumerate(loader):
        if i >= n:
            break
        imgs = imgs.to(device, non_blocking=True)
        tgts = move_targets(tgts, device)
        out = criterion(model(imgs), tgts)
        v = float(out["loss"])
        if math.isfinite(v):
            for k in acc:
                acc[k] += float(out[k])
            cnt += 1
    return ({k: v / max(cnt, 1) for k, v in acc.items()}, cnt)


@torch.no_grad()
def compare_with_float(float_model, qat_model, loader, device, n_batch=3):
    """QAT 完事后最关键的验收：这两个数必须回到 cos>0.99 / rel-L2<0.05。"""
    float_model.eval()
    qat_model.eval()
    rows = []
    for i, (imgs, _, _) in enumerate(loader):
        if i >= n_batch:
            break
        x = imgs.to(device)
        fo = [o.float() for o in float_model(x)]
        qo = [o.float() for o in qat_model(x)]
        for a, b in zip(fo, qo):
            rows.append((
                (torch.norm(a - b) / (torch.norm(a) + 1e-12)).item(),
                torch.nn.functional.cosine_similarity(
                    a.reshape(1, -1), b.reshape(1, -1), dim=1).item()))
    rel = max(r[0] for r in rows)
    cos = min(r[1] for r in rows)
    print("\n[qat] QAT 模型 vs 浮点模型（%d 个 batch）" % n_batch)
    print("      rel-L2 最大 = %.6f     cos-sim 最低 = %.6f" % (rel, cos))
    print("      参考：PTQ 时 rel-L2=0.663 / cos=0.785；")
    print("           QAT 若有效应到 rel-L2<0.05 / cos>0.99")
    return rel, cos


def main():
    ap = argparse.ArgumentParser("gate QAT 微调")
    ap.add_argument("--data_root", default=os.path.join(DEFAULT_ROOT, "data"))
    ap.add_argument("--train_ann", default=None,
                    help="默认 annotations/train.json；想要与训权重时一致的"
                         "小集合可用 train_gate.json（3277 张）")
    ap.add_argument("--ckpt", default="best_sd.pt")
    ap.add_argument("--input_size", type=int, default=640,
                    help="必须与部署分辨率一致（本权重是 640 训练的）")
    ap.add_argument("--calib_dir", default="qr640",
                    help="已有 PTQ 标定目录，用来初始化激活量程（不加会慢很多）")
    ap.add_argument("--out_dir", default="qat640")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--quant_lr_ratio", type=float, default=10.0,
                    help="量化步进参数的学习率倍率（官方示例用单独一组更大 lr）")
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--min_lr_ratio", type=float, default=0.01)
    ap.add_argument("--warmup_epochs", type=int, default=1)
    ap.add_argument("--clip_grad", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_images", type=int, default=0, help=">0 只用前 N 张冒烟")
    ap.add_argument("--scale_max", type=float, default=1.2)
    ap.add_argument("--device", default=None,
                    help="默认自动：有 CUDA 用 cuda，否则 cpu")
    ap.add_argument("--no_mkldnn", action="store_true", default=True,
                    help="CPU 上端到端跑 backward 时，深度可分离卷积在部分 oneDNN "
                         "上会抛 'could not create a primitive descriptor "
                         "iterator'，关掉 mkldnn 走原生实现（略慢但能跑）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze_bn", action="store_true", default=True,
                    help="冻结 BN 统计量（微调默认开，避免小 batch 下统计量漂移）")
    ap.add_argument("--export_xmodel", action="store_true")
    ap.add_argument("--deploy", action="store_true", default=True,
                    help="训练结束导出 deployable 模型")
    ap.add_argument("--dnnl_isa", default=_ISA,
                    help="限制 oneDNN 指令集 AVX512_CORE/AVX2/SSE41，"
                         "用来避开 'could not create a primitive descriptor'")
    ap.add_argument("--dry_run", action="store_true",
                    help="跳过 QatProcessor，直接训原始浮点模型。"
                         "用来验证数据/损失/显存/吞吐，先于真正的 QAT 跑一遍")
    args = ap.parse_args()

    seed_everything(args.seed)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.no_mkldnn:
        torch.backends.mkldnn.enabled = False
    args.train_ann = args.train_ann or os.path.join(
        args.data_root, "annotations", "train.json")
    out_dir = ensure_dir(args.out_dir)
    ckpt_dir = ensure_dir(os.path.join(out_dir, "ckpt"))
    dev = torch.device(args.device)
    print("[env] device=%s | torch=%s | mkldnn=%s | threads=%d | oneDNN ISA=%s"
          % (dev, torch.__version__, torch.backends.mkldnn.enabled,
             torch.get_num_threads(), os.environ.get("ONEDNN_MAX_CPU_ISA")))

    model = build_model(args.ckpt, dev)
    # QatProcessor 会改写模块结构，先把纯浮点副本留着，最后用来算验收指标
    float_ref = copy.deepcopy(model)

    # Vitis QAT 不认裸的 torch.mul / torch.add / torch.cat，先换成 nn.Module。
    # 数学等价、权重不变，只在 QAT 路径生效。
    if not args.dry_run:
        from qat_patch import patch_for_qat
        patch_for_qat(model)
        # nndct 在 trainable_model 里做 conv+bn 折叠，内部有 assert conv.training，
        # 而 build_model 返回的是 eval 状态的模型，这里必须先切回 train
        model.train()

    tl, vl = make_loaders(args)

    if args.dry_run:
        print("[dry-run] 跳过 QatProcessor，直接训练原始浮点模型")
        proc, qm = None, model
    else:
        from pytorch_nndct import QatProcessor
        inputs = torch.zeros(args.batch, 3, args.input_size, args.input_size,
                             device=dev)
        print("[qat] QatProcessor 初始化（内部会强制关闭 CLE 与范围优化）...")
        proc = QatProcessor(model, inputs, bitwidth=8)
        calib = args.calib_dir if args.calib_dir and os.path.isdir(
            args.calib_dir) else ""
        # fpn 在 forward 里被多次调用（多尺度分支共用），必须显式允许复用，
        # 否则报 QUANTIZER_TORCH_REUSED_MODULE
        qm = proc.trainable_model(calib_dir=calib, allow_reused_module=True)
        print("[qat] 已用 %s 初始化激活量程" % (calib or "（无标定，从零学）"))
        if args.freeze_bn and hasattr(qm, "freeze_bn_stats"):
            qm.freeze_bn_stats()
    qm.to(dev)

    # ---- 优化器：权重一组、量化步进参数一组 ----
    try:
        qparams = list(qm.quantizer_parameters())
        wparams = list(qm.non_quantizer_parameters())
    except Exception as e:                              # 兜底：全当权重
        print("[warn] 取量化参数组失败: %r，退化为单一参数组" % (e,))
        qparams, wparams = [], list(qm.parameters())
    print("[qat] 权重参数 %d 组张量 / 量化参数 %d 组张量,  lr=%.2e quant_lr=%.2e"
          % (len(wparams), len(qparams), args.lr,
             args.lr * args.quant_lr_ratio))
    groups = [{"name": "weight", "params": wparams, "lr": args.lr,
               "weight_decay": args.wd}]
    groups.append({"name": "quantizer", "params": qparams,
                   "lr": args.lr * args.quant_lr_ratio, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, lr=args.lr)

    criterion = DetectionLoss()          # 注意：它不是 nn.Module，没有 .to()
    best = float("inf")
    history = []

    for ep in range(args.epochs):
        qm.train()
        t0 = time.time()
        spe = len(tl)
        run = {"loss": 0.0, "obj": 0.0, "box": 0.0, "lm": 0.0}
        cnt = 0
        bad = 0
        for i, (imgs, tgts, _) in enumerate(tl):
            cur = lr_at(ep, i, spe, args)
            for g in opt.param_groups:
                g["lr"] = cur if g.get("name") == "weight" \
                    else cur * args.quant_lr_ratio
            imgs = imgs.to(dev, non_blocking=True)
            tgts = move_targets(tgts, dev)
            out = criterion(qm(imgs), tgts)
            v = float(out["loss"].detach())
            if not math.isfinite(v):
                continue
            opt.zero_grad(set_to_none=True)
            try:
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(qm.parameters(), args.clip_grad)
                opt.step()
            except RuntimeError as e:
                # oneDNN 在部分 CPU 上会给深度可分离卷积挑不支持的 kernel，
                # backward 抛 "could not create a primitive descriptor iterator"。
                # 这是纯微调，丢掉个别 batch 不影响整体收敛，跳过继续。
                bad += 1
                if bad <= 3:
                    print("  [skip] step %d backward 失败: %s"
                          % (i, str(e).split("\n")[0]), flush=True)
                continue
            for k in run:
                run[k] += float(out[k].detach())
            cnt += 1
            if i % 50 == 0:
                print("  ep%d step %d/%d loss=%.4f obj=%.4f box=%.4f "
                      "lm=%.4f lr=%.2e (%.0fs)"
                      % (ep + 1, i, spe, v, float(out["obj"]),
                         float(out["box"]), float(out["lm"]), cur,
                         time.time() - t0), flush=True)
        train = {k: v / max(cnt, 1) for k, v in run.items()}
        val, vn = validate(qm, vl, criterion, dev)
        rec = {"epoch": ep + 1, "train_loss": round(train["loss"], 4),
               "train_obj": round(train["obj"], 4),
               "val_loss": round(val["loss"], 4),
               "val_obj": round(val["obj"], 4),
               "val_batches": vn, "skip_batches": bad,
               "sec": round(time.time() - t0, 1)}
        history.append(rec)
        print("[epoch %d/%d] %s" % (ep + 1, args.epochs, rec), flush=True)

        torch.save({"model": qm.state_dict(), "epoch": ep + 1,
                    "history": history}, os.path.join(ckpt_dir, "last.pt"))
        if val["loss"] < best:
            best = val["loss"]
            torch.save({"model": qm.state_dict(), "epoch": ep + 1},
                       os.path.join(ckpt_dir, "best.pt"))

    qm.eval()
    compare_with_float(float_ref, qm, vl, dev)

    if args.deploy and proc is not None:
        dep_dir = ensure_dir(os.path.join(out_dir, "deploy"))
        print("\n[qat] 导出 deployable 模型 ->", dep_dir)
        proc.to_deployable(qm, dep_dir)
        if args.export_xmodel:
            _ = proc.deployable_model(dep_dir, used_for_xmodel=True)
            proc.export_xmodel(dep_dir, deploy_check=False)
            print("[qat] xmodel 已导出 ->", dep_dir)
    torch.save(history, os.path.join(out_dir, "history.pt"))
    print("[done] 输出目录:", out_dir)


if __name__ == "__main__":
    main()
