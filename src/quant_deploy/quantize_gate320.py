#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gate320 (MobileNetV3-small + FPN-lite + 3 head) 量化脚本。

运行环境：Vitis-AI 3.0 docker 的 conda 环境 vitis-ai-pytorch（torch 1.12 / py3.7）
对应文档：UG1414 v3.0 第 3 章 p77-80（vai_q_pytorch）

三段流程：
  1) calib  用校准图跑前向，统计激活分布      -> quantize_result/Quant_info.json
  2) test   量化模型 vs 浮点模型 逐层输出比对  -> 判断量化损失（不训练、不需标签）
  3) deploy batch=1 跑一次，导出部署件         -> *.xmodel / 量化 *.onnx / *.pt

用法（容器 /workspace 下）：
  conda activate vitis-ai-pytorch

  # 1) 校准（100~1000 张图，无需标签）
  python quantize_gate320.py --quant_mode calib --n_calib 200 \
      --calib_dir /mnt/data1/face_bench/data/images/300w

  # 2) 量化后精度比对
  python quantize_gate320.py --quant_mode test --n_calib 50 \
      --calib_dir /mnt/data1/face_bench/data/images/300w

  # 3) 导出（batch 必须为 1）
  python quantize_gate320.py --quant_mode test --n_calib 1 --batch_size 1 --deploy \
      --calib_dir /mnt/data1/face_bench/data/images/300w

注意：
  * 量化 = 训练后量化(PTQ)，只做前向校准，不做反向传播、不需要标签、不训练
  * 预处理必须与训练一致：letterbox(320) -> RGB -> /255 -> (x-0.5)/0.5
  * 解码/NMS(decode_preds) 不在图内，量化不涉及
  * deploy 必须用 batch_size=1
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common.models.detector import build_detector, decode_preds  # noqa: E402

MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
PAD_VALUE = 114
EXTS = (".jpg", ".jpeg", ".png", ".bmp")


# --------------------------------------------------------------------------- #
# 数据：与训练一致的前处理（letterbox -> /255 -> (x-0.5)/0.5 -> CHW）
# --------------------------------------------------------------------------- #
def list_images(d, limit):
    paths = []
    for root, _, files in os.walk(d):
        for f in files:
            if f.lower().endswith(EXTS):
                paths.append(os.path.join(root, f))
    paths.sort()
    random.Random(0).shuffle(paths)
    return paths[:limit]


def letterbox(im, size):
    """等比缩放 + 居中贴到 114 灰底画布（与 common/dataset.letterbox 一致）。"""
    w, h = im.size
    s = size / float(max(h, w))
    nw, nh = int(round(w * s)), int(round(h * s))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (PAD_VALUE, PAD_VALUE, PAD_VALUE))
    canvas.paste(im, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def preprocess_one(path, size):
    """单张图 -> Tensor(3,size,size)，值域 [-1,1]。"""
    im = Image.open(path).convert("RGB")   # PIL 读出即 RGB，无需再 BGR2RGB
    im = letterbox(im, size)
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def make_batches(paths, size, batch_size):
    """返回 list[Tensor(B,3,320,320)]，值域 [-1,1]。"""
    xs = []
    for p in paths:
        im = Image.open(p).convert("RGB")          # PIL 读出即 RGB，无需再 BGR2RGB
        im = letterbox(im, size)
        x = np.asarray(im, dtype=np.float32) / 255.0
        x = (x - MEAN) / STD
        x = np.ascontiguousarray(x.transpose(2, 0, 1))
        xs.append(torch.from_numpy(x))
    return [torch.stack(xs[i:i + batch_size]) for i in range(0, len(xs), batch_size)]


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
def build_model(weights, device, qat_patch=False):
    model = build_detector("mobilenetv3", pretrained=False, width=1.0)
    sd = torch.load(weights, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    ret = model.load_state_dict(sd, strict=True)
    print("[model] load_state_dict ->", ret)
    if qat_patch:
        # 与 QAT 用同一套图结构（eltwise 都换成 nn.Module），
        # 这样生成的 quant_info.json 节点名才能被 QatProcessor 对上
        from qat_patch import patch_for_qat
        patch_for_qat(model)
    model.eval().to(device)
    return model


@torch.no_grad()
def run(model, batches, device):
    outs = []
    for x in batches:
        outs.append([o.detach().float().cpu() for o in model(x.to(device))])
    return outs


def cat_levels(outs):
    return [torch.cat([o[i] for o in outs], dim=0) for i in range(len(outs[0]))]


def compare(float_outs, quant_outs):
    print("\n[compare] 量化 vs 浮点（三个输出：stride 8/16/32）")
    print("  lvl  shape                cos-sim    mean|d|      rel-L2")
    worst = 0.0
    for i, (a, b) in enumerate(zip(float_outs, quant_outs)):
        d = (a - b).abs()
        rel = (torch.norm(a - b) / (torch.norm(a) + 1e-12)).item()
        cos = torch.nn.functional.cosine_similarity(
            a.reshape(1, -1), b.reshape(1, -1), dim=1).item()
        worst = max(worst, rel)
        print("  %d   %-18s  %.6f   %.6f   %.6f"
              % (i, str(tuple(b.shape)), cos, d.mean().item(), rel))
    print("  最大相对误差 rel-L2 = %.6f" % worst)
    print("  参考：cos>0.99 且 rel-L2<0.05 才算健康；")
    print("        cos≈0.87 / rel-L2≈0.5 表示「幅度对但内容被糊掉」\n")

    # ---- 分通道：ch0=obj, ch1-4=box(l,t,r,b), ch5-14=landmark(5x2) ----
    print("[compare] 分通道 rel-L2  (ch0=obj | ch1..4=box | ch5..14=landmark)")
    for i, (a, b) in enumerate(zip(float_outs, quant_outs)):
        ch = []
        for c in range(a.shape[1]):
            av, bv = a[:, c].reshape(-1), b[:, c].reshape(-1)
            ch.append((torch.norm(av - bv) / (torch.norm(av) + 1e-12)).item())
        print("  lvl%d  obj=%.3f" % (i, ch[0]))
        print("        box= " + " ".join("%.3f" % x for x in ch[1:5]))
        print("        lm = " + " ".join("%.3f" % x for x in ch[5:]))
    print("  → 若 obj/box 小、lm 大：检测还在，只是关键点糊了；")
    print("    若全通道都大：是 backbone 级的问题，不是 head 的问题\n")


def purge_hooks(processor):
    """清掉 AdaQuant 在失败层上残留的 forward hook。

    optimize_layer_v2 内部是「注册 hook → 前向 → clean_hooks(handlers)」；
    一旦中途抛异常，clean_hooks 就执行不到，残留 hook 会在下一层的前向里
    提前抛 StopForward，导致 _cached_output 取到脏值（表现为 qin 变成 list）。
    这里直接把两个模型上所有 forward hook 摘掉，并把单层缓存清空。
    """
    n = 0
    for attr in ("_float_model", "quant_model", "_quant_model"):
        root = getattr(processor, attr, None)
        if root is None or not hasattr(root, "modules"):
            continue
        for mod in root.modules():
            for h in ("_forward_hooks", "_forward_pre_hooks"):
                d = getattr(mod, h, None)
                if isinstance(d, dict) and d:
                    n += len(d)
                    d.clear()
    cache = getattr(processor, "_cached_output", None)
    if isinstance(cache, dict) and cache:
        n += len(cache)
        cache.clear()
    if n:
        print("[ft-guard] 已清理残留 hook/缓存 %d 个" % n, flush=True)


def install_ft_guard():
    """给 AdvancedQuantProcessor.optimize_layer_v2 套一层 try/except。

    AdaQuant 逐层跑 loss.backward()，个别 conv（深度可分离最常见）在某些 CPU
    的 oneDNN 上会抛 "could not create a primitive descriptor iterator"。
    包一层后：失败的层跳过、权重不更新，其余层继续微调，整体流程不中断。
    """
    try:
        import pytorch_nndct                                 # 先确保包已导入
        from pytorch_nndct.qproc import adaquant as _aq
    except Exception as e:                                   # pragma: no cover
        print("[ft-guard] 导入 adaquant 失败: %r" % (e,), flush=True)
        return
    print("[ft-guard] 模块:", getattr(_aq, "__file__", "?"))

    # 注意：optimize_layer_v2 / finetune_v2 属于 AdvancedQuantProcessor，
    # 不是 AdaQuant —— 这里按“谁有这个方法就 patch 谁”来找，避免硬编码类名
    targets = [c for c in vars(_aq).values()
               if isinstance(c, type) and hasattr(c, "optimize_layer_v2")]
    if not targets:
        print("[ft-guard][err] 没找到带 optimize_layer_v2 的类", flush=True)
        return

    for cls in targets:
        orig = cls.optimize_layer_v2
        if getattr(orig, "_guarded", False):
            print("[ft-guard] %s 已安装过，跳过" % cls.__name__, flush=True)
            continue

        def make(orig=orig):
            def guarded(self, qnode, *a, **k):
                name = getattr(qnode, "name", None) or str(qnode)
                try:
                    return orig(self, qnode, *a, **k)
                except RuntimeError as e:
                    print("[ft-guard] 跳过层 %s -> %s"
                          % (name, str(e).split("\n")[0]), flush=True)
                    purge_hooks(self)
                    # 位置参数第 5 个（下标 4）是 net_loss，原样返回
                    return a[4] if len(a) > 4 else None
            guarded._guarded = True
            return guarded

        cls.optimize_layer_v2 = make()
        print("[ft-guard] 安装成功 -> %s.optimize_layer_v2（单层失败自动跳过）"
              % cls.__name__, flush=True)


def describe(tag, levels):
    """打印三路输出的绝对值统计；并跑一次解码，看还能不能检出人脸。

    用途：先确认「浮点模型本身是好的」，再判断量化误差。
    如果浮点模型自己就检不出人脸，那量化误差再小也没意义。
    """
    print("\n[%s] 三路输出统计（未解码的原始 head 输出）" % tag)
    print("  lvl  shape                ||y||       mean       std       max")
    for i, y in enumerate(levels):
        y = y.detach().float()
        print("  %d   %-18s  %9.4f  %8.4f  %8.4f  %8.4f"
              % (i, str(tuple(y.shape)), y.norm().item(), y.mean().item(),
                 y.std().item(), y.max().item()))
    try:
        dets = decode_preds([l[:1] for l in levels])
        boxes, scores = dets[0][0], dets[0][1]
        n = len(scores)
        top = float(scores.max()) if n else 0.0
        print("  解码(第 1 张图, conf>0.05 + NMS): %d 个框, 最高分 %.4f"
              % (n, top))
        if n:
            b = boxes[0].tolist()
            print("  首选框 xyxy = [%.1f, %.1f, %.1f, %.1f]" % tuple(b))
    except Exception as e:                                  # 解码失败不阻断主流程
        print("  [warn] 解码失败: %r" % (e,))


def build_ft_set(model, paths, size, device):
    """快速微调用的蒸馏数据集：(输入, 浮点模型输出)，不需要标注。"""
    data = []
    model.eval()
    with torch.no_grad():
        for p in paths:
            x = preprocess_one(p, size).unsqueeze(0).to(device)
            y = [o.detach().float().cpu() for o in model(x)]
            data.append((x.cpu(), y))
    return data


# AdaQuant 加权配置，由 main() 按命令行填入。
# 默认值（lm_w=1, fg_w=0, lv_w 全 1）与「全图等权 MSE」逐项等价，
# 即保持历史行为不变，只有显式传参才会改变优化目标。
FT_CFG = {"lm_w": 1.0, "fg_w": 0.0, "lv_w": (1.0, 1.0, 1.0), "ch_w": None}


def _build_ch_w(n_ch, lm_w):
    """通道权重：cls(1) + box(4) + lm(10)。通道数不符或 lm_w=1 时退化为等权。"""
    if n_ch != 15 or lm_w == 1.0:
        return None
    w = torch.ones(n_ch, dtype=torch.float32)
    w[5:15] = lm_w
    return w


def weighted_mse(o, y, li):
    """按 前景 / 通道 / 层级 加权的误差。

    y 是浮点模型的输出（蒸馏目标），它自带的 cls 通道正好可以当
    「这个位置是不是人脸」的指示器 —— 所以不需要任何标注。
    """
    o = o.float()
    y = y.float()
    sq = (o - y) ** 2
    w = None
    if FT_CFG["fg_w"] > 0:
        fg = torch.sigmoid(y[:, 0:1].detach())
        w = 1.0 + FT_CFG["fg_w"] * fg
    cw = FT_CFG["ch_w"]
    if cw is not None and cw.numel() == sq.shape[1]:
        cw = cw.to(sq.device).view(1, -1, 1, 1)
        w = cw if w is None else w * cw
    lw = FT_CFG["lv_w"][li] if li < len(FT_CFG["lv_w"]) else 1.0
    return ((sq * w).mean() if w is not None else sq.mean()) * lw


def ft_evaluate(model, loader, loss_fn):
    """AdaQuant 回调：只做前向 + 求 loss（反向由 NNDCT 内部完成）。"""
    model.eval()
    total, n = 0.0, 0
    plain = (FT_CFG["lm_w"] == 1.0 and FT_CFG["fg_w"] == 0.0
             and all(v == 1.0 for v in FT_CFG["lv_w"]))
    with torch.no_grad():
        for x, ys in loader:
            outs = model(x)
            if plain:
                loss = sum(loss_fn(o.float(), y) for o, y in zip(outs, ys))
            else:
                loss = sum(weighted_mse(o, y, li)
                           for li, (o, y) in enumerate(zip(outs, ys)))
            total += loss.item()
            n += 1
    return total / max(n, 1), 0.0, 0.0


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser("gate320 量化 (vai_q_pytorch)")
    ap.add_argument("--weights", default="best_sd.pt",
                    help="纯 state_dict（宿主机已用 pickle_protocol=2 转存）")
    ap.add_argument("--input_size", type=int, default=320)
    ap.add_argument("--calib_dir", required=True, help="校准图目录（无需标签）")
    ap.add_argument("--n_calib", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--quant_mode", default="calib", choices=["calib", "test"])
    ap.add_argument("--deploy", action="store_true", help="导出 xmodel/onnx/pt")
    ap.add_argument("--output_dir", default="quantize_result")
    ap.add_argument("--net_name", default="gate320")
    ap.add_argument("--qat_patch", action="store_true",
                    help="用 qat_patch 改写 eltwise 算子，使图结构与 QAT 一致"
                         "（只在为 QAT 准备校准文件时需要）")
    ap.add_argument("--config", default=None,
                    help="量化策略 json（per_channel / CLE / method 等）")
    ap.add_argument("--fast_finetune", action="store_true",
                    help="校准前做快速微调（蒸馏式，无需标签）")
    ap.add_argument("--ft_images", type=int, default=256,
                    help="快速微调图数（官方 ImageNet 用 5120；CPU 可先取 256）")
    ap.add_argument("--ft_lm_weight", type=float, default=1.0,
                    help="AdaQuant loss 中 landmark 通道(5~14)的权重；"
                         "1.0 = 与历史行为一致（等权）")
    ap.add_argument("--ft_fg_weight", type=float, default=0.0,
                    help="AdaQuant loss 中前景位置的额外权重，用浮点模型"
                         "自带的 cls 分数作指示（无需标注）；0 = 不加权")
    ap.add_argument("--ft_level_weight", default="1,1,1",
                    help="各特征层级的 loss 权重，逗号分隔，如 2,1,1 "
                         "（level0 负责小脸，误差最大，可加重）")
    ap.add_argument("--no_mkldnn", action="store_true",
                    help="关闭 oneDNN(MKLDNN)（AdaQuant 反向在深度可分离卷积上"
                         "可能报 could not create a primitive descriptor iterator）")
    ap.add_argument("--ft_skip_error", action="store_true",
                    help="AdaQuant 逐层优化时，对 backward 抛异常的层跳过而不是中断")
    # 用 known_args：剩下的 --nndct_* 交给 pytorch_nndct 自己解析
    args, unknown = ap.parse_known_args()
    if unknown:
        print("[opt] 透传给 nndct:", unknown)

    # AdaQuant 加权目标（默认与历史等权 MSE 完全一致）
    FT_CFG["lm_w"] = args.ft_lm_weight
    FT_CFG["fg_w"] = args.ft_fg_weight
    FT_CFG["lv_w"] = tuple(float(v) for v in args.ft_level_weight.split(","))
    FT_CFG["ch_w"] = _build_ch_w(15, args.ft_lm_weight)
    if not (args.ft_lm_weight == 1.0 and args.ft_fg_weight == 0.0
            and all(v == 1.0 for v in FT_CFG["lv_w"])):
        print("[ft] 加权 loss: lm×%.1f  前景+%.1f  level=%s"
              % (args.ft_lm_weight, args.ft_fg_weight, FT_CFG["lv_w"]))

    # AdaQuant 需要对每个 conv 做 backward，深度可分离卷积在某些 CPU 的 oneDNN
    # 上会抛 "could not create a primitive descriptor iterator"。
    # 关掉 mkldnn 走 PyTorch 原生实现即可（略慢，但能跑）。
    if args.no_mkldnn or args.fast_finetune:
        torch.backends.mkldnn.enabled = False

    if args.deploy and args.batch_size != 1:
        raise SystemExit("[err] deploy 时 batch_size 必须为 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[env] device =", device, "| torch =", torch.__version__,
          "| mkldnn =", torch.backends.mkldnn.enabled)

    # 1) 浮点模型
    model = build_model(args.weights, device, args.qat_patch)

    # 2) 校准数据
    paths = list_images(args.calib_dir, args.n_calib)
    if not paths:
        raise SystemExit("[err] 校准目录没有图片: %s" % args.calib_dir)
    batches = make_batches(paths, args.input_size, args.batch_size)
    print("[data] %d 张校准图 -> %d 个 batch" % (len(paths), len(batches)))

    # 3) 量化器
    from pytorch_nndct.apis import torch_quantizer
    dummy = torch.randn(args.batch_size, 3, args.input_size, args.input_size,
                        device=device)
    kw = dict(output_dir=args.output_dir, device=device)
    if args.config:
        kw["quant_config_file"] = args.config
    try:
        quantizer = torch_quantizer(args.quant_mode, model, (dummy,), **kw)
    except TypeError:  # 老版本 torch_quantizer 无 device 参数
        kw.pop("device", None)
        quantizer = torch_quantizer(args.quant_mode, model, (dummy,), **kw)
    quant_model = quantizer.quant_model
    quant_model.eval().to(device)

    if args.ft_skip_error or args.fast_finetune:
        install_ft_guard()

    # 快速微调（AdaQuant）：calib 阶段微调权重，test 阶段载入微调后的参数
    if args.fast_finetune:
        if args.quant_mode == "calib":
            ft_paths = list_images(args.calib_dir, args.ft_images)
            ft_set = build_ft_set(model, ft_paths, args.input_size, device)
            print("[ft] 快速微调：%d 张图，loss = MSE(量化输出, 浮点输出)"
                  % len(ft_set))
            try:
                quantizer.fast_finetune(ft_evaluate,
                                        (quant_model, ft_set, torch.nn.MSELoss()))
            except RuntimeError as e:
                # 兜底：微调整体失败也不让流程中断，用（可能已被部分微调的）
                # 权重继续做校准
                print("[ft][warn] 快速微调中断，改用未微调权重继续校准：%s"
                      % str(e).split("\n")[0], flush=True)
        else:
            quantizer.load_ft_param()
            print("[ft] 已载入快速微调参数")

    # 4) 前向
    print("[run] quant_mode = %s ..." % args.quant_mode)
    quant_outs = run(quant_model, batches, device)
    print("[run] 输出: %s" % [tuple(b.shape) for b in quant_outs[0]])
    float_outs = None
    if args.quant_mode == "test":
        float_outs = run(model, batches, device)
        describe("float", cat_levels(float_outs))
        describe("quant", cat_levels(quant_outs))
        compare(cat_levels(float_outs), cat_levels(quant_outs))

    # 5) 产物
    if args.quant_mode == "calib":
        quantizer.export_quant_config()
        print("[ok] 校准完成 -> %s/quant_info.json" % args.output_dir)
        print("     下一步: --quant_mode test 看量化误差")

    if args.deploy:
        # 必须显式传 output_dir：nndct 的导出接口默认是 "quantize_result"，
        # 不传的话所有分辨率都会往同一个目录覆盖（APIs.py:129/132/141）
        os.makedirs(args.output_dir, exist_ok=True)
        quantizer.export_torch_script(output_dir=args.output_dir)
        quantizer.export_onnx_model(output_dir=args.output_dir)
        quantizer.export_xmodel(output_dir=args.output_dir, deploy_check=False)
        print("[ok] 部署件已导出 -> %s/" % args.output_dir)
        for f in sorted(os.listdir(args.output_dir)):
            print("     ", f, "%.2f MB" % (os.path.getsize(
                os.path.join(args.output_dir, f)) / 1e6))
        xm = [f for f in os.listdir(args.output_dir) if f.endswith("_int.xmodel")]
        print("     下一步(主机编译): vai_c_xir -x %s/%s -a arch.json "
              "-o compiled -n %s"
              % (args.output_dir, xm[0] if xm else "*.xmodel", args.net_name))


if __name__ == "__main__":
    main()
