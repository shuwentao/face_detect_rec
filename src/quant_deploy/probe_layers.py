#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐层定位「量化误差第一次爆炸」的位置。

用法（test 模式已有 quant_info.json 之后）：
  conda activate vitis-ai-pytorch
  python -u probe_layers.py --output_dir quantize_result --n_img 4 \
      --calib_dir /mnt/data1/face_bench/data/images/300w 2>&1 | tr -d '\r' | tee probe.log

  # 只看名字（排障用，确认两边模块树怎么对应）
  python -u probe_layers.py --output_dir quantize_result --dump_names \
      --calib_dir /mnt/data1/face_bench/data/images/300w
"""

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from quantize_gate320 import (build_model, list_images,     # noqa: E402
                              make_batches)


import re

_NODE_SEG = re.compile(r"[^\[\]/]*\[([^\]]+)\]")


def node_path_candidates(node, prefer_roots=()):
    """从 NNDCT 图节点名反推浮点模型的模块路径。

    节点名形如
        FaceDetector::FaceDetector/TVBackbone[backbone]/
        Conv2dNormActivation[features]/Conv2dNormActivation[0]/Conv2d[0]/input.3
    方括号里的部分恰好是浮点模型里的属性名，拼起来就是 backbone.features.0.0。

    头部可能多出顶层类名、尾部可能带 input.3/output 这类张量名，所以取所有
    连续子路径作为候选；以浮点模型顶层模块名开头的候选优先（最接近真实路径）。
    """
    name = getattr(node, "name", None) or str(node)
    if "::" in name:
        name = name.split("::", 1)[1]
    segs = []
    for seg in name.split("/"):
        m = _NODE_SEG.match(seg.strip())
        v = m.group(1) if m else seg.strip()
        if v:
            segs.append(v)
    roots = set(prefer_roots)
    out = []
    for k in range(len(segs)):
        for t in range(len(segs), k, -1):
            cand = ".".join(segs[k:t])
            out.append((cand, cand.split(".")[0] in roots, -(t - k)))
    # 以顶层模块名开头的优先，其次取更长的（信息更完整）
    out.sort(key=lambda r: (not r[1], r[2]))
    return [c for c, _a, _b in out]


def pair_by_node(quant_model, float_names, verbose=False):
    """量化侧模块 -> 浮点侧模块路径，靠 module.node.name 的方括号部分还原。"""
    pairs, how = [], ""
    with_node = 0
    seen = {}
    roots = {n.split(".")[0] for n in float_names}
    for qn, qm in quant_model.named_modules():
        if not qn:
            continue
        node = getattr(qm, "node", None)
        if node is None:
            continue
        with_node += 1
        for cand in node_path_candidates(node, roots):
            if cand in float_names:
                seen[cand] = qn          # 同一路径重复时保留最后一个（外层输出）
                break
    pairs = list(seen.items())
    if pairs:
        how = "按 NNDCT 图节点名还原模块路径匹配"
    if verbose:
        print("[probe] 量化侧带 .node 的模块 %d 个，其中 %d 个还原出浮点路径"
              % (with_node, len(pairs)))
    return pairs, how


def named_all(model):
    """所有非根模块的名字（含容器）。不再用 is_leaf —— 量化侧每个算子都会
    多出子模块（inplace 包装），按叶子判定会得到空集。"""
    return [n for n, _ in model.named_modules() if n]


@torch.no_grad()
def collect(model, names, x, with_input=False):
    """跑一次前向，抓每个算子的输出（已是 float 值域）。

    with_input=True 时同时抓第一个输入张量 —— 用来看「误差是不是在上一层的
    输出/量化器就已经产生」，尤其是首层的输入就是 QuantStub 的输出。
    """
    feats, ins = {}, {}
    ok = 0

    def make_hook(name):
        def hook(_m, inp, out):
            if with_input and inp and torch.is_tensor(inp[0]):
                ins[name] = inp[0].detach().float().cpu()
            if torch.is_tensor(out):
                feats[name] = out.detach().float().cpu()
            elif isinstance(out, (list, tuple)) and out and torch.is_tensor(out[0]):
                feats[name] = torch.cat(
                    [o.detach().float().cpu().reshape(-1) for o in out])
        return hook

    hs = []
    for n, m in model.named_modules():
        if n not in names:
            continue
        try:
            hs.append(m.register_forward_hook(make_hook(n)))
            ok += 1
        except Exception as e:                       # ScriptModule 等不允许挂 hook
            print("  [warn] 无法挂 hook: %s -> %r" % (n, e))
    model(x)
    for h in hs:
        h.remove()
    return feats, ins, ok


def rel(a, b):
    return (torch.norm(a - b) / (torch.norm(a) + 1e-12)).item()


def build_quant(model, x, output_dir, config, device):
    from pytorch_nndct.apis import torch_quantizer
    kw = dict(output_dir=output_dir, device=device)
    if config:
        kw["quant_config_file"] = config
    try:
        q = torch_quantizer("test", model, (x,), **kw)
    except TypeError:                                # 老版本无 device 参数
        kw.pop("device", None)
        q = torch_quantizer("test", model, (x,), **kw)
    return q.quant_model.eval().to(device)


def main():
    ap = argparse.ArgumentParser("逐层量化误差定位")
    ap.add_argument("--weights", default="best_sd.pt")
    ap.add_argument("--input_size", type=int, default=320)
    ap.add_argument("--calib_dir", required=True)
    ap.add_argument("--n_img", type=int, default=2,
                    help="探针图片数。所有中间张量都要留在内存，默认 2 保守")
    ap.add_argument("--output_dir", default="quantize_result")
    ap.add_argument("--config", default=None)
    ap.add_argument("--thresh", type=float, default=0.05)
    ap.add_argument("--dump_names", action="store_true",
                    help="只打印两边模块名映射，用来确认对应关系")
    ap.add_argument("--with_input", action="store_true",
                    help="同时抓每个模块的输入张量，判断误差是上一层带来的"
                         "还是本层产生的（内存翻倍，建议配 --n_img 1）")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.weights, device)

    paths = list_images(args.calib_dir, max(args.n_img, 1))
    x = make_batches(paths, args.input_size, args.n_img)[0].to(device)
    print("[probe] %d 张图, shape=%s, device=%s"
          % (len(paths), tuple(x.shape), device))

    quant_model = build_quant(model, x, args.output_dir, args.config, device)

    # NNDCT 是 inplace=True 原地替换算子：量化包装器会把原 module 收成自己的
    # 子模块（见 TorchQuantizer.__init__），所以量化侧"叶子"数量会是 0，
    # 但【名字保持不变】。因此这里对所有具名模块挂 hook，按完全同名配对。
    lf = named_all(model)
    lq = named_all(quant_model)
    print("[probe] 具名模块：浮点 %d / 量化 %d" % (len(lf), len(lq)))

    sf = set(lf)
    # 量化侧名字全是 module_0/1/2，信息已丢；靠 module.node.name 的方括号部分
    # 还原成浮点侧的真实模块路径（见 node_path_candidates）
    pairs, how = pair_by_node(quant_model, sf, verbose=True)
    if not pairs:                                     # 退化：剥掉唯一外层前缀
        def strip1(ns):
            if len({n.split(".")[0] for n in ns}) == 1:
                return {n.split(".", 1)[1]: n for n in ns if "." in n}
            return {n: n for n in ns}
        a, b = strip1(lf), strip1(lq)
        common = set(a) & set(b)
        if common:
            pairs = [(a[k], b[k]) for k in common]
            how = "剥离外层 wrapper 后同名匹配"
    if not pairs:                                     # 再退化：按名字后缀互相包含
        used = set()
        for q in lq:
            for f in lf:
                if f in used:
                    continue
                if f.endswith("." + q) or q.endswith("." + f) or f == q:
                    pairs.append((f, q))
                    used.add(f)
                    break
        if pairs:
            how = "按名字后缀匹配（两边差若干层前缀）"

    print("[probe] 模块树：%s -> %d 对可比对" % (how, len(pairs)))

    # dump 必须放在前面：配对失败时才更需要看真实命名
    if args.dump_names or not pairs:
        print("\n[probe] 量化模型 Diagnosis")
        print("  type          =", type(quant_model).__name__,
              "| mro =", [c.__name__ for c in type(quant_model).__mro__[:4]])
        print("  named_modules =", len(lq), "（浮点侧 %d）" % len(lf))
        for attr in ("module", "model", "_model", "_float_model", "graph"):
            v = getattr(quant_model, attr, None)
            if v is not None:
                print("  有属性 %-14s -> %s" % (attr, type(v).__name__))
        print("\n--- 浮点侧名字（前 30） ---")
        for n in lf[:30]:
            print("   ", n)
        print("--- 量化侧名字（前 30） ---")
        for n in lq[:30]:
            print("   ", n)
        if pairs:
            print("\n--- 前 10 对映射 ---")
            for a, b in pairs[:10]:
                print("   %-40s <-> %s" % (a, b))
        return

    ff, fi, n_hf = collect(model, set(lf), x, args.with_input)
    fq, qi, n_hq = collect(quant_model, set(lq), x, args.with_input)
    print("[probe] 取到输出：浮点 %d/%d / 量化 %d/%d"
          % (len(ff), n_hf, len(fq), n_hq))
    if not ff or not fq:
        print("[probe][err] hook 没取到输出，无法定位")
        return

    rows = []
    skipped = []
    for nf, nq in pairs:
        if nf not in ff or nq not in fq:
            continue
        a, b = ff[nf], fq[nq]
        if a.shape != b.shape:                 # 配对错了才会形状不同
            skipped.append("shape %s vs %s : %s <-> %s"
                           % (tuple(a.shape), tuple(b.shape), nf, nq))
            continue
        a, b = a.reshape(-1).float(), b.reshape(-1).float()
        # in_rel：本层【输入】的误差。用量判定误差是上一层带来的还是本层产生的
        in_rel = None
        if args.with_input and nf in fi and nq in qi:
            ia, ib = fi[nf], qi[nq]
            if ia.shape == ib.shape:
                in_rel = rel(ia.reshape(-1).float(), ib.reshape(-1).float())
        rows.append((rel(a, b), nf, torch.norm(a).item(), torch.norm(b).item(),
                     in_rel))

    print("[probe] 可比对 %d / 形状不匹配跳过 %d" % (len(rows), len(skipped)))
    if skipped:
        print("[probe][warn] 下列配对形状不一致（前 5 条），说明模块对应关系有误：")
        for s in skipped[:5]:
            print("        ", s)
    if len(skipped) > len(pairs) * 0.2:
        print("[probe][err] 超过 20%% 的对形状不一致 -> 配对方式 (%s) 不可靠，"
              "下面的结论不成立，请跑 --dump_names 把映射贴回来" % how)
        return
    if not rows:
        print("[probe][err] 取不到任何可比对的层")
        return

    print("\n[probe] 前向顺序逐层 rel-L2（'>' 超过 %.3f）" % args.thresh)
    hdr = "  %-9s  %-12s  %-11s  %s" % ("rel-L2", "||y_float||", "||yq||/||yf||", "layer")
    if args.with_input:
        hdr = ("  %-9s  %-9s  %-12s  %-11s  %s"
               % ("out-rel", "in-rel", "||y_float||", "||yq||/||yf||", "layer"))
    print(hdr)
    first = None
    for i, row in enumerate(rows):
        r, n, nrm, nrmq, in_rel = row
        if r > args.thresh and first is None:
            first = n
        if r > 0.005 or i < 3:
            tail = "%s" % n
            if args.with_input:
                print("  %s%.6f  %9s  %12.4f  %11.4f  %s"
                      % (">" if r > args.thresh else " ", r,
                         "-" if in_rel is None else "%.6f" % in_rel,
                         nrm, nrmq / (nrm + 1e-12), tail))
            else:
                print("  %s%.6f  %12.4f  %11.4f  %s"
                      % (">" if r > args.thresh else " ", r, nrm,
                         nrmq / (nrm + 1e-12), tail))

    if args.with_input:
        print("\n[probe] 首层输入误差 —— 用来判定误差源头")
        for r, n, nrm, nrmq, in_rel in rows[:1]:
            print("  %s : out-rel=%.6f  in-rel=%s"
                  % (n, r, "-" if in_rel is None else "%.6f" % in_rel))
        print("  in-rel 是 QuantStub 输出的量化误差：")
        print("    in-rel >= 0.02 -> 输入量化就是元凶，改前处理让值域吃满 int8")
        print("    in-rel < 0.005 -> 输入干净，误差是首个 conv 的权重造成的")

    top = sorted(rows, reverse=True)
    print("\n[probe] 误差最大 15 层")
    for r, n, nrm, nrmq, in_rel in top[:15]:
        print("  %.6f  ||y||=%10.4f  幅值比=%.4f  %s"
              % (r, nrm, nrmq / (nrm + 1e-12), n))

    print("\n[probe] 分段聚合（每段内最大 rel-L2）")
    seg = {}
    for r, n, _a, _b, _ir in rows:
        key = ".".join(n.split(".")[:3]) or n
        seg[key] = max(seg.get(key, 0.0), r)
    for key, r in sorted(seg.items()):
        print("  %-34s %.4f %s" % (key, r, "#" * int(min(60, max(1, r * 60)))))

    print("\n[probe] 全层峰值 rel-L2 = %.6f -> %s" % (top[0][0], top[0][1]))
    if first:
        print("[probe] 首次超过 %.3f 的层 = %s" % (args.thresh, first))
        print("        之前基本忠实 -> 查这层权重分布 / 上一层激活量程")
    elif top[0][0] < 0.02:
        print("[probe] 结论：单层都干净，误差是被 head 放大的。")
        print("        回归 head 常是「大数相减得小数」，backbone 上 2% 噪声")
        print("        到输出能放大成 50% -> PTQ 没救，上 AdaQuant / QAT")
    else:
        print("[probe] 结论：误差逐层累积，无单点爆炸（典型 MobileNetV3 per_tensor")
        print("        量化天花板）-> 上 AdaQuant / QAT")


if __name__ == "__main__":
    main()
