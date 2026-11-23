"""把训练好的骨干导出成 ONNX，并做 DPU 兼容性体检。

为什么单独做一次体检
--------------------------------------------------------------------------
DPUCZDX8G 只认一部分算子。一旦图里混进 LayerNorm / GELU / Attention 这类
算子，量化时它们会被丢到 CPU 上（甚至直接编译失败），板子上速度会崩。
本脚本在**上板之前**把算子清单打出来，并对照 DPU 支持表给出警告。

导出约定
--------------------------------------------------------------------------
* 输入  `input`  : float32 [N, 3, 112, 112]，值域 [-1, 1]，RGB
* 输出  `embedding` : float32 [N, 512] —— **未归一化**的原始特征。
  L2 归一化放在 CPU 端做（ReduceL2 在 DPU 上支持有限，且归一化只有
  512 次乘加，放在 ARM 上几乎不花时间）。若确实想把归一化塞进图里，
  加 `--l2norm`。

用法
--------------------------------------------------------------------------
    python export_onnx.py --ckpt checkpoints/backbone_ep20.pth --out face.onnx
    python export_onnx.py --ckpt ... --out face.onnx --l2norm --batch 1
"""
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.ghostfacenet import build_backbone, fuse_for_deploy  # noqa: E402

# DPUCZDX8G (Vitis AI 3.0) 硬件算子。未列出的算子会退到 CPU 或直接报错。
DPU_OK = {
    "Conv", "ConvTranspose", "BatchNormalization", "Relu", "Relu6",
    "LeakyRelu", "PRelu", "HardSigmoid", "HardSwish", "MaxPool",
    "AveragePool", "GlobalAveragePool", "Add", "Mul", "Sub", "Div",
    "Concat", "Resize", "Upsample", "MatMul", "Gemm", "Softmax",
    "Flatten", "Reshape", "Transpose", "Pad", "Slice", "Clip", "Mean",
    "Sigmoid", "Tanh", "Max", "Min", "Sqrt", "Pow", "Split",
    "DepthToSpace", "SpaceToDepth", "Constant", "Identity",
}
# 明确会掉 CPU（或量化不稳）的算子，看到就提醒
DPU_BAD = {"LayerNormalization", "InstanceNormalization", "Gelu", "Erf",
           "Einsum", "NonMaxSuppression", "RoiAlign", "GridSample",
           "ReduceL2", "Sin", "Cos", "Exp", "Log", "Softplus", "ScatterND"}


class WrapNCHW(torch.nn.Module):
    """包一层固定输入形状，让 trace 更稳，并可选在图内做 L2 归一化。"""

    def __init__(self, backbone, l2norm: bool = False):
        super().__init__()
        self.backbone = backbone
        self.l2norm = l2norm

    def forward(self, x):
        f = self.backbone(x)
        if self.l2norm:
            f = torch.nn.functional.normalize(f, dim=1)
        return f


def load_backbone(ckpt: str, arch=None, width=None, emb=None, device="cpu"):
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "backbone" in sd:
        arch = arch or sd.get("arch", "ghostfacenet")
        width = width or sd.get("width", 1.0)
        emb = emb or sd.get("embedding_size", 512)
        sd = sd["backbone"]
    arch = arch or "ghostfacenet"
    width = width or 1.0
    emb = emb or 512
    m = build_backbone(arch, emb, width)
    m.load_state_dict(sd)
    m.eval().to(device)
    return m, arch, width, emb


def check_ops(onnx_path: str):
    import onnx
    model = onnx.load(onnx_path)
    ops = {}
    for node in model.graph.node:
        ops[node.op_type] = ops.get(node.op_type, 0) + 1
    print("[ops] 算子统计:")
    bad, warn = [], []
    for k in sorted(ops):
        tag = ""
        if k in DPU_BAD:
            tag = "  ← DPU 风险"
            bad.append(k)
        elif k not in DPU_OK:
            tag = "  ← 未知，需确认"
            warn.append(k)
        print(f"    {k:<24s} x{ops[k]}{tag}")
    if bad:
        print(f"[warn] 这些算子在 DPUCZDX8G 上会掉到 CPU: {bad}")
    if warn:
        print(f"[info] 这些算子不在已知白名单里，建议上板前确认: {warn}")
    if not bad and not warn:
        print("[ok] 全部算子均落在 DPU 支持范围内")
    n_params = sum(int(np.prod(t.dims)) for t in model.graph.initializer)
    print(f"[ops] 权重参数总量 ≈ {n_params/1e6:.2f}M")
    return bad, warn


def verify_numeric(m, onnx_path: str, emb: int, device="cpu",
                   batch: int = 1, l2norm: bool = False):
    """用 onnxruntime 跑一遍，和 PyTorch 对齐数值。"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[skip] 未安装 onnxruntime，跳过数值校验")
        return
    x = torch.randn(batch, 3, 112, 112)
    with torch.no_grad():
        y_pt = m(x).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    y_ort = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    diff = float(np.abs(y_pt - y_ort).max())
    denom = max(float(np.abs(y_pt).max()), 1e-6)
    print(f"[check] PyTorch vs onnxruntime: 最大绝对误差={diff:.3e} "
          f"(相对 {diff/denom:.2e})")
    if l2norm:
        n = np.linalg.norm(y_ort, axis=1)
        print(f"[check] 输出 L2 范数 = {n}（应为 1.0）")
    else:
        print(f"[check] 输出特征范数 ≈ {np.linalg.norm(y_ort, axis=1)}"
              f"（未归一化，部署时在 CPU 做 L2）")
    assert diff < 1e-3, "导出结果与 PyTorch 不一致，请检查模型定义"


def main():
    ap = argparse.ArgumentParser("导出 ONNX")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="face.onnx")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--width", type=float, default=None)
    ap.add_argument("--embedding_size", type=int, default=None)
    ap.add_argument("--batch", type=int, default=1, help="固定批大小（DPU 用 1）")
    ap.add_argument("--l2norm", action="store_true",
                    help="在图内做 L2 归一化（默认不开，交给 CPU）")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--dynamo", action="store_true",
                    help="用新版 dynamo 导出器（默认关闭，legacy 更稳）")
    ap.add_argument("--no_check", action="store_true")
    ap.add_argument("--keep_norm_cap", action="store_true",
                    help="保留训练用的特征范数上限（默认摘除，L2 归一化后等价）")
    args = ap.parse_args()

    backbone, arch, width, emb = load_backbone(
        args.ckpt, args.arch, args.width, args.embedding_size)
    # 融合 bnneck(BN1d) 到后面的 Linear：Vitis AI 会提示 BN1d 无法与 CONV 融合，
    # 融掉之后少一个算子、少一处量化误差。
    fuse_for_deploy(backbone)
    # 摘掉训练时的特征范数上限：它按样本把特征整体缩放到 ≤64，
    # 而推理端本来就要做 L2 归一化再比余弦，整体缩放不改变方向、语义等价。
    # 留着反而多出 Reciprocal/Mul 两个算子，且 Reciprocal 不在 DPU 算子白名单里。
    if getattr(backbone, "max_feat_norm", 0) > 0 and not args.keep_norm_cap:
        backbone.max_feat_norm = 0.0
        print("[model] 已摘除特征范数上限（L2 归一化后等价，少 2 个算子）")
    model = WrapNCHW(backbone, args.l2norm).eval()
    print(f"[model] {arch} w={width} emb={emb}  ckpt={args.ckpt}  (已融合 bnneck)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    dummy = torch.randn(args.batch, 3, 112, 112)
    export_kw = dict(
        input_names=["input"], output_names=["embedding"],
        opset_version=args.opset,
        dynamic_axes=None,                  # 固定形状，DPU 编译更省事
        do_constant_folding=True,
    )
    # 默认走 legacy exporter：产出的图更「朴素」（少一堆 Slice/Reshape 花活），
    # Vitis AI 解析起来更稳。
    if not args.dynamo:
        export_kw["dynamo"] = False
    torch.onnx.export(model, dummy, args.out, **export_kw)
    print(f"[done] 已导出 -> {args.out}  "
          f"({os.path.getsize(args.out)/1e6:.2f} MB)")

    if not args.no_check:
        check_ops(args.out)
        verify_numeric(backbone, args.out, emb, l2norm=args.l2norm,
                       batch=args.batch)
    print("\n下一步：\n"
          "  PC 端验证   python tools/verify_align.py --onnx %s --help\n"
          "  量化编译    见 deploy/README.md（vai_q_pytorch / vai_q_onnx）"
          % args.out)


if __name__ == "__main__":
    main()
