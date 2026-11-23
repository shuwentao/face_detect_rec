"""在 Vitis AI 3.0 容器里把浮点模型量化成 INT8，并导出 xmodel。

**这个脚本要在 docker 容器里跑，不是在本机跑。**

    docker run --rm -v $(pwd):/workspace -w /workspace \
        xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106 \
        /bin/bash -lc "source /opt/vitis_ai/conda/etc/profile.d/conda.sh && \
                       conda activate vitis-ai-pytorch && \
                       python deploy/quantize_pt.py \
                           --ckpt checkpoints/backbone_ep20.pth \
                           --calib deploy/calib.npy \
                           --out_dir quantize_result"

环境事实（实测自本机镜像 xilinx/vitis-ai-pytorch-cpu:3.0.0.106）
--------------------------------------------------------------------------
    容器里是 Python 3.7.12 + torch 1.12.1 + pytorch_nndct
所以本工程的模型定义必须能在 3.7 下 import —— 这也是
models/ghostfacenet.py 顶部写 `from __future__ import annotations` 的原因
（否则 `int | None` 这类注解在 3.7 会直接抛 TypeError）。

产出
--------------------------------------------------------------------------
    quantize_result/quantize_model.xmodel   ← 给 vai_c_xir 编译
    quantize_result/quant_info.json         ← 量化参数，便于复查
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from models.ghostfacenet import build_backbone, fuse_for_deploy  # noqa: E402


def load_backbone(ckpt: str, arch=None, width=None, emb=None):
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "backbone" in sd:
        arch = arch or sd.get("arch", "ghostfacenet")
        width = width or sd.get("width", 1.0)
        emb = emb or sd.get("embedding_size", 512)
        sd = sd["backbone"]
    m = build_backbone(arch or "ghostfacenet", emb or 512, width or 1.0)
    m.load_state_dict(sd)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser("Vitis AI 3.0 PyTorch 量化")
    ap.add_argument("--ckpt", default=None,
                    help="backbone_epN.pth；不给则配合 --from_scratch 用随机权重")
    ap.add_argument("--from_scratch", action="store_true",
                    help="不加载权重，仅用随机初始化验证图结构（快速试编译用）")
    ap.add_argument("--calib", required=True, help="make_calib.py 产出的 npy")
    ap.add_argument("--out_dir", default="quantize_result")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--width", type=float, default=None)
    ap.add_argument("--embedding_size", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8)
    # 注意：Vitis AI 3.0 的 nndct 不认 "DPUCZDX8G" 这种泛称，必须是带 ISA/BANK 的完整名。
    # 实测容器报的合法名里，DPUCZDX8G_ISA1_B4096 => 0x101000056010407，
    # 正是 deploy/arch.json（板子 IP 配置）里的指纹，所以默认就用它。
    ap.add_argument("--target", default="DPUCZDX8G_ISA1_B4096")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--keep_norm_cap", action="store_true",
                    help="保留训练用的特征范数上限（默认摘除，L2 归一化后等价）")
    ap.add_argument("--deploy_check", action="store_true",
                    help="导出时做量化前后数值比对（慢一点，但能看到量化误差）")
    args = ap.parse_args()

    from pytorch_nndct.apis import torch_quantizer

    device = torch.device(args.device)
    if args.from_scratch:
        # 只验证图结构是不是能被 DPU 吃掉，权重无所谓
        model = build_backbone(args.arch or "ghostfacenet",
                               args.embedding_size or 512, args.width or 1.0)
    else:
        if not args.ckpt:
            raise SystemExit("需要 --ckpt（或加 --from_scratch 用随机权重试跑）")
        model = load_backbone(args.ckpt, args.arch, args.width,
                              args.embedding_size)
    model = model.to(device)
    fuse_for_deploy(model)          # 融掉 BN1d / Dropout，减少量化节点
    # 摘掉训练用的特征范数上限：推理端要做 L2 归一化，整体缩放语义等价；
    # 留着会多出 Reciprocal，而它不在 DPU 算子白名单里、会掉到 CPU 上。
    if getattr(model, "max_feat_norm", 0) > 0 and not args.keep_norm_cap:
        model.max_feat_norm = 0.0
        print("[model] 已摘除特征范数上限（L2 归一化后等价，少 2 个算子）")
    print(f"[model] {type(model).__name__}  "
          f"参数={sum(p.numel() for p in model.parameters())/1e6:.2f}M  (已融合 bnneck)")

    calib = np.load(args.calib)
    if calib.ndim == 4 and calib.shape[1] != 3:
        calib = np.transpose(calib, (0, 3, 1, 2))
    print(f"[calib] {calib.shape} 值域 [{calib.min():.3f}, {calib.max():.3f}]")

    dummy = torch.from_numpy(calib[:1]).to(device)
    os.makedirs(args.out_dir, exist_ok=True)

    # 不同小版本的 torch_quantizer 参数名略有出入，逐个尝试
    kw = dict(output_dir=args.out_dir, device=device, bitwidth=8)
    quantizer = None
    for extra in ({"target": args.target},
                  {"target": args.target, "quant_config_file": None},
                  {}):
        try:
            quantizer = torch_quantizer("calib", model, (dummy,), **kw, **extra)
            print(f"[nndct] torch_quantizer 创建成功 (extra={list(extra)})")
            break
        except TypeError as e:                              # noqa: BLE001
            print(f"[nndct] 参数组合 {list(extra)} 不适用: {e}")
    if quantizer is None:
        raise SystemExit("无法创建 torch_quantizer，请检查容器内的 nndct 版本")

    # ===== 阶段 1：calib —— 统计激活分布，确定量化阈值 =====
    quant_model = quantizer.quant_model
    n = len(calib)
    for i in range(0, n, args.batch):
        x = torch.from_numpy(np.ascontiguousarray(calib[i:i + args.batch])).to(device)
        with torch.no_grad():
            quant_model(x)
    print(f"[calib] 完成 {n} 张")
    try:
        quantizer.export_quant_config()
    except Exception as e:                                  # noqa: BLE001
        print(f"[warn] export_quant_config: {e}")

    # ===== 阶段 2：test —— 用量化参数重建图，才能导出 xmodel =====
    # 这是容易踩的坑：export_xmodel 只在 'test' 模式下产出 .xmodel，
    # 在 'calib' 模式下调用不会报错，但也不会生成任何文件。
    q_test = torch_quantizer("test", model, (dummy,), **kw, target=args.target)
    qm_test = q_test.quant_model
    # 必须只用 batch=1 跑一次：nndct 会把最后这次 forward 的 batch 记进图里，
    # batch>1 时会报 "Batch size must be 1 when exporting xmodel"。
    with torch.no_grad():
        qm_test(dummy)
    print("[test] 已重建量化图")
    try:
        q_test.export_xmodel(output_dir=args.out_dir,
                             deploy_check=args.deploy_check)
    except TypeError:
        q_test.export_xmodel()
    print(f"[done] 量化完成，产物在 {args.out_dir}/")
    for f in sorted(os.listdir(args.out_dir)):
        p = os.path.join(args.out_dir, f)
        if os.path.isfile(p):
            print(f"    {f}  ({os.path.getsize(p)/1e3:.1f} KB)")
    # 产物名跟模型类名有关（如 GhostFaceNet_int.xmodel），不写死
    import glob as _glob
    xms = _glob.glob(os.path.join(args.out_dir, "*_int.xmodel"))
    if xms:
        print(f"\n下一步: bash deploy/compile_xmodel.sh {xms[0]}")
    else:
        print("\n[warn] 没找到 *_int.xmodel，请检查上面的日志")


if __name__ == "__main__":
    main()
