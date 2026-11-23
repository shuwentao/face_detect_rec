"""导出 ARM CPU 上跑的「软件版」模型（不依赖 DPU）。

产出的两个东西
--------------------------------------------------------------------------
1. `*.ncnn.param` + `*.ncnn.bin`  —— NCNN 格式，**ARM CPU 上最快的路子**
   （纯 CPU 推理，NEON 汇编优化，无需 DPU、无需 Vitis AI 运行时）。
2. `*.pt`  —— TorchScript，转 NCNN 的中间产物，留着方便二次转换。

和 `export_onnx.py`（DPU 那条线）的差别
--------------------------------------------------------------------------
* 这里默认**把 L2 归一化塞进图里**（`--no_l2norm` 可关）。
  原因是 ReduceL2 在 DPU 上支持有限，所以 DPU 版把它留在外面；
  而 ARM CPU 上这点开销可以忽略，塞进图里调用方少一步、更不容易出错。
* 输入一律固定 [1,3,112,112]，值域 [-1,1] RGB。

用法
--------------------------------------------------------------------------
    python deploy/export_arm.py --ckpt checkpoints_136k/backbone_ep14.pth
    python deploy/export_arm.py --ckpt ... --out_dir deploy/arm --no_l2norm
"""
from __future__ import annotations

import os
import sys
import shutil
import argparse
import subprocess
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from export_onnx import load_backbone, WrapNCHW               # noqa: E402
from models.ghostfacenet import fuse_for_deploy               # noqa: E402


class WrapL2(torch.nn.Module):
    """固定输入形状 + 可选图内 L2 归一化。"""

    def __init__(self, backbone, l2norm: bool = True):
        super().__init__()
        self.backbone = backbone
        self.l2norm = l2norm

    def forward(self, x):
        f = self.backbone(x)
        return torch.nn.functional.normalize(f, dim=1) if self.l2norm else f


def main():
    ap = argparse.ArgumentParser("导出 ARM CPU 软件版（NCNN）")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out_dir", default="deploy/arm")
    ap.add_argument("--name", default="face_arm")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--width", type=float, default=None)
    ap.add_argument("--embedding_size", type=int, default=None)
    ap.add_argument("--no_l2norm", action="store_true",
                    help="关闭图内 L2 归一化（默认开启）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    l2 = not args.no_l2norm

    backbone, arch, width, emb = load_backbone(
        args.ckpt, args.arch, args.width, args.embedding_size)
    fuse_for_deploy(backbone)
    if getattr(backbone, "max_feat_norm", 0) > 0:
        backbone.max_feat_norm = 0.0          # 推理端 L2 归一化后等价，去掉省算子
    model = WrapL2(backbone, l2).eval()
    n_param = sum(p.numel() for p in backbone.parameters())
    print(f"[model] {arch} w={width} emb={emb}  参数={n_param/1e6:.2f}M  "
          f"图内L2={'开' if l2 else '关'}")

    # 1) TorchScript
    pt_path = os.path.join(args.out_dir, args.name + ".pt")
    dummy = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy)
        traced = torch.jit.freeze(traced)
    traced.save(pt_path)
    print(f"[done] TorchScript -> {pt_path} "
          f"({os.path.getsize(pt_path)/1e6:.2f} MB)")

    # 2) pnnx → NCNN
    pnnx = shutil.which("pnnx")
    if pnnx is None:
        print("[warn] 未找到 pnnx（pip install pnnx），跳过 NCNN 转换")
        return
    work = os.path.join(args.out_dir, "_pnnx")
    os.makedirs(work, exist_ok=True)
    shutil.copy(pt_path, os.path.join(work, args.name + ".pt"))
    cmd = [pnnx, args.name + ".pt", "inputshape=[1,3,112,112]"]
    print("[run ]", " ".join(cmd))
    r = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise SystemExit("[fail] pnnx 转换失败")
    for suf in (".ncnn.param", ".ncnn.bin"):
        src = os.path.join(work, args.name + suf)
        dst = os.path.join(args.out_dir, args.name + suf)
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"[done] NCNN -> {dst} ({os.path.getsize(dst)/1e6:.2f} MB)")
    shutil.rmtree(work, ignore_errors=True)

    # 3) 数值对齐：NCNN vs PyTorch
    param = os.path.join(args.out_dir, args.name + ".ncnn.param")
    if os.path.exists(param):
        verify_ncnn(param, os.path.join(args.out_dir, args.name + ".ncnn.bin"),
                    model, l2)


def verify_ncnn(param, binf, model, l2=True, atol=2e-3):
    try:
        import ncnn
    except ImportError:
        print("[skip] 未安装 ncnn，跳过数值校验")
        return
    x = torch.randn(1, 3, 112, 112)
    with torch.no_grad():
        ref = model(x).numpy()
    net = ncnn.Net()
    net.opt.num_threads = 1
    net.load_param(param)
    net.load_model(binf)
    # NCNN 的 Mat 从 numpy 构造时按 (c, h, w) 解释，直接给 CHW，别转 HWC
    xh = np.ascontiguousarray(x.numpy()[0])
    with net.create_extractor() as ex:
        ex.input("in0", ncnn.Mat(xh))
        ret, out = ex.extract("out0")
    got = np.array(out).reshape(-1)
    if got.shape != ref.shape:
        got = got.reshape(ref.shape)
    diff = float(np.abs(got - ref).max())
    print(f"[check] PyTorch vs NCNN: 最大绝对误差={diff:.3e}  "
          f"范数={np.linalg.norm(got):.4f}")
    assert diff < atol, f"NCNN 与 PyTorch 不一致 (diff={diff:.3e})"
    print("[ok] NCNN 数值对齐通过")


if __name__ == "__main__":
    main()
