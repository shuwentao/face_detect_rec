"""对比 GhostFaceNet / MobileFaceNet 的算力开销。

报：参数量、MACs、GPU 延迟、CPU 延迟（x86 代理，非 ARM 实测）。
重点看深度可分离卷积的数量 —— 这类算子在 ARM CPU 上是访存瓶颈，
FLOPs 低不代表跑得快。
"""
import sys
import time
import torch
import torch.nn as nn

sys.path.insert(0, "/home/tiny/tmp/face_rec")
from models.ghostfacenet import build_backbone  # noqa: E402


def count_macs(model, x):
    """用 hook 统计 Conv2d / Linear 的乘加数。"""
    total = {"macs": 0}

    def hook_conv(m, inp, out):
        kh, kw = m.kernel_size
        oc, ic = m.out_channels, m.in_channels // m.groups
        oh, ow = out.shape[-2], out.shape[-1]
        total["macs"] += oc * ic * kh * kw * oh * ow

    def hook_linear(m, inp, out):
        total["macs"] += m.in_features * m.out_features

    hs = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hs.append(m.register_forward_hook(hook_conv))
        elif isinstance(m, nn.Linear):
            hs.append(m.register_forward_hook(hook_linear))
    with torch.no_grad():
        model(x)
    for h in hs:
        h.remove()
    return total["macs"]


def bench(model, x, n=200):
    model.eval()
    with torch.no_grad():
        for _ in range(30):
            model(x)
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(n):
            model(x)
        torch.cuda.synchronize()
    return (time.time() - t) / n * 1000


def bench_cpu(model, x, threads, n=30):
    torch.set_num_threads(threads)
    model.eval()
    xc = x.cpu()
    mc = model.cpu()
    with torch.no_grad():
        for _ in range(5):
            mc(xc)
        t = time.time()
        for _ in range(n):
            mc(xc)
    return (time.time() - t) / n * 1000


def main():
    print(f"{'模型':<16}{'参数':>9}{'MACs':>10}{'深度可分离':>11}"
          f"{'GPU bs1':>10}{'GPU bs32':>10}{'CPU 1T':>9}{'CPU 4T':>9}")
    print("-" * 84)

    for arch in ["mobilefacenet", "ghostfacenet"]:
        m = build_backbone(arch, 512, 1.0).cuda().eval()
        params = sum(p.numel() for p in m.parameters())
        dw = sum(1 for x in m.modules()
                 if isinstance(x, nn.Conv2d) and x.groups > 1)
        macs = count_macs(m, torch.randn(1, 3, 112, 112).cuda())

        g1 = bench(m, torch.randn(1, 3, 112, 112).cuda())
        g32 = bench(m, torch.randn(32, 3, 112, 112).cuda(), n=50)

        xc = torch.randn(1, 3, 112, 112).cuda()
        m.cpu()
        c1 = bench_cpu(m, xc, 1)
        c4 = bench_cpu(m, xc, 4)

        print(f"{arch:<16}{params/1e6:>8.2f}M{macs/1e6:>9.1f}M{dw:>11}"
              f"{g1:>9.2f}ms{g32:>9.2f}ms{c1:>8.1f}ms{c4:>8.1f}ms")
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
