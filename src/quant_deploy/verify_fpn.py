"""验证 qat_patch 的 FPN 实例拆分：patch 前后输出必须逐位相同。

因为 add / interpolate 都是无参数模块（forward 就是 torch.add / F.interpolate），
把「复用 1 个实例」改成「每次调用各 1 个实例」不改变任何计算，只是让 Vitis
QAT 能为每个融合点学各自的量化刻度。
"""
import sys
import torch

sys.path.insert(0, "/workspace/mymodel/gate320")

from common.models.detector import build_detector  # noqa: E402
import qat_patch  # noqa: E402

m = build_detector("mobilenetv3", pretrained=False, width=1.0)
sd = torch.load("/workspace/mymodel/gate320/best_sd.pt", map_location="cpu")
if isinstance(sd, dict) and "model" in sd:
    sd = sd["model"]
print("[model] load_state_dict ->", m.load_state_dict(sd, strict=True))
m.eval()

torch.manual_seed(0)
x = torch.randn(1, 3, 320, 320)

with torch.no_grad():
    y0 = [t.clone() for t in m(x)]

stats = qat_patch.patch_for_qat(m)
print("[qat-patch]", stats)

with torch.no_grad():
    y1 = [t.clone() for t in m(x)]

md = 0.0
for a, b in zip(y0, y1):
    md = max(md, (a - b).abs().max().item())
print("输出组数:", len(y0), "->", len(y1))
print("max|diff| =", md)
print("严格等价:", md == 0.0)
