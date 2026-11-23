"""把 QAT 训好的权重剥掉量化 step，映射回浮点模型。

目的：单独评估「QAT 训出来的权重本身」的浮点精度。
- 若接近浮点上限 0.6076 -> QAT 训练是好的，问题全在量化导出
- 若明显偏低          -> QAT 训练本身就有问题，不值得再修导出
"""
import sys

import torch

BASE = "/workspace/mymodel/gate320"
sys.path.insert(0, BASE)

from common.models.detector import build_detector  # noqa: E402

ck = torch.load(BASE + "/qat320e/ckpt/best.pt", map_location="cpu")
sd = ck["model"]
print("[qat] epoch =", ck.get("epoch"), "| n_keys =", len(sd), flush=True)

m = build_detector("mobilenetv3", pretrained=False, width=1.0)
msd = m.state_dict()
print("[float] n_keys =", len(msd), flush=True)

extra = [k for k in sd if k not in msd]
missing = [k for k in msd if k not in sd]
print("[extra  ] %d  (量化 step 等): %s" % (len(extra), extra[:6]), flush=True)
print("[missing] %d : %s" % (len(missing), missing[:6]), flush=True)

filtered = {k: v for k, v in sd.items() if k in msd}
ret = m.load_state_dict(filtered, strict=False)
print("[load] missing_keys =", len(ret.missing_keys),
      "| unexpected =", len(ret.unexpected_keys), flush=True)

out = BASE + "/qat320e_float_sd.pt"
torch.save({"model": filtered}, out)
print("[saved]", out, flush=True)
