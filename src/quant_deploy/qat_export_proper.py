"""用 QatProcessor 自带刻度导出 QAT 的 int.pt。

之前那条导出路径（quantize_gate320.py --quant_mode calib/test）会用校准图
重新 PTQ 算一套刻度，把 QAT 学到的量化参数整个丢掉，导出的是
「QAT 权重 + PTQ 刻度」，必然不匹配。这里改走 QatProcessor：
  to_deployable()  -> 固定 QAT 自己的量化参数
  export_torch_script() -> 带着这套参数导出 jit
"""
import os
import sys

import torch

BASE = "/workspace/mymodel/gate320"
sys.path.insert(0, BASE)

from common.models.detector import build_detector  # noqa: E402

CKPT_FLOAT = os.path.join(BASE, "best_sd.pt")
CKPT_QAT = os.path.join(BASE, "qat320e/ckpt/best.pt")
CALIB = os.path.join(BASE, "qr_qat320_v3")
OUT = os.path.join(BASE, "qat320e/qtest_proper")
DEP = os.path.join(BASE, "qat320e/deploy_proper")

# 1) 浮点模型
model = build_detector("mobilenetv3", pretrained=False, width=1.0)
sd = torch.load(CKPT_FLOAT, map_location="cpu")
if isinstance(sd, dict) and "model" in sd:
    sd = sd["model"]
print("[model] float ->", model.load_state_dict(sd, strict=True), flush=True)

from qat_patch import patch_for_qat  # noqa: E402
patch_for_qat(model)
model.train()                       # nndct 折叠 conv+bn 时要求 training

# 2) QatProcessor
from pytorch_nndct import QatProcessor  # noqa: E402
proc = QatProcessor(model, torch.zeros(8, 3, 320, 320), bitwidth=8)
qm = proc.trainable_model(calib_dir=CALIB, allow_reused_module=True)

# 3) 载入 QAT 训好的权重（含量化 step）
ck = torch.load(CKPT_QAT, map_location="cpu")
print("[qat] 载入 ckpt epoch =", ck.get("epoch"), flush=True)
qm.load_state_dict(ck["model"], strict=True)
qm.eval()

# 导出前必须先让 quant_model 跑一次 forward，否则报
# VAIQ_ERROR[QUANTIZER_TORCH_NO_FORWARD]
with torch.no_grad():
    _ = qm(torch.zeros(8, 3, 320, 320))
print("[qat] forward ok", flush=True)

# 4) 用 QAT 自己的刻度导出
os.makedirs(DEP, exist_ok=True)
proc.to_deployable(qm, DEP)
os.makedirs(OUT, exist_ok=True)
proc.export_torch_script(output_dir=OUT, verbose=True)
print("[done] ->", OUT, flush=True)
for f in sorted(os.listdir(OUT)):
    print("   ", f, "%.2f MB" % (os.path.getsize(os.path.join(OUT, f)) / 1e6))
