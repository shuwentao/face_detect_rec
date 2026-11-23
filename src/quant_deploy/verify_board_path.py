"""模拟 DPU 数据链路（跳过 DPU 计算本身），验证：
   1) NCHW <-> NHWC 转换   2) 输出顺序相反   3) int8 定点(fix=4)换算
这三个在容器里用 PyTorch 后端是验证不出来的，但到了板上会直接毁掉结果。
"""
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, "/workspace/mymodel/gate320")
import pytorch_nndct  # noqa: F401
from face_video_bench import decode_preds, preprocess   # noqa: E402

IMG = "/mnt/data1/face_bench/data/images/300w/300w_indoor_001.png"
INT_PT = "/workspace/mymodel/gate320/qr_ft_E/FaceDetector_int.pt"

im = cv2.imread(IMG)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
x = preprocess(im)                      # (1,3,320,320) float32

m = torch.jit.load(INT_PT)
m.eval()
with torch.no_grad():
    outs_pt = [o.numpy() for o in m(torch.from_numpy(x))]

print("PyTorch 输出顺序:", [tuple(o.shape) for o in outs_pt])

# ---- 模拟 DPU：量化到 int8 + 转 NHWC + 输出顺序反过来(10,20,40)
IN_FIX, OUT_FIX = 6, 4                  # 实测自 xmodel
xq = np.clip(np.round(x * (1 << IN_FIX)), -128, 127).astype(np.int8)
print("输入量化: 范围[%d,%d] 溢出像素 %d"
      % (xq.min(), xq.max(), int((np.abs(x * 64) > 127).sum())))

dpu_outs, clipped = [], 0
for o in reversed(outs_pt):              # 40,20,10 -> 10,20,40
    q = np.clip(np.round(o * (1 << OUT_FIX)), -128, 127).astype(np.int8)
    clipped += int((np.abs(o * (1 << OUT_FIX)) > 127).sum())
    dpu_outs.append(np.ascontiguousarray(q.transpose(0, 2, 3, 1)))  # -> NHWC
print("DPU 输出顺序:", [tuple(o.shape) for o in dpu_outs], "(NHWC)")
print("输出量化溢出点数:", clipped)

# ---- 模拟 VartBackend.infer 的后半段：反量化 + 转回 NCHW
recovered = []
for o in dpu_outs:
    v = o.astype(np.float32) / float(1 << OUT_FIX)
    recovered.append(np.ascontiguousarray(v.transpose(0, 3, 1, 2)))

# ---- 两条路径各自解码
b1, s1, l1 = decode_preds(outs_pt)      # PyTorch 路径（顺序 40,20,10）
b2, s2, l2 = decode_preds(recovered)    # 模拟 DPU 路径（顺序 10,20,40）

print()
print("PyTorch 路径检出 %d 框 | 模拟 DPU 路径检出 %d 框" % (len(b1), len(b2)))

def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw, ih = max(0, x2 - x1), max(0, y2 - y1)
    inter = iw * ih
    a1 = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    a2 = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (a1 + a2 - inter)


# 按 IoU 配对：int8 量化后大量 score 并列，按 score 排序对齐是不可靠的
pairs = []
for i in range(len(b1)):
    best, bj = 0.9, -1
    for j in range(len(b2)):
        v = iou(b1[i], b2[j])
        if v > best:
            best, bj = v, j
    if bj >= 0:
        pairs.append((i, bj))

print("IoU 配对成功: %d 对 (b1 %d / b2 %d)" % (len(pairs), len(b1), len(b2)))
n = len(pairs)
if n:
    bd = np.mean([np.abs(b1[i] - b2[j]).max() for i, j in pairs])
    sd = np.mean([abs(s1[i] - s2[j]) for i, j in pairs])
    ld = np.mean([np.abs(l1[i] - l2[j]).mean() for i, j in pairs])
    print("框坐标最大差 : %.4f px" % bd)
    print("score 平均差 : %.6f" % sd)
    print("关键点平均差 : %.4f px" % ld)
    print()
    print("判定：三项都应 ≈0（差异只来自 int8 量化损失）")
    print("      若框差几十像素 => 布局或 stride 顺序处理错了")
else:
    print("有一路没检出框，说明链路有问题")
