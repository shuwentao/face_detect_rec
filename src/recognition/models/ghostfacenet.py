"""GhostFaceNet —— 近两年主流的轻量人脸识别骨干（IEEE Access 2023）。

选它的理由（对照本项目「ZynqMPSoC + DPUCZDX8G」的部署约束）
--------------------------------------------------------------------------
1. **算子极简**：只用 3×3/1×1 卷积、depthwise 卷积、BN、ReLU、Concat、Add、
   Global Average Pool、全连接。这些全部在 DPUCZDX8G 的硬件算子表内，
   不会像 Transformer 那样因为 LayerNorm / GELU / Attention 掉到 CPU 上。
2. **够轻**：宽 1.0 时约 3~4M 参数、~0.2 GFLOPs（112×112），
   DPU 上可以做到百 fps 级别，也便于单张 8GB 显卡训练。
3. **精度够用**：论文在 Glint360K/MS1M 上训练，LFW 99.7%+、CFP-FP 97%+，
   属于「轻量模型里的 SOTA 档」。

结构来源说明
--------------------------------------------------------------------------
GhostNetV1 的 stage 配置表与 GhostBottleneck 取自 Huawei-Noah 的官方实现
（GhostNet: More Features from Cheap Operations, CVPR 2020）；
人脸头部（去掉 1000 类分类头，接 GAP + BN + Dropout + Linear → embedding）
按 GhostFaceNets 论文的做法。

与原作的两处工程性改动，都是为了 DPU 兼容：
  * SE 模块里的 sigmoid 换成 **hardsigmoid**（DPU 对 hard 版支持最稳），
    且默认 `se_ratio=0` 完全关闭 SE —— 人脸特征图分辨率本来就小，
    SE 的收益有限，关掉能进一步降低量化风险。
  * GDC(7×7 global depthwise conv) 换成 **GAP**：112×112 输入经 5 次 stride2
    后特征图只有 4×4，7×7 的 GDC 无从谈起；GAP 更省参数也更好量化。
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _make_divisible(v: float, divisor: int = 8, min_value: int | None = None) -> int:
    """把通道数对齐到 divisor 的倍数（硬件友好，也方便 DPU 切分）。"""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:      # 避免向下取整掉太多
        new_v += divisor
    return int(new_v)


class HSwish(nn.Module):
    def forward(self, x):
        return x * torch.nn.functional.hardswish(x)


class HardSigmoid(nn.Module):
    def forward(self, x):
        return torch.clamp(x / 6.0 + 0.5, 0.0, 1.0)


class SqueezeExcite(nn.Module):
    """压缩-激励：GAP → 1×1 → ReLU → 1×1 → HardSigmoid → 逐通道相乘。

    DPU 对 mul + hard-sigmoid 有硬件支持；这里刻意不用原生 sigmoid。
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = _make_divisible(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced, 1, 1, 0, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced, channels, 1, 1, 0, bias=True)
        self.gate = HardSigmoid()

    def forward(self, x):
        s = self.gap(x)
        s = self.gate(self.fc2(self.act(self.fc1(s))))
        return x * s


def _ghost_split(oup: int, ratio: int = 2, align: int = 8):
    """把目标通道数拆成 (主分支通道, 廉价分支通道)。

    三个约束必须同时满足，否则要么网络构造报错、要么上不了 DPU：
      ① 廉价分支是 depthwise（groups == 主分支通道数），
         所以 `new % init == 0` —— 否则 PyTorch 直接抛
         "out_channels must be divisible by groups"；
      ② 主分支通道按 align(8) 对齐，DPUCZDX8G 的 depthwise 要求通道 8 对齐，
         不对齐时该层上不了 DPU，整网会被切成十几段子图；
      ③ 两路 Concat 后正好是实际输出通道，不需要截断（Slice 会拖慢图解析）。

    满足三者的办法就是 `new = init * (ratio-1)`，代价是实际输出
    = `init * ratio`，可能和目标 `oup` 差几个通道 —— 差得不明显，
    但调用方要用 `out_channels` 而不是 `oup` 去接下一层。
    """
    init = _make_divisible(int(round(oup / ratio)), align)
    init = max(align, init)
    new = init * (ratio - 1)
    assert init > 0 and new % init == 0, f"通道拆分非法: oup={oup} init={init} new={new}"
    return init, new


class GhostModule(nn.Module):
    """Ghost 模块：一半通道用普通卷积，另一半用「廉价」深度卷积生成。

    对外暴露 `out_channels`（= 主分支 + 廉价分支），它是真实输出通道数。
    """

    def __init__(self, inp: int, oup: int, kernel_size: int = 1, ratio: int = 2,
                 dw_size: int = 3, stride: int = 1, relu: bool = True):
        super().__init__()
        self.oup = oup
        self.init_ch, self.new_ch = _ghost_split(oup, ratio)
        self.out_channels = self.init_ch + self.new_ch
        self.primary = nn.Sequential(
            nn.Conv2d(inp, self.init_ch, kernel_size, stride,
                      kernel_size // 2, bias=False),
            nn.BatchNorm2d(self.init_ch),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(self.init_ch, self.new_ch, dw_size, 1, dw_size // 2,
                      groups=self.init_ch, bias=False),
            nn.BatchNorm2d(self.new_ch),
        ) if self.new_ch > 0 else None
        self.relu = nn.ReLU(inplace=True)
        self.use_relu = relu

    def forward(self, x):
        x1 = self.primary(x)
        if self.cheap is None:
            out = x1
        else:
            out = torch.cat([x1, self.cheap(x1)], dim=1)
        return self.relu(out) if self.use_relu else out


class GhostBottleneck(nn.Module):
    """Ghost 残差块：expand → (dwconv) → (SE) → project，外加 stride 对齐捷径。"""

    def __init__(self, in_chs: int, mid_chs: int, out_chs: int,
                 dw_kernel_size: int = 3, stride: int = 1, se_ratio: float = 0.0,
                 ghost_ratio: int = 2):
        super().__init__()
        self.stride = stride
        # GhostModule 的真实输出可能和传入的目标通道差一点，后续层一律按真实值接
        mid_act = sum(_ghost_split(mid_chs, ghost_ratio))
        self.out_channels = sum(_ghost_split(out_chs, ghost_ratio))

        layers = [GhostModule(in_chs, mid_chs, 1, ghost_ratio, relu=True)]
        if stride > 1:
            layers.append(nn.Sequential(
                nn.Conv2d(mid_act, mid_act, dw_kernel_size, stride,
                          dw_kernel_size // 2, groups=mid_act, bias=False),
                nn.BatchNorm2d(mid_act),
            ))
        if se_ratio > 0:
            layers.append(SqueezeExcite(mid_act, reduction=int(1 / se_ratio)))
        layers.append(GhostModule(mid_act, out_chs, 1, ghost_ratio, relu=False))
        self.conv = nn.Sequential(*layers)

        if stride > 1 or in_chs != self.out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride,
                          dw_kernel_size // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, self.out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.shortcut(x) + self.conv(x)


# GhostNetV1 官方配置：[dw_kernel, expand, out_ch, se_ratio, stride]
GHOSTNETV1_CFG = [
    [3, 16, 16, 0, 1],
    [3, 48, 24, 0, 2],
    [3, 72, 24, 0, 1],
    [5, 72, 40, 1, 2],
    [5, 120, 40, 1, 1],
    [3, 240, 80, 0, 2],
    [3, 200, 80, 0, 1],
    [3, 184, 80, 0, 1],
    [3, 184, 80, 0, 1],
    [3, 480, 112, 1, 1],
    [3, 672, 112, 1, 1],
    [5, 672, 160, 1, 2],
    [5, 960, 160, 0, 1],
    [5, 960, 160, 1, 1],
    [5, 960, 160, 0, 1],
    [5, 960, 160, 1, 1],
]


class GhostFaceNet(nn.Module):
    """面向前两阶段（PC 浮点）的轻量人脸特征提取器。

    输入  (N, 3, 112, 112)，值域 [-1, 1]
    输出  (N, embedding_size) —— **未归一化**的特征向量。
        训练侧：head 需要原始范数来算 AdaFace 的自适应 margin，所以这里不归一化。
        推理侧：比对前再做 L2 归一化（见 export_onnx.py / tools/verify_align.py）。

    width=1.3 大约是 GhostFaceNet「1.3×」档；DPU 上若算力吃紧可用 0.5/0.75。
    """

    def __init__(self, embedding_size: int = 512, width: float = 1.0,
                 se_ratio_default: float = 0.0, dropout: float = 0.4,
                 cfg=None, max_feat_norm: float = 64.0):
        super().__init__()
        cfg = cfg if cfg is not None else GHOSTNETV1_CFG
        # AdaFace 的 margin 依赖特征范数，模型会倾向于把范数越推越大
        # （实测能涨到 1e4，loss 随之反弹）。这里给范数一个硬上限：
        # 只有超过阈值时才按比例缩回，不改动小范数样本。
        # 在 ONNX 里就是 Div+Clip+Mul，DPU 都支持。
        self.max_feat_norm = float(max_feat_norm)

        stem_ch = _make_divisible(16 * width, 4)
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU(inplace=True),
        )

        blocks = []
        in_ch = stem_ch
        for k, exp, c, se, s in cfg:
            out_ch = _make_divisible(c * width, 8)
            mid_ch = _make_divisible(exp * width, 8)
            bn = GhostBottleneck(
                in_ch, mid_ch, out_ch, k, s,
                se_ratio=se * se_ratio_default if se_ratio_default else 0.0)
            in_ch = bn.out_channels        # 用真实输出通道接下一层
            blocks.append(bn)
        self.blocks = nn.Sequential(*blocks)

        head_ch = _make_divisible(960 * width, 4)
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_ch, head_ch, 1, 1, 0, bias=False),
            nn.BatchNorm2d(head_ch),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.bnneck = nn.BatchNorm1d(head_ch)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(head_ch, embedding_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.bnneck(x)
        x = self.dropout(x)
        feats = self.linear(x)
        if self.max_feat_norm > 0:
            n = feats.norm(dim=1, keepdim=True).clamp(min=1e-6)
            feats = feats * (self.max_feat_norm / n).clamp(max=1.0)
        return feats


class MobileFaceNetLite(nn.Module):
    """极轻备选骨干（约 1M 参数），算力紧张时用。

    结构：depthwise-separable 卷积堆叠 + SE + GAP + Linear。
    同样只用 DPU 友好的算子。
    """

    def __init__(self, embedding_size: int = 256, width: float = 1.0,
                 dropout: float = 0.2, max_feat_norm: float = 64.0, **kw):
        super().__init__()
        self.max_feat_norm = float(max_feat_norm)

        def dwsep(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        c = [_make_divisible(v * width, 8) for v in
             (64, 64, 128, 128, 256, 256, 512, 512)]
        self.stem = nn.Sequential(
            nn.Conv2d(3, c[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(c[0]), nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            dwsep(c[0], c[1], 1),
            dwsep(c[1], c[2], 2),
            dwsep(c[2], c[3], 1),
            dwsep(c[3], c[4], 2),
            dwsep(c[4], c[5], 1),
            dwsep(c[5], c[6], 2),
            dwsep(c[6], c[6], 1),
            dwsep(c[6], c[7], 2),
        )
        self.head_conv = nn.Sequential(
            nn.Conv2d(c[7], c[7], 1, 1, 0, bias=False),
            nn.BatchNorm2d(c[7]), nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.bnneck = nn.BatchNorm1d(c[7])
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(c[7], embedding_size, bias=False)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.bnneck(x)
        x = self.dropout(x)
        feats = self.linear(x)
        if self.max_feat_norm > 0:
            n = feats.norm(dim=1, keepdim=True).clamp(min=1e-6)
            feats = feats * (self.max_feat_norm / n).clamp(max=1.0)
        return feats


def fuse_for_deploy(model: nn.Module) -> nn.Module:
    """推理前的结构融合：把 1D 的 BatchNorm 融进后面的 Linear，Dropout 置空。

    Vitis AI 的量化器会提示
    ``BatchNorm1d[bnneck] cannot be fused into CONV layers, not quantization friendly``。
    BN 后面正好是 Linear，两者可以数学等价地合并成一个 Linear：
        y = W · (γ·(x−μ)/σ + β) = (W·γ/σ) · x + (W·β − W·γ·μ/σ)
    合并后少一个算子，也就少一处量化误差来源；Dropout 在 eval 下本来就是
    恒等映射，直接换成 Identity 免得量化器多留一个节点。

    注意：融合会改变 state_dict 结构，所以**只在导出 / 量化前调用**，
    不要拿融合后的模型去续训。
    """
    import math
    model.eval()
    bn, lin = getattr(model, "bnneck", None), getattr(model, "linear", None)
    if isinstance(bn, nn.BatchNorm1d) and isinstance(lin, nn.Linear):
        with torch.no_grad():
            var = bn.running_var + bn.eps
            sigma = torch.sqrt(var)
            scale = bn.weight / sigma                       # (C,)
            shift = bn.bias - bn.weight * bn.running_mean / sigma
            w = lin.weight * scale.unsqueeze(0)             # (D, C)
            b = lin.weight @ shift if lin.bias is None else lin.bias + lin.weight @ shift
            new_lin = nn.Linear(lin.in_features, lin.out_features, bias=True)
            new_lin.weight.copy_(w)
            new_lin.bias.copy_(b)
        model.bnneck = nn.Identity()
        model.linear = new_lin
        model.eval()
    if isinstance(getattr(model, "dropout", None), nn.Dropout):
        model.dropout = nn.Identity()
    return model


def build_backbone(arch: str = "ghostfacenet", embedding_size: int = 512,
                   width: float = 1.0, **kw) -> nn.Module:
    arch = arch.lower()
    if arch in ("ghostfacenet", "ghostface", "ghost"):
        return GhostFaceNet(embedding_size=embedding_size, width=width, **kw)
    if arch in ("mobilefacenet", "mbf", "mobilefacenetlite"):
        return MobileFaceNetLite(embedding_size=embedding_size, width=width, **kw)
    raise ValueError(f"未知骨干: {arch}（可选 ghostfacenet / mobilefacenet）")


if __name__ == "__main__":
    for arch, w in (("ghostfacenet", 1.0), ("mobilefacenet", 1.0)):
        m = build_backbone(arch, 512, w).eval()
        n = sum(p.numel() for p in m.parameters())
        x = torch.randn(1, 3, 112, 112)
        with torch.no_grad():
            y = m(x)
        print(f"{arch:<14} width={w}  参数={n/1e6:.2f}M  输出={tuple(y.shape)}")
