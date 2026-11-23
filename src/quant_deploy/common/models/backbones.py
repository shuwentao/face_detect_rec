"""4 种 backbone：YuNet(PyTorch 复现) / MobileNetV1 / V2 / V3。

统一接口：forward(x) -> (c3, c4, c5)，下采样倍率分别为 8 / 16 / 32。
检测头完全一致（见 detector.py），因此对比的只有 backbone。
"""
import torch
import torch.nn as nn
from torchvision import models as tvm

# --------------------------------------------------------------------------- #
# 基础模块
# --------------------------------------------------------------------------- #
class ConvBnAct(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, groups=1, act=True):
        super().__init__()
        p = (k - 1) // 2
        self.conv = nn.Conv2d(cin, cout, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWBlock(nn.Module):
    """depthwise separable: DW(3x3) + PW(1x1)，stride 在 DW 上。"""

    def __init__(self, cin, cout, s=1):
        super().__init__()
        self.dw = ConvBnAct(cin, cin, 3, s, groups=cin)
        self.pw = ConvBnAct(cin, cout, 1, 1)

    def forward(self, x):
        return self.pw(self.dw(x))


# --------------------------------------------------------------------------- #
# YuNet backbone（PyTorch 复现：depthwise stem + 4 stage，参照 libfacedetection）
# --------------------------------------------------------------------------- #
class YuNetBackbone(nn.Module):
    def __init__(self, width=1.0):
        super().__init__()
        c = [max(8, int(v * width)) for v in (16, 32, 64, 128, 256)]
        c0, c1, c2, c3, c4 = c
        self.stem = nn.Sequential(ConvBnAct(3, c0, 3, 2), DWBlock(c0, c0, 1))
        self.stage1 = nn.Sequential(DWBlock(c0, c1, 2))                 # 1/4
        self.stage2 = nn.Sequential(DWBlock(c1, c2, 2), DWBlock(c2, c2, 1))   # 1/8
        self.stage3 = nn.Sequential(DWBlock(c2, c3, 2), DWBlock(c3, c3, 1))   # 1/16
        self.stage4 = nn.Sequential(DWBlock(c3, c4, 2), DWBlock(c4, c4, 1))   # 1/32
        self._out_channels = (c2, c3, c4)
        self.init_weights()

    @property
    def out_channels(self):
        return self._out_channels

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return c3, c4, c5


# --------------------------------------------------------------------------- #
# MobileNetV1（torchvision 未提供，按论文实现）
# --------------------------------------------------------------------------- #
class MobileNetV1Backbone(nn.Module):
    def __init__(self, width_mult=1.0):
        super().__init__()
        def c(v):
            return max(8, int(v * width_mult))

        def dw(cin, cout, s):
            return nn.Sequential(
                ConvBnAct(cin, cin, 3, s, groups=cin),
                ConvBnAct(cin, cout, 1, 1))
        self.s1 = nn.Sequential(ConvBnAct(3, c(32), 3, 2), dw(c(32), c(64), 1))   # 1/2
        self.s2 = nn.Sequential(dw(c(64), c(128), 2), dw(c(128), c(128), 1))      # 1/4
        self.s3 = nn.Sequential(dw(c(128), c(256), 2), dw(c(256), c(256), 1))     # 1/8
        self.s4 = nn.Sequential(*[dw(c(256), c(512), 2)] +
                                [dw(c(512), c(512), 1) for _ in range(5)])       # 1/16
        self.s5 = nn.Sequential(dw(c(512), c(1024), 2), dw(c(1024), c(1024), 1))  # 1/32
        self._out_channels = (c(256), c(512), c(1024))
        self.init_weights()

    @property
    def out_channels(self):
        return self._out_channels

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.s1(x)
        x = self.s2(x)
        c3 = self.s3(x)
        c4 = self.s4(c3)
        c5 = self.s5(c4)
        return c3, c4, c5


# --------------------------------------------------------------------------- #
# MobileNetV2 / V3：直接取 torchvision 的 features，按下采样率自动分层
# --------------------------------------------------------------------------- #
class TVBackbone(nn.Module):
    """把 torchvision mobilenet_v2 / v3 的 features 按 1/8,1/16,1/32 切成三段。"""

    def __init__(self, kind="v2", pretrained=True, width_mult=1.0):
        super().__init__()
        if kind == "v2":
            m = tvm.mobilenet_v2(weights=tvm.MobileNet_V2_Weights.DEFAULT
                                 if pretrained else None,
                                 width_mult=width_mult)
        elif kind == "v3_small":
            m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.DEFAULT
                                       if pretrained else None)
        elif kind == "v3_large":
            m = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.DEFAULT
                                       if pretrained else None,
                                       width_mult=width_mult)
        else:
            raise ValueError(kind)
        self.features = m.features
        self._out_channels = self._probe()

    @property
    def out_channels(self):
        return self._out_channels

    @torch.no_grad()
    def _probe(self):
        """跑一张假图，找出 stride 8/16/32 处最后一层的通道数。"""
        x = torch.zeros(1, 3, 640, 640)
        outs, cur = {}, 1
        for layer in self.features:
            x = layer(x)
            cur = 640 // x.shape[-1]
            if cur in (8, 16, 32):
                # 持续覆盖：同一 stride 下后出现的层语义更强，取最后一个
                outs[cur] = x.shape[1]
        assert 8 in outs and 16 in outs, f"分层失败: {outs}"
        if 32 not in outs:
            outs[32] = x.shape[1]
        return (outs[8], outs[16], outs[32])

    def forward(self, x):
        h = x.shape[-1]
        c3 = c4 = c5 = None
        for layer in self.features:
            x = layer(x)
            s = h // x.shape[-1]
            # 与 _probe 保持一致：同 stride 下持续覆盖，取该分辨率的最后一层
            if s == 8:
                c3 = x
            elif s == 16:
                c4 = x
            elif s == 32:
                c5 = x
        if c5 is None:
            c5 = x        # backbone 最深只到 1/16：兜底用最后一层
        if c4 is None:
            c4 = c3 if c3 is not None else c5
        if c3 is None:
            c3 = c4
        return c3, c4, c5


def build_backbone(name, pretrained=True, width=1.0):
    name = name.lower()
    if name == "yunet":
        return YuNetBackbone(width=width)
    if name == "mobilenetv1":
        return MobileNetV1Backbone(width_mult=width)
    if name == "mobilenetv2":
        return TVBackbone("v2", pretrained=pretrained, width_mult=width)
    if name == "mobilenetv3":
        return TVBackbone("v3_small", pretrained=pretrained)
    if name == "mobilenetv3_large":
        return TVBackbone("v3_large", pretrained=pretrained, width_mult=width)
    raise ValueError(f"未知 backbone: {name}")
