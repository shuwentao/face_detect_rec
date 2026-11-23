"""把模型里「裸的函数式算子」换成 nn.Module，让 Vitis QAT 能接管。

背景
----
Vitis 的 QatProcessor trace 之后会逐个量化节点做检查（quant_aware_training.py
的 _check_module_quantized）：

    if not module_cls.__module__.startswith('pytorch_nndct') \
       and module_cls.__name__ != op_type_name:
        报错 QUANTIZER_TORCH_NOT_A_MODULE

其中 op_type_name 是 torch 算子名的最后一段：'torch.mul' -> 'mul'。
所以通过检查有两条路：
  1. 用 pytorch_nndct 的模块（__module__ 以 pytorch_nndct 开头）
  2. 模块的类名**恰好等于**算子名（小写 mul / add / cat）

torchvision MobileNetV3 的 SE 用 `scale * input`、残差用 `result + input`，
FPN 用 `+`、检测头用 torch.cat，trace 后都是裸算子，于是报错。

为什么不用 pytorch_nndct.nn 的 Mul/Add
--------------------------------------
它们是 _PrimModule，forward 里访问 self.node.in_quant_part，而 node 要等
trace 之后才注入；trace 阶段是 None，直接崩：
    [VAIQ_ERROR] 'NoneType' object has no attribute 'in_quant_part'
（pytorch_nndct.nn.ChannelScale 也走这条路，且它只吃构造时固定的 scale，
SE 的 scale 是输入相关的，本来也用不了。）

所以走第 2 条路：自己写最朴素的包装模块，**类名用小写**。

只在 QAT 时调用。数学上完全等价（实测前后差 0.0），不动权重，
state_dict 的 key 不变，所以 PTQ / 浮点推理 / 已训权重都不受影响。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# torch.add / torch.cat 在 nndct 的 replacement_map 里，必须用它点名的类：
#     ADD    -> pytorch_nndct.nn.modules.functional.Add
#     CONCAT -> pytorch_nndct.nn.modules.functional.Cat
# 这两个是普通 nn.Module（forward 就是 torch.add / torch.cat），不依赖 node，
# trace 阶段不会崩——跟 pytorch_nndct.nn.Add（_PrimModule，会崩）不是一回事。
try:
    from pytorch_nndct.nn.modules.functional import (
        Add as _Add, Cat as _Cat, Interpolate as _Interpolate)
except ImportError:
    _Add = _Cat = _Interpolate = None


class mul(nn.Module):
    """torch.mul 不在 replacement_map，走「类名 == 算子名」那条路，所以必须小写。"""

    def forward(self, a, b):
        return a * b


if _Add is not None:
    add = _Add
else:
    class add(nn.Module):
        def forward(self, a, b):
            return a + b


if _Cat is not None:
    cat = _Cat
else:
    class cat(nn.Module):
        def forward(self, xs, dim=1):
            return torch.cat(xs, dim=dim)


if _Interpolate is not None:
    interpolate = _Interpolate
else:
    class interpolate(nn.Module):
        def forward(self, x, size=None, mode="nearest"):
            return F.interpolate(x, size=size, mode=mode)


class SqueezeExcitationEx(nn.Module):
    """torchvision SqueezeExcitation 的等价改写：乘法走 nn.Module。"""

    def __init__(self, src):
        super().__init__()
        self.avgpool = src.avgpool
        self.fc1 = src.fc1
        self.activation = src.activation
        self.fc2 = src.fc2
        self.scale_activation = src.scale_activation
        self.mul = mul()

    def forward(self, x):
        s = self.avgpool(x)
        s = self.fc1(s)
        s = self.activation(s)
        s = self.fc2(s)
        s = self.scale_activation(s)
        return self.mul(s, x)


class InvertedResidualEx(nn.Module):
    """torchvision InvertedResidual 的等价改写：残差加法走 nn.Module。"""

    def __init__(self, src):
        super().__init__()
        self.block = src.block
        self.use_res_connect = src.use_res_connect
        self.add = add()

    def forward(self, x):
        out = self.block(x)
        return self.add(out, x) if self.use_res_connect else out


class FPNLiteEx(nn.Module):
    """FPNLite 的等价改写：上采样相加走 nn.Module。"""

    def __init__(self, src):
        super().__init__()
        self.lat5, self.lat4, self.lat3 = src.lat5, src.lat4, src.lat3
        self.smooth5 = src.smooth5
        self.smooth4 = src.smooth4
        self.smooth3 = src.smooth3
        # relu 同样要拆：原实现复用 src.relu 一个实例，被 smooth3/4/5 调用 3 次，
        # 一样会触发 VAIQ_WARN: Reused module (fpn.relu)。ReLU 无参数，拆实例等价。
        self.relu3 = nn.ReLU(inplace=True)
        self.relu4 = nn.ReLU(inplace=True)
        self.relu5 = nn.ReLU(inplace=True)
        # FPN 的两个融合点必须各用独立实例：Vitis 对「复用模块」只会学一套量化刻度，
        # 并报 VAIQ_WARN: Reused module (fpn.add / fpn.interpolate) may lead to
        # low accuracy of QAT。各层级特征尺度差异很大，共用刻度必然掉精度。
        # 拆实例不改变任何计算（两个类都是无参数模块，forward 就是 torch.add /
        # F.interpolate），因此数学上严格等价，state_dict 也不变。
        self.add4 = add()
        self.add3 = add()
        self.up5 = interpolate()
        self.up4 = interpolate()

    def forward(self, feats):
        c3, c4, c5 = feats
        p5 = self.lat5(c5)
        p4 = self.add4(self.lat4(c4),
                       self.up5(p5, size=c4.shape[-2:], mode="nearest"))
        p3 = self.add3(self.lat3(c3),
                       self.up4(p4, size=c3.shape[-2:], mode="nearest"))
        return (self.relu3(self.smooth3(p3)), self.relu4(self.smooth4(p4)),
                self.relu5(self.smooth5(p5)))


class DetHeadEx(nn.Module):
    """DetHead 的等价改写：concat 走 nn.Module。"""

    def __init__(self, src):
        super().__init__()
        self.stem = src.stem
        self.cls = src.cls
        self.box = src.box
        self.lm = src.lm
        self.cat = cat()

    def forward(self, x):
        f = self.stem(x)
        return self.cat([self.cls(f), self.box(f), self.lm(f)], 1)


def _patch_children(mod, stats):
    """自底向上：先深入子模块，再替换自己（SE 藏在 InvertedResidual 里面）。"""
    for name, child in list(mod.named_children()):
        _patch_children(child, stats)
        cls = type(child).__name__
        if cls == "SqueezeExcitation":
            mod._modules[name] = SqueezeExcitationEx(child)
            stats["se"] += 1
        elif cls == "InvertedResidual":
            mod._modules[name] = InvertedResidualEx(child)
            stats["ir"] += 1
        elif cls == "FPNLite":
            mod._modules[name] = FPNLiteEx(child)
            stats["fpn"] += 1
        elif cls == "DetHead":
            mod._modules[name] = DetHeadEx(child)
            stats["head"] += 1


def patch_for_qat(model):
    """就地改造 model，返回改了多少处。"""
    stats = {"se": 0, "ir": 0, "fpn": 0, "head": 0}
    _patch_children(model, stats)
    print("[qat-patch] SE×%d  残差×%d  FPN×%d  检测头×%d" %
          (stats["se"], stats["ir"], stats["fpn"], stats["head"]))
    return stats
