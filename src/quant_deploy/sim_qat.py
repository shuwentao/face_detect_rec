"""在本机 GPU 上做「可微分版 Vitis 量化」的 QAT 微调，产出的权重交回容器做标定/导出。

为什么不用容器里的 nndct QAT：那个环境 torch 1.12 CPU 版，70/70 个 Conv2d 的
backward 都抛 "could not create a primitive descriptor"，没法训练。
而容器做标定和 test 只需要前向（这一直是好的），所以分工：
    本机 GPU 训练 -> qat_float.pt -> 容器 calib/test/deploy（纯前向）

仿真规则照抄 Vitis：
  torch_qalgo.py:491  step = max(x_min/-128.5, x_max/127.5)
                      pos  = floor(log2(1/step))，量化步长 = 2^-pos
  nndct_max_fix_position 限制 pos 上限（默认这里用 16，比导出时更粗 => 更保守）
并：
  * 仿.considers = Conv+BN 折叠后的等效权重（Vitis 导出就是按折叠后的权重去量化，
    不折叠的话你训的压根不是它最终量化的那个东西）
  * 用直通估计（STE）让量化可导

用法：
  # 先自检：仿真出的 rel-L2 应当 ≈ 容器里测到的 0.663
  python sim_qat.py --eval_only --input_size 640

  # 微调
  python sim_qat.py --input_size 640 --epochs 2 --lr 1e-5 --max_images 2000 \\
                    --out_dir simqat640
"""
import argparse
import copy
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.loss import DetectionLoss
from common.utils import seed_everything, ensure_dir
from qat_gate import make_loaders, move_targets, compare_with_float

DEFAULT_ROOT = "/mnt/data1/face_bench"


# ---------------------------------------------------------------- 仿真实量化
class Pow2STE(torch.autograd.Function):
    """per-tensor + 幂次步长 + 对称 int8，前向照 Vitis 量化，反向直通。"""

    @staticmethod
    def forward(ctx, x, bits, max_fp):
        det = x.detach()
        x_max, x_min = float(det.max()), float(det.min())
        ab = 1 << (bits - 1)              # 128
        step = max(x_min / (-ab - 0.5), x_max / (ab - 0.5))
        if step <= 0 or math.isnan(step):
            return x
        pos = int(math.floor(math.log2(1.0 / step)))
        if pos > max_fp:
            pos = max_fp
        s = 2.0 ** (-pos)
        return torch.round(det / s).clamp(-ab, ab - 1) * s

    @staticmethod
    def backward(ctx, g):
        return g, None, None


class Pow2STEPos(torch.autograd.Function):
    """pos 由外部（校准/滑动统计）给定，前向量化、反向直通。"""

    @staticmethod
    def forward(ctx, x, pos):
        s = 2.0 ** (-pos)
        ab = 128
        return torch.round(x / s).clamp(-ab, ab - 1) * s

    @staticmethod
    def backward(ctx, g):
        return g, None


def pos_from_range(x_min, x_max, max_fp, bits=8):
    """照抄 Vitis torch_qalgo.py:491 的 pos 公式。"""
    ab = 1 << (bits - 1)
    step = max(x_min / (-ab - 0.5), x_max / (ab - 0.5))
    if step <= 0 or math.isnan(step) or math.isinf(step):
        return max_fp
    pos = int(math.floor(math.log2(1.0 / step)))
    return min(pos, max_fp)


class _QConv(nn.Module):
    """包一层 Conv2d（若后面紧跟 BN 则同时仿真折叠），参数对象与原模块共享，
    这样 model.state_dict() 的键名保持不变，容器脚本能 strict=True 直接加载。"""

    def __init__(self, conv, bn=None, bits=8, max_fp=16, q_bias=False,
                 q_act=False, act_max_fp=16, act_mom=0.999, act_pos=None,
                 act_ema=True):
        super().__init__()
        self.weight = conv.weight
        self.bias = conv.bias
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.bits = bits
        self.max_fp = max_fp
        self.q_bias = q_bias
        self.q_act = q_act
        self.act_max_fp = act_max_fp
        self.act_mom = act_mom
        self.act_pos = act_pos      # 若给了就用容器标定的固定值，否则滑动估计
        self.act_ema = act_ema
        # 不能把 bn 注册成子模块（否则 state_dict 键名会变），塞进 __dict__
        self.__dict__["_bn"] = bn
        # 激活的滑动 min/max 同样不能注册成 buffer（会污染 state_dict）
        self.__dict__["_rmin"] = None
        self.__dict__["_rmax"] = None

    def _act_fixed(self, x):
        """用容器标定出来的固定 pos（和部署完全一致）。"""
        return Pow2STEPos.apply(x, self.act_pos)

    def _act(self, x):
        """按 Vitis 的做法量化激活：pos 由校准得到的固定量程决定，这里用滑动
        极值近似那份校准统计（部署时是固定的，不能逐 batch 变）。"""
        det = x.detach()
        x_min, x_max = float(det.min()), float(det.max())
        rmin, rmax = self.__dict__["_rmin"], self.__dict__["_rmax"]
        if rmin is None or rmin.device != det.device:
            rmin = torch.tensor(x_min, device=det.device)
            rmax = torch.tensor(x_max, device=det.device)
        else:
            m = self.act_mom
            rmax = torch.maximum(rmax * m, torch.tensor(x_max, device=det.device))
            rmin = torch.minimum(rmin * m, torch.tensor(x_min, device=det.device))
        self.__dict__["_rmin"], self.__dict__["_rmax"] = rmin, rmax
        pos = pos_from_range(float(rmin), float(rmax), self.act_max_fp, self.bits)
        return Pow2STEPos.apply(x, pos)

    def eff(self):
        w = self.weight
        bn = self.__dict__["_bn"]
        if bn is None:
            return w, self.bias
        s1 = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        w = w * s1.reshape([-1] + [1] * (w.dim() - 1))
        if self.bias is not None:
            b = (self.bias - bn.running_mean) * s1 + bn.bias
        else:
            b = -bn.running_mean * s1 + bn.bias
        return w, b

    def forward(self, x):
        # 只量化 Vitis 真正会量化的那些张量：给了标定 pos 的用固定 pos，
        # 没给标定数据的（--q_act 但没 --act_pos_json）才退化成滑动估计
        if self.q_act and self.act_pos is not None:
            x = self._act_fixed(x)
        elif self.q_act and self.act_ema:
            x = self._act(x)
        w, b = self.eff()
        wq = Pow2STE.apply(w, self.bits, self.max_fp)
        if b is not None and self.q_bias:
            b = Pow2STE.apply(b, self.bits, self.max_fp)
        return F.conv2d(x, wq, b, self.stride, self.padding,
                        self.dilation, self.groups)


def load_act_pos(json_path):
    """从容器标定出的 quant_info.json 里读出每个 conv 输入激活的 pos。
    节点名形如 .../Conv2d[0]/input.3，方括号里就是模块名。"""
    import json
    import re
    qi = json.load(open(json_path)) if isinstance(json_path, str) else json_path
    pos = {}
    for k, v in qi.get("output", {}).items():
        if "/input." not in k:
            continue
        path = ".".join(re.findall(r"\[([^\]]+)\]", k))
        if not path:
            continue
        p = int(v[0][1])
        pos[path] = min(pos.get(path, p), p)
    return pos


def wrap_all_conv(model, bits=8, max_fp=16, q_bias=False, q_act=False,
                  act_max_fp=16, act_pos=None):
    """把所有 Conv2d 换成 _QConv；conv 后面紧跟的 BN 改成恒等（由 _QConv 承担）。"""
    n_conv = n_pair = 0
    full = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            full[id(m)] = name
    hit = 0
    for parent in list(model.modules()):
        ch = list(parent.children())
        names = [n for n, _ in parent.named_children()]
        for i, c in enumerate(ch):
            if not isinstance(c, nn.Conv2d):
                continue
            bn = ch[i + 1] if (i + 1 < len(ch) and isinstance(
                ch[i + 1], (nn.BatchNorm2d, nn.BatchNorm1d))) else None
            ap = act_pos.get(full.get(id(c), "")) if act_pos else None
            hit += ap is not None
            setattr(parent, names[i],
                    _QConv(c, bn, bits=bits, max_fp=max_fp, q_bias=q_bias,
                           q_act=q_act, act_max_fp=act_max_fp, act_pos=ap,
                           act_ema=(act_pos is None)))
            n_conv += 1
            if bn is not None:
                n_pair += 1
                # 改成恒等，避免重复 affine。
                # 注意：实例属性上的函数不会自动绑定 self，所以这里只能是单参
                bn.forward = lambda x: x
    return model, n_conv, n_pair, hit


# ---------------------------------------------------------------- 训练
def evaluate_det(model, loader, criterion, device, limit=None):
    model.eval()
    n = limit or max(1, int(0.3 * len(loader)))
    acc, cnt, t0 = 0.0, 0, time.time()
    with torch.no_grad():
        for i, (imgs, tgts, _) in enumerate(loader):
            if i >= n:
                break
            out = criterion(model(imgs.to(device, non_blocking=True)),
                            move_targets(tgts, device))
            v = float(out["loss"].detach())
            if math.isfinite(v):
                acc += v
                cnt += 1
    return acc / max(cnt, 1), cnt


def main():
    ap = argparse.ArgumentParser("本机 GPU 的 Vitis 语义 QAT")
    ap.add_argument("--data_root", default=os.path.join(DEFAULT_ROOT, "data"))
    ap.add_argument("--train_ann", default=None)
    ap.add_argument("--ckpt", default="best_sd.pt")
    ap.add_argument("--input_size", type=int, default=640)
    ap.add_argument("--out_dir", default="simqat640")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--scale_max", type=float, default=1.2)
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--max_fix_position", type=int, default=16,
                    help="限制 pos 上限，<=0 表示不限制。默认 16 比导出时更粗，偏保守")
    ap.add_argument("--q_bias", action="store_true", help="连 bias 一起仿真量化")
    ap.add_argument("--q_act", action="store_true",
                    help="连激活一起仿真量化（容器实测表明权重之外还有约 0.26 的误差来自激活）")
    ap.add_argument("--act_max_fix_position", type=int, default=20)
    ap.add_argument("--act_pos_json", default=None,
                    help="容器标定出的 quant_info.json，直接用里面的激活 pos（最贴近部署）")
    ap.add_argument("--kd_weight", type=float, default=1.0,
                    help="让量化模型输出向冻结的浮点模型靠拢的强度")
    ap.add_argument("--det_weight", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(args.device)
    args.train_ann = args.train_ann or os.path.join(
        args.data_root, "annotations", "train.json")
    print("[env] device=%s | torch=%s" % (dev, torch.__version__))

    from common.models.detector import build_detector
    model = build_detector("mobilenetv3", pretrained=False, width=1.0)
    sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    print("[model]", model.load_state_dict(sd, strict=True))
    model.eval().to(dev)

    float_ref = copy.deepcopy(model).eval()      # 冻结的浮点参照
    for p in float_ref.parameters():
        p.requires_grad_(False)

    act_pos = load_act_pos(args.act_pos_json) if args.act_pos_json else None
    model, n_conv, n_pair, hit = wrap_all_conv(
        model, bits=args.bits, max_fp=args.max_fix_position or 999,
        q_bias=args.q_bias, q_act=args.q_act,
        act_max_fp=args.act_max_fix_position or 999, act_pos=act_pos)
    print("[仿真] 包装 Conv2d %d 个，其中 %d 个做了 Conv+BN 折叠仿真；"
          "权重量化=开 激活量化=%s bias量化=%s；激活 pos 命中 %d/%d"
          % (n_conv, n_pair, args.q_act, args.q_bias, hit, n_conv))
    model.to(dev)

    tl, vl = make_loaders(args)
    criterion = DetectionLoss()

    # ---- 保真度自检：这里的 rel-L2 应该接近容器里测到的 0.663 ----
    rel0, cos0 = compare_with_float(float_ref, model, vl, dev, n_batch=3)
    print("[自检] 若上面这行 ≈ 0.66 / 0.78，说明仿真和 Vitis 对得上；"
          "差太远就别往下训，先告诉我")
    if args.eval_only:
        return

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.wd)
    out_dir = ensure_dir(args.out_dir)
    best = float("inf")

    for ep in range(args.epochs):
        model.train()
        t0, n, run_d, run_k = time.time(), 0, 0.0, 0.0
        for i, (imgs, tgts, _) in enumerate(tl):
            x = imgs.to(dev, non_blocking=True)
            tg = move_targets(tgts, dev)
            out = model(x)
            loss = 0.0
            if args.det_weight:
                ld = criterion(out, tg)["loss"]
                loss = loss + args.det_weight * ld
                run_d += float(ld.detach())
            if args.kd_weight:
                with torch.no_grad():
                    fout = float_ref(x)
                lk = sum((o.float() - f.float()).pow(2).mean()
                         for o, f in zip(out, fout))
                loss = loss + args.kd_weight * lk
                run_k += float(lk.detach())
            if not math.isfinite(float(loss.detach())):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            n += 1
            if i % 50 == 0:
                print("  ep%d step %d/%d det=%.4f kd=%.4f (%.0fs)"
                      % (ep + 1, i, len(tl), run_d / max(n, 1),
                         run_k / max(n, 1), time.time() - t0), flush=True)

        rel, cos = compare_with_float(float_ref, model, vl, dev, n_batch=3)
        vdet, vn = evaluate_det(model, vl, criterion, dev)
        print("[epoch %d/%d] det_train=%.4f kd_train=%.4f val_det_loss=%.4f "
              "| 仿真 rel-L2 %.4f -> %.4f cos %.4f -> %.4f | %.0fs"
              % (ep + 1, args.epochs, run_d / max(n, 1), run_k / max(n, 1), vdet,
                 rel0, rel, cos0, cos, time.time() - t0), flush=True)

        if rel < best:
            best = rel
            torch.save({"model": model.state_dict()},
                       os.path.join(out_dir, "qat_float.pt"))
    torch.save({"model": model.state_dict()},
               os.path.join(out_dir, "qat_float_last.pt"))
    print("[done] 最佳仿真实量化 rel-L2 = %.4f（训练前 %.4f）" % (best, rel0))
    print("[done] 把 %s/qat_float.pt 交给容器：--weights qat_float.pt" % out_dir)


if __name__ == "__main__":
    main()
