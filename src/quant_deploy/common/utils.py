import os
import json
import time
import random
import numpy as np
import torch


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def save_json(obj, p):
    ensure_dir(os.path.dirname(p))
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model):
    """参数按 float32 计的模型体积(MB)。"""
    return count_params(model) * 4 / 1024 / 1024


class FlopCounter:
    """轻量 FLOPs 统计(conv2d/linear/bn)，单位 MAC，返回 GFLOPs。"""

    def __init__(self, model, input_size=640):
        self.total = 0
        self.hooks = []
        self.model = model
        self.input_size = input_size

    def _hook(self, module, inp, out):
        if isinstance(module, torch.nn.Conv2d):
            x = inp[0]
            kh, kw = module.kernel_size
            oc = module.out_channels
            groups = module.groups
            h, w = out.shape[-2:]
            macs = (kh * kw) * (x.shape[1] // groups) * oc * h * w // 1
            self.total += macs
        elif isinstance(module, torch.nn.Linear):
            self.total += inp[0].shape[-1] * module.out_features

    def count(self, device="cuda"):
        self.total = 0
        for m in self.model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                self.hooks.append(m.register_forward_hook(self._hook))
        self.model.eval()
        x = torch.zeros(1, 3, self.input_size, self.input_size,
                        device=device if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            self.model(x)
        for h in self.hooks:
            h.remove()
        return self.total / 1e9   # GMACs(常称 GFLOPs)


@torch.no_grad()
def measure_latency(model, input_size=640, device="cuda", repeat=50, warmup=10):
    """返回 (平均 ms, fps)。device='cuda' 或 'cpu'。"""
    dev = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
    m = model.to(dev).eval()
    x = torch.zeros(1, 3, input_size, input_size, device=dev)
    for _ in range(warmup):
        m(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        m(x)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / repeat * 1000
    return dt, 1000.0 / dt


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, v, n=1):
        self.val = v
        self.sum += v * n
        self.count += n
        self.avg = self.sum / max(1, self.count)
