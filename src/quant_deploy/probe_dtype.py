import sys

import torch
import pytorch_nndct  # noqa: F401  注册 vai::fix_neuron 等自定义算子

p = sys.argv[1]
m = torch.jit.load(p)
m.eval()
with torch.no_grad():
    outs = m(torch.zeros(1, 3, 320, 320))
print("model:", p)
if isinstance(outs, (list, tuple)):
    print("返回 list/tuple，长度", len(outs))
    for o in outs:
        print("   shape", tuple(o.shape), "dtype", o.dtype)
else:
    print("单张量 shape", tuple(outs.shape), "dtype", outs.dtype)
