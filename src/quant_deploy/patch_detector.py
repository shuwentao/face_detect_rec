"""把关键点分支从 15 通道 concat 里拆出来，让 lm 拿到独立的量化刻度。

改前后对外行为完全一致（返回的仍是 list[(B,15,H,W)]，decode_preds 不用改），
权重也完全不变（stem/cls/box/lm 都在），只是 lm 走自己的 DeQuantStub。
"""
import shutil

P = "/workspace/mymodel/gate320/common/models/detector.py"
shutil.copy(P, P + ".bak_splitlm")

s = open(P).read()

old1 = """    def forward(self, x):
        f = self.stem(x)
        return torch.cat([self.cls(f), self.box(f), self.lm(f)], dim=1)"""
new1 = """    def forward(self, x):
        f = self.stem(x)
        # det(cls+box) 与 lm 分开返回：二者各走一个 DeQuantStub，才能拿到
        # 各自独立的量化刻度。原先 concat 成 15 通道只能共用一把尺子。
        return torch.cat([self.cls(f), self.box(f)], dim=1), self.lm(f)"""
assert old1 in s, "DetHead.forward 未匹配"
s = s.replace(old1, new1)

old2 = """        self.quant = _QS()
        self.dequant = _DQS()"""
new2 = """        self.quant = _QS()
        self.dequant = _DQS()        # 保留字段以兼容旧 ckpt/脚本，不参与 forward
        self.dequant_det = _DQS()    # det 分支独立刻度
        self.dequant_lm = _DQS()     # lm 分支独立刻度"""
assert old2 in s, "__init__ 未匹配"
s = s.replace(old2, new2)

old3 = """        return [self.dequant(head(f)) for head, f in zip(self.heads, feats)]"""
new3 = """        outs = []
        for head, f in zip(self.heads, feats):
            det, lm = head(f)                    # (B,5,H,W) / (B,10,H,W)
            outs.append(torch.cat([self.dequant_det(det),
                                   self.dequant_lm(lm)], dim=1))
        return outs"""
assert old3 in s, "forward 未匹配"
s = s.replace(old3, new3)

open(P, "w").write(s)
print("[ok] patched:", P)
