"""训练用分类头：ArcFace / AdaFace(+Sub-center) + Partial FC 类采样。

为什么分类头必须做「类采样」(Partial FC)
--------------------------------------------------------------------------
Glint360K 有 36 万类。若每个 step 都拿全部类中心算 softmax：
    360000 × 512 × 4B  ≈ 737MB 权重 + 同样大的梯度 + 优化器状态 ≈ 2.2GB
单张 8GB 显卡还没算激活就爆了。Partial FC（InsightFace, TPAMI 2022）的做法是
每步只抽取一部分类中心参与计算，梯度只更新被抽中的类，收敛几乎无损。

本文件的实现额外保证：**batch 内出现过的正类一定在抽中的子集里**，
否则正样本的 logit 会被当成负类算，直接破坏训练。

AdaFace 的自适应 margin（CVPR 2022 Oral）  ⚠️ 实测不可用，见下
--------------------------------------------------------------------------
核心：用特征范数 z=‖f‖ 近似「图像质量」，让 margin 随质量变化。

⚠️ 警告：本文件的 AdaFace 是自研变体（批内 z-score 质量度 + 特征范数硬上限），
实测在 3 万类 / 46 万张上 LFW 卡在 55% 不再涨，而同配置 CosFace 第 3 个 epoch
就到 92.5%。训练默认已改成 cosface；AdaFace 保留作参考，别直接上生产。

    ẑ         = clip((z − μ_z) / σ_z, −1, 1)          # 批内 z-score
    m_i       = −m · ẑ_i                               # ∈ [−m, +m]，默认
    logits    = s · cos(θ + m_i·1[正类])

        高质量 (ẑ→+1) → m_i→−m  ：样本已经很「干净」，把 margin 放松
        低质量 (ẑ→−1) → m_i→+m  ：样本含噪、类内方差大，用更大 margin 逼它学
    这也是 AdaFace 在 TinyFace / CPLFW / IJB-S 这类低质量集上提升最大的原因。

    两个细节是实测调出来的，和论文写法不同，但都更稳：

    ① ẑ 用「批内 z-score」而不是绝对范数的 EMA。用绝对值会留一条捷径：
       模型只要把特征范数整体拉大，ẑ 就变大、margin 就变小，难度被自己调低，
       实测范数能涨到 1e4，loss 随之反弹。批内相对值下整体放大不改变 ẑ。
    ② 允许负 margin 就还有拉大范数的动机，所以骨干侧再配一道硬上限
       `max_feat_norm`（默认 64，见 ghostfacenet.py），把捷径彻底堵死。
    这两条加起来才能既保留 AdaFace 的快收敛、又不让它躺平。

    若想彻底保守，可传 `allow_negative_margin=False`（CLI: --adaface_nonneg），
    此时 m_i = m·(1−ẑ)/2 ∈ [0, m]，代价是收敛慢一个量级。

    与 ArcFace 的差别仅在 m_i 是否随样本变化；`--loss arcface` 时 m_i ≡ m。

μ_z、σ_z 用 EMA 维护（不参与梯度），第一次前向用批内统计初始化。
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialFCHead(nn.Module):
    """AdaFace / ArcFace 分类头，内置 sub-center 与类采样。

    参数
    ----
    emb_dim       : 特征维度（= backbone 输出维度）
    num_classes   : 身份数
    loss_type     : 'adaface' | 'arcface' | 'cosface'
    m / s         : margin 幅度 / logit 缩放
    sub_centers   : 每个身份的类中心个数（Sub-center ArcFace, 1 = 关闭）
    sample_rate   : 每步参与计算的类比例（1.0 = 全类，等于不用 Partial FC）

    前向
    ----
    labels 给定时返回 (loss, top1_acc)；labels=None 时返回全类余弦（用于验证/检索）。
    """

    def __init__(self, emb_dim: int, num_classes: int, loss_type: str = "adaface",
                 m: float = 0.4, s: float = 64.0, sub_centers: int = 1,
                 sample_rate: float = 0.1, norm_ema: float = 0.99,
                 label_smooth: float = 0.0, adaface_warmup: int = 0,
                 allow_negative_margin: bool = True):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_classes = num_classes
        self.loss_type = loss_type.lower()
        self.m = m
        self.s = s
        self.sub_centers = max(1, sub_centers)
        self.sample_rate = float(sample_rate)
        self.norm_ema = norm_ema
        self.label_smooth = label_smooth
        # 前 N 步先用固定 margin：初始特征范数还没稳定，自适应 margin 容易抖
        self.adaface_warmup = int(adaface_warmup)
        # True（默认）→ margin ∈ [-m, +m]，论文原式 -m·ĥ：高质量样本放松、
        #   低质量样本加严。实测 4409 类 7 个 epoch 就到 loss 7.1 / acc 0.24。
        # False → margin ∈ [0, m] 的保守版，任何样本都不放松。实测同样 14 个
        #   epoch 才 loss 16.5 且 acc 恒为 0，学习慢一个量级。
        # 负 margin 的副作用（拉大范数换放松）由骨干的 max_feat_norm 兜住。
        self.allow_negative_margin = bool(allow_negative_margin)

        self.weight = nn.Parameter(
            torch.empty(num_classes, self.sub_centers, emb_dim))
        nn.init.normal_(self.weight, 0, 0.01)

        # AdaFace 的范数统计（EMA）。初始 None，首次前向用批内统计填。
        self.register_buffer("norm_mean", torch.tensor(0.0))
        self.register_buffer("norm_std", torch.tensor(0.0))
        self.register_buffer("norm_inited", torch.tensor(0, dtype=torch.int64))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.int64))
        # Partial FC 下每步只有部分类行参与前向，对应梯度是稀疏的。
        # 若把整块 weight 交给普通 SGD(momentum)，未采样行的动量缓冲会
        # 一直被 momentum 衰减、同时持续推动权重漂移 —— 这类漂移会让
        # 未参与训练的类「悄悄跑偏」，若干 epoch 后训练 acc 直接崩到 0。
        # 所以这里自己维护动量，并且只更新本步采样到的行。
        # persistent=False：动量缓冲不写进 state_dict，否则 36 万类的
        # checkpoint 会凭空多出 700MB。
        self.manual_update = self.sample_rate < 1.0
        if self.manual_update:
            self.register_buffer("weight_mom", torch.zeros_like(self.weight),
                                 persistent=False)
        else:
            self.weight_mom = None      # 全类参与时交给普通 SGD，省一份显存
        self.last_indices = None

    # ------------------------------------------------------------------ #
    @property
    def _cos_eps(self):
        return 1e-4

    def _cosine(self, feats: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """feats (B,D) 与 weight (K,S,D) 的余弦，sub-center 取最大 → (B,K)。"""
        f = F.normalize(feats, dim=1)
        w = F.normalize(weight, dim=2)
        k, s_, d = w.shape
        cos = f @ w.reshape(k * s_, d).t()          # (B, K*S)
        if s_ > 1:
            cos = cos.view(-1, k, s_).max(dim=2).values
        return cos

    def _update_norm_stats(self, norms: torch.Tensor) -> None:
        with torch.no_grad():
            bmean = norms.mean()
            bstd = norms.std(unbiased=False).clamp_min(1e-3)
            if self.norm_inited.item() == 0:
                self.norm_mean.copy_(bmean)
                self.norm_std.copy_(bstd)
                self.norm_inited.fill_(1)
            else:
                a = self.norm_ema
                self.norm_mean.mul_(a).add_(bmean * (1 - a))
                self.norm_std.mul_(a).add_(bstd * (1 - a))

    def _sample_classes(self, labels: torch.Tensor
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (采样后的类下标, 正类在采样下标里的位置)。

        保证 batch 内出现的真实类一定被选中。
        """
        k = self.num_classes
        if self.sample_rate >= 1.0 or self.sample_rate * k >= k:
            idx = torch.arange(k, device=labels.device)
            return idx, labels

        n_sample = int(math.ceil(self.sample_rate * k))
        n_sample = max(n_sample, 2 * int(labels.unique().numel()))
        n_sample = min(n_sample, k)

        rand = torch.randperm(k, device=labels.device)[:n_sample]
        sel = torch.cat([labels.unique(), rand]).unique()
        # sel 里的元素 → 其在 sel 中的位置
        pos = torch.searchsorted(sel, labels)
        return sel, pos

    # ------------------------------------------------------------------ #
    def forward(self, feats: torch.Tensor, labels: torch.Tensor | None = None):
        feats = feats.float()
        if labels is None:
            # 推理/验证：对全部类算余弦
            return self._cosine(feats, self.weight).clamp(-1.0 + 1e-6, 1.0 - 1e-6)

        norms = feats.norm(p=2, dim=1)
        sel, pos = self._sample_classes(labels)
        self.last_indices = sel          # 供外部做「只更新采样行」的动量 SGD
        weight = self.weight.index_select(0, sel)

        cos = self._cosine(feats, weight).clamp(-1.0 + self._cos_eps,
                                                1.0 - self._cos_eps)
        cos = cos.float()

        onehot = torch.zeros_like(cos)
        onehot.scatter_(1, pos.view(-1, 1), 1.0)

        if self.loss_type == "arcface":
            margin = torch.full_like(norms, self.m)
        elif self.loss_type == "cosface":
            margin = None
        elif self.loss_type == "adaface":
            # 用「批内相对范数」而不是绝对范数的 EMA 统计。
            # 用绝对值会留一条致命捷径：模型只要把特征范数整体拉大，
            # ĥ 就跟着变大、margin 就跟着变小，难度被自己调低。
            # 实测这么跑特征范数会一路涨到 1e4，loss 从 14 反弹到 18 并持续恶化。
            # 改成批内 z-score 后，整体放大范数不再改变 ĥ，捷径被堵死。
            bmean = norms.mean().detach()
            bstd = norms.std().clamp(min=1e-3).detach()
            h_hat = ((norms - bmean) / bstd).clamp(-1.0, 1.0)
            # 顺带把统计写进 buffer，方便训练日志观察范数是否在漂移
            self.norm_mean.mul_(0.99).add_(bmean * 0.01)
            self.norm_std.mul_(0.99).add_(bstd * 0.01)
            # AdaFace 的核心：低质量样本（ĥ 小）承担更大 margin。
            if self.allow_negative_margin:
                # 论文原式：高质量样本 margin 变负（放松），低质量变正（加严）
                margin = -self.m * h_hat                    # ∈ [-m, +m]
                warm_m = 0.0
            else:
                # 保守版：任何样本都不放松，margin ∈ [0, m]。
                # 允许负 margin 时会留一条捷径 —— 模型把特征范数整体拉大就能
                # 让 margin 变负、难度下调，实测 3 万类上训练 acc 从 0.90 掉到 0.42、
                # 特征范数一路涨到 1e4。配合骨干的 max_feat_norm 上限可以兜住，
                # 但默认仍走保守版更稳。
                margin = self.m * (1.0 - h_hat) * 0.5       # ∈ [0, m]
                warm_m = self.m * 0.5
            if self.adaface_warmup > 0 and self.step_count.item() < self.adaface_warmup:
                margin = torch.full_like(norms, warm_m)
            self.step_count += 1
        else:
            raise ValueError(f"未知 loss_type: {self.loss_type}")

        if self.loss_type == "cosface":
            logits = cos - self.m * onehot
        else:
            # cos(θ + m) 的闭式展开，省掉 acos，梯度更稳、更快
            cos_m = torch.cos(margin).unsqueeze(1)
            sin_m = torch.sin(margin).unsqueeze(1)
            # 下限给 1e-4 而不是 1e-9：cos→±1 时 sqrt(1-cos²)→0，
            # 其倒数会让梯度爆炸（训练后期特征与类中心对准时尤其明显）。
            sin_theta = torch.sqrt(torch.clamp(1.0 - cos ** 2, min=1e-4))
            cos_tm = cos * cos_m - sin_theta * sin_m
            logits = torch.where(onehot > 0, cos_tm, cos)

        logits = logits * self.s

        if self.label_smooth > 0:
            loss = F.cross_entropy(logits, pos, label_smoothing=self.label_smooth)
        else:
            loss = F.cross_entropy(logits, pos)

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == pos).float().mean()
        return loss, acc


def build_head(emb_dim: int, num_classes: int, loss_type: str = "adaface",
               **kw) -> PartialFCHead:
    return PartialFCHead(emb_dim, num_classes, loss_type=loss_type, **kw)
