"""TopoFR（NeurIPS 2024）—— 拓扑对齐正则项。

论文：TopoFR: A Closer Look at Topology Alignment on Face Recognition
      Jun Dan, Yang Liu, Jiankang Deng, Haoyu Xie, Siyuan Li, Baigui Sun, Shan Luo
      arXiv:2410.10587

核心思想
--------
margin-based softmax 只约束「样本 → 类中心」的局部关系，不约束整个 batch
的特征空间**形状**。TopoFR 额外要求：特征空间的拓扑结构（用 0 维持久同调
刻画，等价于单连接聚类的合并顺序）与**输入图像空间**的拓扑结构一致。

实现上只做三件事：
  1. 分别对 batch 的输入图像、batch 的 embedding 算 B×B 距离矩阵；
  2. 各跑一次 0 维持久同调（= 最小生成树），取出按权重升序排列的合并边；
  3. 比较两个「合并边序列」，差异平方和即拓扑损失。

这是**纯训练期正则**：推理时骨干网络结构、输出维度、比对方式全部不变，
因此可以叠加在任意骨干（GhostFaceNet / MobileFaceNet）和任意 margin 损失
（CosFace / ArcFace）之上，不影响部署。

用法
----
    topo = TopoFRLoss(weight=0.1)
    loss = loss_cls + topo(imgs, feats)

与官方实现的两处工程性改动
--------------------------
* **尺度归一化**：官方用可学习的 `latent_norm` 缩放两个距离矩阵，但它是在
  forward 里 `torch.nn.Parameter` 现造的，每步都是全新的 1.0，等价于没学。
  这里改成两个矩阵各自除以自身均值（默认 `scale_mode='mean'`），
  尺度无关、数值有界，不会像官方那样因为 512-d 特征范数是 20 而输入距离
  只有 0.3 导致 loss 被放大到与分类损失同量级。
* **子采样**：持久同调是 Python 循环，代价 O(B²)。`topo_sample` 可以只在
  部分样本上算，默认关闭（用满整个 batch）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

try:
    from scipy.sparse.csgraph import minimum_spanning_tree
except ImportError:  # pragma: no cover
    minimum_spanning_tree = None


class UnionFind:
    """并查集，用于 0 维持久同调（连通分量合并）。"""

    def __init__(self, n_vertices: int):
        self._parent = np.arange(n_vertices, dtype=np.int64)
        self._size = np.ones(n_vertices, dtype=np.int64)

    def find(self, u: int) -> int:
        # 路径压缩
        root = u
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[u] != root:
            self._parent[u], u = root, self._parent[u]
        return int(root)

    def merge(self, u: int, v: int):
        ru, rv = self.find(u), self.find(v)
        if ru == rv:
            return
        # 按大小合并，保证树矮
        if self._size[ru] < self._size[rv]:
            ru, rv = rv, ru
        self._parent[rv] = ru
        self._size[ru] += self._size[rv]

    def roots(self):
        return np.array([self.find(i) for i in range(len(self._parent))])


class PersistentHomologyCalculation:
    """0 维持久同调：取最小生成树，按边权升序返回合并边序列。

    返回 `persistence_pairs`，形状 (n-1, 2)，顺序即「合并发生时的边权升序」，
    这正是单连接层次聚类的合并序列。

    官方实现是 Python 写的 Kruskal，要遍历 B(B-1)/2 条边：B=128 时是 8128 次
    循环，**实测 519 ms/步**，会让一轮训练从 2.4 小时变成 20 小时，不可用。
    数学上「Kruskal 按升序取到的边」就是最小生成树的边，所以这里直接用
    scipy 的 C 实现 MST 再按权排序，结果等价但快两个数量级。
    """

    def __call__(self, matrix: np.ndarray):
        matrix = np.asarray(matrix, dtype=np.float64)
        n = matrix.shape[0]
        if n < 2:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 4), dtype=np.int64)

        mst = minimum_spanning_tree(matrix).tocoo()
        order = np.argsort(mst.data, kind="stable")
        rows, cols = mst.row[order], mst.col[order]
        lo = np.minimum(rows, cols)
        hi = np.maximum(rows, cols)
        pairs = np.stack([lo, hi], axis=1).astype(np.int64)
        return pairs, np.zeros((0, 4), dtype=np.int64)


class TopologicalSignatureDistance(nn.Module):
    """比较两个距离矩阵的持久同调签名。"""

    def __init__(self):
        super().__init__()
        self.signature_calculator = PersistentHomologyCalculation()

    def _get_pairings(self, distances: torch.Tensor):
        matrix = distances.detach().float().cpu().numpy()
        return self.signature_calculator(matrix)

    @staticmethod
    def _select_distances_from_pairs(distance_matrix: torch.Tensor, pairs):
        pairs_0, _ = pairs
        if len(pairs_0) == 0:
            return distance_matrix.new_zeros(0)
        idx = torch.as_tensor(pairs_0, device=distance_matrix.device)
        return distance_matrix[idx[:, 0], idx[:, 1]]

    @staticmethod
    def sig_error(signature1: torch.Tensor, signature2: torch.Tensor):
        n = min(signature1.numel(), signature2.numel())
        return ((signature1[:n] - signature2[:n]) ** 2).sum()

    def forward(self, distances1: torch.Tensor, distances2: torch.Tensor):
        pairs1 = self._get_pairings(distances1)
        pairs2 = self._get_pairings(distances2)
        sig1 = self._select_distances_from_pairs(distances1, pairs1)
        sig2 = self._select_distances_from_pairs(distances2, pairs2)
        return self.sig_error(sig1, sig2)


def compute_distance_matrix(x: torch.Tensor, p: int = 2) -> torch.Tensor:
    """batch 内两两距离矩阵，形状 (B, B)。

    官方写法 `torch.norm(x[:, None] - x[None, :], dim=2)` 会先展开出
    B×B×D 的临时张量：B=128、D=3×112×112=37632 时是 **2.4 GB**，
    光这一步就 480 ms/步，是整条链路的瓶颈。
    改用 ‖a-b‖² = ‖a‖² + ‖b‖² − 2·aᵀb，只做一次 matmul，不展开中间量。
    """
    x_flat = x.reshape(x.size(0), -1).float()
    sq = (x_flat * x_flat).sum(dim=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (x_flat @ x_flat.t())
    # 这里 clamp 的下界必须是 1e-12 而不是 0：
    # d(sqrt(x))/dx = 1/(2·sqrt(x))，在 x=0 处是 inf；距离矩阵对角线恒为 0，
    # 而 autograd 对「未被选中」的元素也会算 0 × inf = nan，
    # 于是整个骨干的梯度被污染成 nan（表现为第一步 |g_bb|=nan、
    # GradScaler 的 scale 从 65536 一路掉到 4096）。给个极小下界即可。
    d2 = d2.clamp(min=1e-12)
    if p == 2:
        return torch.sqrt(d2)
    return d2 ** (1.0 / p)


class TopoFRLoss(nn.Module):
    """TopoFR 的拓扑对齐正则。

    参数
    ----
    weight      : 叠加系数（论文用 0.1，即 loss = loss_cls + 0.1 * loss_topo）
    scale_mode  : 'mean'（各自除以自身均值，默认，尺度无关）
                  'max_distance'（官方写法：输入除以理论最大距离，特征不缩放）
    topo_sample : 只在前 k 个样本上算持久同调，None = 用满整个 batch
    """

    def __init__(self, weight: float = 0.1, scale_mode: str = "mean",
                 topo_sample: int | None = None):
        super().__init__()
        self.weight = float(weight)
        self.scale_mode = scale_mode
        self.topo_sample = topo_sample
        self.signature = TopologicalSignatureDistance()

    def _norm(self, d: torch.Tensor) -> torch.Tensor:
        if self.scale_mode == "max_distance":
            return d
        m = d.detach().mean().clamp(min=1e-8)
        return d / m

    def forward(self, input_space: torch.Tensor,
                feature_space: torch.Tensor) -> torch.Tensor:
        n = self.topo_sample
        if n is not None and n < input_space.size(0):
            input_space = input_space[:n]
            feature_space = feature_space[:n]

        input_distances = compute_distance_matrix(input_space.float())
        feature_distances = compute_distance_matrix(feature_space.float())

        if self.scale_mode == "max_distance" and input_space.dim() == 4:
            b, ch, h, w = input_space.shape
            input_distances = input_distances / float((4.0 * ch * h * w) ** 0.5)
        else:
            input_distances = self._norm(input_distances)
        feature_distances = self._norm(feature_distances)

        error = self.signature(input_distances, feature_distances)
        return error / float(input_space.size(0))
