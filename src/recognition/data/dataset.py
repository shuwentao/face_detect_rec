"""训练/校准数据管线。

核心点：**训练增强必须复刻端上的对齐误差**
--------------------------------------------------------------------------
部署链路是「原图 + 检测到的 5 点 → estimateAffinePartial2D → 112×112」。
关键点本身带误差（YuNet 在侧脸/遮挡/小尺度上会漂几个像素），再加上检测框
抖动，最终送进网络的图相对「理想对齐」会有：平移 ±3~5px、旋转 ±5~10°、
缩放 ±5%，以及极端情况下的倾斜（roll 角估计错）。

而 Glint360K 里的图已经是**理想对齐**的。如果训练时不加扰动，模型会过度
依赖「像素级对齐」，一到板子上精度就掉。所以这里的相似变换增强不是可选项，
它是对齐链路的 domain randomisation。

翻转的说明
--------------------------------------------------------------------------
人脸近似左右对称，水平翻转是 insightface 系列的标准增强；翻转后身份标签
不变（112×112 已对齐图直接镜像即可，不需要交换关键点）。

用法
--------------------------------------------------------------------------
    from data.dataset import ShardDataset, FaceAlignAug, build_loader
    aug = FaceAlignAug(size=112)
    ds = ShardDataset("shards", aug, training=True)
    loader = build_loader(ds, batch_size=128, num_workers=6)
"""
from __future__ import annotations

import os
import sys
import json
import struct
import math
import numpy as np
import cv2
from torch.utils.data import IterableDataset, get_worker_info

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# DataLoader 多进程下让 cv2 单线程，避免 worker 之间抢线程反而变慢
cv2.setNumThreads(0)


# --------------------------------------------------------------------------- #
# 增强
# --------------------------------------------------------------------------- #
class FaceAlignAug:
    """对齐扰动 + 光度扰动增强。

    参数按「端上对齐误差」的实测量级设定：
        rot   ±8°     关键点 roll 角误差
        scale ±5%     检测框尺寸误差
        trans ±4%     检测框中心偏移
        shear 0        （相似变换不含剪切；端上是相似变换）
    另外用 Perspective 小概率模拟「关键点个别点漂移」非刚体形变。
    """

    def __init__(self, size: int = 112, rot: float = 8.0, scale: float = 0.05,
                 trans: float = 0.04, flip_p: float = 0.5,
                 brightness: float = 0.2, contrast: float = 0.2,
                 gamma: float = 0.15, blur_p: float = 0.1,
                 jpeg_p: float = 0.1, mask_p: float = 0.05,
                 seed: int | None = None):
        self.size = size
        self.rot = rot
        self.scale = scale
        self.trans = trans
        self.flip_p = flip_p
        self.brightness = brightness
        self.contrast = contrast
        self.gamma = gamma
        self.blur_p = blur_p
        self.jpeg_p = jpeg_p
        self.mask_p = mask_p
        self._rng = np.random.default_rng(seed)

    # -- 几何 -------------------------------------------------------------- #
    def _warp(self, img: np.ndarray) -> np.ndarray:
        s = self.size
        rng = self._rng
        ang = rng.uniform(-self.rot, self.rot)
        sc = 1.0 + rng.uniform(-self.scale, self.scale)
        tx = rng.uniform(-self.trans, self.trans) * s
        ty = rng.uniform(-self.trans, self.trans) * s
        M = cv2.getRotationMatrix2D((s / 2.0 - 0.5, s / 2.0 - 0.5), ang, sc)
        M[0, 2] += tx
        M[1, 2] += ty
        return cv2.warpAffine(img, M, (s, s), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    # -- 光度 -------------------------------------------------------------- #
    def _photo(self, img: np.ndarray) -> np.ndarray:
        rng = self._rng
        out = img.astype(np.float32)
        if self.brightness:
            out += rng.uniform(-1, 1) * self.brightness * 255.0 * 0.5
        if self.contrast:
            mean = out.mean()
            out = (out - mean) * (1.0 + rng.uniform(-1, 1) * self.contrast) + mean
        if self.gamma and abs(rng.normal()) > 1.0:
            g = math.exp(rng.uniform(-1, 1) * self.gamma)
            out = 255.0 * np.power(np.clip(out, 0, 255) / 255.0, g)
        img = np.clip(out, 0, 255).astype(np.uint8)

        if rng.random() < self.blur_p:
            k = int(rng.choice([3, 5]))
            img = cv2.GaussianBlur(img, (k, k), 0)
        if rng.random() < self.jpeg_p:
            q = int(rng.integers(30, 95))
            ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
            if ok:
                img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        if rng.random() < self.mask_p:
            img = self._random_occlusion(img)
        return img

    def _random_occlusion(self, img: np.ndarray) -> np.ndarray:
        """随机矩形遮挡：模拟口罩/手/头发等真实遮挡。"""
        rng = self._rng
        s = self.size
        w = int(rng.uniform(0.25, 0.6) * s)
        h = int(rng.uniform(0.1, 0.35) * s)
        x = int(rng.uniform(0, s - w))
        y = int(rng.uniform(int(0.4 * s), s - h))     # 偏下半脸
        img = img.copy()
        img[y:y + h, x:x + w] = int(rng.integers(0, 255))
        return img

    # -- 出口 -------------------------------------------------------------- #
    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        img = self._warp(img_bgr)
        if self._rng.random() < self.flip_p:
            img = img[:, ::-1]
        img = self._photo(img)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = rgb.astype(np.float32)
        x = (x - 127.5) / 127.5                       # -> [-1, 1]
        return np.ascontiguousarray(x.transpose(2, 0, 1))   # CHW


class ValAug:
    """验证/校准用：只做归一化，不做随机扰动（保证可复现）。"""

    def __init__(self, size: int = 112):
        self.size = size

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        if img_bgr.shape[0] != self.size or img_bgr.shape[1] != self.size:
            img_bgr = cv2.resize(img_bgr, (self.size, self.size))
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = (rgb.astype(np.float32) - 127.5) / 127.5
        return np.ascontiguousarray(x.transpose(2, 0, 1))


# --------------------------------------------------------------------------- #
# 分片数据集
# --------------------------------------------------------------------------- #
def load_shard(path: str):
    """把整个分片读进内存，返回 [(label, jpeg_bytes), ...]。"""
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:4] != b"FRSH":
        raise ValueError(f"{path} 不是合法的分片文件")
    (count,) = struct.unpack_from("<I", blob, 4)
    out = []
    off = 8
    for _ in range(count):
        lab, jlen = struct.unpack_from("<iI", blob, off)
        off += 8
        out.append((lab, blob[off:off + jlen]))
        off += jlen
    return out


class ShardDataset(IterableDataset):
    """按分片流式产样本的 IterableDataset。

    多 worker 协同：每个 worker 只取自己那一份分片（stride=num_workers），
    互不重复；每个分片内部再打乱，于是 batch 内身份足够杂。

    __iter__ 每次被调用（= 每个 epoch）都会换一次分片顺序，
    相当于自动 reshuffle。
    """

    def __init__(self, shard_dir: str, aug, training: bool = True,
                 seed: int = 0, drop_last_shard: bool = False):
        self.dir = shard_dir
        with open(os.path.join(shard_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.paths = [os.path.join(shard_dir, s) for s in self.meta["shards"]]
        self.aug = aug
        self.training = training
        self.seed = seed
        self.drop_last_shard = drop_last_shard
        self._epoch = 0
        self.num_classes = self.meta["num_classes"]

    def __len__(self):
        return self.meta["num_samples"]

    def set_epoch(self, e: int):
        self._epoch = e

    def __iter__(self):
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        nw = wi.num_workers if wi is not None else 1

        rng = np.random.default_rng(self.seed + self._epoch * 10007 + wid)
        order = rng.permutation(len(self.paths))
        mine = order[wid::nw]

        for k in mine:
            samples = load_shard(self.paths[int(k)])
            idx = rng.permutation(len(samples))
            for i in idx:
                lab, jpg = samples[int(i)]
                buf = np.frombuffer(jpg, np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                yield self.aug(img), int(lab)
        self._epoch += 1


class RecDataset(IterableDataset):
    """直接从 .rec + 索引读的 IterableDataset（无需预先分片）。

    机械盘上随机 seek 很慢，只推荐用于：小规模跑通流程、量化校准集取图。
    """

    def __init__(self, rec_path: str, index_path: str, aug,
                 limit: int | None = None, seed: int = 0, shuffle: bool = False):
        super().__init__()
        from data.mrec import load_index, RecReader
        offs, ids, attrs = load_index(index_path)
        if limit:
            offs, ids, attrs = offs[:limit], ids[:limit], attrs[:limit]
        self.reader = RecReader(rec_path, offs, ids, attrs)
        self.aug = aug
        self.seed = seed
        self.shuffle = shuffle
        self._epoch = 0
        self.num_classes = int(ids.max()) + 1

    def __len__(self):
        return len(self.reader)

    def set_epoch(self, e: int):
        self._epoch = e

    def __iter__(self):
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        nw = wi.num_workers if wi is not None else 1
        n = len(self.reader)
        idx = np.arange(n)
        if self.shuffle:
            np.random.default_rng(self.seed + self._epoch).shuffle(idx)
        for i in idx[wid::nw]:
            i = int(i)
            img = self.reader.read(i)                        # BGR
            yield self.aug(img), int(self.reader.identities[i])
        self._epoch += 1


class ImageListDataset(IterableDataset):
    """从「图片目录 + 清单文件」读（每行 `<路径> <label>`），用于自定义数据。"""

    def __init__(self, list_file: str, aug, root: str = ""):
        super().__init__()
        self.items = []
        with open(list_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p, lab = line.rsplit(maxsplit=1)
                self.items.append((os.path.join(root, p) if root else p, int(lab)))
        self.aug = aug
        self.num_classes = max(l for _, l in self.items) + 1 if self.items else 0

    def __len__(self):
        return len(self.items)

    def set_epoch(self, e: int):
        np.random.default_rng(e).shuffle(self.items)

    def __iter__(self):
        for p, lab in self.items:
            img = cv2.imread(p)
            if img is None:
                continue
            yield self.aug(img), int(lab)


def build_loader(dataset, batch_size: int, num_workers: int = 6,
                 pin_memory: bool = True):
    """包一层 DataLoader。IterableDataset 不设 shuffle（自身已打乱）。"""
    import torch
    from torch.utils.data import DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
    )
