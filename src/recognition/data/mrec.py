"""MXNet RecordIO (.rec) 读取 —— Glint360K / MS1M 训练集的容器格式。

为什么自己解析而不装 mxnet
--------------------------------------------------------------------------
mxnet 已停止维护，且与 numpy>=2 冲突。RecordIO 只是一层很薄的封装，
逐字节解析只有几十行，还能避免把 200MB 的运行时拖进环境。

实测记录布局（本文件即按此实现，已用 17 万条样本抽样 + 图像肉眼核对验证）
--------------------------------------------------------------------------
    偏移      字段          类型/值                 说明
    --------------------------------------------------------------------
    0         magic         0xCED7230A              固定魔数（小端 0A 23 D7 CE）
    4         lrecord       uint32                  = payload 长度 + 32
    8         flag          uint32                  恒 2
    12        -             uint32                  恒 0
    16        image_id      int64                   全局序号，从 1 递增（≠ 类别）
    24        -             int64                   恒 0
    32        identity      float32                 ★ 身份标签（0 ~ 360231）
    36        attr          float32                 附带属性，取值 0/1（性别或域标记）
    40        payload       bytes                   JPEG 字节流
    tail      pad           0~3 字节                补 0 使整条记录 4 字节对齐

整条记录长度 = 8 + lrecord，再向上取整到 4 的倍数。

> 注意：字段「identity」正是 insightface 侧 `header.label` 的落点，
> 但在此数据集里它并不在标准 MXNet 的 label 偏移上（那里恒为 0），
> 这也是很多现成解析代码读出来标签全是 0 的原因。

参考点 112×112 的标准 5 点模板（ArcFace）与本格式中图像已经过的对齐一致：
    [[38.2946, 51.6963], [73.5318, 51.6963], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.3655]]
"""
from __future__ import annotations

import os
import io
import struct
import numpy as np
from PIL import Image

MAGIC = 0xCED7230A
HEADER_SIZE = 40          # magic(4) + lrecord(4) + IRHeader(32)

_IDENTITY_OFF = 32        # header 内 identity(float32) 的偏移
_ATTR_OFF = 36            # header 内 attr(float32) 的偏移
_PAYLOAD_OFF = HEADER_SIZE


def rec_len_at(f, offset: int) -> int:
    """返回 offset 处整条记录的字节长度（含尾部对齐填充）。"""
    f.seek(offset)
    head = f.read(8)
    if len(head) < 8:
        raise EOFError("读取记录头失败（已到文件末尾？）")
    magic, lrecord = struct.unpack("<II", head)
    if magic != MAGIC:
        raise ValueError(f"offset {offset} 处魔数错误: 0x{magic:08X}")
    return (8 + lrecord + 3) & ~3


def read_record(f, offset: int, want_attr: bool = False):
    """从 offset 读一条记录。

    返回 (identity:int, img_bytes:bytes)；want_attr=True 时返回
    (identity, attr:int, img_bytes)。
    """
    f.seek(offset)
    head = f.read(HEADER_SIZE)
    if len(head) < HEADER_SIZE:
        raise EOFError("读取记录头失败")
    magic, lrecord = struct.unpack("<II", head[:8])
    if magic != MAGIC:
        raise ValueError(f"offset {offset} 处魔数错误: 0x{magic:08X}")
    identity = int(struct.unpack("<f", head[_IDENTITY_OFF:_IDENTITY_OFF + 4])[0])
    attr = int(struct.unpack("<f", head[_ATTR_OFF:_ATTR_OFF + 4])[0])
    payload_len = (8 + lrecord) - HEADER_SIZE
    payload = f.read(payload_len)
    if len(payload) < payload_len:
        raise EOFError("读取 payload 失败")
    if want_attr:
        return identity, attr, payload
    return identity, payload


def decode_rgb(jpeg_bytes: bytes) -> np.ndarray:
    """JPEG 字节流 -> RGB uint8 (H, W, 3)。

    用 PIL 而不是 cv2：本工程训练侧不依赖 opencv，少一个依赖。
    """
    img = Image.open(io.BytesIO(jpeg_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def load_idx(idx_path: str, limit: int | None = None) -> np.ndarray:
    """解析 train.idx（InsightFace 文本格式：每行 "序号\\t字节偏移"）。

    返回 int64 数组 offsets，offsets[i] 即第 i 条记录（0 基）的字节偏移。
    """
    offs = []
    with open(idx_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace("\t", " ").split()
            offs.append(int(parts[-1]))
            if limit is not None and len(offs) >= limit:
                break
    return np.asarray(offs, dtype=np.int64)


def scan_index(rec_path: str, progress_every: int = 500_000):
    """顺序扫一遍 train.rec，产出 (offsets, identities, attrs)。

    这是整个流程里唯一一次「读满 139GB」的操作。本机数据盘是机械盘
    (ST2000DM001)，实测顺序读 43.5MB/s，全量扫描约 55 分钟。
    结果请务必缓存成 .npz，后续训练只依赖索引。

    为了少解码，这里只解析 header，跳过 payload。
    """
    offsets, ids, attrs = [], [], []
    fsize = os.path.getsize(rec_path)
    with open(rec_path, "rb") as f:
        pos = 0
        n = 0
        while pos < fsize:
            f.seek(pos + _IDENTITY_OFF)
            h = f.read(8)                      # identity + attr
            if len(h) < 8:
                break
            identity = int(struct.unpack("<f", h[:4])[0])
            attr = int(struct.unpack("<f", h[4:8])[0])
            offsets.append(pos)
            ids.append(identity)
            attrs.append(attr)
            f.seek(pos + 4)
            lrecord = struct.unpack("<I", f.read(4))[0]
            pos += (8 + lrecord + 3) & ~3
            n += 1
            if progress_every and n % progress_every == 0:
                print(f"  [scan] {n} 条  {pos/1e9:.1f}/{fsize/1e9:.1f} GB "
                      f"({100*pos/fsize:.1f}%)  label={identity}")
    return (np.asarray(offsets, dtype=np.int64),
            np.asarray(ids, dtype=np.int32),
            np.asarray(attrs, dtype=np.int8))


def save_index(path: str, offsets, identities, attrs) -> None:
    np.savez(path, offsets=offsets, identities=identities, attrs=attrs)


def load_index(path: str):
    z = np.load(path)
    return z["offsets"], z["identities"], z["attrs"]


class RecReader:
    """带索引的随机访问读取器（线程内独立打开文件句柄，可放心用于 DataLoader）。"""

    def __init__(self, rec_path: str, offsets: np.ndarray, identities: np.ndarray,
                 attrs: np.ndarray | None = None):
        self.rec_path = rec_path
        self.offsets = offsets
        self.identities = identities
        self.attrs = attrs
        self._f = None

    def __len__(self):
        return len(self.offsets)

    @property
    def f(self):
        if self._f is None:
            self._f = open(self.rec_path, "rb")
        return self._f

    def _read_raw(self, i: int):
        return read_record(self.f, int(self.offsets[i]),
                           want_attr=self.attrs is not None)

    def read(self, i: int) -> np.ndarray:
        """返回第 i 条记录解码后的 RGB 图像。"""
        r = self._read_raw(i)
        return decode_rgb(r[-1])

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_f"] = None
        return s
