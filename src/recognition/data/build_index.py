"""扫描 train.rec，抽出每条记录的身份标签，缓存成索引。

为什么必须做这一步
--------------------------------------------------------------------------
`train.idx` 只有「序号 → 字节偏移」，**没有标签**；标签埋在每条记录的
header 里（见 mrec.py 的格式说明）。要按身份采样（每类取 N 张）就必须先
拿到标签序列。

代价：全量 139GB 顺序扫一遍。本机数据盘是机械盘，实测 43.5MB/s，
全量约 55 分钟。**只需要跑一次**，产物 index.npz 只有几百 MB。

想快速跑通流程，可以 `--max_records 2000000` 只扫前 200 万条（约 4 万身份），
几十秒就能出结果。

用法
----
    python -m data.build_index \
        --rec  /media/tiny/423ADD10728E128E/glint360k_release/glint360k/train.rec \
        --idx  /media/tiny/423ADD10728E128E/glint360k_release/glint360k/train.idx \
        --out  index/glint360k.npz
"""
from __future__ import annotations

import os
import sys
import argparse
import struct
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.mrec import MAGIC, HEADER_SIZE, save_index  # noqa: E402

_ID_OFF = 32
_ATTR_OFF = 36
_CHUNK = 32 << 20          # 32MB 顺序读块


def scan(rec_path: str, max_records: int | None = None,
         progress_every: int = 500_000):
    """按物理顺序扫 rec，返回 (offsets, identities, attrs)。

    用大块顺序读 + 内存内解析记录边界：机械盘上比「每 40 字节一次 seek+read」
    快一个数量级。
    """
    fsize = os.path.getsize(rec_path)
    offsets, ids, attrs = [], [], []
    pos = 0
    t0 = time.time()
    next_report = progress_every      # 按「达到 N 条就报一次」推进，比取模可靠
    f = open(rec_path, "rb")
    try:
        while pos < fsize:
            f.seek(pos)
            buf = f.read(_CHUNK)
            if not buf:
                break
            base = pos
            n = len(buf)
            i = 0
            while i + HEADER_SIZE <= n:
                magic, lrecord = struct.unpack_from("<II", buf, i)
                if magic != MAGIC:
                    raise ValueError(
                        f"offset {base+i} 魔数错误 0x{magic:08X}，文件可能损坏")
                identity = int(struct.unpack_from("<f", buf, i + _ID_OFF)[0])
                attr = int(struct.unpack_from("<f", buf, i + _ATTR_OFF)[0])
                offsets.append(base + i)
                ids.append(identity)
                attrs.append(attr)
                i += (8 + lrecord + 3) & ~3
                if max_records is not None and len(offsets) >= max_records:
                    break
            pos = base + i
            if max_records is not None and len(offsets) >= max_records:
                break
            if not i:                    # 保护：单条记录大于块大小
                raise RuntimeError("记录超过读块大小，请调大 _CHUNK")
            if progress_every and len(offsets) >= next_report:
                el = time.time() - t0
                mb = pos / 1e6
                print(f"  [scan] {len(offsets):>9,} 条  {mb/1000:7.1f} GB  "
                      f"{mb/max(el,1e-9):5.1f} MB/s  已用 {el/60:.1f} min  "
                      f"label={identity}", flush=True)
                while next_report <= len(offsets):
                    next_report += progress_every
    finally:
        f.close()
    return (np.asarray(offsets, dtype=np.int64),
            np.asarray(ids, dtype=np.int32),
            np.asarray(attrs, dtype=np.int8))


def main():
    ap = argparse.ArgumentParser(description="构建 Glint360K RecordIO 索引")
    ap.add_argument("--rec", required=True)
    ap.add_argument("--idx", default=None, help="可选，用于交叉校验记录数")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_records", type=int, default=None,
                    help="只扫前 N 条（快速验证用）")
    args = ap.parse_args()

    offs, ids, attrs = scan(args.rec, args.max_records)
    print(f"[done] 扫描到 {len(offs):,} 条记录")

    if args.idx and os.path.exists(args.idx) and args.max_records is None:
        with open(args.idx) as f:
            n_idx = sum(1 for _ in f)
        print(f"[check] train.idx 行数={n_idx:,}  vs  扫描记录数={len(offs):,}"
              f"  {'一致 OK' if n_idx == len(offs) else '不一致（末尾可能有非图像记录）'}")

    uniq = np.unique(ids)
    print(f"[stat] 身份数={len(uniq):,}  label 范围=[{ids.min()}, {ids.max()}]")
    cnt = np.bincount(ids[ids >= 0])
    tmp = np.sort(cnt[cnt > 0])
    print(f"[stat] 每身份样本数: 中位={int(np.median(tmp))} "
          f"均值={tmp.mean():.1f} 最小={tmp.min()} 最大={tmp.max()}")
    for thr in (5, 10, 20, 50):
        kept = tmp[:len(tmp)]                      # 保留全部身份
        print(f"[stat] 每类<= {thr:>3} 张时总量 = "
              f"{int(np.minimum(kept, thr).sum()):,}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_index(args.out, offs, ids, attrs)
    print(f"[done] 索引已保存 -> {args.out}")


if __name__ == "__main__":
    main()
