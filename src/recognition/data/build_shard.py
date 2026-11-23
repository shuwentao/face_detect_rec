"""把 train.rec 的选定子集重打包成「分片(shard)」文件。

为什么需要分片
--------------------------------------------------------------------------
数据盘是机械盘（顺序读 43.5MB/s，随机读只有约 100 次/s）。训练时若直接
按索引随机 seek 读 .rec，每个 batch 要十几次随机 IO，GPU 会长期空转。

分片做法：把样本重排成若干个大文件，**每个分片几十 MB，训练时整片读入内存**。
于是磁盘访问退化为「顺序大块读」，机械盘也能喂饱 GPU。

分片内部格式（FRSH）
    4B  magic  'FRSH'
    4B  count
    count × { 4B label(int32), 4B jpeg_len(uint32), jpeg_len 字节 JPEG }

轮转分配（关键设计）
--------------------------------------------------------------------------
样本是「按身份连续」排布的，若顺序填充分片，一个分片会几乎只含少数身份，
batch 内缺少负类，会明显拖慢 margin loss 的收敛。
这里改成**轮转写入**：第 i 个样本写进 shard[i % K]，使每个分片都能拿到
各式各样的身份。代价只是同时打开 K 个文件句柄。

用法
--------------------------------------------------------------------------
    # 全部 36 万身份，每身份最多 10 张（约 300 万张，24GB）
    python -m data.build_shard --index index/glint360k.npz \
        --rec  /media/.../glint360k/train.rec --out_dir shards --per_class 10

    # 快速验证：只取前 4 万身份、每类 20 张
    python -m data.build_shard --index index/glint360k.npz \
        --rec  /media/.../glint360k/train.rec --out_dir shards_quick \
        --per_class 20 --max_classes 40000
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import struct
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.mrec import load_index, HEADER_SIZE  # noqa: E402

MAGIC = b"FRSH"
CHUNK = 32 << 20          # 顺序读块
_INVALID_LABEL = 1 << 20  # 超过这个值一定是非图像记录（本数据集实测末条为 17091560）


def pick_samples(idents: np.ndarray, per_class: int, max_classes: int | None,
                 min_faces: int, seed: int) -> np.ndarray:
    """返回选中样本的下标数组（按下标升序，保持物理顺序）。

    数据已按身份排布，因此可以先把每个身份的样本下标收集起来，
    再统一决定保留哪些下标，最后排序即可保证「升序」，供顺序读使用。
    """
    valid = (idents >= 0) & (idents < _INVALID_LABEL)
    if max_classes is not None:
        valid &= (idents < max_classes)
    idx = np.flatnonzero(valid)
    labels = idents[idx]

    # 用 bincount + 分块的方式按类采样，避免对 1700 万条做 Python 循环
    order = np.argsort(labels, kind="stable")
    idx_sorted = idx[order]
    lab_sorted = labels[order]
    uniq, starts, counts = np.unique(lab_sorted, return_index=True,
                                     return_counts=True)

    rng = np.random.default_rng(seed)
    keep = []
    dropped_classes = 0
    for s, c in zip(starts, counts):
        if c < min_faces:
            dropped_classes += 1
            continue
        n = min(int(c), per_class)
        block = idx_sorted[s:s + c]
        if n == c:
            keep.append(block)
        else:
            sel = rng.choice(c, size=n, replace=False)
            keep.append(block[sel])
    if dropped_classes:
        print(f"[pick] 丢弃样本数 < {min_faces} 的身份 {dropped_classes} 个")
    keep = np.sort(np.concatenate(keep))
    return keep


def remap_labels(sel_labels: np.ndarray) -> tuple[np.ndarray, int]:
    """把原始身份 id 压缩成 0..K-1 的连续类别下标。"""
    uniq = np.unique(sel_labels)
    lut = np.full(sel_labels.max() + 1, -1, dtype=np.int64)
    lut[uniq] = np.arange(len(uniq))
    return lut[sel_labels].astype(np.int32), len(uniq)


def main():
    ap = argparse.ArgumentParser(description="把 rec 子集重打包成分片")
    ap.add_argument("--index", required=True, help="build_index.py 产出的 npz")
    ap.add_argument("--rec", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--per_class", type=int, default=10,
                    help="每个身份最多保留多少张（默认 10）")
    ap.add_argument("--max_classes", type=int, default=None,
                    help="只用 label < N 的身份（快速验证）")
    ap.add_argument("--min_faces", type=int, default=2,
                    help="样本数少于该值的身份直接丢弃（清长尾噪声）")
    ap.add_argument("--shard_size", type=int, default=4000)
    ap.add_argument("--max_total", type=int, default=None, help="总量上限")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    offsets, idents, _ = load_index(args.index)
    print(f"[load] 索引 {len(offsets):,} 条")

    sel = pick_samples(idents, args.per_class, args.max_classes,
                       args.min_faces, args.seed)
    if args.max_total and len(sel) > args.max_total:
        rng = np.random.default_rng(args.seed)
        sel = np.sort(rng.choice(sel, size=args.max_total, replace=False))
    sel_labels, num_classes = remap_labels(idents[sel])
    # pick_samples 返回的是「记录序号」，这里映射成真正的字节偏移。
    # offsets 随记录序号单调递增，所以 sel_off 同样保持升序，可配合顺序读使用。
    sel_off = offsets[sel]
    print(f"[pick] 选中 {len(sel):,} 张 / {num_classes:,} 个身份  "
          f"(每类 <= {args.per_class})")

    n_shard = max(1, int(np.ceil(len(sel) / args.shard_size)))
    os.makedirs(args.out_dir, exist_ok=True)
    writers = []
    counts = [0] * n_shard
    for k in range(n_shard):
        p = os.path.join(args.out_dir, f"shard_{k:05d}.bin")
        f = open(p, "wb")
        f.write(MAGIC)
        f.write(struct.pack("<I", 0))          # count 占位，最后回填
        writers.append(f)

    rec_size = os.path.getsize(args.rec)
    n_written = 0
    si = 0
    f = open(args.rec, "rb")
    try:
        pos = 0
        while pos < rec_size and si < len(sel_off):
            f.seek(pos)
            buf = f.read(CHUNK)
            if not buf:
                break
            base = pos
            n = len(buf)
            while si < len(sel_off):
                off = int(sel_off[si])
                rel = off - base
                if rel < 0:                     # 理论上不会发生（升序）
                    si += 1
                    continue
                if rel + HEADER_SIZE > n:
                    break                       # 本块盖不到，下一块再处理
                lrecord = struct.unpack_from("<I", buf, rel + 4)[0]
                rec_len = (8 + lrecord + 3) & ~3
                if rel + rec_len > n:
                    break                       # 整条记录跨块，下一块再处理
                plen = (8 + lrecord) - HEADER_SIZE
                payload = buf[rel + HEADER_SIZE: rel + HEADER_SIZE + plen]
                k = si % n_shard
                writers[k].write(struct.pack("<iI", int(sel_labels[si]), plen))
                writers[k].write(payload)
                counts[k] += 1
                n_written += 1
                si += 1
            # 还有没处理完、且落在本块内的样本 → 从它开始重读；否则整块推进
            if si < len(sel_off) and int(sel_off[si]) - base < n:
                pos = int(sel_off[si])
            else:
                pos = base + n
            if n_written and n_written % 100_000 < 300:
                el = time.time() - t0
                print(f"  [shard] {n_written:,}/{len(sel_off):,}  "
                      f"已用 {el/60:.1f} min", flush=True)
    finally:
        f.close()
        for k, w in enumerate(writers):
            w.seek(4)
            w.write(struct.pack("<I", counts[k]))
            w.close()

    if n_written != len(sel_off):
        raise RuntimeError(f"写入条数 {n_written} != 选中条数 {len(sel_off)}，"
                           f"分片可能不完整，请检查 --rec 是否与索引同一份数据")

    meta = {
        "num_classes": int(num_classes),
        "num_samples": int(n_written),
        "shard_size": args.shard_size,
        "num_shards": n_shard,
        "img_size": 112,
        "per_class": args.per_class,
        "max_classes": args.max_classes,
        "source_rec": os.path.abspath(args.rec),
        "shards": [f"shard_{k:05d}.bin" for k in range(n_shard)],
        "counts": counts,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as fp:
        json.dump(meta, fp, indent=2)
    print(f"[done] 写出 {n_written:,} 张 / {n_shard} 个分片 / "
          f"{num_classes:,} 类 -> {args.out_dir}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
