"""在标准验证集（bin 格式）上评测人脸识别精度。

支持 InsightFace 发布的 8 个经典评测集，Glint360K 发行包里已经带齐：
    lfw.bin  cfp_fp.bin  cfp_ff.bin  agedb_30.bin
    calfw.bin  cplfw.bin  vgg2_fp.bin (可选)

bin 文件格式（实测确认，两种都存在）
--------------------------------------------------------------------------
文件是 pickle 流，load 出来是元组 `(bins, issame_list)`：
  * bins：长度 2N，奇数位是「第 (i+1)/2 对的第一张」，偶数位是第二张
  * issame_list：长度 N 的布尔列表，True 表示这两张是同一身份
  * 元素类型不统一：
        lfw.bin                → bytes（直接是 JPEG 字节流）
        cfp_*/agedb_*/vgg2_*   → np.ndarray，shape (n,1) uint8，同样是 JPEG 字节流
    所以统一 tobytes() 后交给 cv2.imdecode 即可。

评测协议
--------------------------------------------------------------------------
LFW / CFP / AgeDB 都用 **10-fold 交叉验证**报准确率：
把 N 对平均切成 10 份，每份轮流当测试集，用另外 9 份搜最佳阈值，
最后取 10 次准确率的平均 —— 这样阈值不是「用测试集挑出来的」，结果可比。

用法
--------------------------------------------------------------------------
    python eval_bin.py --ckpt checkpoints/backbone_ep20.pth \
        --arch ghostfacenet --width 1.0 \
        --bins /media/tiny/423ADD10728E128E/glint360k_release/glint360k \
        --flip
"""
from __future__ import annotations

import os
import sys
import glob
import pickle
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

cv2.setNumThreads(0)

# 评测集名称 → 直观的参考线（GhostFaceNet 级别轻量模型在 Glint360K 训练后的量级）
REFERENCE = {
    "lfw": 99.7, "cfp_fp": 96.5, "cfp_ff": 99.5, "agedb_30": 97.5,
    "calfw": 95.8, "cplfw": 92.0, "vgg2_fp": 94.5,
}


def load_bin(path: str):
    """读 bin → (list[bytes], np.ndarray[bool])。"""
    with open(path, "rb") as f:
        data = pickle.load(f, encoding="bytes")
    if isinstance(data, dict):
        data = (data.get(b"bins", data.get("bins")),
                data.get(b"issame_list", data.get("issame_list")))
    bins, issame = data
    out = []
    for e in bins:
        out.append(e if isinstance(e, bytes) else np.asarray(e).tobytes())
    return out, np.asarray(issame, dtype=bool).reshape(-1)


def _decode_norm(jpeg: bytes, size: int = 112) -> np.ndarray:
    """JPEG → RGB CHW float32 [-1,1]（与训练/部署完全一致的预处理）。"""
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError("无法解码 bin 中的图像")
    if img.shape[0] != size or img.shape[1] != size:
        img = cv2.resize(img, (size, size))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    x = (rgb - 127.5) / 127.5
    return np.ascontiguousarray(x.transpose(2, 0, 1))


@torch.no_grad()
def extract_features(backbone, jpegs, batch_size: int = 256,
                     device=None, flip: bool = False, size: int = 112):
    """对一串 JPEG 提特征，返回 L2 归一化后的 (N, D) numpy。

    flip=True 时同时提原图和镜像图的特征再相加（TTA），约 +0.1~0.2%。
    """
    if device is None:
        device = next(backbone.parameters()).device
    backbone.eval()
    feats = []
    for i in range(0, len(jpegs), batch_size):
        chunk = jpegs[i:i + batch_size]
        arr = np.stack([_decode_norm(b, size) for b in chunk])
        x = torch.from_numpy(arr).to(device)
        f = backbone(x)
        if flip:
            f = f + backbone(torch.flip(x, dims=[3]))
        feats.append(F.normalize(f.float(), dim=1).cpu().numpy())
    return np.concatenate(feats, axis=0)


def _best_threshold(sim: np.ndarray, same: np.ndarray, steps: int = 4000):
    """在候选阈值里挑准确率最高的那个（向量化，很快）。

    判定规则：相似度 > 阈值 → 认为是同一身份。于是
        正对被判对 = #{s ∈ pos : s > thr} = len(pos) − #{s ≤ thr}
        负对被判对 = #{s ∈ neg : s ≤ thr}
    """
    pos = np.sort(sim[same])
    neg = np.sort(sim[~same])
    thrs = np.linspace(-1.0, 1.0, steps)
    pos_ok = len(pos) - np.searchsorted(pos, thrs, side="right")
    neg_ok = np.searchsorted(neg, thrs, side="right")
    acc = (pos_ok + neg_ok) / (len(pos) + len(neg))
    k = int(np.argmax(acc))
    return float(thrs[k]), float(acc[k])


def _acc_at(sim: np.ndarray, same: np.ndarray, thr: float) -> float:
    pred = sim > thr
    return float((pred == same).mean())


def eval_10fold(sim: np.ndarray, same: np.ndarray, n_folds: int = 10):
    """10-fold：每折用其余 9 折选阈值，在该折上报准确率。"""
    n = len(same)
    fold = n // n_folds
    accs, thrs = [], []
    for f in range(n_folds):
        lo, hi = f * fold, (f + 1) * fold
        te = np.arange(lo, hi)
        tr = np.concatenate([np.arange(0, lo), np.arange(hi, n)])
        thr, _ = _best_threshold(sim[tr], same[tr])
        accs.append(_acc_at(sim[te], same[te], thr))
        thrs.append(thr)
    return float(np.mean(accs)), float(np.mean(thrs))


def evaluate_bin(backbone, bin_path: str, batch_size: int = 256,
                 device=None, flip: bool = False, verbose: bool = True):
    """在单个 bin 上评测，返回 (accuracy, threshold)。"""
    jpegs, same = load_bin(bin_path)
    assert len(jpegs) == 2 * len(same), "bins 长度应为 issame 的两倍"
    f = extract_features(backbone, jpegs, batch_size, device, flip=flip)
    sim = (f[0::2] * f[1::2]).sum(axis=1)          # 余弦相似度
    acc, thr = eval_10fold(sim, same)
    if verbose:
        name = os.path.basename(bin_path).replace(".bin", "")
        ref = REFERENCE.get(name)
        tail = f"  (参考量级 {ref}%)" if ref else ""
        print(f"[eval] {name:<10s} acc={acc*100:.3f}%  thr={thr:.3f}  "
              f"n_pair={len(same)}{tail}")
    return acc, thr


def evaluate_all(backbone, bin_dir: str, names=None, batch_size: int = 256,
                 device=None, flip: bool = False):
    names = names or ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
    res = {}
    for n in names:
        p = os.path.join(bin_dir, n + ".bin")
        if not os.path.exists(p):
            continue
        acc, thr = evaluate_bin(backbone, p, batch_size, device, flip)
        res[n] = acc
    if res:
        print("[eval] 平均 = %.3f%%" % (100 * float(np.mean(list(res.values())))))
    return res


def main():
    ap = argparse.ArgumentParser("bin 验证集评测")
    ap.add_argument("--ckpt", required=True, help="backbone_epN.pth")
    ap.add_argument("--arch", default=None)
    ap.add_argument("--width", type=float, default=None)
    ap.add_argument("--embedding_size", type=int, default=None)
    ap.add_argument("--bins", required=True, help="bin 文件所在目录")
    ap.add_argument("--names", default=None, help="逗号分隔，默认全部常见集")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--flip", action="store_true", help="水平翻转 TTA")
    args = ap.parse_args()

    from models.ghostfacenet import build_backbone
    sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "backbone" in sd:
        arch = args.arch or sd.get("arch", "ghostfacenet")
        width = args.width or sd.get("width", 1.0)
        emb = args.embedding_size or sd.get("embedding_size", 512)
        sd = sd["backbone"]
    else:
        arch, width, emb = (args.arch or "ghostfacenet",
                            args.width or 1.0, args.embedding_size or 512)
    backbone = build_backbone(arch, emb, width)
    backbone.load_state_dict(sd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device).eval()
    print(f"[env] {device}  {arch}(w={width}, {emb}-d)")

    names = args.names.split(",") if args.names else None
    evaluate_all(backbone, args.bins, names, args.batch_size, device, args.flip)


if __name__ == "__main__":
    main()
