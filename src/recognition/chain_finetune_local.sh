#!/bin/bash
# 识别复现链 —— 适配当前机器 (data3)，可直接在当前目录运行。
#
# 用法:
#   bash chain_finetune_local.sh prep-shards    # 从 train.rec 重建 30k + 136k 分片(只需一次)
#   bash chain_finetune_local.sh 30k            # 阶段1: GhostFaceNet 在 30k 子集从零训
#   bash chain_finetune_local.sh finetune136k   # 阶段2: 挑 LFW 最佳的 epoch 微调到 136k
#   bash chain_finetune_local.sh direct136k     # 阶段3: GhostFaceNet 直接在 136k 训 lr=0.003
#   bash chain_finetune_local.sh all            # prep-shards -> 30k -> finetune136k -> direct136k
#
# 与旧 chain_finetune.sh 的区别:
#   - cd 到本脚本所在目录(不再是 /home/tiny/tmp/face_rec)
#   - 数据路径: /mnt/data3/glint360k_release/glint360k/ (不再是 /mnt/data1)
#   - shards 由 prep-shards 从 train.rec 重建(不再是 /mnt/data1/shards_*)
#   - 去掉旧的 "等 pid 19392" 逻辑(改成顺序执行)

set -e
cd "$(dirname "$0")"

REC=/mnt/data3/glint360k_release/glint360k/train.rec
LFW=/mnt/data3/glint360k_release/glint360k/lfw.bin
IDX=index/glint360k_2M.npz
SHARDS30K=shards_30k
SHARDS136K=shards_136k

[ -f "$REC" ] || { echo "[err] 找不到 $REC"; exit 1; }
[ -f "$LFW" ] || { echo "[err] 找不到 $LFW"; exit 1; }
IDX30K=index/glint360k_2M.npz      # 30727 身份 -> "30k" 子集
IDX136K=index/local_60G.npz        # 136418 身份 -> "136k" 子集(用错会掉 6~7 个点!)
[ -f "$IDX136K" ] || { echo "[err] 找不到索引 $IDX136K"; exit 1; }

prep_shards() {
  echo "[prep] 从 train.rec 重建 30k / 136k 分片..."
  [ -d "$SHARDS30K" ] || python3 -m data.build_shard \
      --index "$IDX30K" --rec "$REC" --out_dir "$SHARDS30K" \
      --per_class 10 --shard_size 4000 --seed 42
  [ -d "$SHARDS136K" ] || python3 -m data.build_shard \
      --index "$IDX136K" --rec "$REC" --out_dir "$SHARDS136K" \
      --per_class 10 --shard_size 4000 --seed 42
  echo "[prep] 分片就绪: $SHARDS30K(3万身份) / $SHARDS136K(13.6万身份)"
}

train_30k() {
  echo "[30k] GhostFaceNet 在 30k 子集从零训..."
  python3 train.py \
      --shards "$SHARDS30K" --arch ghostfacenet \
      --loss cosface --sample_rate 1.0 \
      --batch_size 128 --epochs 20 --lr 0.01 \
      --num_workers 6 --amp --log_every 1500 --save_every_epochs 1 \
      --out_dir checkpoints_gfn30k --lfw_bin "$LFW"
}

finetune_136k() {
  echo "[finetune136k] 挑 30k 最佳 epoch 微调到 136k..."
  [ -d "$SHARDS136K" ] || { echo "[err] 先跑 prep-shards"; exit 1; }
  BEST=$(python3 -c "
import json
h=json.load(open('checkpoints_gfn30k/history.json'))
print(max(h,key=lambda r:r.get('lfw',0))['epoch'])")
  CKPT="checkpoints_gfn30k/backbone_ep${BEST}.pth"
  [ -f "$CKPT" ] || CKPT="checkpoints_gfn30k/last.pth"
  echo "[finetune136k] 起点权重=$CKPT"
  python3 train.py \
      --shards "$SHARDS136K" --arch ghostfacenet \
      --loss cosface --sample_rate 1.0 \
      --batch_size 128 --epochs 10 --lr 0.003 \
      --num_workers 6 --amp --log_every 1500 --save_every_epochs 1 \
      --finetune "$CKPT" \
      --out_dir checkpoints_gfn136_ft --lfw_bin "$LFW"
}

direct_136k() {
  echo "[direct136k] GhostFaceNet 直接在 136k 训 lr=0.003..."
  [ -d "$SHARDS136K" ] || { echo "[err] 先跑 prep-shards"; exit 1; }
  python3 train.py \
      --shards "$SHARDS136K" --arch ghostfacenet \
      --loss cosface --sample_rate 1.0 \
      --batch_size 128 --epochs 12 --lr 0.003 \
      --num_workers 6 --amp --log_every 1500 --save_every_epochs 1 \
      --out_dir checkpoints_gfn136_lr3 --lfw_bin "$LFW"
}

case "${1:-all}" in
  prep-shards)  prep_shards ;;
  30k)          train_30k ;;
  finetune136k) finetune_136k ;;
  direct136k)   direct_136k ;;
  all)          prep_shards; train_30k; finetune_136k; direct_136k ;;
  *) echo "未知阶段: $1"; sed -n '2,12p' "$0"; exit 1 ;;
esac
