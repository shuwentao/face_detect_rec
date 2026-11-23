#!/bin/bash
# 等 30k 训练结束 → 自动挑 LFW 最佳 epoch → finetune 到 136k
cd /home/tiny/tmp/face_rec

echo "[chain] $(date '+%F %T') 等待 30k 训练结束 (pid 19392)..."
while kill -0 19392 2>/dev/null; do sleep 60; done
sleep 20
echo "[chain] $(date '+%F %T') 30k 已结束"

BEST=$(python3 -c "
import json
h = json.load(open('checkpoints_gfn30k/history.json'))
b = max(h, key=lambda r: r.get('lfw', 0))
print(b['epoch'])
")
BESTLFW=$(python3 -c "
import json
h = json.load(open('checkpoints_gfn30k/history.json'))
print(max(r.get('lfw', 0) for r in h))
")
echo "[chain] 30k 最佳 epoch=$BEST  LFW=$BESTLFW"

CKPT="checkpoints_gfn30k/backbone_ep${BEST}.pth"
if [ ! -f "$CKPT" ]; then
    echo "[chain] 找不到 $CKPT，改用 last.pth"
    CKPT="checkpoints_gfn30k/last.pth"
fi

echo "[chain] $(date '+%F %T') 启动 136k finetune，权重=$CKPT"
python3 train.py \
    --shards /mnt/data1/shards_136k \
    --arch ghostfacenet \
    --loss cosface --sample_rate 1.0 \
    --batch_size 128 --epochs 10 --lr 0.003 \
    --num_workers 6 --amp --log_every 1500 --save_every_epochs 1 \
    --finetune "$CKPT" \
    --out_dir checkpoints_gfn136_ft \
    --lfw_bin /mnt/data1/val_bins/lfw.bin \
    > /tmp/train_gfn136_ft.log 2>&1

echo "[chain] $(date '+%F %T') finetune 结束，退出码=$?"
grep -E "^\[eval\] LFW" /tmp/train_gfn136_ft.log | tail -20

# 第二阶段：GhostFaceNet 从零直接训 136k，lr 0.003 / 12 epoch（对齐 cf2 的最佳配置）
echo "[chain] $(date '+%F %T') 启动 136k 直接训练 lr=0.003"
python3 train.py \
    --shards /mnt/data1/shards_136k \
    --arch ghostfacenet \
    --loss cosface --sample_rate 1.0 \
    --batch_size 128 --epochs 12 --lr 0.003 \
    --num_workers 6 --amp --log_every 1500 --save_every_epochs 1 \
    --out_dir checkpoints_gfn136_lr3 \
    --lfw_bin /mnt/data1/val_bins/lfw.bin \
    > /tmp/train_gfn136_lr3.log 2>&1

echo "[chain] $(date '+%F %T') 直接训练结束，退出码=$?"
grep -E "^\[eval\] LFW" /tmp/train_gfn136_lr3.log | tail -20
