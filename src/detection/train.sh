#!/bin/bash 

# 冒烟：确认 pipeline + 数据路径通（用你给的路径）
python train.py --data_dir /workspace/mymodel/face_trainer/dataset/WideFace \
    --input_size 320 --max_records 200 --epochs 1 --out_dir checkpoints

## 正式训练（约 12k 图）
#python train.py --data_dir /workspace/mymodel/face_trainer/dataset/WideFace \
#    --input_size 320 --batch 16 --epochs 50 --out_dir checkpoints

