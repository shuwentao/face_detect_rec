#!/bin/bash
# 等实验 A（PID 1678299，calib+test 完整链）跑完，自动接 600 张评估
while kill -0 1678299 2>/dev/null; do sleep 60; done
sleep 20
cd /workspace/mymodel/gate320
source /opt/vitis_ai/conda/etc/profile.d/conda.sh
conda activate vitis-ai-pytorch
export OMP_NUM_THREADS=16
python eval_dump.py --jit qr_ft_lmA/FaceDetector_int.pt \
    --max_images 600 --batch 8 --dump eval_lmA.jsonl > eval_lmA.log 2>&1
echo "[sentinel] 评估完成 $(date)" >> eval_lmA.log
