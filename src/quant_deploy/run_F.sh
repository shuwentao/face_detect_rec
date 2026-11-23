#!/bin/bash
# 实验 F: 关键点分支独立量化刻度（lm 走单独 DeQuantStub）
# 其余配置与当前最优的 E 完全一致：diffs 校准 + 128 张微调图
#   + lm×3 前景加权 + level 2,1,1
cd /workspace/mymodel/gate320
source /opt/vitis_ai/conda/etc/profile.d/conda.sh
conda activate vitis-ai-pytorch
export OMP_NUM_THREADS=16

COMMON="--calib_dir /mnt/data1/face_bench/data/images/300w \
  --output_dir qr_ft_F --config quant_config_diffs.json \
  --fast_finetune --ft_images 128 --ft_lm_weight 3 \
  --ft_fg_weight 3 --ft_level_weight 2,1,1 --no_mkldnn"

echo "[F] start $(date)" > run_F_status.log

python quantize_gate320.py --quant_mode calib --n_calib 200 $COMMON > calib_F.log 2>&1
echo "[F] calib rc=$? $(date)" >> run_F_status.log

python quantize_gate320.py --quant_mode test --n_calib 1 --batch_size 1 \
  --deploy $COMMON > test_F.log 2>&1
echo "[F] test rc=$? $(date)" >> run_F_status.log

python eval_dump.py --jit qr_ft_F/FaceDetector_int.pt \
  --max_images 600 --batch 8 --dump eval_F.jsonl > eval_F.log 2>&1
echo "[F] eval done $(date)" >> run_F_status.log
