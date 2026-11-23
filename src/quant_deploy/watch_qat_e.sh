#!/bin/bash
# 等 qat320e 训练结束（deployable.pth 出现）→ 自动导出 int.pt → 600 张评估
cd /workspace/mymodel/gate320
source /opt/vitis_ai/conda/etc/profile.d/conda.sh
conda activate vitis-ai-pytorch
export OMP_NUM_THREADS=8

while [ ! -f qat320e/deploy/deployable.pth ]; do sleep 60; done
sleep 30
echo "[watch] train done $(date)" > watch_qat_e_status.log

python quantize_gate320.py --quant_mode test --n_calib 200 --batch_size 1 \
  --deploy --qat_patch --weights qat320e/deploy/deployable.pth \
  --output_dir qat320e/qtest \
  --calib_dir /mnt/data1/face_bench/data/images/300w > qat320e_export.log 2>&1
echo "[watch] export rc=$? $(date)" >> watch_qat_e_status.log

python eval_dump.py --jit qat320e/qtest/FaceDetector_int.pt \
  --max_images 600 --batch 8 --dump eval_qat320e.jsonl > eval_qat320e.log 2>&1
echo "[watch] eval done $(date)" >> watch_qat_e_status.log
