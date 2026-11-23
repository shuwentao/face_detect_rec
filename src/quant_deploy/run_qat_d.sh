#!/bin/bash
# 实验 D-qat：用修复后的 qat_patch（FPN 的 add/interpolate 拆成独立实例，
# 消除 Vitis 的 Reused module 警告）重跑 QAT。
cd /workspace/mymodel/gate320
source /opt/vitis_ai/conda/etc/profile.d/conda.sh
conda activate vitis-ai-pytorch
export OMP_NUM_THREADS=8

# 等新的校准文件生成完毕（图结构变了，旧的 qr_qat320_patched 不能再用）
while [ ! -f qr_qat320_v2/quant_info.json ]; do sleep 20; done
sleep 30
echo "[qat-d] calib ready $(date)" > run_qat_d_status.log

python qat_gate.py --input_size 320 --epochs 10 --lr 2e-5 --batch 8 \
  --workers 4 --calib_dir qr_qat320_v2 --out_dir qat320d --no_mkldnn \
  > qat320d.log 2>&1
echo "[qat-d] train rc=$? $(date)" >> run_qat_d_status.log
