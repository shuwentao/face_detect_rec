#!/bin/bash
# 实验 qat320e：完整修复版 QAT（FPN 的 add / interpolate / relu 全部拆成独立实例）
# 与 qat320d 的区别：qat320d 只拆了 add/interpolate，漏了 relu
# （VAIQ_WARN: Reused module (fpn.relu) 仍在），故重跑。
cd /workspace/mymodel/gate320
source /opt/vitis_ai/conda/etc/profile.d/conda.sh
conda activate vitis-ai-pytorch
export OMP_NUM_THREADS=10

# 图结构又变了（relu 拆成 3 个），必须重新生成校准文件
python quantize_gate320.py --quant_mode calib --n_calib 200 --qat_patch \
  --calib_dir /mnt/data1/face_bench/data/images/300w \
  --output_dir qr_qat320_v3 > calib_qat_v3.log 2>&1
echo "[qat-e] calib rc=$? $(date)" > run_qat_e_status.log

python qat_gate.py --input_size 320 --epochs 10 --lr 2e-5 --batch 8 \
  --workers 3 --calib_dir qr_qat320_v3 --out_dir qat320e --no_mkldnn \
  > qat320e.log 2>&1
echo "[qat-e] train rc=$? $(date)" >> run_qat_e_status.log
