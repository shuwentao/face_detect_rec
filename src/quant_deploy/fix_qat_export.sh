#!/bin/bash
# 修正 qat320e 导出：必须先跑 calib 生成 bias_corr.pth，再跑 test --deploy
cd /workspace/mymodel/gate320
source /opt/vitis_ai/conda/etc/profile.d/conda.sh
conda activate vitis-ai-pytorch
export OMP_NUM_THREADS=16

W=qat320e/deploy/deployable.pth
OUT=qat320e/qtest
CAL=/mnt/data1/face_bench/data/images/300w

python quantize_gate320.py --quant_mode calib --n_calib 200 --qat_patch \
  --weights $W --output_dir $OUT --calib_dir $CAL > qat320e_calib2.log 2>&1
echo "calib rc=$? $(date)" > qat320e_fix_status.log

python quantize_gate320.py --quant_mode test --n_calib 200 --batch_size 1 \
  --deploy --qat_patch --weights $W --output_dir $OUT \
  --calib_dir $CAL > qat320e_export2.log 2>&1
echo "export rc=$? $(date)" >> qat320e_fix_status.log

python eval_dump.py --jit $OUT/FaceDetector_int.pt --max_images 600 \
  --batch 8 --dump eval_qat320e.jsonl > eval_qat320e.log 2>&1
echo "eval done $(date)" >> qat320e_fix_status.log
