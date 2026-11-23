#!/bin/bash
# 用 QAT 自身刻度导出 int.pt，再跑 600 张评估
cd /workspace/mymodel/gate320
source /opt/vitis_ai/conda/etc/profile.d/conda.sh
conda activate vitis-ai-pytorch
export OMP_NUM_THREADS=16

python qat_export_proper.py > qat_export_proper.log 2>&1
echo "export rc=$? $(date)" > run_qat_proper_status.log

python eval_dump.py --jit qat320e/qtest_proper/FaceDetector_int.pt \
  --max_images 600 --batch 8 --dump eval_qat_proper.jsonl \
  > eval_qat_proper.log 2>&1
echo "eval done $(date)" >> run_qat_proper_status.log
