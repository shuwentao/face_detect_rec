#!/bin/bash
# 宿主机上一键跑完「量化 + 编译」，内部自动进 Vitis AI 3.0 容器。
#
#   bash deploy/run_vitis.sh                        # 随机权重，只验证图结构能否上 DPU
#   bash deploy/run_vitis.sh checkpoints/backbone_ep20.pth
#
# 产物：
#   quantize_result/GhostFaceNet_int.xmodel   量化后的模型
#   compiled/face_rec.xmodel                  编译好的、板端可直接加载的模型
set -e

CKPT=${1:-}
ARCH=${2:-ghostfacenet}          # ghostfacenet / mobilefacenet
if [ -n "$CKPT" ]; then
    EXTRA="--ckpt $CKPT"
    echo "== 量化: $CKPT =="
else
    EXTRA="--from_scratch"
    echo "== 量化: 随机权重（仅验证图结构） =="
fi
echo "== 骨干: $ARCH =="

docker run --rm \
    -v "$(pwd)":/workspace -w /workspace \
    xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106 \
    /bin/bash -lc "source /opt/vitis_ai/conda/etc/profile.d/conda.sh && \
                   conda activate vitis-ai-pytorch && \
                   rm -rf quantize_result compiled && \
                   python deploy/quantize_pt.py $EXTRA --arch $ARCH \
                       --calib deploy/calib.npy --out_dir quantize_result && \
                   XMODEL=\$(ls quantize_result/*_int.xmodel | head -1) && \
                   bash deploy/compile_xmodel.sh \"\$XMODEL\" compiled face_rec"

echo
echo "== 产物 =="
ls -lh compiled 2>/dev/null || true
