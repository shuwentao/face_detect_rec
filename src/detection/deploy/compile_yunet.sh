#!/usr/bin/env bash
# YuNet 编译到 DPU（Vitis-AI 3.0 环境内运行）。
# 注意：DPU 只接受 backbone+head 的 4 个原始输出，解码/NMS 在 CPU 端做。
set -e

ARCH_JSON=${1:-./arch.json}          # 板子对应的 arch.json，需你提供
QUANT_H5=${2:-yunet_quantized.h5}
OUT_XMODEL=${3:-yunet.xmodel}

if [ ! -f "$ARCH_JSON" ]; then
  echo "缺少 arch.json，请从 Vitis-AI 拷贝对应板子："
  echo "  cp /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/*.json ./arch.json"
  echo "（ACU7EV/XCZU7EV 对照 ZCU104；fingerprint/B 档须与你的 DPU IP 一致）"
  exit 1
fi

echo "编译 $QUANT_H5 -> $OUT_XMODEL (arch=$ARCH_JSON)"
vai_c_tensorflow2 -a "$ARCH_JSON" \
  --quantize "$QUANT_H5" \
  -o "$(dirname "$OUT_XMODEL")" \
  --net_name yunet

echo "[ok] 生成 $OUT_XMODEL"
echo "下一步：在 PS(ARM) 上用 models/yunet.decode_predictions() 对 4 个输出做"
echo "anchor->框 解码 + NMS（不要放进 DPU）。"
