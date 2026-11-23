#!/bin/bash
# 把量化后的 xmodel 编译成能在 DPUCZDX8G 上跑的 .xmodel
#
# 用法（在 Vitis AI 3.0 容器里）:
#   bash deploy/compile_xmodel.sh quantize_result/quantize_model.xmodel
#
# 参数说明:
#   -a arch.json  板子的 DPU 指纹。本工程已放好，
#                 与 ~/tmp/ACU7EV_DPUCZDX8G 与 ~/tmp/DPUCZDX8G_VAI_v3.0
#                 两份 Vivado 工程里的 IP 配置一致: 0x101000056010407
#   -n 网络名     板上 VART 加载时要用的名字，和 C++ 里保持一致
set -e

XMODEL=${1:-quantize_result/GhostFaceNet_int.xmodel}
OUTDIR=${2:-compiled}
NETNAME=${3:-face_rec}
ARCH=${ARCH:-$(dirname "$0")/arch.json}

if [ ! -f "$XMODEL" ]; then
    echo "找不到 $XMODEL，请先跑 deploy/quantize_pt.py"
    exit 1
fi

mkdir -p "$OUTDIR"

echo "== 编译 =="
echo "   输入 : $XMODEL"
echo "   架构 : $ARCH  ($(cat "$ARCH"))"
echo "   输出 : $OUTDIR/$NETNAME.xmodel"

vai_c_xir \
    -x "$XMODEL" \
    -a "$ARCH" \
    -o "$OUTDIR" \
    -n "$NETNAME" 2>&1 | tee "$OUTDIR/${NETNAME}_compile_log.txt"

echo "== 编译结果 =="
ls -lh "$OUTDIR"

# 编译日志里会打印 "Total DPU subgraph number"，
# 若该数 > 1 通常说明部分算子没被 DPU 吃掉（会掉 CPU 执行）。
if grep -qi "Total DPU subgraph number" "$OUTDIR/${NETNAME}_compile_log.txt" 2>/dev/null; then
    echo "== DPU 子图统计 =="
    grep -i "Total DPU subgraph number" "$OUTDIR/${NETNAME}_compile_log.txt"
fi
echo "编译产物: $OUTDIR/$NETNAME.xmodel"
