#!/usr/bin/env bash
# ============================================================
# 从零复现 qr_ft_E 检测器 (MobileNetV3-small + FPN-lite + 3 head, 输入320)
# 三阶段: ① float 训练  →  ② QAT 量化(calib/test/deploy)  →  ③ 导出部署件
# 数据已就绪: WIDER FACE(bbox) + 600 张 68 点关键点(landmark_300w.json)
#
# 用法:  编辑下面路径后, 逐步执行(本脚本默认全注释, 避免误跑)
#   bash reproduce.sh            # 跑阶段①
#   bash -x reproduce.sh         # 调试
#
# ⚠ 当前按"先别跑"原则: 阶段① 也用 set -e 但请你确认后再解除注释执行。
# ============================================================
set -e
cd /mnt/data3/cv_face_baseline/src/quant_deploy

# ---------------- 路径 ----------------
DATA_ROOT=$PWD/common/data/facebench_wider
EXTRA_ANN=$DATA_ROOT/annotations/landmark_300w.json
EXTRA_IMG=/mnt/data3/300W/300W
TRAIN_OUT=$PWD/repro_mobilenetv3
CALIB_DIR=/mnt/data3/WideFace/WIDER_train/images   # 量化校准图目录(无需标签)

# ---------------- 阶段① 从零训练 (float) ----------------
# train_cli.py 没有 __main__, 用 -c 调它的 train(algo, argv) 函数
# algo='mobilenetv3' 即 torchvision MobileNetV3-small(与 qr_ft_E 参数量 1.19M 吻合)
# 骨干默认 ImageNet 预训练; 数据 = WIDER FACE(bbox) + 600 关键点(走 --extra_ann)
python -c "from common.scripts.train_cli import train; train('mobilenetv3', [
  '--data_root', '$DATA_ROOT',
  '--input_size', '320',
  '--epochs', '20', '--lr', '5e-4', '--wd', '5e-4', '--batch', '8', '--width', '1.0',
  '--extra_ann', '$EXTRA_ANN', '--extra_img', '$EXTRA_IMG',
  '--out_dir', '$TRAIN_OUT'])"
# → 产出 $TRAIN_OUT/ckpt/best.pt (float 权重)

# ---------------- 阶段② QAT 量化 → int (quantize_gate320.py, 当年产物同款) ----------------
# 2a 校准(无需标签, 从 CALIB_DIR 随机抽 n_calib 张)
python quantize_gate320.py --weights $TRAIN_OUT/ckpt/best.pt \
  --input_size 320 --calib_dir $CALIB_DIR --n_calib 200 --batch_size 8 \
  --quant_mode calib --output_dir $TRAIN_OUT/quant
# 2b 精度测试
python quantize_gate320.py --weights $TRAIN_OUT/ckpt/best.pt \
  --input_size 320 --calib_dir $CALIB_DIR --n_calib 50 --batch_size 8 \
  --quant_mode test --output_dir $TRAIN_OUT/quant
# 2c 导出部署件 (xmodel / 量化 onnx / pt) —— 与 github 的 FaceDetector_0_int.xmodel 同格式
python quantize_gate320.py --weights $TRAIN_OUT/ckpt/best.pt \
  --input_size 320 --calib_dir $CALIB_DIR --n_calib 1 --batch_size 1 \
  --quant_mode test --deploy --output_dir $TRAIN_OUT/quant
# → 产出 $TRAIN_OUT/quant/*.xmodel

# ---------------- 阶段③ 规范导出 ONNX (qat_export_proper.py, 可选) ----------------
# 注意: 该脚本内 CKPT_QAT 默认指向 qat320e/ckpt/best.pt, 复现前需改成本次权重:
#   sed -i "s#qat320e/ckpt/best.pt#$TRAIN_OUT/ckpt/best.pt#" qat_export_proper.py
# python qat_export_proper.py

echo "[reproduce] 完成。最终 xmodel 在 $TRAIN_OUT/quant/"
