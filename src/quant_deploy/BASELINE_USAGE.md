# 复现 baseline (`repro_mobilenetv3`) 使用说明

所有命令都在本目录执行:
```bash
cd /mnt/data3/cv_face_baseline/src/quant_deploy
```

权重(baseline 本体):`repro_mobilenetv3/ckpt/best.pt`(4.9MB,float)
- `final.pt` 最后一轮、`last.pt` 含优化器可续训、`config.json` 配置、`history.json` loss 曲线

## 1. 单图检测(画框 + 5 关键点)
```bash
python3 infer_baseline.py --img /path/to/face.jpg --out out.jpg --conf 0.3
```
`--conf` 调高可减少误检(默认 0.05 会出很多低分框,建议 0.3~0.5)。

## 2. 批量检测整个目录
```bash
python3 infer_baseline.py --img_dir /path/to/images --out_dir out/ --conf 0.3
```

## 3. 在 300W 600 张上评估成绩(mAP / NME)
```bash
python3 eval_dump.py --ckpt repro_mobilenetv3/ckpt/best.pt \
  --ann common/data/facebench_wider/annotations/landmark_300w.json \
  --img_dir /mnt/data3/300W/300W \
  --max_images 600 --input_size 320 --batch 8
```
当前实测:mAP@0.5=0.6801, NME(框归一)=0.0768, 失效率 20.2%。

## 4. 从零重训(复现实验本身)
```bash
python3 repro_train.py cuda     # 20 epoch, ~48 分钟, 覆盖写 repro_mobilenetv3/
```

## 5. 量化到 DPU(未跑,需 Vitis-AI docker 环境)
```bash
# 需进入 xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106 容器,
# 激活 vitis-ai-pytorch conda 环境后:
python quantize_gate320.py --weights repro_mobilenetv3/ckpt/best.pt \
  --input_size 320 --calib_dir /mnt/data3/WideFace/WIDER_train/images \
  --n_calib 200 --batch_size 8 --quant_mode calib \
  --output_dir repro_mobilenetv3/quant
# 再 --quant_mode test --deploy 导出 xmodel
```

## 6. 识别模型复现(数据已从 train.rec 重建验证)
```bash
cd /mnt/data3/cv_face_baseline/src/recognition
bash chain_finetune_local.sh prep-shards    # 从 train.rec 重建 30k/136k 分片(~22秒)
bash chain_finetune_local.sh 30k            # 或 all 一次跑完三阶段
bash chain_finetune_local.sh finetune136k
bash chain_finetune_local.sh direct136k
```
数据源 `/mnt/data3/glint360k_release/glint360k/{train.rec,lfw.bin}` + `index/*.npz` 都在。
各实验最佳 backbone(6.9MB)和 history.json 保留在 `checkpoints_*/`。

## 环境
- 训练/推理/评估:当前 python3(torch 2.14.1+cu130, RTX 4070)
- 量化:需 Vitis-AI docker(镜像已在本地,见第 5 节)
