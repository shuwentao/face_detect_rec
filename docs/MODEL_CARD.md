# 人脸识别模型文档（MobileFaceNet 136k）

> 更新日期：2026-11-05
> 权重来源：`checkpoints_136k/backbone_ep11.pth`（项目内最佳）

## 1. 模型概况

| 项目 | 值 |
|---|---|
| 骨干 | MobileFaceNetLite（自研简化版，8 个 depthwise-separable block，无 SE） |
| 参数量 | **1.33M** |
| 输入 | `float32 [1, 3, 112, 112]`，RGB，值域 **[-1, 1]** |
| 输出 | `float32 [1, 512]` 嵌入向量，**未归一化** |
| 训练数据 | 13.6 万身份 shards（`/mnt/data1/shards_136k`） |
| 损失 | CosFace |
| 最佳 epoch | ep11 |

**注意**：这不是论文里的 MobileFaceNetV2（2021），是项目自建的简化 MobileFaceNet。若需要论文版需重新实现并训练。

## 2. 精度（ONNX 实测，10-fold）

| 评测集 | 精度 | 最佳阈值 | 备注 |
|---|---|---|---|
| LFW | **98.50%** | 0.281 | 与 PyTorch 完全一致，导出零损失 |
| CFP-FF | 97.99% | 0.289 | 正面照 |
| CALFW | 90.95% | 0.237 | 跨年龄 |
| CFP-FP | 90.09% | 0.196 | 跨姿态 |
| VGG2-FP | 89.90% | 0.171 | 跨姿态 |
| AgeDB-30 | 86.53% | 0.213 | 跨年龄 |
| CPLFW | 84.98% | 0.202 | 跨姿态 |

**已知短板**：跨姿态（CFP-FP）与跨年龄（AgeDB/CALFW）明显弱于大数据集训练的 SOTA 参考值（差 7~11 个点）。LFW 成绩好看主要因为同姿态居多；实际部署场景若姿态/年龄跨度大，请以 CFP-FP / AgeDB 的表现为准。

## 3. 部署产物

| 文件 | 用途 |
|---|---|
| `deploy/mobilefacenet.onnx` | batch=1 固定，供 Vitis AI 量化编译（DPU 线） |
| `deploy/arm/model.onnx` | ARM CPU 用 onnxruntime 直接跑 |
| `deploy/arm/face_arm.ncnn.param/.bin` | ncnn 格式（ARM 上通常更快） |
| `checkpoints_136k/backbone_ep11.pth` | PyTorch 原始权重（量化走这条线） |

算子清单：`Conv×18 + Relu×18 + Gemm + GlobalAveragePool + Flatten`，全部落在 DPUCZDX8G 白名单内，BatchNorm 已融合。

## 4. 链路与预处理约定

```
原图 → YuNet 检测(5点) → 相似变换对齐 112×112 → 模型 → L2 归一化 → 余弦比对
```

预处理（必须与训练/评测一致）：

1. BGR → RGB
2. `(x - 127.5) / 127.5` → [-1, 1]
3. HWC → CHW，加 batch 维

对齐模板：ArcFace 标准 5 点（左眼、右眼、鼻尖、左嘴角、右嘴角），见 `tools/verify_align.py` 中的 `TEMPLATE`。

**L2 归一化在模型外做**（两个 onnx 输出都未归一化；`deploy/arm/infer.py` 已内置，输出恒为单位向量）。

## 5. 使用命令

```bash
cd /home/tiny/tmp/face_rec/deploy/arm

# 两图比对（onnxruntime）
python3 infer.py --model model.onnx --img a.jpg --img2 b.jpg

# 原图 + 5 点（左眼,右眼,鼻尖,左嘴角,右嘴角）
python3 infer.py --model model.onnx \
    --img "a.jpg:38,52,74,52,56,72,42,92,71,92" \
    --img2 "b.jpg:40,50,75,51,57,73,43,93,72,93"

# CPU 测速（onnx / ncnn 均支持）
python3 bench.py --model model.onnx --runs 200
python3 bench.py --model face_arm.ncnn.param --runs 200

# PC 端端到端验证（检测+对齐+识别）
python3 tools/verify_align.py --onnx deploy/mobilefacenet.onnx \
    --detector ../face_detector_yunet/face_detection_yunet_2023mar.onnx \
    --img a.jpg --img b.jpg
```

**比对阈值**：1:1 场景起步用 **0.28**（LFW 10-fold 均值 0.281）。正式业务需按目标 FAR 用 FAR/FRR 曲线重新标定，不可直接照抄。

## 6. DPU 量化（上板时）

量化走 `vai_q_pytorch`，直接吃 `.pth`，不经过 onnx：

```bash
python3 deploy/quantize_pt.py \
    --ckpt checkpoints_136k/backbone_ep11.pth \
    --calib deploy/calib.npy \
    --target DPUCZDX8G_ISA1_B4096 \
    --deploy_check

bash deploy/compile_xmodel.sh quantize_result/MobileFaceNetLite_int.xmodel
```

量化后精度损失未验证，上板前务必用 `eval_bin.py` 复测一遍。

## 7. 对照实验记录（同数据 30k/136k）

| 实验 | 骨干 | 损失 | lr | 最佳 LFW |
|---|---|---|---|---|
| checkpoints_cf30k | MobileFaceNet | cosface | 0.01 | 96.67% |
| checkpoints_cf2 | MobileFaceNet | cosface | **0.003** | **97.98%**（30k 档最佳） |
| checkpoints_30k20 | MobileFaceNet | adaface | 0.004 | 69.58%（塌陷） |
| checkpoints_gfn30k | GhostFaceNet | cosface | 0.01 | 96.40% |
| **checkpoints_136k** | **MobileFaceNet** | cosface | — | **98.50%**（本文档模型） |
| checkpoints_gfn136_lr3 | GhostFaceNet | cosface | 0.003 | 96.75% |

关键结论：**lr 0.003 显著优于 0.01**（同数据 +1.31 点）；GhostFaceNet 在 136k 身份下出现维度塌陷（有效维度 ep0 即降至 1.0），故未采用。
