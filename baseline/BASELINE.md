# 基线模型（Baseline）

本文件把**当前可用的人脸识别模型**正式定为项目基线。后续所有实验 / 改进都以它为准对照。

## 基线标识
- 名称：`baseline_mobilefacenet_cosface_glint360k_ep11`
- 权重文件：`baseline/baseline_mobilefacenet_cosface_glint360k_ep11.pth`
  （从 `checkpoints/recognition_backbone_ep11.pth` **冻结拷贝**，原文件保持不变，互不干扰）
- 导出 ONNX（端到端推理用）：`src/recognition/deploy/mobilefacenet.onnx`
- 检测端 ONNX：`src/recognition/deploy/yunet_2023mar.onnx`

## 模型规格（来自 checkpoint 元数据，已确认）
| 项 | 值 |
|---|---|
| 骨架 | mobilefacenet（width = 1.0） |
| 嵌入维度 | 512 |
| 分类头损失 | cosface（margin `m=0.4`, scale `s=64`，train.py 默认） |
| 类别数 num_classes | 136,418（Glint360K 采样子集身份数） |
| 训练 epoch | 11（ep11 权重） |

## 训练数据
- Glint360K（MXNet RecordIO `train.rec`；原始 17.4M 图 / 360K 身份，本次采样 136,418 身份）
- 数据位置（本盘）：`/mnt/data3/glint360k_release/glint360k`
- 验证集：lfw / cfp_ff / cfp_fp / agedb_30 / calfw / cplfw / vgg2_fp（同目录下 7 个 `.bin`）
- **完整数据清单（来源 / 规模 / 选择方式 / 逐集明细）见 [DATA_MANIFEST.md](./DATA_MANIFEST.md)。**

## 实测精度（2026-11-13，data3 上 `eval_bin.py --flip` 复测）
| 验证集 | acc |
|---|---|
| LFW | 98.383% |
| CFP-FF | 98.171% |
| CFP-FP | 90.371% |
| AgeDB-30 | 87.017% |
| CALFW | 91.233% |
| CPLFW | 85.533% |
| **平均** | **91.785%** |

## 如何使用基线
```bash
cd /mnt/data3/cv_face_baseline/src/recognition

# 评测基线
python3 eval_bin.py \
   --ckpt /mnt/data3/cv_face_baseline/baseline/baseline_mobilefacenet_cosface_glint360k_ep11.pth \
   --bins /mnt/data3/glint360k_release/glint360k --flip

# 端到端（检测→对齐→识别）
python3 ../pipeline/verify_align.py --onnx deploy/mobilefacenet.onnx --img A.jpg --img B.jpg
```

## 说明
- 这是"当前可用即基线"决策下的**冻结快照**：未来若训出更优模型，请另存新基线并更新本文件，保留历史可比性。
- 原训练超参（batch_size / lr / sample_rate / 优化器 / 数据采样方式）以原 `face_rec` 训练记录为准；本文件仅记录 checkpoint 确认项与本次复测指标。
- 运行前务必在 **data3 目录下**启动 python（本机 shell 的持久 CWD 曾停在已掉线的 `/mnt/data2`，会导致 import 扫到死盘而崩溃；只要 `cd` 进 data3 任意目录即可）。
