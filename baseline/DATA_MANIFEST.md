# 基线数据清单（DATA MANIFEST）

本文件是基线 `baseline_mobilefacenet_cosface_glint360k_ep11` 所用数据的正式清单，用于可复现与审计。所有数据均位于 **data3 本盘**，不依赖旧盘 data2。

## 1. 数据总览
| 用途 | 数据集 | 位置（data3） | 规模 | 是否基线训练数据 |
|---|---|---|---|---|
| 识别训练 | Glint360K 采样子集 | `/mnt/data3/glint360k_release/glint360k/train.rec` (+ `train.idx`) | 136,418 身份 / ≈17.4M 图 | **是** |
| 识别验证 | 验证 bin | `/mnt/data3/glint360k_release/glint360k/*.bin` | 见第 3 节 | 验证用（非训练） |
| 检测训练/评测 | WideFace | `/mnt/data3/WideFace` | 32,203 张 jpg + 标注 | 检测（YuNet）用 |
| 冒烟调试 | shards_30k / shards_quick | `cv_face_baseline/src/recognition/shards_*` | 见第 5 节 | 否（仅调试） |

## 2. 识别训练数据：Glint360K 采样子集
- **文件**：`train.rec`（130G）+ `train.idx`（343M）
- **类别数（采样后身份数）**：136,418（来自 baseline checkpoint 的 `num_classes`）
- **图像数**：≈17.4M（Glint360K 原始 360K 身份经 `--sample_rate` 采样）
- **选择方式**：训练时以 `--sample_rate` 随机采样生成，**结果固化在 `train.rec` 内部**；**未使用 `--list_file`**，因此工程内**无独立的逐身份/逐图清单文件**。
- **来源（迁移前）**：原 USB 盘 `/media/tiny/423ADD10728E128E/glint360k_release/glint360k/train.rec`（现数据已整体迁至 data3 的 `glint360k_release/`）
- **说明**：`train.rec` 即"所选数据"本身。若需可复现的逐身份清单，可从该 rec 倒出标签并落地为 `selected_identities.json`，重训时加 `--list_file` 引用（见第 6 节）。

## 3. 识别验证数据：验证 bin（人脸对验证集）
位于 `/mnt/data3/glint360k_release/glint360k/`，每个 `.bin` 含若干人脸对。

| bin | 人脸对数 | 基线 acc（eval_bin.py --flip，2026-11-13） |
|---|---|---|
| lfw | 6,000 | 98.383% |
| cfp_ff | 7,000 | 98.171% |
| cfp_fp | 7,000 | 90.371% |
| agedb_30 | 6,000 | 87.017% |
| calfw | 6,000 | 91.233% |
| cplfw | 6,000 | 85.533% |
| vgg2_fp | （bin 存在，57M） | **未纳入本次评测**（eval_bin.py 默认未评此项） |

> 基线评测实际覆盖上述 6 项，**平均 91.785%**。`vgg2_fp.bin` 文件存在但本次未评；若需补全，可单独运行 `eval_bin.py` 指定该 bin。

## 4. 检测数据：WideFace
- 位置：`/mnt/data3/WideFace`
- 内容：`WIDER_train/`、`WIDER_val/`、`WIDER_test/`（共 **32,203** 张 jpg）+ `wider_face_split/`（标注 `.mat` 与 filelist）
- 用途：YuNet 检测器（`yunet_2023mar.onnx`）训练/评测

## 5. 冒烟调试分片（非基线训练数据）
- `shards_30k`：`meta.json` 记 30,727 类 / 268,674 样本 / 每类 10 张 / 34 分片
- `shards_quick`：`meta.json` 记 4,409 类 / 67,131 样本 / 每类 20 张 / 17 分片
- 仅用于快速冒烟训练/调试，**不是**基线模型训练数据。
- 注：`meta.json` 中 `source_rec` 仍指向旧 USB 路径（迁移前），仅为生成记录，不参与训练。

## 6. 如何保证数据可复现
1. 当前基线训练数据以 `train.rec` 为载体，采样子集已固化其中，无需外部清单即可复现训练。
2. 若要求**逐身份可审计**：可从 `train.rec` 倒出身份标签（读取接口见 `src/recognition/data/mrec.py`），落地为 `baseline/selected_identities.json`，重训时加 `--list_file selected_identities.json` 即可锁定同一子集。需要我写这个倒出脚本可随时说。
3. 全部数据已随工程迁至 data3，训练/评测用显式绝对路径，完全不依赖旧盘 data2。
