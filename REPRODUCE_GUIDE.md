# 完整复现指南（从零重建 + 训练 + 使用）

本仓库包含**人脸检测**和**人脸识别**两套基线的完整复现代码。本文是一份自包含的操作手册：
从 clone 后的初始状态开始，如何重建数据、重建被删的大文件、执行训练、使用与评估基线。

所有命令默认在项目根目录执行：`cd <仓库根目录>`（当前机器为 `/mnt/data3/cv_face_baseline`）。

---

## 目录

- [0. 前提：外部数据集（不在仓库内）](#0)
- [1. 环境](#1)
- [2. 重建被删的两个大文件](#2)
- [3. 检测：从零复现](#3)
- [4. 识别：从零复现](#4)
- [5. 使用已固化的 baseline（不重训）](#5)
- [6. 选中的图片数据清单](#6)
- [7. 复现结果与验证记录](#7)
- [8. 体积与上传说明](#8)

---

<a name="0"></a>
## 0. 前提：外部数据集（不在仓库内，需自己保有）

三个数据集都不随仓库上传（体积大），复现前确认它们在你的机器上：

| 用途 | 数据 | 路径 | 规模 |
|---|---|---|---|
| 检测 bbox | WIDER FACE | `/mnt/data3/WideFace/WIDER_{train,val}/images` + `wider_face_split/*.txt` | train 12880 / val 3226 |
| 检测关键点 | 300W (68 点) | `/mnt/data3/300W/300W`（01_Indoor + 02_Outdoor） | 600 张 |
| 识别训练 | GLINT360K | `/mnt/data3/glint360k_release/glint360k/{train.rec,train.idx}` | 130G，811 万条 / 13.6 万身份 |
| 识别验证 | LFW | `/mnt/data3/glint360k_release/glint360k/lfw.bin` | 6000 对 |

路径写死在脚本里；如果你的数据不在 `/mnt/data3`，按实际路径改脚本顶部的变量。

---

<a name="1"></a>
## 1. 环境

- 训练/推理/评估：当前 `python3`（torch 2.14.1+cu130），GPU RTX 4070（识别/检测用 `--amp` 混合精度）
- 依赖：`torch torchvision numpy opencv-python pillow scipy`（见 `requirements/`）
- 检测量化到 DPU（可选）：本机 Vitis-AI docker `xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106`，当前 python 缺 `pytorch_nndct`，须进容器跑

---

<a name="2"></a>
## 2. 重建被删的两个大文件（为 GitHub 上传删除，可重建）

这两个文件超过 GitHub 单文件 100MB 限制而删除。源数据 `train.rec` 在，按下面重建。**顺序：先索引 → 再分片 → 再校准。**

### 2.1 `src/recognition/index/local_60G.npz`（原 101M，136418 身份索引）
用于重建 136k 识别分片。（`index/glint360k_2M.npz` 和 `glint360k_head300k.npz` 未删，30k 可直接用。）
```bash
cd src/recognition
python -m data.build_index \
  --rec /mnt/data3/glint360k_release/glint360k/train.rec \
  --idx /mnt/data3/glint360k_release/glint360k/train.idx \
  --out index/local_60G.npz
```
全量扫描 130G rec 约 **55 分钟**。不重建则只能跑 30k 阶段，跑不了 136k。

### 2.2 `src/recognition/deploy/calib.npy`（原 144M，INT8 量化校准集）
仅检测/识别**量化部署**时需要；训练和普通推理不需要。
```bash
cd src/recognition
bash chain_finetune_local.sh prep-shards    # 从 train.rec 重建 shards
python deploy/make_calib.py --shards shards_136k --out deploy/calib.npy --num 1000
```

> 重建出的这两个文件已被 `.gitignore` 排除，不会误提交。

---

<a name="3"></a>
## 3. 检测：从零复现

### 3.1 重建训练标注 JSON（仓库里已有；丢了才重跑）
```bash
cd src/quant_deploy
python3 common/scripts/widerface_to_facebench.py \
  --wider_root /mnt/data3/WideFace \
  --out_dir common/data/facebench_wider \
  --sample_rate 1.0
python3 common/scripts/landmark_300w_to_json.py \
  --root /mnt/data3/300W/300W \
  --out common/data/facebench_wider/annotations/landmark_300w.json
```
产物：`annotations/{train,val,landmark_300w}.json`（COCO 格式，图片路径内嵌，不依赖额外 images 目录）。

### 3.2 从零训练（20 epoch，约 48 分钟）
```bash
cd src/quant_deploy
python3 repro_train.py cuda
```
配方：MobileNetV3-small（ImageNet 预训练，1.193M）/ 输入 320 / lr 5e-4 / wd 5e-4 / batch 8 / epoch 20 / seed 42；
数据：WIDER FACE 12880（bbox）+ 300W 600（关键点）= 13480 张。产物：`repro_mobilenetv3/ckpt/best.pt`。

### 3.3 推理与评估
```bash
# 单图检测（画框+5关键点）
python3 infer_baseline.py --img 你的图.jpg --out out.jpg --conf 0.3
# 批量检测
python3 infer_baseline.py --img_dir 图片目录 --out_dir out/ --conf 0.3
# 在 300W 600 张上评估 mAP / NME
python3 eval_dump.py --ckpt repro_mobilenetv3/ckpt/best.pt \
  --ann common/data/facebench_wider/annotations/landmark_300w.json \
  --img_dir /mnt/data3/300W/300W --max_images 600
```

### 3.4 （可选）量化到 DPU
需进 Vitis-AI docker，激活 `vitis-ai-pytorch` 环境后：
```bash
python quantize_gate320.py --weights repro_mobilenetv3/ckpt/best.pt \
  --input_size 320 --calib_dir /mnt/data3/WideFace/WIDER_train/images \
  --n_calib 200 --batch_size 8 --quant_mode calib \
  --output_dir repro_mobilenetv3/quant
# 再 --quant_mode test --deploy 导出 xmodel
```

---

<a name="4"></a>
## 4. 识别：从零复现

> 前提：已按 [2.1](#2) 重建 `local_60G.npz`（136k 阶段需要）。

### 4.1 一键全链（推荐）
```bash
cd src/recognition
bash chain_finetune_local.sh all
```
依次执行：`prep-shards`（重建分片）→ `30k`（从零训）→ `finetune136k`（微调）→ `direct136k`（直接训），约 1.5~2.5 小时。

### 4.2 分阶段执行
```bash
bash chain_finetune_local.sh prep-shards    # 从 train.rec 重建 30k/136k 分片（seed=42）
bash chain_finetune_local.sh 30k            # 30k 子集 20ep lr0.01, GhostFaceNet+CosFace
bash chain_finetune_local.sh finetune136k   # 挑 30k 最佳 epoch 微调 136k 10ep lr0.003
bash chain_finetune_local.sh direct136k     # 136k 直接训 12ep lr0.003
```
训练通用参数：`--arch ghostfacenet --loss cosface --sample_rate 1.0 --batch_size 128 --amp --num_workers 6 --lfw_bin <lfw.bin>`，每 epoch 末跑 LFW 评测。

**关键教训**：136k 分片必须用 `index/local_60G.npz`（136418 身份）。用 `glint360k_2M.npz`（30727 身份）会掉约 6 个点，脚本已修正。

### 4.3 选图机制（可复现的原因）
- 用 `data/build_shard.py` 的 `pick_samples(per_class=10, seed=42)` 从 index 挑图，选中结果固化在 shard 里。
- 训练 `torch.manual_seed(42)` / `np.random.seed(42)`，DataLoader 每轮 shuffle + 在线增强，都固定 seed。
- **同一 seed 重跑，逐 epoch LFW 完全一致（已验证 3 次）**。选中的具体记录见 [第 6 节](#6)。

---

<a name="5"></a>
## 5. 使用已固化的 baseline（不重训）

仓库已包含最佳权重，直接加载即可：

| 部分 | 权重 | 成绩 |
|---|---|---|
| 检测 | `src/quant_deploy/repro_mobilenetv3/ckpt/best.pt` | mAP@0.5 0.6801 / NME 0.0768 |
| 识别 30k | `src/recognition/checkpoints_gfn30k/backbone_ep18.pth` | LFW 95.15% |
| 识别两阶段（主） | `src/recognition/checkpoints_gfn136_ft/backbone_ep7.pth` | LFW 96.77% |
| 识别直接训 | `src/recognition/checkpoints_gfn136_lr3/backbone_ep11.pth` | LFW 84.28% |

权重为 dict 结构 `{'backbone': {...}, 'arch', 'width', 'embedding_size', 'num_classes'}`，加载方式见 `src/recognition/diag_why.py` / `export_onnx.py`。检测权重见 `infer_baseline.py`。

---

<a name="6"></a>
## 6. 选中的图片数据清单

`SELECTED_DATA/` 下有 5 个清单，是实际训练用到的数据子集（固定 seed 挑出来的结果，显式化）：

| 文件 | 行数 | 格式 |
|---|---|---|
| `detection_train_list.txt` | 12880 | 每行一张图的完整路径 |
| `detection_val_list.txt` | 3226 | 同上 |
| `detection_landmark_300w_list.txt` | 600 | 同上 |
| `recognition_30k_selected.txt` | 268674 | `记录序号 身份ID 字节偏移`（首行注明 rec 路径） |
| `recognition_136k_selected.txt` | 1182654 | 同上 |

识别清单是从 index 用与训练相同的 `pick_samples` 重放导出，数字与训练日志吻合。

---

<a name="7"></a>
## 7. 复现结果与验证记录

### 检测（300W 600 张）
| 指标 | 我们 | qr_ft_E 目标（量化） |
|---|---|---|
| mAP@0.5 | 0.6801 | 0.5365 |
| NME（框归一） | 0.0768 | 0.1534 |
| 失效率 | 20.2% | 56.7% |

### 识别（LFW）
| 阶段 | 我们 | 当年目标 |
|---|---|---|
| 30k | 95.15% | 96.4% |
| finetune136k | 96.77% | 97.53% |
| direct136k | 84.28% | 85.2% |

### 可复现性实证
- 手动重复训练 2 次 + 自动 1 次，共 3 次，42 个 epoch 的 LFW **逐位完全一致（最大差异 0.0）**。
- shards 从 `train.rec` 重建验证通过（30k 3 万身份 / 136k 13.6 万身份）。

---

<a name="8"></a>
## 8. 体积与上传说明

- 当前仓库 **254M**，无 >100MB 文件（GitHub 单文件硬限制），283 个文件。
- 为上传删除的两个大文件及其重建见 [第 2 节](#2)。
- `.gitignore` 排除了缓存、超大重建文件、shards、训练中间产物；4 个核心权重确认不会被误排除。
- 完整历史备份（8.4G，含删除前所有内容）：`/mnt/data3/cv_face_baseline.tar.gz`。

### 上传到 GitHub
```bash
cd <仓库根目录>
git init
git add .
git commit -m "复现基线: 检测+识别 (from-scratch)"
# 在 GitHub 建空仓库后：
git remote add origin https://github.com/<账号>/<仓库名>.git
git push -u origin main
```
上传前建议 `git status --short | wc -l` 看一眼文件数，确认没有误加临时文件。clone 后若要跑识别 136k，先按 [第 2.1 节](#2) 重建 `local_60G.npz`（约 55 分钟）。

---

## 附：命令速查（逐条带注释）

**检测**（`cd src/quant_deploy`）
```bash
python3 repro_train.py cuda              # 从零训练 20ep（已有 best.pt 则不必）
python3 infer_baseline.py --img 图.jpg --out out.jpg --conf 0.3  # 单图检测
python3 infer_baseline.py --img_dir 目录 --out_dir out/ --conf 0.3  # 批量检测
python3 eval_dump.py --ckpt repro_mobilenetv3/ckpt/best.pt \   # 评估 mAP/NME
  --ann common/data/facebench_wider/annotations/landmark_300w.json \
  --img_dir /mnt/data3/300W/300W --max_images 600
```
**识别**（`cd src/recognition`）
```bash
bash chain_finetune_local.sh all         # 一键全链
bash chain_finetune_local.sh prep-shards  # 只重建分片
bash chain_finetune_local.sh 30k          # 只跑 30k 阶段
bash chain_finetune_local.sh finetune136k # 只跑微调阶段
bash chain_finetune_local.sh direct136k   # 只跑直接训阶段
```

## 附：权重文件结构
识别 backbone 权重是 dict：`{'backbone': {486 个张量}, 'arch', 'width', 'embedding_size', 'num_classes}`，加载方式见 `src/recognition/diag_why.py` / `export_onnx.py`。检测权重见 `infer_baseline.py`（自动取 `model` 键）。

## 附：原始工程文档
`README.md` 是原项目作者写的工程说明（模块划分、识别技术细节、历史实验、DPU 部署），与复现无关但保留。
