# 环境保存与恢复指南

> 记录于 2026-11-05。本文件 + `requirements.txt` + 项目代码 + `/mnt/data1` 数据盘
> 构成了"将来继续"所需的最小自包含集合。

## 1. 运行时环境（写这份文档时的真实状态）

| 项 | 值 |
|---|---|
| 操作系统 | Linux（根盘 `/dev/sdb3`，193G，已用 92%） |
| Python | 3.10.12（系统 `/usr/bin/python3`） |
| 包管理 | pip 26.2.1，安装到用户级 `/home/tiny/.local` |
| 虚拟环境 | **无 conda、无 venv**（直接装在用户 site-packages） |
| CUDA / GPU | CUDA 13.0；torch `2.14.1+cu130`，torchvision `0.29.1` |
| 数据盘 | `/mnt/data1`（`/dev/sda2`，916G，持久，剩 ~51G） |

**关键依赖**（其余见 `requirements.txt`）：

| 包 | 版本 |
|---|---|
| torch | 2.14.1+cu130（**需 CUDA 13.0 驱动**） |
| torchvision | 0.29.1 |
| onnx | 1.23.2 |
| onnxruntime | 1.23.2 |
| opencv-python | 5.0.0.93 |
| numpy | 2.2.6 |
| scipy / matplotlib / pillow / tqdm | 见 `requirements.txt` |

## 2. 项目位置与目录用途

项目根：`/home/tiny/tmp/face_rec`

> ⚠️ **风险**：项目在 `/home/tiny/tmp/`，属于 `/tmp` 体系。系统 `tmpfiles`
> 清理策略**可能删除 10 天以上未访问的文件**，且重装系统会丢失。
> 长期保存请移至持久位置（见第 5 节）。

| 路径 | 用途 |
|---|---|
| `models/` | 骨干定义（GhostFaceNet / MobileFaceNetLite）、head、TopoFR |
| `train.py` | 训练入口（支持 `--arch/--loss/--shards/--finetune`） |
| `export_onnx.py` | PyTorch → ONNX + DPU 算子体检 |
| `eval_bin.py` | LFW/CFP/AgeDB 等 bin 评测 |
| `tools/verify_align.py` | 端到端：检测+对齐+识别+余弦 |
| `diagnose_*.py` | 维度塌陷等诊断脚本 |
| `deploy/` | **部署产物与脚本**（核心） |
| `deploy/arm/infer.py` | ARM CPU 推理（ncnn / onnxruntime 双后端） |
| `deploy/arm/bench.py` | CPU 测速（ncnn / ort 双后端） |
| `deploy/mobilefacenet.onnx` | batch=1，给 Vitis AI 量化（DPU 线） |
| `deploy/arm/model.onnx` | ARM CPU 用（onnxruntime） |
| `deploy/arm/face_arm.ncnn.{param,bin}` | ncnn 格式（ARM 更快） |
| `checkpoints_136k/` | **本文档模型来源**（ep11，LFW 98.50%，621M） |
| `data/` | 数据集相关脚本 |
| `MODEL_CARD.md` | 模型卡（精度、链路、命令） |

## 3. 数据

训练/评测数据在**持久数据盘**，不随项目丢失：

- 训练 shards：`/mnt/data1/shards_136k`（13.6 万身份）、`/mnt/data1/shards_30k20`
- 评测 bin：`/mnt/data1/val_bins/*.bin`（lfw / cfp_fp / agedb_30 / …）

恢复训练/评测时，确保这些路径存在即可。

## 4. 恢复步骤

```bash
# 1. 拿到项目代码与 requirements.txt（见第 5 节持久化位置）
# 2. 装依赖
pip install -r requirements.txt
#   注意：若需 GPU，torch 必须为 cu130 版，必要时：
#   pip install torch==2.14.1 torchvision==0.29.1 \
#       --index-url https://download.pytorch.org/whl/cu130

# 3. 确认数据盘路径 /mnt/data1/{shards_136k,val_bins} 存在
# 4. 验证环境
python3 -c "import torch,onnx,onnxruntime,cv2,numpy; \
print(torch.__version__, onnx.__version__, onnxruntime.__version__)"
```

## 5. 持久化（避免 /tmp 丢失）

把项目整体搬到数据盘（剩 51G，足够），并保留原样：

```bash
# 复制（保留 /tmp 原副本，安全）
mkdir -p /mnt/data1/face_rec
cp -a /home/tiny/tmp/face_rec /mnt/data1/face_rec/face_rec_$(date +%Y%m%d)

# 或建立软链接，让后续命令无需改路径
# ln -s /mnt/data1/face_rec/face_rec_20261105 /home/tiny/tmp/face_rec_persist
```

> 旧实验 checkpoints（`cf30k`/`cf2`/`gfn*`/`30k20` 等）多为对照，占用约 3.4G。
> 若只保留复现 98.50% 结果所需，可只保留：
> `checkpoints_136k/` + `deploy/` + `models/` + `train.py` + `eval_bin.py`
> + `tools/` + `data/` + `*.py` + `*.md` + `requirements.txt`（约 1G）。

## 6. 模型快速复现

```bash
cd /mnt/data1/face_rec/face_rec_20261105/deploy/arm
python3 infer.py --model model.onnx --img a.jpg --img2 b.jpg   # 1:1 比对
python3 bench.py  --model model.onnx --runs 200                # CPU 测速
```

完整链路、精度、阈值见 `MODEL_CARD.md`。
