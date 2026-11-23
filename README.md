# cv_face_baseline —— 人脸系统 CV 工程基线

> 由 `face_rec_20261105`（PyTorch 识别）与 `mymodel`（TF 检测 / DPU 量化）整合而成。
> 作为将来继续开发与人脸论文（PyTorch 主线）的 baseline。

## 模块划分

| 目录 | 内容 | 框架 | 角色 |
|---|---|---|---|
| `src/recognition/` | 识别训练/推理/ONNX/DPU 量化 | **PyTorch** | 识别主线（最佳模型 LFW 98.5%） |
| `src/detection/` | YuNet 人脸检测（float onnx + 训练） | TensorFlow | 检测入口 |
| `src/quant_deploy/` | gate320 检测 DPU INT8 量化部署 | TensorFlow | 检测上板 |
| `src/pipeline/` | `verify_align.py` 端到端（检测→对齐→识别） | — | 验收脚本，检测器路径已参数化 |
| `checkpoints/` | 精选模型：`recognition_backbone_ep11.pth` + `yunet_2023mar.onnx` | — | 即拿即用 |
| `experiments/` | 历史实验归档位（本次未归档，实验仍在 `src/*/checkpoints_*`、`src/quant_deploy/qat*/qr*`） | — | — |
| `docs/` | RECOGNITION_README / ENVIRONMENT / MODEL_CARD / RESTORE / MYMODEL_REQUIRELIST | — | 文档合集 |
| `requirements/` | recognition.txt / detection.txt / quant_deploy.txt | — | 分框架依赖 |
| `data/` | 外部数据集入口（软链接入，见下方"数据集"与 `data/README.md`） | — | — |

## 数据集

本工程**不搬运大文件**，统一通过 `data/` 下的软链接入。实际用到/接入的数据集：

| 数据集 | 软链 | 格式 | 规模 | 用途 | 被谁消费 |
|---|---|---|---|---|---|
| **Glint360K** | `data/glint360k` → `../../glint360k_release/glint360k` | MXNet RecordIO（`train.rec` 130G + `train.idx`） | 17.4M 图 / 360K 身份 | 识别训练主数据 | `src/recognition/train.py`（`--shards`/`--lfw_bin`）、`eval_bin.py` |
| └ 验证集（7 个 bin） | 同上目录内 `lfw.bin` `cfp_ff.bin` `cfp_fp.bin` `agedb_30.bin` `calfw.bin` `cplfw.bin` `vgg2_fp.bin` | 二进制 pair/bin | — | 识别精度评测 | `eval_bin.py` |
| **WideFace** | `data/widerface` → `../../WideFace` | 单张 jpg（WIDER_train/val/test，共 32,203 张）+ 标注 | 9G | 人脸**检测**训练/评测 | `src/detection/`（YuNet）、`src/quant_deploy/`（gate320） |
| **Glint360K 本地分片** | `data/glint360k_shards_quick` / `glint360k_shards_30k` → `../src/recognition/shards_*` | 自定义 `.bin` 分片 | 小 | 小规模冒烟训练/调试（无需外置大数据） | `train.py --shards` |

要点：
- **识别数据保持整文件**（RecordIO / `.bin` 分片），不展开成单张图——机械盘随机 IO 会拖垮 GPU 吞吐（详见 `data/README.md`）。
- Glint360K 标签不在 MXNet 标准偏移，而在 RecordIO header 第 32 字节，解析见 `src/recognition/data/mrec.py`。
- `imagenet数据集`（266G，位于**已退役的旧盘 data2**）与人脸无关，**不接入**；`glint360k`（gzip 分卷 ~120G）与 `glint360k_release` 重复，仅存在于旧盘、本盘无需处理。

## 数据软链维护（本盘自洽）

数据集已随工程迁移到**本盘**（与 `cv_face_baseline/` 同级）：`glint360k_release/`、`WideFace/`，以及 `src/recognition/shards_*`。
`data/` 下的软链应指向这些**本盘本地副本**，而非旧盘 data2。因 `data/` 目录属主为 root，修正需 sudo：

```bash
sudo bash -c 'cd /mnt/data3/cv_face_baseline/data && \
  ln -sfn ../../glint360k_release/glint360k glint360k && \
  ln -sfn ../../WideFace               widerface && \
  ln -sfn ../src/recognition/shards_quick   glint360k_shards_quick && \
  ln -sfn ../src/recognition/shards_30k    glint360k_shards_30k'
```

> 不修正也能训练/测试：命令直接传绝对路径（如 `--lfw_bin /mnt/data3/glint360k_release/glint360k`），不依赖软链。

## 快速上手

```bash
# 识别训练/评测/导出（PyTorch）
cd src/recognition
python test_smoke.py
python train.py --shards <shards> --arch mobilefacenet --loss cosface ...
python eval_bin.py --ckpt checkpoints_136k/backbone_ep11.pth ...

# 端到端（检测→对齐→识别）：
#   --onnx 是【识别】模型（下面用已导出的 mobilefacenet.onnx）
#   --detector 是 YuNet 检测器，默认已指向 src/detection/yunet_2023mar.onnx
python src/pipeline/verify_align.py \
    --onnx src/recognition/deploy/mobilefacenet.onnx --img A.jpg --img B.jpg
# 也可显式指定检测器 / 用环境变量覆盖：
#   --detector <onnx>   或   export FACE_DETECTOR_ONNX=<onnx>

# 识别上 DPU（Vitis AI 3.0 pytorch 后端，容器 vitis-ai-pytorch）
cd src/recognition && bash deploy/run_vitis.sh checkpoints_136k/backbone_ep11.pth

# 检测上 DPU（Vitis AI 3.0 tensorflow2 后端，在 docker 内）
cd src/quant_deploy && python quantize_gate320.py ...
```

## 关于 TensorFlow

- **识别**：PyTorch 全程（含 DPU 量化走 `pytorch_nndct`），不依赖 TF。
- **检测量化**：`src/quant_deploy`（gate320）走 Vitis-AI `vai-q-tensorflow2`，是 YuNet 历史包袱。
- 论文主线用 PyTorch 即可；TF 仅停留在检测部署工具链，无需写进论文。
- 最新 Vitis AI（6.1）仍支持 PyTorch（BYOM 明确列出），旧 `pytorch_nndct` 在新一代 NPU 上被 AMD Quark 取代，但本项目的 Zynq DPUCZDX8G 仍用 Vitis AI 3.x + `pytorch_nndct`，无需改动。

## 未决事项

1. `mymodel/face_trainer`（TF 版识别训练）与 PyTorch 主线重复，**已随工程迁移至本盘 `/mnt/data3/mymodel/face_trainer`，未并入 `cv_face_baseline`**。确认后可归档或丢弃。
2. 数据集已随工程迁移至本盘（`glint360k_release/` 与 `WideFace/` 在工程根目录同级），通过 `data/` 软链接入，详见"数据集"/"数据软链维护"小节；`glint360k`(gzip 分卷) 与 `imagenet数据集` 仅存在于**已退役的旧盘 data2**，本盘无需清理。
