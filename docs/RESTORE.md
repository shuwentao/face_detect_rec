# 人脸系统备份 · 恢复说明 (RESTORE)

本目录存放人脸识别系统的两份归档及其恢复方法。两份归档**互补**：
一份是**识别**（PyTorch 训练/推理），一份是**检测 + DPU 量化 + 训练**（Vitis-AI 框架）。

```
/media/tiny/423ADD10728E128E/
├── face_rec_20261105.tar.gz   识别（训练 + 推理 + ONNX 导出）
├── mymodel_20261110.tar.gz    检测 + 量化部署 + 训练
└── RESTORE.md                 本文件
```

---

## 1. 两个包分别是什么

### `face_rec_20261105.tar.gz` — 识别
- PyTorch 人脸识别**训练 + 推理**代码
- 最佳模型：`checkpoints_136k/backbone_ep11.pth`（MobileFaceNet，LFW **98.50%**）
- 已导出的识别 ONNX：`deploy/mobilefacenet.onnx`（即上面那个权重的 ONNX）
- 本地小分片：`shards_quick/`、`shards_30k/`（3 万类，可直接冒烟训练，无需外置数据集）
- `requirements.txt`：完整 pip 环境 freeze（torch / onnx / onnxruntime / insightface / opencv / numpy …）
- **不含**全量训练数据集（Glint360K 在另外的外置盘，见第 5 节）

### `mymodel_20261110.tar.gz` — 检测 + 量化 + 训练
- `face_detector_yunet/`：YuNet 人脸**检测**（含 `face_detection_yunet_2023mar.onnx` + 训练代码）
- `face_trainer/`：Glint360K 识别**训练**（TensorFlow）
- `gate320/`：人脸检测 **DPU 量化部署**（320 分辨率，INT8 `FaceDetector_int.onnx` + 量化日志）
- `requirelist.txt`：**顶层依赖/恢复清单**，解压后先看这个
- 注意：这些子项目跑在 **Vitis-AI 3.0 docker** 内，依赖在镜像里，不是本地 pip 列表

---

## 2. 恢复（解压）

```bash
tar -xzf face_rec_20261105.tar.gz  -C /你的恢复父目录
tar -xzf mymodel_20261110.tar.gz   -C /你的恢复父目录
```

解压后结构：
```
恢复父目录/
├── face_rec_20261105/
│   ├── train.py  eval_bin.py  export_onnx.py  requirements.txt  README.md  ...
│   ├── checkpoints_136k/  deploy/  shards_quick/  shards_30k/  tools/  ...
└── mymodel/
    ├── face_detector_yunet/
    ├── face_trainer/
    ├── gate320/
    └── requirelist.txt
```

---

## 3. 让两个包连起来（关键一步）

`face_rec` 的 `tools/verify_align.py` **写死**了检测模型的相对路径：

```python
# tools/verify_align.py
--detector ../face_detector_yunet/face_detection_yunet_2023mar.onnx
```

意思是：检测模型必须在 `face_rec_20261105` 的**上一级目录**里。
所以把 `mymodel` 里的检测器单独提出来摆好：

```bash
cd /你的恢复父目录
cp -r mymodel/face_detector_yunet ./face_detector_yunet
```

得到：
```
恢复父目录/
├── face_rec_20261105/
├── face_detector_yunet/   ← 从 mymodel 取出，与 face_rec_20261105 同级
└── mymodel/
```

之后即可跑端到端对齐验证（检测 → 对齐 → 识别）：
```bash
cd face_rec_20261105
python tools/verify_align.py \
    --onnx deploy/mobilefacenet.onnx \
    --detector ../face_detector_yunet/face_detection_yunet_2023mar.onnx \
    --img 图A.jpg --img 图B.jpg
```

---

## 4. 运行环境

### 4.1 识别 `face_rec`（纯 Python）
- 按 `requirements.txt` 装环境：`pip install -r requirements.txt`
  （或 `conda create` 后安装；清单来自原机 `pip freeze`，含 nvidia / tensorflow / torch / onnx 等）
- 仅做**推理/ONNX 导出**只需：`onnxruntime` + `opencv-python` + `numpy`

### 4.2 检测/量化/训练 `mymodel`（需 Vitis-AI 3.0 框架）
- 全部在 **Vitis-AI 3.0 docker** 内运行
- 量化工具：`vitis-ai-tensorflow2` 环境的 `vai-q-tensorflow2` / `vitis_quantize`
- 板端部署 `gate320` 的 `*_int.onnx` 需要 **Vitis AI DPU 运行时**
- Vitis-AI 仓库**本身不含在包内**，需另行 `git clone` 官方（提供 `dpu/ examples/ src/` 等工具链）
- 详见包内 `mymodel/requirelist.txt`

---

## 5. 训练数据集（不在包内，需另备）

| 数据集 | 用途 | 路径 | 对应参数 |
|---|---|---|---|
| Glint360K (RecordIO) | 识别训练（推荐） | `/media/tiny/423ADD10728E128E/glint360k_release/glint360k` | `face_trainer --rec_dir` |
| Glint360K (图像树) | 识别训练（后备） | `/media/tiny/glint360k_extracted/glint360k` | `face_trainer --data_root` |
| WIDER FACE | 检测训练 | `/data/WIDER` | `face_detector_yunet --data_dir` |

- `face_rec` 的全量 36 万类分片也源自 Glint360K；本地已有 `shards_quick` / `shards_30k` 可直接冒烟，无需先挂外置盘。
- ⚠️ 这些路径都在**外置盘**上，**盘未挂载时不可达**。换机器时要重新挂载，或把数据集拷过去后修改对应参数路径。

---

## 6. 快速验证

```bash
# ① 校验归档完整性
tar -tzf face_rec_20261105.tar.gz  >/dev/null && echo "face_rec OK"
tar -tzf mymodel_20261110.tar.gz   >/dev/null && echo "mymodel OK"

# ② 识别代码冒烟（确认 Python 环境可用）
cd face_rec_20261105 && python test_smoke.py

# ③ 端到端（检测+对齐+识别），见第 3 节
```

---

## 7. 备注

- 两份归档当前**双重冗余**：一份在本外置盘（`/media/tiny/423ADD10728E128E/`），另一份在 `/mnt/data1/`（`face_rec` 为目录 `face_rec_20261105/`，`mymodel` 为 `mymodel_20261110.tar.gz`）。
- `/mnt/data1` 已 96% 满（`face_rec` 源目录占 7.2G）。**确认外置盘副本无误后**，可删除 `/mnt/data1/face_rec/face_rec_20261105` 腾空间（mymodel 的 662M tar 留本地当副本即可）。
- 更多细节：
  - `face_rec/README.md`、`face_rec/ENVIRONMENT.md`
  - `mymodel/requirelist.txt`
- 打包日期：`face_rec` 2026-11-05，`mymodel` 2026-11-10。
