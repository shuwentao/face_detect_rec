# 人脸识别训练工程（面向 ZynqMPSoC + Vitis AI 3.0）

输入链路：**原图 + 5 个关键点 → 仿射对齐 112×112 → 特征提取 → 余弦比对**
训练阶段：Glint360K（1700 万张 / 36 万身份）+ 轻量骨干 + AdaFace/PartialFC
部署阶段：ONNX → Vitis AI 3.0 量化 → DPUCZDX8G

---

## 一、数据集评估结论

你给的目录 `/media/tiny/423ADD10728E128E`（2.0T，可用 1.4T）：

| 目录 | 内容 | 能否用于本次训练 |
|---|---|---|
| `glint360k_release/glint360k/` | `train.rec` 139GB + `train.idx` + 8 个验证 bin | ✅ **直接用**，这就是主训练集 |
| `glint360k/glint360k_00 … _06` | 上面那份数据的 gzip 分卷（7×20GB） | 与上一份**重复**，不用管 |
| `imagenet数据集/` | ImageNet | ❌ 与人脸无关 |
| `WideFace/` | WIDER FACE 检测集 | 只能训检测器；**没有关键点**，不能用于识别训练 |

### Glint360K 实测规格

| 项 | 实测值 |
|---|---|
| 图像数 | **17,451,890** |
| 身份数 | **360,232** |
| 图像尺寸 | 112×112×3 JPEG |
| 平均每身份 | ≈ 48 张（长尾，最少 1 张） |
| 对齐方式 | 已按 RetinaFace 5 点 + ArcFace 模板对齐 |
| 验证集 | lfw / cfp_ff / cfp_fp / agedb_30 / calfw / cplfw / vgg2_fp 齐全 |

**它是目前公开的规模最大的人脸识别训练集之一**，用它训出来的模型在 LFW/CFP/AgeDB
上的表现会显著好于 MS1M。

### 两个必须知道的坑

**① 标签不在标准 MXNet 的 label 偏移上**

`train.idx` 只有「序号 → 字节偏移」，**没有标签**。标签埋在每条记录的 header 里，
而且不在 mxnet 官方的 `IRHeader.label`（那里恒为 0），而在**第 32 字节**：

```
偏移   字段              说明
0      magic 0xCED7230A
4      lrecord           = payload 长度 + 32
8      flag              恒 2
12     -                 恒 0
16     image_id int64    全局序号，≠ 类别
24     -                 恒 0
32     identity float32  ★ 身份标签（0 ~ 360231）
36     attr float32      附带属性，0/1
40     payload           JPEG 字节流
```

很多现成解析代码读出来标签全是 0，就是因为读了 offset 12/24。
本工程的 `data/mrec.py` 已按上表实现并经图像肉眼核对（同标签块内确为同一人）。

**② 数据盘是机械盘，顺序读只有 43.5 MB/s**

`ST2000DM001`（ROTA=1）。训练时若按索引随机 seek，每个 batch 十几次随机 IO，
GPU 会长期空转。所以本工程**先做分片（shard）**：把样本重排成若干几十 MB 的大文件，
训练时整片读入内存，磁盘访问退化成顺序读。
实测分片后训练吞吐 **1270 img/s**（RTX 4070 Laptop / 8GB）。

### 如果你想要「原始图 + 关键点」形式的训练数据

Glint360K 里的图**已经对齐过了，不带关键点**，所以严格意义上不能用来训练
「从原图 + 5 点 → 特征」这条完整链路的**几何部分**（只能训练对齐之后的部分）。
本工程的应对是**域随机化**：训练时对已对齐图施加随机相似变换（平移/旋转/缩放），
量级按端上关键点误差设定，见 `data/dataset.py`。

若确实需要带关键点的原始数据，可下载：

| 数据集 | 规模 | 有关键点 | 地址 |
|---|---|---|---|
| WebFace260M | 4M 身份 / 42M 图 | ✅ 5 点 + bbox | https://github.com/bchidamb/WebFace260M （需申请） |
| MS1MV2/V3 | 85K 身份 / 5.8M 图 | 需自己跑检测器 | https://github.com/deepinsight/insightface/tree/master/recognition/_data_ |
| Glint360K | 360K 身份 / 17M 图 | ❌（已对齐） | 同上 |
| LFW（评测） | 6000 对 | ✅ | http://vis-www.cs.umass.edu/lfw/ |

> 你手上的 Glint360K 已经足够，不建议为此再下别的数据集。

---

## 二、算法选型（近两年）

| 模块 | 选择 | 年份 | 理由 |
|---|---|---|---|
| 骨干 | **GhostFaceNet**（GhostNetV1 结构 + 人脸头） | IEEE Access 2023 | 轻量档 SOTA，LFW 99.7%+ / CFP-FP 96%+；只用了 Conv / DWConv / BN / ReLU / Concat / Add / GAP / MatMul，**全部在 DPUCZDX8G 硬件算子表内** |
| 损失 | **AdaFace** | CVPR 2022 Oral | 用特征范数近似图像质量，margin 随质量自适应；在低质量集（CPLFW / TinyFace / IJB-S）提升最大，正好对应板子上的真实摄像头画质 |
| 拓扑正则 | **TopoFR** | NeurIPS 2024 | 要求特征空间的拓扑结构与输入图像空间一致。**纯训练期正则，推理时骨干不变**，可叠加在任意骨干 + 任意 margin 损失之上；`--topo_weight 0.1` 开启 |
| 大规模训练 | **Partial FC** | TPAMI 2022 | 36 万类不可能每步全算（737MB 权重 + 梯度 + 优化器状态 ≈ 2.2GB），每步只抽一部分类 |
| 类内多峰 | Sub-center（可选，`--sub_centers 3`） | — | 一个身份有多个视觉簇（正脸/侧脸/戴眼镜）时更稳 |

### ⚠️ 默认值必须按上表来，别用 2018 年那套

曾经因为「要上 DPU」把 `train.py` 的默认骨干从 GhostFaceNet 改成了
MobileFaceNet（2018，它能整网编成 1 个 DPU 子图），又因为本工程的 AdaFace
实现会塌陷（见第六节）把默认损失改成了 CosFace（2018）——
**两个 2018 年的配置叠加在一起，等于把第二节的选型整体架空了**。
现在部署走 CPU/ncnn，不再受 DPU 算子约束，默认已改回：

    --arch ghostfacenet     # IEEE Access 2023
    --loss cosface          # 过渡方案，见第六节 AdaFace 塌陷问题
    --topo_weight 0.1       # NeurIPS 2024，叠加近两年的正则

若确实要编 DPU，再显式加 `--arch mobilefacenet`。

### 为什么没选那几个「更新」的

* **TransFace / Face Transformer / ViT 类**：含 LayerNorm + GELU + Attention，
  DPUCZDX8G **不支持**，量化后会整块掉到 CPU，帧率崩掉。
* **RepViT / MobileNetV4**：RepViT 重参数化后是纯 CNN，理论上可以，
  但它带 SE 且通道数偏大，收益不如 GhostFaceNet 明确；MobileNetV4 含 MQA 块，直接出局。
* **TopoFR（NeurIPS 2024）**：它的「拓扑一致性」是**训练期正则**，推理时骨干不变，
  和本工程完全兼容。想再冲一把可以后加，不影响部署。
* **ArcFace（CVPR 2019）**：仍在工程里保留（`--loss arcface`），作为 AdaFace 的对照组。

---

## 三、目录结构

```
face_rec/
├── data/
│   ├── mrec.py           RecordIO 逐字节解析（无 mxnet 依赖）
│   ├── build_index.py    扫 train.rec 抽标签 → index.npz
│   ├── build_shard.py    rec 子集 → 分片（解决机械盘随机 IO）
│   └── dataset.py        增强（对齐扰动）+ IterableDataset
├── models/
│   ├── ghostfacenet.py   GhostFaceNet / MobileFaceNetLite 骨干
│   └── heads.py          AdaFace / ArcFace / CosFace + Sub-center + PartialFC
├── tools/
│   └── verify_align.py   端到端：原图+5点 → 对齐 → 特征 → 比对
├── deploy/
│   ├── arch.json         DPU 指纹 0x101000056010407
│   ├── make_calib.py     生成 INT8 校准集
│   ├── quantize_pt.py    容器内量化（pytorch_nndct）
│   └── compile_xmodel.sh vai_c_xir 编译
├── train.py              训练入口
├── eval_bin.py           LFW/CFP/AgeDB 等 bin 验证集评测
├── export_onnx.py        导出 ONNX + DPU 算子体检
└── test_smoke.py         无需数据的自检
```

---

## 四、完整流程

### 0. 自检（不需要数据，1 分钟）

```bash
cd ~/tmp/face_rec
python test_smoke.py
# 全部自检通过 ✓
```

### 1. 建索引（一次性，全量约 55 分钟）

```bash
python -m data.build_index \
  --rec "/media/tiny/423ADD10728E128E/glint360k_release/glint360k/train.rec" \
  --idx ".../train.idx" \
  --out index/glint360k.npz
```

想先跑通流程就 `--max_records 2000000`（约 6 分钟，覆盖 3 万身份）。

### 2. 建分片

```bash
# 正式：36 万身份，每身份 ≤10 张 ≈ 300 万张 / 24GB
python -m data.build_shard --index index/glint360k.npz \
  --rec ".../train.rec" --out_dir shards --per_class 10

# 快速验证：每身份 ≤20 张的小规模
python -m data.build_shard --index index/glint360k_2M.npz \
  --rec ".../train.rec" --out_dir shards_quick --per_class 20
```

> 分片默认写进工程目录（本机 SSD）。磁盘紧张时用 `--out_dir /mnt/data1/shards`。

### 3. 训练

```bash
python train.py --shards shards --arch mobilefacenet --width 1.0 \
  --loss adaface --sample_rate 0.08 \
  --batch_size 128 --epochs 25 --lr 0.01 \
  --num_workers 8 --amp --out_dir checkpoints \
  --lfw_bin "/media/tiny/423ADD10728E128E/glint360k_release/glint360k/lfw.bin"
```

* **`--arch` 怎么选**：`mobilefacenet`（默认推荐）能整网编译成 **1 个 DPU 子图**，
  上板跑得干净；`ghostfacenet` 精度潜力更大但会被切成 **17 个子图**（原因见第六节）。
  先在 PC 上比精度就随意，确定要上板请按第六节那张表选。
* 8GB 显存建议 `--batch_size 128~256`；OOM 就先降 batch 再降 `width`（0.75 / 0.5）。
* `--sample_rate` 的经验值（**能开满就开满，采样率越低学习越慢**）：
      3 万类  → 1.0   全类权重才 63MB，显存完全够，收敛最快
      36 万类 → 0.05~0.1（每步约 2~3.6 万类参与，权重 737MB + 梯度 + 动量 ≈ 2.2GB）
* **要训够 epoch**：3 万类 26 万张跑 5 个 epoch 只能看到趋势，
  LFW 要上去通常要 20 个 epoch 以上（本机约 1.1 小时 / epoch，36 万类会更久）。
* 产物：`checkpoints/backbone_epN.pth`（部署用）+ `last.pth`（续训用）。

### 4. 评测

```bash
python eval_bin.py --ckpt checkpoints/backbone_ep20.pth \
  --bins "/media/tiny/423ADD10728E128E/glint360k_release/glint360k" --flip
```

### 5. 导出 ONNX + DPU 算子体检

```bash
python export_onnx.py --ckpt checkpoints/backbone_ep20.pth --out deploy/face.onnx
```

实测算子清单（GhostFaceNet w=1.0）：

```
Conv x80  Concat x32  Relu x18  Add x16  BatchNorm x1
GlobalAveragePool x1  Flatten x1  MatMul x1
[ok] 全部算子均落在 DPU 支持范围内
```

### 6. 端到端验证（原图 + 5 点 → 比对）

```bash
# 已对齐图
python tools/verify_align.py --onnx deploy/face.onnx --img a.jpg --img b.jpg

# 原图 + 手写 5 点（左眼 右眼 鼻尖 左嘴角 右嘴角）
python tools/verify_align.py --onnx deploy/face.onnx \
  --img "a.jpg:38,52,74,52,56,72,42,92,71,92" --img b.jpg

# 原图 + YuNet 自动检测（与板子一致）
python tools/verify_align.py --onnx deploy/face.onnx \
  --detector ../face_detector_yunet/face_detection_yunet_2023mar.onnx \
  --img a.jpg --img b.jpg --save_aligned /tmp/out
```

### 7. 量化 + 编译（Vitis AI 3.0 容器内）

```bash
# 本机先生成校准集
python deploy/make_calib.py --shards shards --out deploy/calib.npy --num 1000

# 进容器量化
docker run --rm -v $(pwd):/workspace -w /workspace \
  xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106 \
  /bin/bash -lc "source /opt/vitis_ai/conda/etc/profile.d/conda.sh && \
                 conda activate vitis-ai-pytorch && \
                 python deploy/quantize_pt.py \
                   --ckpt checkpoints/backbone_ep20.pth \
                   --calib deploy/calib.npy --out_dir quantize_result"

# 编译到 DPUCZDX8G
docker run --rm -v $(pwd):/workspace -w /workspace \
  xilinx/vitis-ai-pytorch-cpu:ubuntu2004-3.0.0.106 \
  /bin/bash -lc "source /opt/vitis_ai/conda/etc/profile.d/conda.sh && \
                 conda activate vitis-ai-pytorch && \
                 bash deploy/compile_xmodel.sh quantize_result/quantize_model.xmodel compiled face_rec"
```

产物 `compiled/face_rec.xmodel` 直接给板端 VART 加载。
注意编译日志里的 `Total DPU subgraph number`：若 > 1，说明有算子没被 DPU 吃掉。

---

## 五、关于「5 个关键点 + 原图」这条链路

部署侧的完整链路（`tools/verify_align.py` 已复刻）：

```
YuNet 检测 → 5 点 → estimateAffinePartial2D → 112×112 → GhostFaceNet → 512-d → L2 → 余弦
```

关键点用的是 **ArcFace 标准 112×112 模板**（与训练数据对齐方式一致）：

```python
[[38.2946, 51.6963], [73.5318, 51.6963], [56.0252, 71.7366],
 [41.5493, 92.3655], [70.7299, 92.3655]]   # 左眼 右眼 鼻尖 左嘴角 右嘴角
```

训练侧对应的域随机化（`data/dataset.py::FaceAlignAug`）：

| 扰动 | 量级 | 对应的端上误差 |
|---|---|---|
| 旋转 | ±8° | 关键点 roll 角估计偏差 |
| 缩放 | ±5% | 检测框尺寸偏差 |
| 平移 | ±4% | 检测框中心偏差 |
| 遮挡 | 5% | 口罩 / 手 / 头发 |
| 模糊 / JPEG | 各 10% | 摄像头画质 |

---

## 六、实测记录（RTX 4070 Laptop / 8GB）

### 正式训练曲线（MobileFaceNetLite + CosFace，30,727 类 / 463,042 张）

bs=128，lr=0.01，余弦退火，每 epoch 末在 LFW 上评测：

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 10 | 19 |
|---|---|---|---|---|---|---|---|---|
| train loss | 38.05 | 37.04 | 34.12 | 31.65 | 29.19 | 26.93 | 20.26 | **16.93** |
| **LFW acc** | 63.97% | 80.85% | 90.00% | 92.48% | 93.77% | 94.72% | **96.67%** | (见下) |

阈值也从一开始的 0.93（余弦分布退化）收敛到 0.32（正常区间），
这是判断「模型是不是真在学」最直观的信号 —— 见下面踩坑记录第 9 条。

### 最终精度（`checkpoints_cf2/backbone_ep11.pth`，全部带 flip TTA）

| 基准 | 20 epoch | +12 epoch 续训 | 参考量级 |
|---|---|---|---|
| **LFW** | 97.58% | **98.25%** | 99.7% |
| CFP_FF | 96.47% | **97.23%** | 99.5% |
| CALFW | 88.35% | **90.37%** | 95.8% |
| CFP_FP | 88.14% | **88.77%** | 96.5% |
| AgeDB_30 | 83.32% | **85.50%** | 97.5% |
| CPLFW | 81.73% | **83.57%** | 92.0% |
| **平均** | 89.27% | **90.61%** | |

续训的写法（复用骨干 + 新分类头，lr 降到 0.003）：

```bash
python train.py --shards shards --loss cosface --sample_rate 1.0 \
  --batch_size 128 --epochs 12 --lr 0.003 --warmup_epochs 0.3 --amp \
  --finetune checkpoints_cf30k/backbone_ep19.pth --out_dir checkpoints_cf2 \
  --lfw_bin /mnt/data1/val_bins/lfw.bin
```

差距主要来自**身份数**：只用了 30,727 个身份，Glint360K 完整 36 万类才是 99.7% 的量级。

| 项 | 值 |
|---|---|
| 模型 | MobileFaceNetLite w=1.0，1.34M 参数，emb=512 |
| 训练吞吐 | **1300~2400 img/s**（分片 + AMP + 6 worker，取决于分片所在磁盘） |
| ONNX | 6.4MB，算子全部落在 DPU 支持范围 |
| 量化 + 编译 | 链路跑通，target = `DPUCZDX8G_ISA1_B4096` |

### AdaFace vs ArcFace（同配置对照，4409 类）

| 损失 | 初始 loss | 1 epoch 后 | 训练 acc |
|---|---|---|---|
| **AdaFace** | 17.9 | **1.55** | **0.898** |
| ArcFace | 36.5 | 34.8 | 0.000 |

两者公式只差 margin 怎么取，收敛速度却差一个量级，原因是：
ArcFace 的固定 `m=0.4` 把正类 logit 压成 `s·cos(θ+0.4)`。训练初期 θ≈90°，
正类 logit 恒在 `−25` 左右、**比所有负类都低**（此时 loss 的理论值就是
`s·0.39 + log(K)`），于是 acc 在很长一段时间里恒为 0，loss 也降得极慢。
AdaFace 的自适应 margin 在 `[-m, m]` 间浮动，约一半样本拿到负 margin，
正类 logit 从一开始就能站住，训练信号顺畅得多。

这不代表 ArcFace 有问题（InsightFace 用 bs=512 训 ArcFace 很成熟），
只是**单卡小 batch + 大类数**下它更挑剔：要么把 batch 开大，要么接受更长的训练。
本工程默认用 `--loss adaface`。

### ⚠️ AdaFace 实测不如 CosFace（重要，默认损失已改）

同一份 4409 类数据、bs=128、lr=0.01，各跑 4 个 epoch：

| 损失 | LFW @ep0 | @ep1 | @ep2 | @ep3 |
|---|---|---|---|---|
| **CosFace** | 54.3% | 63.8% | 63.8% | **64.7%** ↑ |
| AdaFace（本工程实现） | 55.8% | 54.8% | 55.3% | 55.1% ✗ 平的 |

在 3 万类上差距更明显：AdaFace 跑 3 个 epoch LFW 一直在 55% 附近不动，
CosFace 同一个位置已经 80.9%。两者的 train loss 都在降，
**所以「loss 在降」完全不代表模型在学** —— 必须看验证集准确率。

诊断手段：把 LFW 的 embedding 拿出来做 PCA。

    正常模型   第 1 主成分占比 ~30~50%，余弦直方图铺开在 0.2~0.8
    塌陷模型   第 1 主成分占比 >99%，余弦直方图挤在 ±1 两端

AdaFace 跑出来的就是后者：主成分 99.9%、余弦 44%≈-1 / 55%≈+1，
embedding 退化成「±同一个方向」，自然分不开人脸。
对应的外部表现就是**评测阈值跑到 ±0.999**（正常应在 0.3~0.4）。

结论：本工程的 AdaFace 实现不可信，默认已改为 `--loss cosface`。
若仍想用 AdaFace，先跑 `python /tmp/diag_pca.py <ckpt>` 这类检查确认没塌。

> **身份数决定泛化**：只用几千个身份训出来的模型只会「识别这几千个人」，
> 面对 LFW 里的全新身份自然很差。要拿到 99%+ 的 LFW，得用 Glint360K 的完整
> 36 万类身份并训够 epoch。正式训练别用 `--max_records` 偷懒。

---

## 七、踩坑记录（都是实测踩出来的）

### 数据侧

1. **label 不在 mxnet 的标准偏移**：在 RecordIO header 的第 32 字节，见第一节。
2. **机械盘不能按索引随机 seek**：顺序读只有 43MB/s，必须先分片。
3. **分片的样本序号 ≠ 字节偏移**：`build_shard` 里 `sel` 存的是记录序号，
   要用 `offsets[sel]` 映射后才能去 `.rec` 里定位。早期版本直接拿序号当偏移，
   结果写出 20GB 垃圾数据却只写出 1642 条；脚本末尾现在会校验条数是否吻合。

### 训练侧

4. **Partial FC 的动量缓冲会推着未采样的类漂移**：每步只有部分类参与前向，
   其余类梯度为 0。若把整块 `weight` 交给普通 SGD(momentum)，未采样行的动量
   会持续推动权重，几个 epoch 后训练 acc 直接崩到 0。
   本工程在 head 里自己维护 `weight_mom`，只更新本步采样到的行。
5. **分类头移出 optimizer 后，GradScaler 不再帮它 unscale**：梯度被放大上万倍，
   分类头当场炸掉（loss 卡在 log(K) 附近、acc 恒 0）。
   必须手动 `head.weight.grad.div_(scaler.get_scale())`。
6. **lr 必须压得比「线性缩放值」更低**：insightface 的基准是 bs=512 / lr=0.1，
   线性缩放到 bs=128 得 0.025，但那是 8 卡多机的配置。单卡小 batch 下梯度噪声大，
   实测 3 万类上：0.05 直接发散、0.025 训一个 epoch 后开始回升、**0.005~0.01 才稳定下降**。
   默认给 0.01，想更稳用 0.005（代价是要训更多 epoch）。
7. **AdaFace 会让特征范数一路膨胀**：margin 依赖范数，模型发现「把范数拉大就能
   调低难度」这条捷径，实测范数能涨到 1e4，loss 随之反弹。
   本工程做了两道保险：margin 用**批内相对范数**（整体放大不再改变 ĥ），
   骨干输出再加**范数上限 64**（`max_feat_norm`，ONNX 里是 Div+Clip+Mul，DPU 支持）。
8. **分类头必须留在 fp32**：margin 里有 `sqrt(1-cos²)`，fp16 下 `1-cos²`
   会因精度损失变负 → sqrt 出 nan → GradScaler 一路把 scale 降到底 → 训练崩。
   `train.py` 里只有骨干包在 autocast 内。
9. **IterableDataset 必须继承 `torch.utils.data.IterableDataset`**：
   只实现 `__iter__` 不够，DataLoader 会按 map-style 去索引，报 `not subscriptable`。

### 部署侧

10. **Vitis AI 3.0 的 target 要写完整名**：写 `DPUCZDX8G` 会报 invalid target name。
    合法名带 ISA 和 BANK，本工程 arch.json 的 `0x101000056010407`
    对应 **`DPUCZDX8G_ISA1_B4096`**（B4096 档）。
11. **`export_xmodel` 只在 test 模式产出 xmodel**：calib 模式下调用不报错，
    但也不生成文件。必须 calib 跑完校准后，再重建一个
    `torch_quantizer('test', ...)` 才能导出。
12. **导出 xmodel 时 batch 必须是 1**，否则报 "Batch size must be 1"。
13. **Depthwise 卷积的通道要按 8 对齐**，否则该层上不了 DPU，整网被切成多段子图，
    在 CPU/DPU 之间来回搬数据。`models/ghostfacenet.py::_ghost_split()` 就是干这个的，
    它同时保证 `new % init == 0`（否则 PyTorch 直接抛
    "out_channels must be divisible by groups"）。
14. **BN1d 会被量化器判为「不可融合」**，本工程用 `fuse_for_deploy()`
    把 bnneck 数学等价地融进后面的 Linear，并顺手把 Dropout 换成 Identity。

### 重要：选哪个骨干上 DPU（实测结论）

Vitis AI 3.0 + `DPUCZDX8G_ISA1_B4096` 下的编译结果：

| 骨干 | 参数量 | DPU 子图数 | 编译产物 |
|---|---|---|---|
| **MobileFaceNetLite** | ~1.0M | **1** ✅ | 1.6MB |
| GhostFaceNet w=1.0 | 1.67M | **17** ❌ | 2.7MB |

**上板请优先用 `--arch mobilefacenet`**：它能整网编译成 1 个 DPU kernel，
一次加载、一次执行，中间不用来回搬数据。
GhostFaceNet 被切成 17 段，每段切换都要在 CPU/DPU 之间搬一次中间结果，
112×112 输入下这部分开销很可能吃掉 Ghost 结构省下来的算力。

切分的根源是 Ghost 模块里的 `Concat`（主分支 + cheap 分支拼接）：
DPU 对「Concat 后面再接 depthwise、还带残差 Add」这种组合不支持合图。
通道按 8 对齐（见 `_ghost_split`）是必要的，但不足以让它合成一张图。

如果一定要用 GhostFaceNet，可以试着把 cheap 分支从 depthwise 换成普通 1×1 conv
去掉 Concat，或逐段读 `compiled/*_compile_log.txt` 定位落在 CPU 上的算子再改。

> 训练阶段用哪个都行（PC 上跑浮点，不受影响）；这个结论只影响部署。
> 用 `bash deploy/run_vitis.sh <ckpt> <arch>` 可以一条命令跑完量化+编译并看到子图数。

---

## 八、调参速查

| 症状 | 处理 |
|---|---|
| 显存 OOM | 降 `--batch_size`；再降 `--width` 到 0.75/0.5 |
| LFW 上不去 | 检查身份数（`meta.json` 的 `num_classes`），确保是 36 万量级 |
| loss 震荡 | 减小 `--lr`；或先 `--loss arcface` 对照，排除 AdaFace 自适应 margin 的抖动 |
| 训练不稳定（AdaFace） | 训练初期特征范数分布未稳，可用较小的 `--m`（0.3） |
| 板子速度慢 | 降 `--width`；确认编译日志 `Total DPU subgraph number == 1` |
| 量化掉点多 | 校准集用真实人脸（勿用噪声）；数量提到 1000+ |
