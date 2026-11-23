# 数据集说明（软链接入，不复制大文件）

本 baseline 的 `data/` 仅作入口占位，**不搬运大文件**。数据集已随工程迁移到**本盘**（与 `cv_face_baseline/` 同级），通过软链接入；也可在脚本参数里直接传绝对路径。

## 现有数据集（位于本盘 `/mnt/data3`）

| 目录（本盘） | 规模 | 用途 | 状态 |
|---|---|---|---|
| `glint360k_release/glint360k` | 139GB `train.rec` + 8 验证 bin | 识别训练（主，17.4M 图 / 360K 身份） | **已迁移，首选** |
| `WideFace` | 9GB | WIDER FACE 检测集，无关键点 | 已迁移，仅检测可用 |
| `glint360k` (gzip 分卷，旧盘 data2) | ~120GB | 与 `glint360k_release` **同一份数据**，重复 | 旧盘，未迁移，无需处理 |
| `imagenet数据集`（旧盘 data2） | 266GB | ImageNet，**与人脸无关** | 旧盘，未迁移，排除 |

## 已接入（软链，不复制大文件）

| 软链 | 指向 | 用途 |
|---|---|---|
| `glint360k` | `../../glint360k_release/glint360k`（本盘） | 识别训练主数据（130GB `train.rec` + 8 验证 bin） |
| `widerface` | `../../WideFace`（本盘） | 检测（3.2 万张单图，原本就是单图） |
| `glint360k_shards_quick` / `_30k` | `../src/recognition/shards_*`（本盘工程内） | 本地小分片，可冒烟训练，无需外置数据 |

> 形态原则：**识别训练数据保持整文件（RecordIO / .bin 分片），不展开成单张图**；
> 检测数据（WideFace）规模小、本就单图，保持单图。

## 待清理（仅旧盘 data2，本盘无需处理）

以下两份**仅存在于已退役的旧盘 `/mnt/data2`**，本盘 `data3` 未迁移、也无需处理；列此仅供追溯：

| 目录（旧盘） | 大小 | 说明 |
|---|---|---|
| `/mnt/data2/glint360k`（gzip 分卷 `glint360k_00..06`） | ~120G | 与 `glint360k_release` **完全重复** |
| `/mnt/data2/imagenet数据集` | 266G | 与人脸**无关**，不接入 |

> 本盘数据已完整（`glint360k_release` + `WideFace` + 本地分片），训练/测试不依赖旧盘。

## 关键点（来自原工程记录）

- `train.rec` 的标签**不在 MXNet 标准偏移**，而在 RecordIO header 第 32 字节（float32 身份标签）。
  解析见 `src/recognition/data/mrec.py`。
- 机械盘随机 seek 慢，需先 `build_shard` 做分片顺序读。
- 验证 bin：lfw / cfp_ff / cfp_fp / agedb_30 / calfw / cplfw / vgg2_fp。
