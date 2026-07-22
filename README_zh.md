# LGA-RCM-YOLO

**面向化学实验室场景的定制化 YOLO11 实例分割模型**

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.3.227-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-red.svg)](https://pytorch.org/)

---

## 项目总述

LGA-RCM-YOLO 是面向**化学实验室场景**的定制化 YOLO11m-seg 改进模型。它同时解决两项高难度视觉任务：

1. **透明玻璃反应容器的分割**（烧杯、烧瓶、试管等）
2. **容器内部多相界面的检测**（气-液、液-液、液-固、气-固等）

传统的实例分割模型在处理透明玻璃器皿时，由于折射、反射以及内部相界面微弱的视觉边界而难以胜任。LGA-RCM-YOLO 通过嵌入 YOLO11 框架的两项架构创新来解决这些挑战。

### 核心创新

| 模块 | 描述 | 嵌入位置 |
|:---|:---|:---|
| **LGA**（局部-全局注意力） | 多尺度注意力模块，同时捕获细粒度局部特征和长距离全局上下文，增强骨干网络对透明容器边界和细小相界面的感知能力。 | 骨干网络（Backbone） |
| **RCM**（矩形自校准模块） | 专为颈部网络设计的矩形自校准注意力结构，显式建模实验室玻璃器皿和相界面的细长矩形特征，提升跨尺度特征对齐。 | 颈部网络（Neck，C3k2_RCM 模块） |

### 完整流程

项目覆盖从数据集准备到实际部署的完整工作流程：

```
数据集标注 → 模型训练 → 定制化评估 → 颜色分类 → 液位实时监测 → 可视化交互 GUI
```

---

## 项目结构

```
LGA-RCM-YOLO/
├── assets/results/              # 训练结果与可视化文件
│   └── args.yaml                #   （需从 Zenodo 下载，见下方说明）
├── configs/
│   ├── models/                  # YOLO 模型结构配置 YAML 文件
│   │   ├── yolo11-seg.yaml                  # 基线 YOLO11m-seg
│   │   ├── yolo11-seg-LGA.yaml              # 仅 LGA 变体（消融实验）
│   │   ├── yoyo11-seg-c3k2RCM.yaml          # 仅 RCM 变体（消融实验）
│   │   └── yolo11-seg-LGA-c3k2RCM.yaml      # 完整 LGA + RCM（推荐）
│   └── datasets/
│       └── instrument-seg-new.yaml          # 数据集配置（30 类）
├── datasets/CTG2.0/             # CTG 2.0 数据集（需从 Zenodo 下载）
├── docs/                        # 评估指标 CSV 文件
├── lga_rcm_yolo/                # 项目核心 Python 包
│   ├── Attention/
│   │   └── LGA.py               # 局部-全局注意力模块
│   ├── modules/
│   │   └── block.py             # RCM、C3k2_RCM、C2f_RCM、ASPP 等模块
│   └── tasks.py                 # YOLO 任务基类（覆盖 ultralytics 官方文件）
├── scripts/                     # 训练、评估与性能基准测试脚本
│   ├── train.py                 # 标准单阶段训练
│   ├── train_lga_only.py        # LGA 消融实验：3 阶段训练流程
│   ├── train_rcm_only.py        # RCM 消融实验：2 阶段训练流程
│   ├── eval_interface.py        # 容器内界面定制化评估
│   └── test_fps.py              # FPS 与推理延迟基准测试
├── tools/
│   ├── color_classifier/        # 有色/无色界面分类器（ResNet18）
│   │   ├── auto_color_binary.py                     # HSV 自动标注颜色标记
│   │   ├── labelme2yolo_with_colorflag.py           # LabelMe JSON → YOLO 分割标签
│   │   ├── train_color_cls.py                       # 训练 ResNet18 颜色分类器
│   │   ├── val_color_cls.py                         # 评估颜色分类器精度
│   │   ├── yolo_interface_color_annotator.py.py     # HSV 规则颜色推理（基线方法）
│   │   └── yolo_seg_color_infer.py.py               # YOLO + ResNet 联合推理（主方法）
│   └── process_monitoring/
│       ├── liquid_level_calculator.py               # 液位比率分析
│       ├── segmentation_ui.py                       # 交互式分割可视化 GUI（PyQt5）
│       └── video_preprocess/
│           ├── video_segmentation.py                # 视频分割 → 输出标注视频
│           ├── extract_frames.py                    # 逐帧拆分（原图、可视化、掩膜、CSV）
│           ├── plot_height_diff.py                  # 气液/液液高度差变化曲线
│           ├── plot_height_diff_corrected.py        # 带液滴修正的高度差曲线
│           └── plot_solid_timeline.py               # 固相界面出现时间轴（甘特图）
├── weights/
│   └── yolo11m-seg.pt            # YOLO11m-seg 预训练权重（从 Ultralytics 下载）
├── samples/                      # 示例图片，用于快速测试
├── source_init.py                # 将 LGA-RCM 模块注入 ultralytics 安装目录
├── pyproject.toml                # 项目元数据与 uv 依赖配置
└── uv.lock                       # 锁定依赖版本
```

---

## CTG 2.0 数据集

**CTG 2.0（Chemical Transparent Glasses 2.0）** 是专为本项目构建的化学实验室图像分割数据集，包含 **30 个类别**的玻璃器皿与多相界面标注：

### 玻璃器皿类（24 种）
烧杯、滴定管、离心管、比色管、锥形瓶、瓶塞、结晶皿、培养皿、滴瓶、量杯、量筒、细口瓶、梨形分液漏斗、移液管、带接头粉末漏斗、圆底烧瓶、砂芯漏斗、螺口玻璃瓶、试管、三口烧瓶、双颈烧瓶、上嘴过滤瓶、容量瓶、广口瓶

### 界面类（5 种）
气-液界面、气-固界面、液-液界面、液-固界面、固-固界面

### 其他（1 种）
标签

> **下载方式**：完整的带 YOLO 多边形分割标签的标注数据集（含 train/val/test 划分）可在 Zenodo 获取。详见下方[外部 Zenodo 资源](#外部-zenodo-资源)。

---

## 环境安装

### 前置条件

- **Linux** 操作系统（安装指南面向 Linux 系统）
- **Python 3.9**
- **NVIDIA GPU** 并已安装 CUDA toolkit
- **Git** 用于克隆仓库

### 使用 uv 逐步安装

[uv](https://docs.astral.sh/uv/) 是本项目推荐的 Python 包管理器，可确保环境可复现。

#### 1. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

安装后**重启终端**，然后验证：

```bash
uv --version
```

#### 2. 克隆仓库

```bash
git clone https://github.com/YourName/LGA-RCM-YOLO.git
cd LGA-RCM-YOLO
```

#### 3. 创建虚拟环境（Python 3.9）

```bash
uv venv --python 3.9 .venv
source .venv/bin/activate
```

#### 4. 安装依赖

```bash
# 完整安装（训练 + GUI + 开发工具）
uv sync

# 或按需安装可选依赖组：
uv sync --extra gui      # 基础依赖 + PyQt5 可视化
uv sync --extra dev      # 基础依赖 + 代码格式化工具（ruff）
```

#### 5. 将自定义模块注入 Ultralytics

安装依赖后，运行初始化脚本，用修改后的 LGA-RCM 版本替换 ultralytics 内部文件：

```bash
uv run python source_init.py
```

此脚本将 `LGA.py`、`block.py` 和 `tasks.py` 复制到已安装的 ultralytics 包中，使自定义模型架构生效。

#### 6. 下载预训练权重

下载官方 YOLO11m-seg 预训练权重：

```bash
# 从 Ultralytics 官方下载：
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt -P weights/
```

#### 7. 验证安装

```bash
python --version           # 应输出 Python 3.9.x
python -c "import ultralytics; print(ultralytics.__version__)"  # 应输出 8.3.227
```

### 常见问题

| 问题 | 解决方法 |
|:---|:---|
| `uv: command not found` | 完全关闭并重新打开终端以重新加载 PATH。 |
| 找不到 Python 3.9 | 通过系统包管理器安装：`apt install python3.9`（Ubuntu）。 |
| CUDA out of memory | 减小 `--batch` 或 `--imgsz` 参数值。 |
| 退出虚拟环境 | 运行 `deactivate`。 |

---

## 模型配置

`configs/models/` 目录下提供四种模型变体：

| 配置文件 | 骨干网络 | 颈部网络 | 说明 |
|:---|:---|:---|:---|
| `yolo11-seg.yaml` | 标准 C3k2 + SPPF + C2PSA | 标准 C3k2 | 基线 YOLO11m-seg |
| `yolo11-seg-LGA.yaml` | C3k2 + SPPF + **LGA** + C2PSA | 标准 C3k2 | 仅 LGA 消融实验 |
| `yolo11-seg-c3k2RCM.yaml` | 标准 C3k2 + SPPF + C2PSA | **C3k2_RCM** | 仅 RCM 消融实验 |
| `yolo11-seg-LGA-c3k2RCM.yaml` | C3k2 + SPPF + **LGA** + C2PSA | **C3k2_RCM** | **完整 LGA-RCM（推荐）** |

所有配置默认使用 `nc: 30` 类别和 `m`（medium）规模。

---

## 使用方法

### 训练

#### 标准单阶段训练

最简单的入口——一次性训练完整的 LGA-RCM 模型：

```bash
uv run python scripts/train.py \
    --model-cfg configs/models/yolo11-seg-LGA-c3k2RCM.yaml \
    --data-yaml configs/datasets/instrument-seg-new.yaml \
    --pretrain weights/yolo11m-seg.pt \
    --epochs 200 \
    --imgsz 768 \
    --batch 8 \
    --lr0 4e-4 \
    --device 0 \
    --project my_experiment \
    --name lga_rcm_run1
```

所有超参数均通过命令行参数暴露。运行 `uv run python scripts/train.py --help` 查看完整列表。

#### 消融实验：LGA-Only 训练（3 阶段流程）

```bash
# 依次运行全部三个阶段
uv run python scripts/train_lga_only.py --stage all

# 运行指定阶段
uv run python scripts/train_lga_only.py --stage 1
uv run python scripts/train_lga_only.py --stage 2
uv run python scripts/train_lga_only.py --stage 3
```

| 阶段 | Epochs | 学习率 | 关键数据增强 |
|:---|:---|:---|:---|
| 1 — 基础训练 | 300 | 0.0008 | Mosaic、mixup、HSV 开启 |
| 2 — 微调一 | 120 | 0.0003 | Mosaic=0.2，HSV 关闭 |
| 3 — 微调二 | 120 | 0.0003 | Mosaic=0.0，全部关闭 |

#### 消融实验：RCM-Only 训练（2 阶段流程）

```bash
# 运行两个阶段
uv run python scripts/train_rcm_only.py --stage all

# 运行单独阶段
uv run python scripts/train_rcm_only.py --stage 1
uv run python scripts/train_rcm_only.py --stage 2
```

| 阶段 | Epochs | 学习率 | 关键数据增强 |
|:---|:---|:---|:---|
| 1 — 基础微调 | 200 | 0.0025 | HSV 开启，mosaic 关闭 |
| 2 — 界面优化 | 120 | 6e-5 | HSV 关闭，copy-paste、erasing、dropout |

> 两个消融脚本默认使用官方 YOLO11m-seg 预训练权重，确保公平一致的基线。任何性能差异均可归因于架构修改本身。

### 评估

#### 标准验证

验证已内置于训练脚本中（使用 `--val` 参数）。独立评估：

```bash
uv run python scripts/train.py --epochs 0 --val
```

#### 容器内界面定制化评估

一种专门设计的评估指标，评估模型在**关联容器内**检测界面的能力。每个预测界面被分配到与之 IoU 最高的真实容器：

```bash
uv run python scripts/eval_interface.py \
    --weights assets/results/weights/best.pt \
    --img_dir datasets/CTG2.0/val \
    --label_dir datasets/CTG2.0/val \
    --output interface_metrics.csv \
    --conf 0.001 \
    --device 0
```

**输出指标**（每个容器-界面对）：
- `mAP50` 和 `mAP50-95`（COCO 风格严格 IoU 匹配）
- `Precision`、`Recall`、`F1`（IoU=0.5 宽松贪心匹配）
- Macro 平均值（全局和按界面类型）

#### FPS 基准测试

```bash
uv run python scripts/test_fps.py \
    --model assets/results/weights/best.pt \
    --image samples/test.jpg \
    --imgsz 640 \
    --repeat 200 \
    --warmup 20
```

---

## 辅助工具

### 颜色分类器（`tools/color_classifier/`）

使用训练好的 ResNet18 二分类模型区分**有色**和**无色**液体界面。

#### 工作流程

```
LabelMe JSON（无 color 标记）
    ↓  auto_color_binary.py（HSV 自动标注）
LabelMe JSON（已带 color 标记）
    ↓  labelme2yolo_with_colorflag.py（转为 YOLO 格式）
YOLO 分割标签（带 color flag）
    ↓  train_color_cls.py（训练 ResNet18）
ResNet18 颜色分类器
    ↓  val_color_cls.py（评估精度）
最终模型权重
```

#### 脚本说明

| 脚本 | 用途 |
|:---|:---|
| `auto_color_binary.py` | 利用 HSV 饱和度规则自动给 LabelMe JSON 批量补充 `colored`/`colorless` 标记。**训练前需人工复核修正。** |
| `labelme2yolo_with_colorflag.py` | 将已带 color 标记的 LabelMe JSON 标注转为 YOLO 多边形分割 `.txt` 标签。 |
| `train_color_cls.py` | 在界面图像块上训练 ResNet18 二分类器，预测有色/无色。 |
| `val_color_cls.py` | 评估训练好的颜色分类器，输出混淆矩阵和误分类样本。 |

#### 推理方式

| 脚本 | 方法 | 描述 |
|:---|:---|:---|
| `yolo_interface_color_annotator.py` | **HSV 基线方法** | 纯 HSV 规则颜色判断（三种模式：`simple`、`overlay`、`native`），无需神经网络。 |
| `yolo_seg_color_infer.py` | **YOLO + ResNet（主方法）** | 联合推理：YOLO 分割 + 训练好的 ResNet18 颜色分类器，实现鲁棒的颜色预测。 |

```bash
# HSV 基线推理
uv run python tools/color_classifier/yolo_interface_color_annotator.py \
    --method native --image samples/test.jpg

# YOLO + ResNet 联合推理（主方法）
uv run python tools/color_classifier/yolo_seg_color_infer.py \
    --image-path samples/test.jpg
```

### 过程监测（`tools/process_monitoring/`）

#### 液位计算器（`liquid_level_calculator.py`）

根据 YOLO 分割结果计算玻璃容器内的液体填充比率，生成安全风险摘要和用于 LLM 分析的结构化提示。

```bash
# 单文件
uv run python tools/process_monitoring/liquid_level_calculator.py \
    --input outputs/json/sample.json

# 批量文件夹
uv run python tools/process_monitoring/liquid_level_calculator.py \
    --input outputs/json --output-dir outputs/liquid_analysis
```

#### 分割可视化 GUI（`segmentation_ui.py`）

基于 PyQt5 的交互式分割可视化界面，支持无头批量模式和 GUI 模式：

```bash
# 无头批量处理（服务器推荐）
uv run python tools/process_monitoring/segmentation_ui.py \
    --headless --model assets/results/weights/best.pt \
    --folder /data/images --output /data/results

# 单张图片（无头模式）
uv run python tools/process_monitoring/segmentation_ui.py \
    --headless --model best.pt --image test.jpg --output ./out

# GUI 模式（需要 X Server：ssh -X 或本地显示）
uv run python tools/process_monitoring/segmentation_ui.py
```

#### 视频预处理流程（`tools/process_monitoring/video_preprocess/`）

处理实验室实验视频的五阶段流程：

| 步骤 | 脚本 | 描述 |
|:---|:---|:---|
| 1 | `video_segmentation.py` | 对视频运行 YOLO 分割，输出带掩膜的标注视频。 |
| 2 | `extract_frames.py` | 逐帧拆分：原始图像、可视化、二值掩膜、CSV 数据。 |
| 3 | `plot_height_diff.py` | 绘制气液/液液高度差变化曲线及分离终点标记。 |
| 4 | `plot_solid_timeline.py` | 绘制固-气/液-固界面出现时间轴（甘特图）。 |
| 5 | `plot_height_diff_corrected.py` | 绘制带液滴事件补偿的高度差修正曲线。 |

---

## 外部 Zenodo 资源

本工作提供两个独立的 Zenodo 存档以支持可复现性：

### 1. 训练结果与颜色分类器权重

对应仓库中的 `assets/results/`。

| 项目 | 链接 / 标识符 |
|:---|:---|
| **全版本 DOI**（始终指向最新版） | `10.5281/zenodo.21455502` |
| **固定 v1 快照** | `10.5281/zenodo.21455503` |
| **永久链接** | [https://zenodo.org/records/21455503](https://zenodo.org/records/21455503) |
| **发布日期** | 2026-07-20 |

**内容**：训练曲线、混淆矩阵、指标日志、预测可视化以及训练好的 `color_classifier_resnet18.pth` 权重。

### 2. CTG 2.0 数据集

| 项目 | 链接 / 标识符 |
|:---|:---|
| **全版本 DOI**（涵盖所有数据集更新） | `10.5281/zenodo.21451320` |
| **固定 v1.0 快照** | `10.5281/zenodo.21451321` |
| **永久链接** | [https://zenodo.org/records/21451321](https://zenodo.org/records/21451321) |
| **发布日期** | 2026-07-20 |

**内容**：完整的带 YOLO 多边形分割标签的 CTG 2.0 标注数据集（train/val/test 划分），30 个类别。

> **注意**：全版本 DOI 始终自动重定向到最新发布版本。同行评审建议使用版本特定的记录链接，以确保实验文件不变。

---

## 引用

如果您在研究中使用 LGA-RCM-YOLO 或 CTG 2.0 数据集，请引用：

### 训练辅助文件

```bibtex
@misc{lgarcm_aux_files,
  title={Supplementary Training Results and ResNet18 Color Classifier Checkpoint for LGA-RCM-YOLO},
  author={Your Full Name},
  year={2026},
  month={7},
  day={20},
  publisher={Zenodo},
  version={v1},
  doi={10.5281/zenodo.21455502},
  url={https://zenodo.org/records/21455503}
}
```

### CTG 2.0 数据集

```bibtex
@misc{ctg2_dataset,
  title={CTG 2.0: Chemical Transparent Glasses Laboratory Image Segmentation Benchmark},
  author={Your Full Name},
  year={2026},
  month={7},
  day={20},
  publisher={Zenodo},
  version={v1.0},
  doi={10.5281/zenodo.21451320},
  url={https://zenodo.org/records/21451321}
}
```

---

## 许可证

本项目采用 **AGPL-3.0** 许可证。详见 [LICENSE](LICENSE) 文件。

YOLO11 基础模型和 ultralytics 框架同样由 Ultralytics 以 AGPL-3.0 协议发布。

---

## 默认路径约定

| 资源 | 默认路径 |
|:---|:---|
| YOLO 分割权重 | `assets/results/weights/best.pt` |
| 颜色分类器权重 | `tools/color_classifier/color_classifier_resnet18.pth` |
| YOLO11m-seg 预训练权重 | `weights/yolo11m-seg.pt` |
| 测试图片 | `samples/` |
| 数据集 | `datasets/CTG2.0/` |
| 模型配置 | `configs/models/` |
| 数据集配置 | `configs/datasets/` |

所有脚本中的路径均相对于项目根目录解析。
