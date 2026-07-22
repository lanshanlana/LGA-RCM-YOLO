# LGA-RCM-YOLO

**Customized YOLO11 Instance Segmentation for Chemical Laboratory Scenes**

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.3.227-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-AGPL--3.0-green.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.0-red.svg)](https://pytorch.org/)

---

## Overview

LGA-RCM-YOLO is a customized YOLO11m-seg model tailored for **chemical laboratory scene understanding**. It addresses two highly challenging visual tasks simultaneously:

1. **Segmentation of transparent glass reaction vessels** (beakers, flasks, test tubes, etc.)
2. **Detection of multiphase interfaces inside containers** (gas-liquid, liquid-liquid, liquid-solid, gas-solid)

Traditional instance segmentation models struggle with transparent glassware due to refraction, reflection, and the subtle visual boundaries of internal phase interfaces. LGA-RCM-YOLO tackles these challenges through two architectural innovations embedded into the YOLO11 framework.

### Core Innovations

| Module | Description | Location |
|:---|:---|:---|
| **LGA** (Local-Global Attention) | A multi-scale attention module that captures both fine-grained local features and long-range global context, enhancing the backbone's ability to perceive transparent vessel boundaries and thin phase interfaces. | Backbone |
| **RCM** (Rectangular Calibration Module) | A rectangular self-calibrating attention structure designed for the neck network. It explicitly models the elongated, rectangular nature of laboratory glassware and phase interfaces, improving feature alignment across scales. | Neck (C3k2_RCM blocks) |

### Full Pipeline

The project covers the complete workflow from dataset preparation to real-world deployment:

```
Dataset Annotation → Model Training → Custom Evaluation → Color Classification → Liquid-Level Monitoring → Visualization GUI
```

---

## Project Structure

```
LGA-RCM-YOLO/
├── assets/results/              # Training outputs & visualization files
│   └── args.yaml                #   (downloadable from Zenodo, see below)
├── configs/
│   ├── models/                  # YOLO model architecture YAML configs
│   │   ├── yolo11-seg.yaml                  # Baseline YOLO11m-seg
│   │   ├── yolo11-seg-LGA.yaml              # LGA-only variant
│   │   ├── yolo11-seg-c3k2RCM.yaml          # RCM-only variant
│   │   └── yolo11-seg-LGA-c3k2RCM.yaml      # Full LGA + RCM (recommended)
│   └── datasets/
│       └── instrument-seg-new.yaml          # Dataset configuration (30 classes)
├── datasets/CTG2.0/             # CTG 2.0 Dataset (downloadable from Zenodo)
├── docs/                        # Evaluation metric CSV files
├── lga_rcm_yolo/                # Core Python package
│   ├── Attention/
│   │   └── LGA.py               # Local-Global Attention module
│   ├── modules/
│   │   └── block.py             # RCM, C3k2_RCM, C2f_RCM, ASPP modules
│   └── tasks.py                 # YOLO task base class (overrides ultralytics)
├── scripts/                     # Training, evaluation & benchmarking
│   ├── train.py                 # Standard single-stage training
│   ├── train_lga_only.py        # LGA ablation: 3-stage training pipeline
│   ├── train_rcm_only.py        # RCM ablation: 2-stage training pipeline
│   ├── eval_interface.py        # Interface-in-container custom evaluation
│   └── test_fps.py              # FPS & latency benchmarking
├── tools/
│   ├── color_classifier/        # Colored/colorless interface classifier (ResNet18)
│   │   ├── auto_color_binary.py                     # Auto-annotate color flags via HSV
│   │   ├── labelme2yolo_with_colorflag.py           # LabelMe JSON → YOLO seg labels
│   │   ├── train_color_cls.py                       # Train ResNet18 color classifier
│   │   ├── val_color_cls.py                         # Evaluate color classifier
│   │   ├── yolo_interface_color_annotator.py.py     # HSV rule-based color inference
│   │   └── yolo_seg_color_infer.py.py               # YOLO + ResNet joint inference
│   └── process_monitoring/
│       ├── liquid_level_calculator.py               # Liquid-level ratio analysis
│       ├── segmentation_ui.py                       # Interactive segmentation GUI (PyQt5)
│       └── video_preprocess/
│           ├── video_segmentation.py                # Video segmentation → annotated video
│           ├── extract_frames.py                    # Frame extraction (images, masks, CSV)
│           ├── plot_height_diff.py                  # Gas-liquid/liquid-liquid height curves
│           ├── plot_height_diff_corrected.py        # Height curves with drop correction
│           └── plot_solid_timeline.py               # Solid interface occurrence timeline
├── weights/
│   └── yolo11m-seg.pt            # YOLO11m-seg pretrained weights (download from Ultralytics)
├── samples/                      # Example images for quick testing
├── source_init.py                # Inject LGA-RCM modules into ultralytics installation
├── pyproject.toml                # Project metadata & uv dependency configuration
└── uv.lock                       # Locked dependency versions
```

---

## CTG 2.0 Dataset

**CTG 2.0 (Chemical Transparent Glasses 2.0)** is a dedicated laboratory image segmentation dataset containing **30 classes** of glassware and multiphase interfaces:

### Glassware Classes (20 types)
Beaker, burette, centrifuge tube, colorimetric tube, conical flask, cork, crystallizing dish, culture dish, dropper bottle, measuring cup, measuring cylinder, narrow mouth bottle, pear-shaped separatory funnel, pipette, powder funnel with joint, round bottle flask, sand core funnel, screw-top glass bottle, test tube, three-mouth flask, two-necked flask, upper nozzle filtering flask, volumetric flask, wide mouth bottle

### Interface Classes (6 types)
Gas-liquid interface, gas-solid interface, liquid-liquid interface, liquid-solid interface, solid-solid interface

### Other (1 type)
Label

> **Download**: The full annotated dataset with YOLO polygon segmentation labels (train/val/test splits) is available on Zenodo. See [External Resources](#external-zenodo-resources) below.

---

## Installation

### Prerequisites

- **Linux** operating system (installation guide is Linux-specific)
- **Python 3.9**
- **NVIDIA GPU** with CUDA toolkit installed
- **Git** for repository cloning

### Step-by-Step Setup with uv

[uv](https://docs.astral.sh/uv/) is the recommended Python package manager for this project, ensuring reproducible environments.

#### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Restart your terminal after installation, then verify:

```bash
uv --version
```

#### 2. Clone the Repository

```bash
git clone https://github.com/YourName/LGA-RCM-YOLO.git
cd LGA-RCM-YOLO
```

#### 3. Create Virtual Environment (Python 3.9)

```bash
uv venv --python 3.9 .venv
source .venv/bin/activate
```

#### 4. Install Dependencies

```bash
# Full installation (training + GUI + dev tools)
uv sync

# Or with optional dependency groups:
uv sync --extra gui      # Base + PyQt5 visualization
uv sync --extra dev      # Base + code formatting (ruff)
```

#### 5. Inject Custom Modules into Ultralytics

After installation, run the initialization script to replace ultralytics internal files with the modified LGA-RCM versions:

```bash
uv run python source_init.py
```

This copies `LGA.py`, `block.py`, and `tasks.py` into the installed ultralytics package, enabling the custom model architectures.

#### 6. Download Pretrained Weights

Download the official YOLO11m-seg pretrained weights:

```bash
# From Ultralytics:
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt -P weights/
```

#### 7. Verify Installation

```bash
python --version           # Should output Python 3.9.x
python -c "import ultralytics; print(ultralytics.__version__)"  # Should output 8.3.227
```

### FAQ

| Problem | Solution |
|:---|:---|
| `uv: command not found` | Close and reopen your terminal to reload PATH. |
| Python 3.9 not found | Install via system package manager: `apt install python3.9` (Ubuntu). |
| CUDA out of memory | Reduce `--batch` size or `--imgsz` in training commands. |
| Deactivate venv | Run `deactivate`. |

---

## Model Configurations

Four model variants are provided under `configs/models/`:

| Config File | Backbone | Neck | Description |
|:---|:---|:---|:---|
| `yolo11-seg.yaml` | Standard C3k2 + SPPF + C2PSA | Standard C3k2 | Baseline YOLO11m-seg |
| `yolo11-seg-LGA.yaml` | C3k2 + SPPF + **LGA** + C2PSA | Standard C3k2 | LGA-only ablation |
| `yolo11-seg-c3k2RCM.yaml` | Standard C3k2 + SPPF + C2PSA | **C3k2_RCM** | RCM-only ablation |
| `yolo11-seg-LGA-c3k2RCM.yaml` | C3k2 + SPPF + **LGA** + C2PSA | **C3k2_RCM** | **Full LGA-RCM (recommended)** |

All configs target `nc: 30` classes and use the `m` (medium) scale by default.

---

## Usage

### Training

#### Standard Single-Stage Training

The simplest entry point — train the full LGA-RCM model in one pass:

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

All hyperparameters are exposed as command-line arguments. Run `uv run python scripts/train.py --help` for the full list.

#### Ablation: LGA-Only Training (3-Stage Pipeline)

```bash
# Run all three stages sequentially
uv run python scripts/train_lga_only.py --stage all

# Run a specific stage
uv run python scripts/train_lga_only.py --stage 1
uv run python scripts/train_lga_only.py --stage 2
uv run python scripts/train_lga_only.py --stage 3
```

| Stage | Epochs | LR | Key Augmentation |
|:---|:---|:---|:---|
| 1 — Base Training | 300 | 0.0008 | Mosaic, mixup, HSV on |
| 2 — Fine-Tune 1 | 120 | 0.0003 | Mosaic=0.2, HSV off |
| 3 — Fine-Tune 2 | 120 | 0.0003 | Mosaic=0.0, all off |

#### Ablation: RCM-Only Training (2-Stage Pipeline)

```bash
# Run both stages
uv run python scripts/train_rcm_only.py --stage all

# Run individual stages
uv run python scripts/train_rcm_only.py --stage 1
uv run python scripts/train_rcm_only.py --stage 2
```

| Stage | Epochs | LR | Key Augmentation |
|:---|:---|:---|:---|
| 1 — Base Fine-Tuning | 200 | 0.0025 | HSV on, mosaic off |
| 2 — Interface-Optimized | 120 | 6e-5 | HSV off, copy-paste, erasing, dropout |

> Both ablation scripts default to official YOLO11m-seg pretrained weights, ensuring fair and consistent baselines. Performance differences can be attributed solely to architectural modifications.

### Evaluation

#### Standard Validation

Validation is built into the training scripts (use `--val`). For standalone evaluation:

```bash
uv run python scripts/train.py --epochs 0 --val
```

#### Interface-in-Container Custom Evaluation

A specialized evaluation metric that assesses how well the model detects interfaces **within their associated containers**. Each predicted interface is assigned to the ground-truth container with the highest IoU:

```bash
uv run python scripts/eval_interface.py \
    --weights assets/results/weights/best.pt \
    --img_dir datasets/CTG2.0/val \
    --label_dir datasets/CTG2.0/val \
    --output interface_metrics.csv \
    --conf 0.001 \
    --device 0
```

**Output metrics** (per container-interface pair):
- `mAP50` and `mAP50-95` (COCO-style strict IoU matching)
- `Precision`, `Recall`, `F1` at IoU=0.5 (relaxed greedy matching)
- Macro averages (global and per-interface type)

#### FPS Benchmarking

```bash
uv run python scripts/test_fps.py \
    --model assets/results/weights/best.pt \
    --image samples/test.jpg \
    --imgsz 640 \
    --repeat 200 \
    --warmup 20
```

---

## Auxiliary Tools

### Color Classifier (`tools/color_classifier/`)

Distinguishes between **colored** and **colorless** liquid interfaces using a trained ResNet18 binary classifier.

#### Pipeline

```
LabelMe JSON (no color flags)
    ↓  auto_color_binary.py (HSV auto-annotation)
LabelMe JSON (with color flags)
    ↓  labelme2yolo_with_colorflag.py (convert to YOLO format)
YOLO seg labels (with color flag)
    ↓  train_color_cls.py (train ResNet18)
ResNet18 color classifier
    ↓  val_color_cls.py (evaluate accuracy)
Final model checkpoint
```

#### Scripts

| Script | Purpose |
|:---|:---|
| `auto_color_binary.py` | Automatically annotate `colored`/`colorless` flags in LabelMe JSON using HSV saturation rules. **Requires manual verification before training.** |
| `labelme2yolo_with_colorflag.py` | Convert LabelMe JSON annotations (with color flags) to YOLO polygon segmentation `.txt` labels. |
| `train_color_cls.py` | Train a ResNet18 binary classifier on interface patches to predict colored vs. colorless. |
| `val_color_cls.py` | Evaluate the trained color classifier; outputs confusion matrix and misclassified samples. |

#### Inference

| Script | Method | Description |
|:---|:---|:---|
| `yolo_interface_color_annotator.py` | **HSV baseline** | Pure HSV rule-based color judgment (3 modes: `simple`, `overlay`, `native`). No neural network needed. |
| `yolo_seg_color_infer.py` | **YOLO + ResNet (main)** | Joint inference: YOLO segmentation + trained ResNet18 color classifier for robust color prediction. |

```bash
# HSV baseline inference
uv run python tools/color_classifier/yolo_interface_color_annotator.py \
    --method native --image samples/test.jpg

# YOLO + ResNet joint inference (main method)
uv run python tools/color_classifier/yolo_seg_color_infer.py \
    --image-path samples/test.jpg
```

### Process Monitoring (`tools/process_monitoring/`)

#### Liquid-Level Calculator (`liquid_level_calculator.py`)

Computes the liquid fill ratio inside glassware from YOLO segmentation results. Generates safety risk summaries and structured prompts for LLM-based analysis.

```bash
# Single file
uv run python tools/process_monitoring/liquid_level_calculator.py \
    --input outputs/json/sample.json

# Batch folder
uv run python tools/process_monitoring/liquid_level_calculator.py \
    --input outputs/json --output-dir outputs/liquid_analysis
```

#### Segmentation Visualization GUI (`segmentation_ui.py`)

An interactive PyQt5-based GUI for visualizing segmentation results. Supports both headless (batch) and GUI modes:

```bash
# Headless batch processing (recommended for servers)
uv run python tools/process_monitoring/segmentation_ui.py \
    --headless --model assets/results/weights/best.pt \
    --folder /data/images --output /data/results

# Single image (headless)
uv run python tools/process_monitoring/segmentation_ui.py \
    --headless --model best.pt --image test.jpg --output ./out

# GUI mode (requires X Server: ssh -X or local display)
uv run python tools/process_monitoring/segmentation_ui.py
```

#### Video Preprocessing Pipeline (`tools/process_monitoring/video_preprocess/`)

A five-stage pipeline for processing laboratory experiment videos:

| Step | Script | Description |
|:---|:---|:---|
| 1 | `video_segmentation.py` | Run YOLO segmentation on video, output annotated video with masks. |
| 2 | `extract_frames.py` | Extract frames frame-by-frame: original images, visualizations, binary masks, CSV data. |
| 3 | `plot_height_diff.py` | Plot gas-liquid/liquid-liquid height difference curves with separation endpoints. |
| 4 | `plot_solid_timeline.py` | Plot solid-gas/liquid-solid occurrence timeline (Gantt chart). |
| 5 | `plot_height_diff_corrected.py` | Plot corrected height difference curves with drop event compensation. |

---

## External Zenodo Resources

Two independent Zenodo archives support the reproducibility of this work:

### 1. Training Results & Color Classifier Checkpoint

Matches `assets/results/` in the repository.

| Item | Link / Identifier |
|:---|:---|
| **All-version DOI** (latest) | `10.5281/zenodo.21455502` |
| **Fixed v1 snapshot** | `10.5281/zenodo.21455503` |
| **Permanent URL** | [https://zenodo.org/records/21455503](https://zenodo.org/records/21455503) |
| **Release date** | 2026-07-20 |

**Contents**: Training curves, confusion matrices, metric logs, prediction visualizations, and the trained `color_classifier_resnet18.pth` checkpoint.

### 2. CTG 2.0 Dataset

| Item | Link / Identifier |
|:---|:---|
| **All-version DOI** (latest) | `10.5281/zenodo.21451320` |
| **Fixed v1.0 snapshot** | `10.5281/zenodo.21451321` |
| **Permanent URL** | [https://zenodo.org/records/21451321](https://zenodo.org/records/21451321) |
| **Release date** | 2026-07-20 |

**Contents**: Full annotated CTG 2.0 dataset with YOLO polygon segmentation labels (train/val/test), 30 categories.

> **Note**: All-version DOIs always resolve to the newest release. Version-specific record links are recommended for peer review to guarantee unchanged experimental files.

---

## Citation

If you use LGA-RCM-YOLO or the CTG 2.0 dataset in your research, please cite:

### Training Auxiliary Files

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

### CTG 2.0 Dataset

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

## License

This project is licensed under the **AGPL-3.0** License. See the [LICENSE](LICENSE) file for details.

The YOLO11 base model and ultralytics framework are also distributed under AGPL-3.0 by Ultralytics.

---

## Default Path Conventions

| Resource | Default Path |
|:---|:---|
| YOLO segmentation weights | `assets/results/weights/best.pt` |
| Color classifier checkpoint | `tools/color_classifier/color_classifier_resnet18.pth` |
| Pretrained YOLO11m-seg weights | `weights/yolo11m-seg.pt` |
| Test images | `samples/` |
| Dataset | `datasets/CTG2.0/` |
| Model configs | `configs/models/` |
| Dataset configs | `configs/datasets/` |

All paths in scripts are resolved relative to the project root directory.
