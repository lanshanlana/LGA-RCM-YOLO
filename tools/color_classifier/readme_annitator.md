#!/usr/bin/env python3
"""
Here is a concise and professional **README.md** in English for your script.

---

```markdown
# YOLO Interface Color Annotator

A YOLO-based segmentation tool that detects gas‑liquid and liquid‑liquid interfaces in images and automatically classifies each detected region as **colored** or **colorless** using HSV color analysis. The annotation is appended directly to the detection label (e.g., `gas_liquid_interface-colored`).

---

## Features

- **Three processing modes**:
  - `simple` – fast thresholding using mean saturation.
  - `overlay` – complex classification with local background correction + semi‑transparent mask overlay.
  - `native` – complex classification + per‑instance labels rendered using YOLO’s built‑in plotter.
- **Default relative paths** (no hardcoded absolute paths).
- **Fully parameterized** via command‑line arguments.

---

## Requirements

Install the dependencies:

```bash
pip install ultralytics opencv-python numpy
```

---

## Usage

```bash
python yolo_interface_color_annotator.py [--method MODE] [--model MODEL] [--image IMAGE] [--output OUTPUT]
```

### Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--method` | `native` | Processing mode: `simple`, `overlay`, or `native` |
| `--model` | `assets/results/weights/best.pt` | Path to the YOLO segmentation weights |
| `--image` | `samples/test.png` | Path to the input image |
| `--output` | `{image_stem}_{method}_output.png` | Path for the output annotated image |

---

## Method Details

- **`simple`**  
  Uses the average saturation of the masked region.  
  → Fast but sensitive to lighting.

- **`overlay`**  
  Applies erosion, local background saturation comparison, hue histogram analysis, and multiple statistical thresholds. Draws a colored overlay (purple/cyan) on the image.

- **`native`**  
  Uses the same complex logic as `overlay`, but injects the `-colored` / `-colorless` suffix directly into the label list and renders the final image via `res.plot()` (clean, native YOLO style).

---

## Examples

Run with default settings:

```bash
python yolo_interface_color_annotator.py
```

Use a custom model and image with the overlay method:

```bash
python yolo_interface_color_annotator.py --method overlay --model ./my_model.pt --image ./test.jpg
```

Save output to a specific location:

```bash
python yolo_interface_color_annotator.py --method native --output ./results/annotated.png
```

---

## Output

The script saves a single annotated image with all detected interfaces labeled with their color status. For the `overlay` method, colored masks are blended over the original detection boxes.
```
"""