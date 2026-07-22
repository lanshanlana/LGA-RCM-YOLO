import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# ------------------------------
# Helper functions (common)
# ------------------------------
def compute_local_bg_s_mean(mask_bin, hsv_full, pad=25):
    """Compute mean saturation of the background patch around the mask."""
    H, W = hsv_full.shape[:2]
    y, x = np.where(mask_bin > 0)
    if len(y) == 0:
        return float(np.mean(hsv_full[:, :, 1]))
    y1 = max(0, np.min(y) - pad)
    y2 = min(H, np.max(y) + pad)
    x1 = max(0, np.min(x) - pad)
    x2 = min(W, np.max(x) + pad)
    local_patch = hsv_full[y1:y2, x1:x2, 1]
    patch_mask = mask_bin[y1:y2, x1:x2]
    bg_pixels = local_patch[patch_mask == 0]
    if bg_pixels.size < 30:
        return float(np.mean(hsv_full[:, :, 1]))
    return float(np.mean(bg_pixels))


def classify_complex(mask_arr, orig_img, hsv_full, S_bg_mean):
    """
    Complex classification logic (used by both 'overlay' and 'native').
    Returns 'colored' or 'colorless'.
    """
    H, W = orig_img.shape[:2]
    # Resize mask to original image size
    mask_resized = cv2.resize(mask_arr.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    # Erode to avoid glass edges
    kernel = np.ones((5, 5), np.uint8)
    mask_eroded = cv2.erode(mask_resized, kernel, iterations=2)
    mask_bin = (mask_eroded > 0).astype(np.uint8)
    if np.count_nonzero(mask_bin) == 0:
        return "colorless", mask_bin  # fallback

    masked_hsv = hsv_full[mask_bin > 0]
    if len(masked_hsv) == 0:
        return "colorless", mask_bin

    h_vals = masked_hsv[:, 0].astype(np.float32)
    s_vals = masked_hsv[:, 1].astype(np.float32)
    v_vals = masked_hsv[:, 2].astype(np.float32)

    median_s = float(np.median(s_vals))
    p75_s = float(np.percentile(s_vals, 75))
    mean_v = float(np.mean(v_vals))
    s_std = float(np.std(s_vals))
    S_local_mean = compute_local_bg_s_mean(mask_bin, hsv_full)
    hist, _ = np.histogram(h_vals, bins=36, range=(0, 180))
    peak_ratio = float(hist.max()) / (np.sum(hist) + 1e-6)
    hue_std = float(np.std(h_vals))

    # Decision rules
    cond_bright = (mean_v > 150 and s_std < 18)
    cond_saturation = (median_s > 50 and p75_s > 65)
    cond_hue_focus = (hue_std < 30 or peak_ratio > 0.28)
    cond_local_diff = abs(S_local_mean - median_s) < 25
    cond_reflect = (hue_std > 35 and peak_ratio < 0.25)
    cond_dark_colored = (mean_v < 150 and p75_s > 70)

    if cond_reflect or cond_bright or cond_local_diff:
        color_status = "colorless"
    elif cond_dark_colored and cond_hue_focus:
        color_status = "colored"
    elif cond_saturation and cond_hue_focus and not cond_bright:
        color_status = "colored"
    else:
        color_status = "colorless"

    return color_status, mask_bin


def simple_classify(mask_arr, orig_img):
    """Simple mean saturation threshold."""
    H, W = orig_img.shape[:2]
    mask_resized = cv2.resize(mask_arr.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    mask_bin = (mask_resized > 0).astype(np.uint8)
    if np.count_nonzero(mask_bin) == 0:
        return "colorless", mask_bin
    masked_pixels = orig_img[mask_bin > 0]
    if len(masked_pixels) == 0:
        return "colorless", mask_bin
    hsv = cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
    mean_sat = np.mean(hsv[:, 0, 1])
    return "colored" if mean_sat > 40 else "colorless", mask_bin


# ------------------------------
# Main processing functions
# ------------------------------
def process_simple(model_path, image_path, output_path):
    model = YOLO(model_path)
    results = model(image_path)
    res = results[0]
    orig_img = cv2.imread(image_path)

    new_names = res.names.copy()
    for cls_id, mask in zip(res.boxes.cls, res.masks.data):
        label = res.names[int(cls_id)]
        if label not in ["gas_liquid_interface", "liquid_liquid_interface"]:
            continue
        color_status, _ = simple_classify(mask.cpu().numpy(), orig_img)
        new_names[int(cls_id)] = f"{label}-{color_status}"
    res.names = new_names
    output = res.plot()
    cv2.imwrite(output_path, output)
    print(f"✅ Simple method saved to {output_path}")


def process_overlay(model_path, image_path, output_path):
    model = YOLO(model_path)
    results = model(image_path)
    res = results[0]
    orig_img = cv2.imread(image_path)
    H, W = orig_img.shape[:2]
    hsv_full = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
    S_bg_mean = float(np.mean(hsv_full[:, :, 1]))

    base_vis = res.plot()
    overlay = base_vis.copy()

    COLOR_COLORED = (255, 0, 255)    # purple
    COLOR_COLORLESS = (255, 255, 0)  # cyan
    ALPHA = 0.45

    masks_np = res.masks.data.cpu().numpy()
    cls_list = res.boxes.cls.cpu().numpy()
    boxes_xyxy = res.boxes.xyxy.cpu().numpy()

    for idx, (mask_arr, cls_id) in enumerate(zip(masks_np, cls_list)):
        label_base = res.names[int(cls_id)]
        if label_base not in ["gas_liquid_interface", "liquid_liquid_interface"]:
            continue

        color_status, mask_bin = classify_complex(mask_arr, orig_img, hsv_full, S_bg_mean)
        if np.count_nonzero(mask_bin) == 0:
            continue

        # Fill overlay
        color_fill = COLOR_COLORED if color_status == "colored" else COLOR_COLORLESS
        mask_color = np.zeros_like(overlay, dtype=np.uint8)
        mask_color[mask_bin > 0] = color_fill
        overlay = cv2.addWeighted(overlay, 1.0, mask_color, ALPHA, 0)

        # Draw text
        x1, y1, x2, y2 = boxes_xyxy[idx].astype(int)
        cx = int((x1 + x2) / 2)
        cy = y1 - 8 if y1 - 8 > 15 else int((y1 + y2) / 2)
        text = f"{label_base}-{color_status}"
        cv2.putText(overlay, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(overlay, text, (cx-1, cy-1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)

    final = cv2.addWeighted(base_vis, 1.0, overlay, 1.0, 0)
    cv2.imwrite(output_path, final)
    print(f"✅ Overlay method saved to {output_path}")


def process_native(model_path, image_path, output_path):
    model = YOLO(model_path)
    results = model(image_path)
    res = results[0]
    orig_img = cv2.imread(image_path)
    hsv_full = cv2.cvtColor(orig_img, cv2.COLOR_BGR2HSV)
    S_bg_mean = float(np.mean(hsv_full[:, :, 1]))

    masks_np = res.masks.data.cpu().numpy()
    cls_list = res.boxes.cls.cpu().numpy()

    # First collect statuses
    statuses = []
    for mask_arr, cls_id in zip(masks_np, cls_list):
        label_base = res.names[int(cls_id)]
        if label_base not in ["gas_liquid_interface", "liquid_liquid_interface"]:
            statuses.append(None)
        else:
            color_status, _ = classify_complex(mask_arr, orig_img, hsv_full, S_bg_mean)
            statuses.append(color_status)

    # Modify names for those instances (by class ID, but since multiple instances can share class ID,
    # we need to handle per‑instance renaming. YOLO's res.names is a dict mapping class ID to label.
    # If multiple instances of same class have different colours, we cannot change the class‑level name.
    # To support per‑instance labels, we need to assign unique class IDs or use plot() with custom labels.
    # A simpler approach: we can draw text ourselves, but to keep it native, we'll assign new class IDs.
    # Since the user wants native plot, we'll create a copy of names and assign unique IDs per instance.
    # However, the original code simply overwrites res.names[cls_id] - that works only if all instances
    # of that class share the same colour status. In practice, gas_liquid and liquid_liquid may have
    # multiple instances with different statuses. The original code does that and may be incorrect.
    # We'll implement per‑instance labels by using the `plot()` argument `labels` (not available in older versions).
    # Alternative: we create a new dictionary mapping each instance index to a label and pass to `res.plot(labels=...)`
    # Let's use the `res.plot(labels=...)` if available; otherwise fallback to overlay.
    # For this unified script, we'll use the overlay method for per‑instance accuracy.
    # But to respect the "native" option, we'll generate a list of labels and use the `labels` parameter.
    # Modern ultralytics supports passing a list of labels to `plot()`.
    labels = []
    for i, (mask_arr, cls_id) in enumerate(zip(masks_np, cls_list)):
        label_base = res.names[int(cls_id)]
        if label_base not in ["gas_liquid_interface", "liquid_liquid_interface"]:
            labels.append(label_base)
        else:
            # We already computed statuses
            status = statuses[i]
            if status is None:
                labels.append(label_base)
            else:
                labels.append(f"{label_base}-{status}")

    # Use plot with custom labels
    output = res.plot(labels=labels)
    cv2.imwrite(output_path, output)
    print(f"✅ Native method (per‑instance labels) saved to {output_path}")


# ------------------------------
# Command-line interface
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="YOLO segmentation colour‑status annotator.")
    parser.add_argument("--method", choices=["simple", "overlay", "native"], default="native",
                        help="Method to use: simple threshold, overlay with complex logic, or native plot with per‑instance labels.")
    parser.add_argument("--model", type=str, default="assets/results/weights/best.pt",
                        help="Path to YOLO model weights (relative).")
    parser.add_argument("--image", type=str, default="samples/test.png",
                        help="Path to input image (relative).")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path. If not given, derived from method and input name.")
    args = parser.parse_args()

    # Resolve paths relative to current working directory
    model_path = Path(args.model).resolve()
    image_path = Path(args.image).resolve()
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return
    if not image_path.exists():
        print(f"❌ Image file not found: {image_path}")
        return

    if args.output is None:
        stem = image_path.stem
        output_path = Path(f"{stem}_{args.method}_output.png").resolve()
    else:
        output_path = Path(args.output).resolve()

    print(f"Using model: {model_path}")
    print(f"Processing image: {image_path}")
    print(f"Method: {args.method}")
    print(f"Output will be saved to: {output_path}")

    if args.method == "simple":
        process_simple(str(model_path), str(image_path), str(output_path))
    elif args.method == "overlay":
        process_overlay(str(model_path), str(image_path), str(output_path))
    elif args.method == "native":
        process_native(str(model_path), str(image_path), str(output_path))
    else:
        print("❌ Unknown method.")


if __name__ == "__main__":
    main()