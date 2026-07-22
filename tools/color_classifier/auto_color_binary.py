# auto_annotate_color_flag.py
import argparse
import os
import json
import numpy as np
import cv2
from pathlib import Path

TARGET_CLASSES = {"gas_liquid_interface", "liquid_liquid_interface"}


def polygon_to_mask(points, img_h, img_w):
    """Convert polygon points to boolean mask."""
    poly = np.array(points, dtype=np.int32)
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    return mask.astype(bool)


def judge_colored_or_colorless(img_bgr, mask) -> str:
    """Classify region as colored / colorless / unknown using HSV saturation."""
    masked_pixels = img_bgr[mask]
    if len(masked_pixels) == 0:
        return "unknown"

    hsv = cv2.cvtColor(masked_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    saturation = hsv[:, 1].astype(float)
    value = hsv[:, 2].astype(float)

    # Empirical thresholds
    if np.mean(saturation) < 30 or np.median(value) > 240 or np.median(value) < 15:
        return "colorless"
    return "colored"


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    json_file_list = [f for f in os.listdir(args.json_dir) if f.lower().endswith(".json")]

    for json_filename in json_file_list:
        json_path = os.path.join(args.json_dir, json_filename)
        with open(json_path, "r", encoding="utf-8") as f:
            labelme_data = json.load(f)

        image_filename = os.path.basename(labelme_data.get("imagePath", ""))
        full_image_path = os.path.join(args.image_dir, image_filename)
        if not os.path.exists(full_image_path):
            print(f"[Warning] Image not found: {full_image_path}, skip {json_filename}")
            continue

        image = cv2.imdecode(np.fromfile(full_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        img_h, img_w = image.shape[:2]
        modified_flag = False

        for shape in labelme_data.get("shapes", []):
            label = shape.get("label", "")
            if label not in TARGET_CLASSES:
                continue
            point_list = shape.get("points", [])
            if not point_list:
                continue

            mask = polygon_to_mask(point_list, img_h, img_w)
            color_tag = judge_colored_or_colorless(image, mask)

            if shape.get("flags", {}).get("color") != color_tag:
                shape.setdefault("flags", {})["color"] = color_tag
                modified_flag = True

        output_json_path = os.path.join(args.output_dir, json_filename)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(labelme_data, f, ensure_ascii=False, indent=2)

        if modified_flag:
            print(f"[Updated] {json_filename} -> color flag assigned")

    print("\nProcessing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Automatically add color flag (colored/colorless) to LabelMe JSON shapes for interface annotations."
    )
    parser.add_argument("--image-dir", type=str, default=r"datasets/CTG2.0/train/images",
                        help="Directory of source images")
    parser.add_argument("--json-dir", type=str, required=True,
                        help="Directory of original LabelMe JSON files (REQUIRED)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for updated LabelMe JSON with color flags")
    args = parser.parse_args()
    main(args)
