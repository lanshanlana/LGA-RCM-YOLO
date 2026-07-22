# labelme2yolo_with_colorflag.py
import os
import json
import glob
import cv2
import argparse
import numpy as np

# Class mapping, consistent with your segmentation data.yaml
CLASS_DICT = {
    "beaker": 0, "burette": 1, "centrifuge_tube": 2, "colorimetric_tube": 3,
    "conical_flask": 4, "cork": 5, "crystallizing_dish": 6, "culture_dish": 7,
    "dropper_bottle": 8, "gas_liquid_interface": 9, "gas_solid_interface": 10,
    "label": 11, "liquid_liquid_interface": 12, "liquid_solid_interface": 13,
    "measuring_cup": 14, "measuring_cylinder": 15, "narrow_mouth_bottle": 16,
    "pear_shaped_separatory_funnel": 17, "pipette": 18, "powder_funnel_with_joint": 19,
    "round_bottle_flask": 20, "sand_core_funnel": 21, "screw_top_glass_bottle": 22,
    "solid_solid_interface": 23, "test_tube": 24, "three_mouth_flask": 25,
    "two_necked_flask": 26, "upper_nozzle_filtering_flask": 27, "volumetric_flask": 28,
    "wide_mouth_bottle": 29
}


def convert_polygon(points: list, img_w: int, img_h: int):
    coords_out = []
    for x, y in points:
        x_norm = min(max(x / img_w, 0.0), 1.0)
        y_norm = min(max(y / img_h, 0.0), 1.0)
        coords_out.extend([x_norm, y_norm])
    return coords_out


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    json_path_list = glob.glob(os.path.join(args.json_dir, "*.json"))
    if len(json_path_list) == 0:
        raise FileNotFoundError(f"No json files found in {args.json_dir}")

    for json_file in json_path_list:
        with open(json_file, "r", encoding="utf-8") as f:
            labelme_data = json.load(f)

        image_filename = os.path.basename(labelme_data.get("imagePath", ""))
        full_image_path = os.path.join(args.image_dir, image_filename)
        if not os.path.exists(full_image_path):
            print(f"[Warning] Missing image: {full_image_path}, skip {os.path.basename(json_file)}")
            continue

        image = cv2.imread(full_image_path)
        if image is None:
            print(f"[Warning] Failed to load image: {full_image_path}")
            continue
        img_h, img_w = image.shape[:2]

        label_lines = []
        for shape in labelme_data.get("shapes", []):
            label_text = shape.get("label")
            if label_text not in CLASS_DICT:
                continue
            class_id = CLASS_DICT[label_text]
            if shape.get("shape_type") != "polygon":
                continue
            point_list = shape.get("points", [])
            if len(point_list) < 3:
                continue

            normalized_coords = convert_polygon(point_list, img_w, img_h)

            # Parse color flag from shape flags
            shape_flags = shape.get("flags", {})
            color_tag = shape_flags.get("color", None)

            color_flag = -1
            if isinstance(color_tag, str):
                tag_lower = color_tag.lower()
                if tag_lower == "colored":
                    color_flag = 1
                elif tag_lower == "colorless":
                    color_flag = 0
            elif isinstance(color_tag, (int, float)):
                color_flag = int(color_tag)

            if color_flag == -1:
                print(f"[Warning] shape {label_text} has no valid color flag in {os.path.basename(json_file)}")

            coord_str = " ".join([f"{v:.6f}" for v in normalized_coords])
            line = f"{class_id} {coord_str} {int(color_flag)}"
            label_lines.append(line)

        txt_basename = os.path.splitext(os.path.basename(json_file))[0] + ".txt"
        txt_save_path = os.path.join(args.output_dir, txt_basename)
        with open(txt_save_path, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines))
        print(f"Saved: {txt_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert LabelMe JSON to YOLO segmentation txt with additional color_flag at the end of each line. "
                    "Set shape flag: color = colored / colorless in LabelMe."
    )
    parser.add_argument("--image-dir", type=str, default=r"datasets/CTG2.0/train/images",
                        help="Directory of source images")
    parser.add_argument("--json-dir", type=str, required=True,
                        help="Directory of LabelMe annotation JSON files (REQUIRED)")
    parser.add_argument("--output-dir", type=str, default=r"datasets/CTG2.0/train/labels_with_colorflag",
                        help="Output directory for YOLO txt labels with color flag")
    args = parser.parse_args()
    main(args)
