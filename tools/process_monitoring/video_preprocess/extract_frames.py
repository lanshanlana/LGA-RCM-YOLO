#!/usr/bin/env python3
"""
Extract every frame, save original image, visualisation, individual masks, and a summary CSV.
"""
import os
import cv2
import csv
import numpy as np
import argparse
from ultralytics import YOLO

CLASS_NAMES = {
    0: "beaker", 1: "burette", 2: "centrifuge_tube", 3: "colorimetric_tube",
    4: "conical_flask", 5: "cork", 6: "crystallizing_dish",
    7: "culture_dish", 8: "dropper_bottle",
    9: "gas_liquid_interface",
    10: "gas_solid_interface",
    11: "label",
    12: "liquid_liquid_interface",
    13: "liquid_solid_interface",
    14: "measuring_cup",
    15: "measuring_cylinder",
    16: "narrow_mouth_bottle",
    17: "pear_shaped_separatory_funnel",
    18: "pipette",
    19: "powder_funnel_with_joint",
    20: "round_bottle_flask",
    21: "sand_core_funnel",
    22: "screw_top_glass_bottle",
    23: "solid_solid_interface",
    24: "test_tube",
    25: "three_mouth_flask",
    26: "two_necked_flask",
    27: "upper_nozzle_filtering_flask",
    28: "volumetric_flask",
    29: "wide_mouth_bottle",
}

def main():
    parser = argparse.ArgumentParser(description="Extract frames and segmentation data from video.")
    parser.add_argument("--video", required=True, help="Input video file.")
    parser.add_argument("--model", default="best.pt", help="YOLO model weights.")
    parser.add_argument("--output_dir", default="./extracted", help="Root output directory.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold.")
    args = parser.parse_args()

    # Create subdirectories
    subdirs = {
        "frames": "frames",
        "vis": "vis",
        "masks": "masks",
        "labels": "labels",
    }
    for d in subdirs.values():
        os.makedirs(os.path.join(args.output_dir, d), exist_ok=True)

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"Cannot open video: {args.video}"

    # CSV summary
    csv_path = os.path.join(args.output_dir, "summary.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_id", "class_id", "class_name", "instance_id", "area_pixels",
                         "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"])

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        frame_name = f"frame_{frame_idx:06d}"

        # Save original frame
        cv2.imwrite(os.path.join(args.output_dir, subdirs["frames"], f"{frame_name}.jpg"), frame)

        # Inference
        results = model(frame, conf=args.conf, iou=args.iou, verbose=False)[0]
        vis_img = results.plot()
        cv2.imwrite(os.path.join(args.output_dir, subdirs["vis"], f"{frame_name}_vis.jpg"), vis_img)

        # If no masks, skip mask saving
        if results.masks is not None:
            mask_dir = os.path.join(args.output_dir, subdirs["masks"], frame_name)
            os.makedirs(mask_dir, exist_ok=True)
            masks = results.masks.data.cpu().numpy()
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy().astype(int)

            for i, (mask, box, cls_id) in enumerate(zip(masks, boxes, classes)):
                class_name = CLASS_NAMES.get(cls_id, "unknown")
                mask_img = (mask * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(mask_dir, f"class_{cls_id}_{class_name}_{i}.png"), mask_img)
                area = int(mask.sum())
                x1, y1, x2, y2 = map(int, box)
                # Append to label file
                label_path = os.path.join(args.output_dir, subdirs["labels"], f"{frame_name}.txt")
                with open(label_path, "a", encoding="utf-8") as f:
                    f.write(f"{cls_id} {class_name} {i} area={area} bbox=({x1},{y1},{x2},{y2})\n")
                csv_writer.writerow([frame_idx, cls_id, class_name, i, area, x1, y1, x2, y2])

    cap.release()
    csv_file.close()
    print(f"✅ Extraction completed. Output in {args.output_dir}")

if __name__ == "__main__":
    main()