#!/usr/bin/env python3
"""
Evaluate interface segmentation per container.
Assigns predicted interfaces to ground-truth containers, then computes metrics
per (container_class, interface_class) group.
"""

import os
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from ultralytics import YOLO
import torch

# ---------- class mappings ----------
CONTAINER_CLASSES = {
    0: "beaker",
    1: "burette",
    2: "centrifuge_tube",
    3: "colorimetric_tube",
    4: "conical_flask",
    6: "crystallizing_dish",
    7: "culture_dish",
    8: "dropper_bottle",
    14: "measuring_cup",
    15: "measuring_cylinder",
    16: "narrow_mouth_bottle",
    17: "pear_shaped_separatory_funnel",
    18: "pipette",
    19: "powder_funnel_with_joint",
    20: "round_bottle_flask",
    21: "sand_core_funnel",
    22: "screw_top_glass_bottle",
    24: "test_tube",
    25: "three_mouth_flask",
    26: "two_necked_flask",
    27: "upper_nozzle_filtering_flask",
    28: "volumetric_flask",
    29: "wide_mouth_bottle"
}

INTERFACE_CLASSES = {
    9: "gas_liquid_interface",
    10: "gas_solid_interface",
    12: "liquid_liquid_interface",
    13: "liquid_solid_interface"
}

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate interface segmentation with container assignment"
    )
    parser.add_argument("--weights", type=str,
                        default="assets/results/weights/best.pt",
                        help="Path to YOLO model weights (relative to cwd)")
    parser.add_argument("--img_dir", type=str,
                        default="datasets/CTG2.0/val/images/",
                        help="Directory containing validation images")
    parser.add_argument("--label_dir", type=str,
                        default="datasets/CTG2.0/val/labels/",
                        help="Directory containing label .txt files")
    parser.add_argument("--output", type=str,
                        default="interface_metrics_pred2gt_containers.csv",
                        help="Output CSV file name (relative to cwd)")
    parser.add_argument("--conf", type=float, default=0.001,
                        help="Detection confidence threshold (low to keep all)")
    parser.add_argument("--iou_thres", nargs="+", type=float,
                        default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                        help="IoU thresholds for mAP (space-separated)")
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device, e.g. '0' or 'cpu'")
    return parser.parse_args()

# ---------- utility functions ----------
def load_gt(lab_path, shape, img_id):
    H, W = shape
    insts = []
    if not os.path.exists(lab_path):
        return insts
    with open(lab_path) as f:
        for line in f:
            d = line.strip().split()
            if len(d) < 3:
                continue
            cls = int(d[0])
            pts = np.array(d[1:], dtype=np.float32).reshape(-1, 2)
            pts[:, 0] *= W
            pts[:, 1] *= H
            mask = np.zeros((H, W), np.uint8)
            cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
            insts.append({"cls": cls, "mask": mask, "image_id": img_id})
    return insts

def parse_pred(result, shape, img_id):
    H, W = shape
    insts = []
    if result.masks is None:
        return insts
    for cls, conf, poly in zip(
        result.boxes.cls.cpu().numpy().astype(int),
        result.boxes.conf.cpu().numpy(),
        result.masks.xy
    ):
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [np.array(poly, dtype=np.int32)], 1)
        insts.append({
            "cls": int(cls),
            "mask": mask,
            "score": float(conf),
            "image_id": img_id
        })
    return insts

def align(mask, shape):
    if mask.shape[0] == shape[0] and mask.shape[1] == shape[1]:
        return mask > 0
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]),
                      interpolation=cv2.INTER_NEAREST) > 0

def mask_iou(m1, m2, shape, strict=True):
    a = align(m1, shape)
    b = align(m2, shape)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if strict:
        return inter / (union + 1e-6)
    else:
        min_area = min(a.sum(), b.sum())
        return inter / (min_area + 1e-6)

def assign_container(iface, containers):
    best_iou, best_cls = 0.0, None
    for c in containers:
        iou = mask_iou(iface["mask"], c["mask"], iface["mask"].shape, strict=True)
        if iou > best_iou:
            best_iou, best_cls = iou, c["cls"]
    return best_cls

def compute_ap(preds, gts, iou_thr):
    if not gts:
        return 0.0
    preds = sorted(preds, key=lambda x: -x["score"])
    gts_by_img = defaultdict(list)
    for g in gts:
        gts_by_img[g["image_id"]].append({"mask": g["mask"], "matched": False})
    tp, fp = [], []
    for p in preds:
        img_id = p["image_id"]
        best_iou, best_idx = 0.0, -1
        if img_id in gts_by_img:
            for j, g in enumerate(gts_by_img[img_id]):
                if g["matched"]:
                    continue
                iou = mask_iou(p["mask"], g["mask"], p["mask"].shape, strict=True)
                if iou > best_iou:
                    best_iou, best_idx = iou, j
        if best_iou >= iou_thr and best_idx >= 0:
            tp.append(1)
            fp.append(0)
            gts_by_img[img_id][best_idx]["matched"] = True
        else:
            tp.append(0)
            fp.append(1)
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    rec = tp / (len(gts) + 1e-6)
    prec = tp / (tp + fp + 1e-6)
    rec = np.concatenate([[0.0], rec, [1.0]])
    prec = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(prec) - 1, 0, -1):
        prec[i - 1] = max(prec[i - 1], prec[i])
    idx = np.where(rec[1:] != rec[:-1])[0]
    return float(np.sum((rec[idx + 1] - rec[idx]) * prec[idx + 1]))

def compute_prf(preds, gts, iou_thr=0.5):
    if not preds and not gts:
        return 0.0, 0.0, 0.0
    preds = sorted(preds, key=lambda x: -x["score"])
    gts_by_img = defaultdict(list)
    for g in gts:
        gts_by_img[g["image_id"]].append({"mask": g["mask"], "matched": False})
    tp = 0
    for p in preds:
        img_id = p["image_id"]
        best_iou, best_idx = 0.0, -1
        if img_id in gts_by_img:
            for j, g in enumerate(gts_by_img[img_id]):
                if g["matched"]:
                    continue
                iou = mask_iou(p["mask"], g["mask"], p["mask"].shape, strict=False)
                if iou > best_iou:
                    best_iou, best_idx = iou, j
        if best_iou >= iou_thr and best_idx >= 0:
            tp += 1
            gts_by_img[img_id][best_idx]["matched"] = True
    fp = len(preds) - tp
    fn = len(gts) - tp
    P = tp / (tp + fp + 1e-6)
    R = tp / (tp + fn + 1e-6)
    F1 = 2 * P * R / (P + R + 1e-6) if (P + R) > 0 else 0.0
    return P, R, F1

def main():
    args = parse_args()

    # Resolve paths relative to current working directory
    weights_path = os.path.join(os.getcwd(), args.weights)
    img_dir = os.path.join(os.getcwd(), args.img_dir)
    label_dir = os.path.join(os.getcwd(), args.label_dir)
    out_path = os.path.join(os.getcwd(), args.output)

    print(f"Loading model from: {weights_path}")
    model = YOLO(weights_path)

    iou_thres = sorted(args.iou_thres)
    buckets = defaultdict(lambda: {"preds": [], "gts": []})

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(img_files)} images in {img_dir}")

    for img_name in tqdm(img_files, desc="Inference & collecting"):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        img_id = img_name

        # Load ground truth
        label_name = img_name.rsplit('.', 1)[0] + ".txt"
        lab_path = os.path.join(label_dir, label_name)
        gt = load_gt(lab_path, (H, W), img_id)

        # Run inference
        with torch.no_grad():
            result = model(img, conf=args.conf, verbose=False)[0]
        pred = parse_pred(result, (H, W), img_id)

        gt_cont = [g for g in gt if g["cls"] in CONTAINER_CLASSES]
        gt_iface = [g for g in gt if g["cls"] in INTERFACE_CLASSES]
        pred_iface = [p for p in pred if p["cls"] in INTERFACE_CLASSES]

        # Assign GT interfaces to GT containers
        for g in gt_iface:
            c = assign_container(g, gt_cont)
            if c is not None:
                buckets[(c, g["cls"])]["gts"].append({"mask": g["mask"], "image_id": img_id})

        # Assign predicted interfaces to GT containers (key correction)
        for p in pred_iface:
            c = assign_container(p, gt_cont)
            if c is not None:
                buckets[(c, p["cls"])]["preds"].append({
                    "mask": p["mask"],
                    "score": p["score"],
                    "image_id": img_id
                })

    # Compute metrics per bucket
    rows = []
    macro_P, macro_R, macro_F1 = [], [], []
    per_iface = defaultdict(lambda: {"P": [], "R": [], "F1": []})

    for (c, iface), d in buckets.items():
        preds, gts = d["preds"], d["gts"]
        aps = [compute_ap(preds, gts, thr) for thr in iou_thres]
        ap50 = aps[0] if len(aps) > 0 else 0.0
        mAP = np.mean(aps) if len(aps) > 0 else 0.0
        P, R, F1 = compute_prf(preds, gts, 0.5)

        rows.append({
            "container": CONTAINER_CLASSES[c],
            "interface": INTERFACE_CLASSES[iface],
            "Precision": P,
            "Recall": R,
            "F1": F1,
            "mAP50": ap50,
            "mAP50-95": mAP
        })
        macro_P.append(P)
        macro_R.append(R)
        macro_F1.append(F1)
        per_iface[iface]["P"].append(P)
        per_iface[iface]["R"].append(R)
        per_iface[iface]["F1"].append(F1)

    df = pd.DataFrame(rows)

    # Global macro
    df.loc["Macro"] = {
        "container": "Macro",
        "interface": "All",
        "Precision": np.mean(macro_P) if macro_P else 0.0,
        "Recall": np.mean(macro_R) if macro_R else 0.0,
        "F1": np.mean(macro_F1) if macro_F1 else 0.0,
        "mAP50": "",
        "mAP50-95": ""
    }

    # Per‑interface macro
    for iface, m in per_iface.items():
        df.loc[f"Macro_{INTERFACE_CLASSES[iface]}"] = {
            "container": "Macro",
            "interface": INTERFACE_CLASSES[iface],
            "Precision": np.mean(m["P"]),
            "Recall": np.mean(m["R"]),
            "F1": np.mean(m["F1"]),
            "mAP50": "",
            "mAP50-95": ""
        }

    df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
    print(df)

if __name__ == "__main__":
    main()