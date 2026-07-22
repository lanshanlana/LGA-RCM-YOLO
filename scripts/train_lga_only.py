#!/usr/bin/env python3
"""
Three‑stage YOLO11m‑seg LGA training pipeline.

Stages:
  1. Base training (strong augmentation, 300 epochs)
  2. First fine‑tuning (reduced augmentation, 120 epochs)
  3. Second fine‑tuning on new dataset (mosaic fully off, 120 epochs)

All paths are resolved relative to the current working directory.
"""

import os
import argparse
from ultralytics import YOLO


# ----------------------------------------------------------------------
# Argument parser (includes all settings from your snippet)
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='YOLO11m-seg LGA pipeline')

    # ---- Stage selection ----
    parser.add_argument('--stage', type=str, default='all',
                        choices=['1', '2', '3', 'all'],
                        help='Which stage to run: 1, 2, 3, or all (default: all)')

    # ---- Paths (relative to cwd) ----
    parser.add_argument('--model_cfg', type=str,
                        default='configs/models/yolo11m-seg-LGA.yaml',
                        help='Model YAML file (relative to cwd)')
    parser.add_argument('--data_yaml', type=str,
                        default='configs/datasets/instrument-seg-new.yaml',
                        help='Dataset YAML file (relative to cwd)')
    parser.add_argument('--pretrain', type=str,
                        default='weights/yolo11m-seg.pt',
                        help='Baseline weights for stage 1 (relative to cwd)')

    # ---- Training hyperparameters (stage‑specific) ----
    parser.add_argument('--epochs1', type=int, default=300, help='Epochs for stage 1')
    parser.add_argument('--epochs2', type=int, default=120, help='Epochs for stage 2')
    parser.add_argument('--epochs3', type=int, default=120, help='Epochs for stage 3')
    parser.add_argument('--imgsz', type=int, default=768, help='Input image size')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--workers', type=int, default=8, help='Data loader workers')

    # ---- Learning rates ----
    parser.add_argument('--lr0_1', type=float, default=0.0008, help='Initial LR for stage 1')
    parser.add_argument('--lr0_2', type=float, default=0.0003, help='Initial LR for stage 2 & 3')
    parser.add_argument('--lrf', type=float, default=0.05, help='Final LR factor')
    parser.add_argument('--momentum', type=float, default=0.937, help='SGD/AdamW momentum')
    parser.add_argument('--weight_decay_1', type=float, default=0.0006, help='Weight decay stage 1')
    parser.add_argument('--weight_decay_23', type=float, default=0.0004, help='Weight decay stage 2 & 3')
    parser.add_argument('--warmup_epochs1', type=int, default=5, help='Warmup epochs stage 1')
    parser.add_argument('--warmup_epochs23', type=int, default=3, help='Warmup epochs stage 2 & 3')
    parser.add_argument('--warmup_momentum', type=float, default=0.8, help='Warmup momentum')
    parser.add_argument('--warmup_bias_lr', type=float, default=0.05, help='Warmup bias LR')

    # ---- Augmentation (all stages) ----
    parser.add_argument('--mosaic', type=float, default=0.0, help='Mosaic probability')
    parser.add_argument('--mixup', type=float, default=0.0, help='Mixup probability')
    parser.add_argument('--copy_paste', type=float, default=0.002, help='Copy-paste probability')
    parser.add_argument('--erasing', type=float, default=0.01, help='Random erasing probability')
    parser.add_argument('--hsv_h', type=float, default=0.0, help='HSV hue')
    parser.add_argument('--hsv_s', type=float, default=0.0, help='HSV saturation')
    parser.add_argument('--hsv_v', type=float, default=0.0, help='HSV value')
    parser.add_argument('--scale', type=float, default=0.10, help='Scale augmentation')
    parser.add_argument('--translate', type=float, default=0.02, help='Translate augmentation')
    parser.add_argument('--fliplr', type=float, default=0.12, help='Horizontal flip probability')
    parser.add_argument('--flipud', type=float, default=0.0, help='Vertical flip probability (disabled)')
    parser.add_argument('--degrees', type=float, default=0.0, help='Rotation (disabled)')
    parser.add_argument('--shear', type=float, default=0.0, help='Shear (disabled)')
    parser.add_argument('--perspective', type=float, default=0.0, help='Perspective (disabled)')
    parser.add_argument('--close_mosaic_1', type=int, default=30, help='Close mosaic epoch for stage 1')
    parser.add_argument('--close_mosaic_23', type=int, default=0, help='Close mosaic epoch for stage 2')
    # For stage 3 we force mosaic=0, so close_mosaic is irrelevant

    # ---- Segmentation & regularization ----
    parser.add_argument('--dropout', type=float, default=0.015, help='Dropout rate')
    parser.add_argument('--mask_ratio', type=int, default=4, help='Mask downsample ratio')
    parser.add_argument('--overlap_mask', action='store_true', help='Allow overlapping masks')

    # ---- Output control ----
    parser.add_argument('--project', type=str, default='retry_experiment',
                        help='Root project folder (relative to cwd)')
    parser.add_argument('--name1', type=str, default='yolo11m-seg-LGA-1',
                        help='Experiment name for stage 1')
    parser.add_argument('--name2', type=str, default='yolo11m-seg-LGA-finetune',
                        help='Experiment name for stage 2')
    parser.add_argument('--name3', type=str, default='yolo11m-seg-LGA-finetune-new',
                        help='Experiment name for stage 3')
    parser.add_argument('--val', action='store_true', help='Enable validation during training')
    parser.add_argument('--patience1', type=int, default=50, help='Patience for stage 1')
    parser.add_argument('--patience23', type=int, default=40, help='Patience for stages 2 & 3')
    parser.add_argument('--save_period', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--resume', action='store_true', help='Resume training (not used in pipeline)')

    return parser.parse_args()


# ----------------------------------------------------------------------
# Helper to build absolute paths
# ----------------------------------------------------------------------
def get_abs_path(rel_path):
    """Convert a relative path (from cwd) to absolute."""
    return os.path.join(os.getcwd(), rel_path)


# ----------------------------------------------------------------------
# Stage functions
# ----------------------------------------------------------------------
def run_stage1(args):
    """Base training with strong augmentation."""
    model_cfg = get_abs_path(args.model_cfg)
    data_yaml = get_abs_path(args.data_yaml)
    pretrain = get_abs_path(args.pretrain)

    # Instantiate model and set segmentation‑specific parameters
    model = YOLO(model_cfg)
    model.load(pretrain)
    model.overlap_mask = args.overlap_mask
    model.mask_ratio = args.mask_ratio

    model.train(
        data=data_yaml,
        epochs=args.epochs1,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer=args.optimizer,
        lr0=args.lr0_1,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay_1,
        warmup_epochs=args.warmup_epochs1,
        warmup_momentum=args.warmup_momentum,
        warmup_bias_lr=args.warmup_bias_lr,
        close_mosaic=args.close_mosaic_1,
        mosaic=args.mosaic,          # 1.0 by default in stage 1 – but we use arg
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        erasing=args.erasing,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        dropout=args.dropout,
        patience=args.patience1,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name1,
        verbose=True,
        save_period=args.save_period,
        val=args.val,
        resume=args.resume,
    )

    metrics = model.val()
    print(f'[Stage 1] Box mAP@0.5-0.95: {metrics.box.map:.4f}')
    print(f'[Stage 1] Seg mAP@0.5-0.95: {metrics.seg.map:.4f}')

    # Return the path to the best weights for the next stage
    return os.path.join(args.project, args.name1, 'weights', 'best.pt')


def run_stage2(args, pretrained_weight):
    """First fine‑tuning with reduced augmentation."""
    model_cfg = get_abs_path(args.model_cfg)
    data_yaml = get_abs_path(args.data_yaml)

    model = YOLO(model_cfg)
    model.load(pretrained_weight)
    model.overlap_mask = args.overlap_mask
    model.mask_ratio = args.mask_ratio

    model.train(
        data=data_yaml,
        epochs=args.epochs2,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer=args.optimizer,
        lr0=args.lr0_2,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay_23,
        warmup_epochs=args.warmup_epochs23,
        warmup_momentum=args.warmup_momentum,
        warmup_bias_lr=args.warmup_bias_lr,
        close_mosaic=args.close_mosaic_23,   # typically 0
        mosaic=args.mosaic,                  # kept low (e.g. 0.2)
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        erasing=args.erasing,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        dropout=args.dropout,
        patience=args.patience23,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name2,
        verbose=True,
        save_period=args.save_period,
        val=args.val,
        resume=False,  # always fresh start from best
    )

    metrics = model.val()
    print(f'[Stage 2] Box mAP@0.5-0.95: {metrics.box.map:.4f}')
    print(f'[Stage 2] Seg mAP@0.5-0.95: {metrics.seg.map:.4f}')

    return os.path.join(args.project, args.name2, 'weights', 'best.pt')


def run_stage3(args, pretrained_weight):
    """Second fine‑tuning on new dataset; mosaic forced to 0."""
    model_cfg = get_abs_path(args.model_cfg)
    data_yaml = get_abs_path(args.data_yaml)

    model = YOLO(model_cfg)
    model.load(pretrained_weight)
    model.overlap_mask = args.overlap_mask
    model.mask_ratio = args.mask_ratio

    model.train(
        data=data_yaml,
        epochs=args.epochs3,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer=args.optimizer,
        lr0=args.lr0_2,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay_23,
        warmup_epochs=args.warmup_epochs23,
        warmup_momentum=args.warmup_momentum,
        warmup_bias_lr=args.warmup_bias_lr,
        close_mosaic=1,            # completely disable
        mosaic=0.0,                # force zero
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        erasing=args.erasing,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        dropout=args.dropout,
        patience=args.patience23,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name3,
        verbose=True,
        save_period=args.save_period,
        val=args.val,
        resume=False,
    )

    metrics = model.val()
    print(f'[Stage 3] Box mAP@0.5-0.95: {metrics.box.map:.4f}')
    print(f'[Stage 3] Seg mAP@0.5-0.95: {metrics.seg.map:.4f}')

    return os.path.join(args.project, args.name3, 'weights', 'best.pt')


# ----------------------------------------------------------------------
# Main dispatcher
# ----------------------------------------------------------------------
def main():
    args = parse_args()

    # Optionally, you can override default augmentation values for each stage
    # by assigning to args before calling the stage functions. For simplicity,
    # we use the same augmentation parameters for all stages, but you can easily
    # set stage‑specific values via command line (e.g., --mosaic 1.0 for stage1,
    # then --mosaic 0.2 for stage2). The script takes them as given.

    if args.stage in ('1', 'all'):
        print('\n===== Running Stage 1: Base training =====')
        stage1_best = run_stage1(args)
    else:
        stage1_best = None  # not used

    if args.stage in ('2', 'all'):
        print('\n===== Running Stage 2: First fine‑tuning =====')
        if args.stage == 'all':
            pretrained_for_2 = stage1_best
        else:
            # If running only stage 2, we expect the stage 1 weights to exist.
            # The user can also specify a different pretrain path via --pretrain,
            # but here we use the standard location.
            pretrained_for_2 = os.path.join(args.project, args.name1, 'weights', 'best.pt')
        stage2_best = run_stage2(args, pretrained_for_2)
    else:
        stage2_best = None

    if args.stage in ('3', 'all'):
        print('\n===== Running Stage 3: Second fine‑tuning (new dataset) =====')
        if args.stage == 'all':
            pretrained_for_3 = stage2_best
        else:
            # Expect stage 2 weights to exist
            pretrained_for_3 = os.path.join(args.project, args.name2, 'weights', 'best.pt')
        run_stage3(args, pretrained_for_3)

    print('\nAll requested stages completed.')


if __name__ == '__main__':
    main()