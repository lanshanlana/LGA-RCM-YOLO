#!/usr/bin/env python3
"""
Two‑stage training pipeline for YOLO11m‑seg with C3k2_RCM module (no LGA).

Stages:
  1. Base fine‑tuning: from native YOLO11m‑seg weights, 200 epochs, HSV on, mosaic off.
  2. Interface‑optimised fine‑tuning: from stage‑1 best, 120 epochs, HSV off, very low LR.

All paths are resolved relative to the current working directory.
"""

import os
import argparse
import torch
from ultralytics import YOLO


# ----------------------------------------------------------------------
# Argument parser (includes all stage‑specific settings)
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='YOLO11m‑seg C3k2_RCM two‑stage pipeline')

    # ---- Stage selection ----
    parser.add_argument('--stage', type=str, default='all',
                        choices=['1', '2', 'all'],
                        help='Which stage to run: 1, 2, or all (default: all)')

    # ---- Paths (relative to cwd) ----
    parser.add_argument('--model_cfg', type=str,
                        default='configs/models/yolo11-seg-c3k2RCM.yaml',
                        help='Model YAML file (relative to cwd)')
    parser.add_argument('--data_yaml', type=str,
                        default='configs/datasets/instrument-seg-new.yaml',
                        help='Dataset YAML file (relative to cwd)')
    parser.add_argument('--pretrain_stage1', type=str,
                        default='weights/yolo11m-seg.pt',
                        help='Native YOLO11m‑seg weights for stage 1 (relative to cwd)')

    # ---- Training hyperparameters (stage‑specific) ----
    parser.add_argument('--epochs1', type=int, default=200, help='Epochs for stage 1')
    parser.add_argument('--epochs2', type=int, default=120, help='Epochs for stage 2')
    parser.add_argument('--imgsz', type=int, default=768, help='Input image size')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--workers', type=int, default=8, help='Data loader workers')

    # ---- Optimizer & learning rates ----
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer')
    parser.add_argument('--lr0_1', type=float, default=0.0025, help='Initial LR for stage 1')
    parser.add_argument('--lr0_2', type=float, default=6e-5, help='Initial LR for stage 2')
    parser.add_argument('--lrf_1', type=float, default=0.01, help='Final LR factor for stage 1')
    parser.add_argument('--lrf_2', type=float, default=0.02, help='Final LR factor for stage 2')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (ignored by AdamW)')
    parser.add_argument('--weight_decay_1', type=float, default=0.01, help='Weight decay for stage 1')
    parser.add_argument('--weight_decay_2', type=float, default=0.012, help='Weight decay for stage 2')
    parser.add_argument('--warmup_epochs1', type=int, default=3, help='Warmup epochs for stage 1')
    parser.add_argument('--warmup_epochs2', type=int, default=2, help='Warmup epochs for stage 2')
    parser.add_argument('--warmup_bias_lr', type=float, default=0.05, help='Warmup bias LR')

    # ---- Augmentation (stage 1) ----
    parser.add_argument('--mosaic', type=float, default=0.0, help='Mosaic probability (both stages)')
    parser.add_argument('--mixup', type=float, default=0.0, help='Mixup probability (both stages)')
    parser.add_argument('--hsv_h_1', type=float, default=0.015, help='HSV hue for stage 1')
    parser.add_argument('--hsv_s_1', type=float, default=0.4, help='HSV saturation for stage 1')
    parser.add_argument('--hsv_v_1', type=float, default=0.4, help='HSV value for stage 1')
    parser.add_argument('--hsv_h_2', type=float, default=0.0, help='HSV hue for stage 2 (disabled)')
    parser.add_argument('--hsv_s_2', type=float, default=0.0, help='HSV saturation for stage 2')
    parser.add_argument('--hsv_v_2', type=float, default=0.0, help='HSV value for stage 2')
    parser.add_argument('--scale_1', type=float, default=0.15, help='Scale for stage 1')
    parser.add_argument('--scale_2', type=float, default=0.10, help='Scale for stage 2')
    parser.add_argument('--translate_1', type=float, default=0.03, help='Translate for stage 1')
    parser.add_argument('--translate_2', type=float, default=0.02, help='Translate for stage 2')
    parser.add_argument('--fliplr_1', type=float, default=0.2, help='Horizontal flip for stage 1')
    parser.add_argument('--fliplr_2', type=float, default=0.15, help='Horizontal flip for stage 2')
    parser.add_argument('--flipud', type=float, default=0.0, help='Vertical flip (disabled)')
    parser.add_argument('--degrees', type=float, default=0.0, help='Rotation (disabled)')
    parser.add_argument('--shear', type=float, default=0.0, help='Shear (disabled)')
    parser.add_argument('--perspective', type=float, default=0.0, help='Perspective (disabled)')
    parser.add_argument('--close_mosaic', type=int, default=1, help='Epoch to close mosaic (both)')

    # ---- Stage‑2 specific augmentations ----
    parser.add_argument('--copy_paste_2', type=float, default=0.005, help='Copy‑paste for stage 2')
    parser.add_argument('--erasing_2', type=float, default=0.02, help='Random erasing for stage 2')

    # ---- Regularisation ----
    parser.add_argument('--dropout_1', type=float, default=0.0, help='Dropout for stage 1')
    parser.add_argument('--dropout_2', type=float, default=0.012, help='Dropout for stage 2')
    parser.add_argument('--amp', action='store_true', default=True, help='Enable AMP (mixed precision)')

    # ---- Output control ----
    parser.add_argument('--project', type=str, default='The_latest_experiment',
                        help='Root project folder (relative to cwd)')
    parser.add_argument('--name1', type=str, default='yolo11m-seg-C3k2_RCM',
                        help='Experiment name for stage 1')
    parser.add_argument('--name2', type=str, default='yolo11m-seg-C3k2_RCM-newdata',
                        help='Experiment name for stage 2')
    parser.add_argument('--val', action='store_true', default=True, help='Enable validation')
    parser.add_argument('--patience1', type=int, default=40, help='Patience for stage 1')
    parser.add_argument('--patience2', type=int, default=50, help='Patience for stage 2')
    parser.add_argument('--save_period', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--resume', action='store_true', help='Resume training (not used in pipeline)')

    return parser.parse_args()


# ----------------------------------------------------------------------
# Helper to resolve paths relative to cwd
# ----------------------------------------------------------------------
def get_abs_path(rel_path):
    return os.path.join(os.getcwd(), rel_path)


# ----------------------------------------------------------------------
# Stage 1: Base fine‑tuning (HSV on, larger LR)
# ----------------------------------------------------------------------
def run_stage1(args):
    print('\n===== Stage 1: Base fine‑tuning (C3k2_RCM) =====')
    model_cfg = get_abs_path(args.model_cfg)
    data_yaml = get_abs_path(args.data_yaml)
    pretrain = get_abs_path(args.pretrain_stage1)

    # File existence checks
    for p in [model_cfg, data_yaml, pretrain]:
        if not os.path.exists(p):
            raise FileNotFoundError(f'Required file not found: {p}')

    model = YOLO(model_cfg)
    model.load(pretrain)
    print(f'>>> Loaded pretrained weights: {pretrain}')

    model.train(
        data=data_yaml,
        epochs=args.epochs1,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,

        optimizer=args.optimizer,
        lr0=args.lr0_1,
        lrf=args.lrf_1,
        weight_decay=args.weight_decay_1,
        momentum=args.momentum,
        warmup_epochs=args.warmup_epochs1,
        warmup_bias_lr=args.warmup_bias_lr,

        # Augmentation
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        mixup=args.mixup,
        hsv_h=args.hsv_h_1,
        hsv_s=args.hsv_s_1,
        hsv_v=args.hsv_v_1,
        degrees=args.degrees,
        shear=args.shear,
        scale=args.scale_1,
        translate=args.translate_1,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr_1,
        dropout=args.dropout_1,
        # copy_paste and erasing not used in stage1 (default 0)

        amp=args.amp,
        patience=args.patience1,
        save_period=args.save_period,
        val=args.val,
        project=args.project,
        name=args.name1,
        resume=args.resume,
        verbose=True,
    )

    metrics = model.val()
    print(f'[Stage 1] Box mAP@0.5-0.95: {metrics.box.map:.4f}')
    print(f'[Stage 1] Seg mAP@0.5-0.95: {metrics.seg.map:.4f}')

    # Return the best weights path for stage 2
    return os.path.join(args.project, args.name1, 'weights', 'best.pt')


# ----------------------------------------------------------------------
# Stage 2: Interface‑optimised fine‑tuning (HSV off, very low LR)
# ----------------------------------------------------------------------
def run_stage2(args, pretrained_weight):
    print('\n===== Stage 2: Interface‑optimised fine‑tuning =====')
    model_cfg = get_abs_path(args.model_cfg)
    data_yaml = get_abs_path(args.data_yaml)

    if not os.path.exists(pretrained_weight):
        raise FileNotFoundError(f'Stage 1 weights not found: {pretrained_weight}')

    model = YOLO(model_cfg)
    model.load(pretrained_weight)
    print(f'>>> Loaded pretrained weights: {pretrained_weight}')

    # Clear cache to avoid OOM
    torch.cuda.empty_cache()

    model.train(
        data=data_yaml,
        epochs=args.epochs2,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,

        optimizer=args.optimizer,
        lr0=args.lr0_2,
        lrf=args.lrf_2,
        weight_decay=args.weight_decay_2,
        momentum=args.momentum,
        warmup_epochs=args.warmup_epochs2,
        warmup_bias_lr=args.warmup_bias_lr,

        # Augmentation (HSV forced to 0, mosaic off)
        mosaic=0.0,               # explicitly off
        close_mosaic=1,
        mixup=0.0,
        hsv_h=args.hsv_h_2,       # 0.0
        hsv_s=args.hsv_s_2,
        hsv_v=args.hsv_v_2,
        degrees=args.degrees,
        shear=args.shear,
        scale=args.scale_2,
        translate=args.translate_2,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr_2,
        dropout=args.dropout_2,
        copy_paste=args.copy_paste_2,
        erasing=args.erasing_2,

        amp=args.amp,
        patience=args.patience2,
        save_period=args.save_period,
        val=args.val,
        project=args.project,
        name=args.name2,
        resume=False,             # fresh start from best
        verbose=True,
    )

    metrics = model.val()
    print(f'[Stage 2] Box mAP@0.5-0.95: {metrics.box.map:.4f}')
    print(f'[Stage 2] Seg mAP@0.5-0.95: {metrics.seg.map:.4f}')

    return os.path.join(args.project, args.name2, 'weights', 'best.pt')


# ----------------------------------------------------------------------
# Main dispatcher
# ----------------------------------------------------------------------
def main():
    args = parse_args()

    if args.stage in ('1', 'all'):
        stage1_best = run_stage1(args)
    else:
        stage1_best = None

    if args.stage in ('2', 'all'):
        if args.stage == 'all':
            pretrained_for_2 = stage1_best
        else:
            # When running only stage 2, we expect stage 1 weights at default location
            pretrained_for_2 = os.path.join(args.project, args.name1, 'weights', 'best.pt')
        run_stage2(args, pretrained_for_2)

    print('\nAll requested stages completed.')


if __name__ == '__main__':
    main()