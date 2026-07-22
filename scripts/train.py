import argparse
import os
from ultralytics import YOLO
import warnings
warnings.filterwarnings("ignore", message="WARNING ⚠️ no model scale passed.")

def main():
    parser = argparse.ArgumentParser(description="Training script for LGA-RCM-YOLO instance segmentation")

    # Path arguments
    parser.add_argument("--model-cfg", type=str, default="configs/models/yolo11-seg-LGA-c3k2RCM.yaml",
                        help="relative path to model yaml config")
    parser.add_argument("--data-yaml", type=str, default="configs/datasets/instrument-seg-new.yaml",
                        help="relative path to dataset yaml")
    parser.add_argument("--pretrain", type=str, default="weights/yolo11m-seg.pt",
                        help="pretrained weight file path")

    # Training basic hyperparameters
    parser.add_argument("--device", type=int, default=0, help="cuda device index")
    parser.add_argument("--epochs", type=int, default=200, help="total training epochs")
    parser.add_argument("--imgsz", type=int, default=768, help="input resolution, must be multiple of 32")
    parser.add_argument("--batch", type=int, default=8, help="training batch size")
    parser.add_argument("--workers", type=int, default=8, help="dataloader worker count")

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="AdamW", help="optimizer type")
    parser.add_argument("--lr0", type=float, default=4e-4, help="initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.05, help="final learning rate factor")
    parser.add_argument("--weight-decay", type=float, default=0.012, help="weight decay")

    # Warm-up
    parser.add_argument("--warmup-epochs", type=int, default=3, help="warmup epochs")
    parser.add_argument("--warmup-bias-lr", type=float, default=0.05, help="warmup bias lr")

    # Augmentation
    parser.add_argument("--mosaic", type=float, default=0.0, help="mosaic augmentation probability")
    parser.add_argument("--mixup", type=float, default=0.0, help="mixup augmentation probability")
    parser.add_argument("--copy-paste", type=float, default=0.002, help="copy-paste augmentation")
    parser.add_argument("--erasing", type=float, default=0.01, help="random erasing probability")

    parser.add_argument("--hsv-h", type=float, default=0.0, help="HSV hue augmentation")
    parser.add_argument("--hsv-s", type=float, default=0.0, help="HSV saturation augmentation")
    parser.add_argument("--hsv-v", type=float, default=0.0, help="HSV value augmentation")

    parser.add_argument("--scale", type=float, default=0.10, help="scale augmentation")
    parser.add_argument("--translate", type=float, default=0.02, help="translate augmentation")
    parser.add_argument("--fliplr", type=float, default=0.12, help="horizontal flip probability")

    # Segmentation & regularization
    parser.add_argument("--dropout", type=float, default=0.015, help="dropout rate")
    parser.add_argument("--mask-ratio", type=int, default=4, help="mask downsample ratio")
    parser.add_argument("--overlap-mask", action="store_true", help="allow overlapping masks")

    # Output control
    parser.add_argument("--project", type=str, default="retry_experiment", help="save root folder")
    parser.add_argument("--name", type=str, default="yolo11m-LGA-RCM-test", help="experiment name")
    parser.add_argument("--val", action="store_true", help="enable validation during training")

    args = parser.parse_args()

    # Build absolute paths
    model_cfg = os.path.join(os.getcwd(), args.model_cfg)
    data_yaml = os.path.join(os.getcwd(), args.data_yaml)
    pretrain_path = os.path.join(os.getcwd(), args.pretrain)

    # File existence check
    if not os.path.exists(model_cfg):
        raise FileNotFoundError(f"Model config not found: {model_cfg}")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")
    if not os.path.exists(pretrain_path):
        raise FileNotFoundError(f"Pretrained weights not found: {pretrain_path}")

    model = YOLO(model_cfg)
    model.load(pretrain_path)

    model.train(
        data=data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        val=args.val,

        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,

        warmup_epochs=args.warmup_epochs,
        warmup_bias_lr=args.warmup_bias_lr,
        amp=True,
        dropout=args.dropout,

        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        erasing=args.erasing,

        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        scale=args.scale,
        translate=args.translate,
        fliplr=args.fliplr,

        overlap_mask=args.overlap_mask,
        mask_ratio=args.mask_ratio,

        project=args.project,
        name=args.name,
    )

    metrics = model.val()
    print("\n========== Final Validation ==========")
    print(f"Box mAP50-95 = {metrics.box.map:.4f}")
    print(f"Seg mAP50-95 = {metrics.seg.map:.4f}")


if __name__ == "__main__":
    main()
