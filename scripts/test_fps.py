import os
import time
import argparse
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="FPS and latency evaluation for YOLO segmentation model")
    parser.add_argument("--model", type=str, default="assets/results/weights/best.pt",
                        help="relative path to model weight file")
    parser.add_argument("--image", type=str, default="samples/test.jpg",
                        help="relative path to test image")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="input image resolution, must be an integer multiple of 32")
    parser.add_argument("--repeat", type=int, default=200, help="inference repeat times")
    parser.add_argument("--warmup", type=int, default=20, help="GPU warm-up rounds")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    args = parser.parse_args()

    model_path = os.path.join(os.getcwd(), args.model)
    img_path = os.path.join(os.getcwd(), args.image)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Test image not found: {img_path}")

    if args.imgsz % 32 != 0:
        raise ValueError(f"imgsz={args.imgsz} is invalid. Input size must be an integer multiple of 32.")

    model = YOLO(model_path)
    device = "cuda"
    model.to(device)

    print("=========== Model Info ===========")
    print("testing fps, please wait...")
    model.info(verbose=False)

    # Warm-up
    for _ in range(args.warmup):
        model(img_path, imgsz=args.imgsz, conf=args.conf, verbose=False)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(args.repeat):
        model(img_path, imgsz=args.imgsz, conf=args.conf, verbose=False)
    torch.cuda.synchronize()
    end = time.time()

    latency = (end - start) / args.repeat
    fps = 1 / latency

    print(f"\nInput size: {args.imgsz}")
    print(f"Average end-to-end latency: {latency * 1000:.2f} ms")
    print(f"Average FPS: {fps:.2f}")


if __name__ == "__main__":
    main()