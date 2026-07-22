#!/usr/bin/env python3
"""
Video segmentation using YOLO InstanceSegmentation (ultralytics.solutions).
Saves a new video with overlaid masks and bounding boxes.
"""
import os
import cv2
import argparse
from ultralytics import solutions

def main():
    parser = argparse.ArgumentParser(description="Run YOLO instance segmentation on a video.")
    parser.add_argument("--video", required=True, help="Path to input video file.")
    parser.add_argument("--model", default="best.pt", help="Path to YOLO segmentation model weights.")
    parser.add_argument("--output_dir", default="./output", help="Directory to save the output video.")
    parser.add_argument("--output_video", default=None, help="Output video filename (if not given, auto-generates).")
    parser.add_argument("--show", action="store_true", help="Show processing window (default: False).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_video is None:
        base = os.path.splitext(os.path.basename(args.video))[0]
        out_path = os.path.join(args.output_dir, f"{base}_segmented.mp4")
    else:
        out_path = args.output_video

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"Cannot open video: {args.video}"

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    print(f"Processing video: {args.video}")
    print(f"Output will be saved to: {out_path}")

    segmenter = solutions.InstanceSegmentation(
        show=args.show,
        model=args.model,
        # optionally set classes=[...]
    )

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        result = segmenter(frame)
        writer.write(result.plot_im)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("✅ Segmentation video saved.")

if __name__ == "__main__":
    main()