#!/usr/bin/env python3
"""
Generate a timeline (Gantt‑style) showing when gas‑solid and liquid‑solid interfaces appear.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
from ultralytics import YOLO

GAS_SOLID_ID = 10
LIQUID_SOLID_ID = 13

def state_to_intervals(times, states):
    intervals = []
    start = None
    for i, s in enumerate(states):
        if s == 1 and start is None:
            start = times[i]
        elif s == 0 and start is not None:
            intervals.append((start, times[i]))
            start = None
    if start is not None:
        intervals.append((start, times[-1]))
    return intervals

def main():
    parser = argparse.ArgumentParser(description="Plot timeline of solid interface presence.")
    parser.add_argument("--video", required=True, help="Input video file.")
    parser.add_argument("--model", default="best.pt", help="YOLO model weights.")
    parser.add_argument("--output_dir", default="./plots", help="Directory to save the plot.")
    parser.add_argument("--output_plot", default=None, help="Output plot filename (default: auto-generated).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_plot is None:
        base = os.path.splitext(os.path.basename(args.video))[0]
        out_path = os.path.join(args.output_dir, f"{base}_solid_timeline.png")
    else:
        out_path = args.output_plot

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"Cannot open video: {args.video}"

    fps = cap.get(cv2.CAP_PROP_FPS)
    times = []
    gas_state, liq_state = [], []

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)[0]
        gs_present = 0
        ls_present = 0
        if results.boxes is not None:
            classes = results.boxes.cls.cpu().numpy().astype(int)
            if GAS_SOLID_ID in classes:
                gs_present = 1
            if LIQUID_SOLID_ID in classes:
                ls_present = 1
        times.append(frame_idx / fps)
        gas_state.append(gs_present)
        liq_state.append(ls_present)
        frame_idx += 1

    cap.release()
    times = np.array(times)

    gs_intervals = state_to_intervals(times, gas_state)
    ls_intervals = state_to_intervals(times, liq_state)

    fig, ax = plt.subplots(figsize=(12, 3))
    y_gs, y_ls = 1, 0
    height = 0.35

    for start, end in gs_intervals:
        ax.broken_barh([(start, end-start)], (y_gs - height/2, height), facecolors="#2ca02c", alpha=0.8)
    for start, end in ls_intervals:
        ax.broken_barh([(start, end-start)], (y_ls - height/2, height), facecolors="#ff7f0e", alpha=0.8)

    ax.set_yticks([y_ls, y_gs])
    ax.set_yticklabels(["Liquid–Solid", "Gas–Solid"])
    ax.set_xlabel("Time (s)")
    ax.set_title("Timeline of Solid Precipitation")
    ax.set_xlim(times.min(), times.max())
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"✅ Solid timeline saved to {out_path}")

if __name__ == "__main__":
    main()