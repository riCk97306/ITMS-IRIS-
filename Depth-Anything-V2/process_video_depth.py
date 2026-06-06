#!/usr/bin/env python3
"""
process_video_depth.py

Usage (from inside Depth-Anything-V2 repo):
    python process_video_depth.py --video /path/to/input_video.mp4 --encoder vits

What it does:
 1) Extracts frames from the input video into ./frames/
 2) Calls run.py to produce grayscale depth maps from frames -> ./depth_frames/
 3) Loads LUT (lut.csv in repo root or provided path) and converts each frame's grayscale intensities to distances
 4) Splits the top band (top_ratio) into top/left/right and computes counts of pixels > intensity_threshold
 5) Classifies each frame according to your rules and writes a summary CSV and annotated video.

Requirements:
 - You already installed requirements and have the checkpoint in checkpoints/depth_anything_v2_vits.pth
 - run.py is the repo inference script (this script calls it)
"""

import os
import sys
import argparse
import subprocess
import shutil
from glob import glob
import time
import math

import cv2
import numpy as np
import pandas as pd

# ---------- Helpers (reused/adapted from earlier script) -------------
def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def load_lut(path):
    # Accept simple CSV as intensity,distance or single-column distance
    df = pd.read_csv(path, header=None)
    if df.shape[1] == 1:
        lut = df.iloc[:,0].to_numpy(dtype=float)
    else:
        # If two columns, we expect intensity,distance
        lut = np.zeros(256, dtype=float)
        for row in df.itertuples(index=False):
            try:
                i = int(row[0]); d = float(row[1])
                if 0 <= i <= 255:
                    lut[i] = d
            except Exception:
                continue
        # if still zeros (not filled), try interpolate from provided rows
        if np.all(lut == 0):
            # fallback: take second column as distances and linearly map
            distances = df.iloc[:,1].to_numpy(dtype=float)
            lut = np.interp(np.arange(256), np.linspace(0,255,len(distances)), distances)
    if lut.shape[0] != 256:
        # interpolate to length 256
        lut = np.interp(np.arange(256), np.linspace(0,255,len(lut)), lut)
    return lut.astype(float)

def compute_regions(img, top_ratio):
    h, w = img.shape
    top_h = max(1, int(round(h * top_ratio)))
    top_region = img[0:top_h, :]
    mid = w // 2
    left_region = img[0:top_h, 0:mid]
    right_region = img[0:top_h, mid:w]
    return top_region, left_region, right_region

def count_pixels_above_threshold(region, thr):
    return int(np.count_nonzero(region > thr))

def classify_obstruction(top_cnt, left_cnt, right_cnt, count_threshold):
    t = top_cnt > count_threshold
    l = left_cnt > count_threshold
    r = right_cnt > count_threshold
    if t and r and l:
        return "NARROW BRIDGE"
    if t and (not r) and (not l):
        return "OBSTRUCTION_FROM_TOP"
    if (not t) and r and l:
        return "OBSTRUCTION_FROM_LEFT_AND_RIGHT"
    if (not t) and r and (not l):
        return "OBSTRUCTION_FROM_RIGHT"
    if (not t) and (not r) and l:
        return "OBSTRUCTION_FROM_LEFT"
    return "UNDETERMINED"

def map_depths(img_gray, lut):
    # img_gray is uint8
    return lut[img_gray]

# ---------- Main pipeline ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--frames_dir", default="frames", help="Where to save extracted frames")
    parser.add_argument("--depth_frames_dir", default="depth_frames", help="Where DepthAnything will save depth images")
    parser.add_argument("--results_dir", default="results", help="Where to save CSVs and annotated video")
    parser.add_argument("--encoder", default="vits", help="Depth-Anything encoder: vits/vitb/vitl")
    parser.add_argument("--lut", default="lut.csv", help="Path to LUT CSV mapping intensity->distance (0..255)")
    parser.add_argument("--top_ratio", type=float, default=0.20, help="Fraction of image height considered top band")
    parser.add_argument("--intensity_threshold", type=int, default=200, help="Intensity threshold for counting bright pixels")
    parser.add_argument("--count_threshold", type=int, default=200, help="Count threshold for classification")
    parser.add_argument("--frame_step", type=int, default=1, help="Process every Nth frame (1 = all frames)")
    parser.add_argument("--annotate", action="store_true", help="Produce an annotated video in results_dir")
    args = parser.parse_args()

    video_path = args.video
    frames_dir = args.frames_dir
    depth_frames_dir = args.depth_frames_dir
    results_dir = args.results_dir
    encoder = args.encoder
    lut_path = args.lut

    ensure_dir(frames_dir)
    ensure_dir(depth_frames_dir)
    ensure_dir(results_dir)

    # --- 1) Extract frames from video using OpenCV (preserve original resolution) ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("ERROR: Cannot open video:", video_path)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Video opened: {video_path} fps={fps:.2f} total_frames={total_frames}")

    frame_idx = 0
    saved_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.frame_step == 0:
            fname = os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(fname, frame)
            saved_frames += 1
        frame_idx += 1
    cap.release()
    print(f"Extracted {saved_frames} frames to {frames_dir}")

    # --- 2) Run Depth-Anything repo run.py on frames_dir to create grayscale depth frames ---
    # This script calls the repo's run.py which handles model input sizes and preprocessing.
    cmd = [
        sys.executable, "run.py",
        "--encoder", encoder,
        "--img-path", frames_dir,
        "--outdir", depth_frames_dir,
        "--pred-only",
        "--grayscale"
    ]
    print("Running Depth-Anything inference (this may take time)...")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, check=True)   # will raise on error
    print("Depth inference complete. Depth frames written to", depth_frames_dir)

    # --- 3) Load LUT ---
    if not os.path.exists(lut_path):
        print(f"ERROR: LUT not found at {lut_path}. Create lut.csv first (linear mapping or calibration).")
        sys.exit(1)
    lut = load_lut(lut_path)
    print("Loaded LUT:", lut_path)

    # --- 4) Process depth frames: summarize and optionally annotate ---
    depth_images = sorted(glob(os.path.join(depth_frames_dir, "*_depth.png")) + glob(os.path.join(depth_frames_dir, "*.png")))
    if len(depth_images) == 0:
        # try a different pattern
        depth_images = sorted(glob(os.path.join(depth_frames_dir, "*depth*.png")))
    if len(depth_images) == 0:
        print("No depth images found in", depth_frames_dir)
        sys.exit(1)

    summaries = []
    annotated_frames = []
    first_frame_shape = None

    for idx, depth_path in enumerate(depth_images):
        # infer frame number from filename if possible
        fname = os.path.basename(depth_path)
        # attempt to find frame index in name
        num = None
        import re
        m = re.search(r"(\d{4,6})", fname)
        if m:
            num = int(m.group(1))
        frame_number = num if num is not None else idx

        gray = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print("Warning: couldn't read", depth_path)
            continue
        if first_frame_shape is None:
            first_frame_shape = (gray.shape[1], gray.shape[0])  # (w,h)

        top_region, left_region, right_region = compute_regions(gray, args.top_ratio)
        top_cnt = count_pixels_above_threshold(top_region, args.intensity_threshold)
        left_cnt = count_pixels_above_threshold(left_region, args.intensity_threshold)
        right_cnt = count_pixels_above_threshold(right_region, args.intensity_threshold)
        classification = classify_obstruction(top_cnt, left_cnt, right_cnt, args.count_threshold)

        # depth distances (mean/min/max) using LUT
        depth_map = map_depths(gray, lut)
        # region slices
        h, w = gray.shape
        top_h = max(1, int(round(h * args.top_ratio)))
        mid = w // 2
        top_stats = {
            "mean_m": float(np.nanmean(depth_map[0:top_h, :])) if depth_map.size else math.nan,
            "min_m": float(np.nanmin(depth_map[0:top_h, :])) if depth_map.size else math.nan,
            "max_m": float(np.nanmax(depth_map[0:top_h, :])) if depth_map.size else math.nan,
        }
        left_stats = {
            "mean_m": float(np.nanmean(depth_map[0:top_h, 0:mid])),
            "min_m": float(np.nanmin(depth_map[0:top_h, 0:mid])),
            "max_m": float(np.nanmax(depth_map[0:top_h, 0:mid])),
        }
        right_stats = {
            "mean_m": float(np.nanmean(depth_map[0:top_h, mid:w])),
            "min_m": float(np.nanmin(depth_map[0:top_h, mid:w])),
            "max_m": float(np.nanmax(depth_map[0:top_h, mid:w])),
        }

        timestamp_s = frame_number / (fps if fps>0 else 1.0)

        summary = {
            "frame_index": frame_number,
            "timestamp_s": timestamp_s,
            "depth_path": depth_path,
            "top_pixel_count_gt_threshold": top_cnt,
            "left_pixel_count_gt_threshold": left_cnt,
            "right_pixel_count_gt_threshold": right_cnt,
            "classification": classification,
            "top_mean_m": top_stats["mean_m"],
            "top_min_m": top_stats["min_m"],
            "top_max_m": top_stats["max_m"],
            "left_mean_m": left_stats["mean_m"],
            "left_min_m": left_stats["min_m"],
            "left_max_m": left_stats["max_m"],
            "right_mean_m": right_stats["mean_m"],
            "right_min_m": right_stats["min_m"],
            "right_max_m": right_stats["max_m"],
        }
        summaries.append(summary)

        # Optional: build annotated image
        if args.annotate:
            # colorize grayscale for visualization
            disp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            # draw top band rectangle and left/right separators
            cv2.rectangle(disp, (0,0), (w, top_h), (0,255,0), 1)
            cv2.line(disp, (mid,0), (mid, top_h), (0,255,0), 1)
            # overlay text
            text = f"Frame {frame_number} t={timestamp_s:.2f}s cls={classification}"
            cv2.putText(disp, text, (10, top_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            text2 = f"Top>{args.intensity_threshold}:{top_cnt} L:{left_cnt} R:{right_cnt}"
            cv2.putText(disp, text2, (10, top_h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,0), 1)
            annotated_frames.append(disp)

    # --- write summary CSV ---
    df = pd.DataFrame(summaries)
    summary_csv = os.path.join(results_dir, "summary_by_frame.csv")
    df.to_csv(summary_csv, index=False)
    print("Saved summary CSV:", summary_csv)

    # --- optionally make an annotated video ---
    if args.annotate and len(annotated_frames) > 0:
        out_video = os.path.join(results_dir, "annotated_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w, h = first_frame_shape
        writer = cv2.VideoWriter(out_video, fourcc, fps / max(1, args.frame_step), (w,h))
        for f in annotated_frames:
            writer.write(f)
        writer.release()
        print("Saved annotated video:", out_video)

    print("All done. Processed", len(summaries), "frames.")

if __name__ == "__main__":
    main()
