"""
video_depth_pipeline.py

Adaptive ROI version — region sizes depend on input video frame resolution.

Place this in the Depth-Anything-V2 repo folder and run:
    python video_depth_pipeline.py

Requirements:
 - checkpoint in checkpoints/
 - torch, opencv, numpy, pandas, tqdm installed
"""
import os
import csv
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# === USER CONFIG ===
VIDEO_PATH = "track.mp4"          # put your video filename here
CHECKPOINT_PATH = "checkpoints/depth_anything_v2_vits.pth"
OUTPUT_DIR = "results"
DEPTH_FRAMES_DIR = "depth_frames"
LUT_PATH = "lut.csv"

# Model / preprocessing
ENCODER = "vits"
MODEL_INPUT_SIZE = 518   # must be multiple of 14 for DINOv2 (518 works)

# Region selection defaults (fractions relative to original video size)
# Defaults chosen to match ~300px top on 1080p and ~600px side on 1920p:
TOP_HEIGHT_FRAC = 300.0 / 1080.0    # ~0.2778
SIDE_WIDTH_FRAC = 600.0 / 1920.0    # ~0.3125

# Detection thresholds
INTENSITY_THRESHOLD = 200    # grayscale > 200 counts as "close"
COUNT_THRESHOLD = 200        # absolute pixel count threshold (in scaled ROI)
FRACTION_THRESHOLD = 0.80    # at least 80% of region pixels must be > INTENSITY_THRESHOLD

# Outputs
ANNOTATE_VIDEO = True
ANNOTATED_VIDEO_PATH = os.path.join(OUTPUT_DIR, "annotated_output.mp4")
SUMMARY_CSV_PATH = os.path.join(OUTPUT_DIR, "summary_by_frame.csv")

# LUT (linear default)
LUT_MAX_DIST = 100.0   # distance at gray 0 (far)
LUT_MIN_DIST = 1.0     # distance at gray 255 (near)
# ===================


def ensure_dir(p: str) -> None:
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def create_linear_lut(path=LUT_PATH, max_dist=LUT_MAX_DIST, min_dist=LUT_MIN_DIST) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        for i in range(256):
            r = i / 255.0
            d = max_dist - (max_dist - min_dist) * r
            w.writerow([i, d])
    print(f"[LUT] Created linear LUT at {path}")


def load_lut(path=LUT_PATH) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"LUT not found: {path}")
    lut = np.zeros(256, dtype=np.float32)
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            i = int(float(row[0])); d = float(row[1])
            if 0 <= i <= 255:
                lut[i] = d
    if np.count_nonzero(lut) < 256:
        nz = np.nonzero(lut)[0]
        if len(nz) >= 2:
            lut = np.interp(np.arange(256), nz, lut[nz]).astype(np.float32)
        else:
            lut = np.linspace(LUT_MAX_DIST, LUT_MIN_DIST, 256).astype(np.float32)
    return lut


def load_depth_anything_model(checkpoint_path=CHECKPOINT_PATH, encoder=ENCODER, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except Exception as e:
        raise ImportError("Cannot import DepthAnythingV2. Edit loader if repo differs.\n" + str(e))
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    }
    if encoder not in model_configs:
        raise ValueError("encoder must be one of " + ", ".join(model_configs.keys()))
    cfg = model_configs[encoder]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = DepthAnythingV2(**cfg)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    print(f"[MODEL] Loaded {checkpoint_path} on {device}")
    return model, device


def preprocess_for_model(bgr_frame: np.ndarray, target_size: int, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return tensor


def model_infer_depth(model, inp_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        out = model(inp_tensor)
    if isinstance(out, (tuple, list)):
        out = out[0]
    depth = out.squeeze().cpu().numpy()
    return depth


def normalize_depth_to_uint8(depth_float: np.ndarray) -> np.ndarray:
    d = depth_float.copy()
    mn, mx = float(np.nanmin(d)), float(np.nanmax(d))
    if mx - mn < 1e-8:
        return np.zeros_like(d, dtype=np.uint8)
    return ((d - mn) / (mx - mn) * 255.0).astype(np.uint8)


def compute_region_sizes_from_orig(orig_w: int, orig_h: int, model_w: int, model_h: int):
    """
    Compute pixel sizes for top and side regions based on original frame resolution,
    then scale those sizes to the model (depth image) resolution.
    Returns: top_h_scaled, side_w_scaled, top_h_orig, side_w_orig
    """
    top_h_orig = max(1, int(round(orig_h * TOP_HEIGHT_FRAC)))
    side_w_orig = max(1, int(round(orig_w * SIDE_WIDTH_FRAC)))

    # scale factors from original to model size
    scale_y = model_h / orig_h
    scale_x = model_w / orig_w

    top_h_scaled = max(1, int(round(top_h_orig * scale_y)))
    side_w_scaled = max(1, int(round(side_w_orig * scale_x)))

    return top_h_scaled, side_w_scaled, top_h_orig, side_w_orig


def compute_counts_and_props(gray_u8: np.ndarray, top_h: int, side_w: int, intensity_thr: int):
    h, w = gray_u8.shape
    top_region = gray_u8[0:top_h, :]
    left_region = gray_u8[:, 0:side_w]
    right_region = gray_u8[:, w - side_w : w]

    top_cnt = int(np.count_nonzero(top_region > intensity_thr))
    left_cnt = int(np.count_nonzero(left_region > intensity_thr))
    right_cnt = int(np.count_nonzero(right_region > intensity_thr))

    top_size = max(1, top_region.size)
    left_size = max(1, left_region.size)
    right_size = max(1, right_region.size)

    top_prop = top_cnt / top_size
    left_prop = left_cnt / left_size
    right_prop = right_cnt / right_size

    return {
        "top_cnt": top_cnt, "left_cnt": left_cnt, "right_cnt": right_cnt,
        "top_size": top_size, "left_size": left_size, "right_size": right_size,
        "top_prop": top_prop, "left_prop": left_prop, "right_prop": right_prop
    }


def classify_from_counts_and_props(top_cnt, left_cnt, right_cnt,
                                   top_prop, left_prop, right_prop,
                                   count_thr=COUNT_THRESHOLD, frac_thr=FRACTION_THRESHOLD) -> str:
    t = (top_cnt > count_thr) and (top_prop >= frac_thr)
    l = (left_cnt > count_thr) and (left_prop >= frac_thr)
    r = (right_cnt > count_thr) and (right_prop >= frac_thr)

    if t and r and l:
        return "NARROW_BRIDGE"
    if t and (not r) and (not l):
        return "OBSTRUCTION_FROM_TOP"
    if (not t) and r and l:
        return "OBSTRUCTION_FROM_LEFT_AND_RIGHT"
    if (not t) and r and (not l):
        return "OBSTRUCTION_FROM_RIGHT"
    if (not t) and (not r) and l:
        return "OBSTRUCTION_FROM_LEFT"
    return "UNDETERMINED"


def process_video(video_path=VIDEO_PATH):
    ensure_dir(OUTPUT_DIR)
    ensure_dir(DEPTH_FRAMES_DIR)

    if not os.path.exists(LUT_PATH):
        create_linear_lut(LUT_PATH, LUT_MAX_DIST, LUT_MIN_DIST)
    lut = load_lut(LUT_PATH)

    model, device = load_depth_anything_model(CHECKPOINT_PATH, ENCODER)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: " + video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"[VIDEO] opened {video_path} fps={fps:.2f} frames={total}")

    writer = None
    results = []
    frame_idx = 0
    pbar = tqdm(total=total or None, desc="Processing frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        orig_h, orig_w = frame.shape[:2]

        # model input is square MODEL_INPUT_SIZE x MODEL_INPUT_SIZE
        inp = preprocess_for_model(frame, MODEL_INPUT_SIZE, device)
        depth_float = model_infer_depth(model, inp)
        depth_gray = normalize_depth_to_uint8(depth_float)

        # Save depth frame
        depth_fname = os.path.join(DEPTH_FRAMES_DIR, f"depth_{frame_idx:06d}.png")
        cv2.imwrite(depth_fname, depth_gray)

        # LUT mapping
        depth_u8 = depth_gray.astype(np.uint8)
        distance_map = lut[depth_u8]

        # compute region sizes scaled from original frame to model depth image
        model_h, model_w = depth_u8.shape
        top_h_scaled, side_w_scaled, top_h_orig, side_w_orig = compute_region_sizes_from_orig(
            orig_w, orig_h, model_w, model_h
        )

        # compute counts/proportions on scaled depth image
        stats = compute_counts_and_props(depth_u8, top_h_scaled, side_w_scaled, INTENSITY_THRESHOLD)
        top_cnt = stats["top_cnt"]; left_cnt = stats["left_cnt"]; right_cnt = stats["right_cnt"]
        top_prop = stats["top_prop"]; left_prop = stats["left_prop"]; right_prop = stats["right_prop"]

        # mean distances in each region (on scaled depth image)
        top_mean = float(np.nanmean(distance_map[0:top_h_scaled, :]))
        left_mean = float(np.nanmean(distance_map[:, 0:side_w_scaled]))
        right_mean = float(np.nanmean(distance_map[:, model_w - side_w_scaled : model_w]))

        classification = classify_from_counts_and_props(
            top_cnt, left_cnt, right_cnt,
            top_prop, left_prop, right_prop,
            COUNT_THRESHOLD, FRACTION_THRESHOLD
        )

        ts = frame_idx / fps if fps > 0 else frame_idx
        row = {
            "frame_index": frame_idx,
            "timestamp_s": ts,
            "depth_frame": depth_fname,
            "orig_frame_w": orig_w,
            "orig_frame_h": orig_h,
            "top_h_orig_px": top_h_orig,
            "side_w_orig_px": side_w_orig,
            "top_h_scaled": top_h_scaled,
            "side_w_scaled": side_w_scaled,
            "top_pixel_count_gt_threshold": top_cnt,
            "left_pixel_count_gt_threshold": left_cnt,
            "right_pixel_count_gt_threshold": right_cnt,
            "top_region_pixels": stats["top_size"],
            "left_region_pixels": stats["left_size"],
            "right_region_pixels": stats["right_size"],
            "top_prop_gt": top_prop,
            "left_prop_gt": left_prop,
            "right_prop_gt": right_prop,
            "classification": classification,
            "top_mean_m": top_mean,
            "left_mean_m": left_mean,
            "right_mean_m": right_mean,
        }
        results.append(row)

        # ANNOTATE
        if ANNOTATE_VIDEO:
            disp = cv2.cvtColor(depth_u8, cv2.COLOR_GRAY2BGR)
            h, w = depth_u8.shape

            # draw top band (scaled)
            cv2.rectangle(disp, (0, 0), (w, top_h_scaled), (0, 255, 0), 2)
            # draw left band (scaled)
            cv2.rectangle(disp, (0, 0), (side_w_scaled, h), (255, 0, 0), 2)
            # draw right band (scaled)
            cv2.rectangle(disp, (w - side_w_scaled, 0), (w, h), (0, 0, 255), 2)

            cv2.putText(disp, f"Frame {frame_idx} t={ts:.2f}s cls={classification}", (10, top_h_scaled + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(disp,
                        f"T>{INTENSITY_THRESHOLD}:{top_cnt} ({top_prop:.2f})  L:{left_cnt} ({left_prop:.2f})  R:{right_cnt} ({right_prop:.2f})",
                        (10, top_h_scaled + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(ANNOTATED_VIDEO_PATH, fourcc, fps, (w, h))
            writer.write(disp)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()

    df = pd.DataFrame(results)
    df.to_csv(SUMMARY_CSV_PATH, index=False)
    print(f"[RESULTS] Saved CSV: {SUMMARY_CSV_PATH} ({len(results)} rows)")
    print(f"[RESULTS] Depth frames saved in: {DEPTH_FRAMES_DIR}")
    if ANNOTATE_VIDEO:
        print(f"[RESULTS] Annotated video saved: {ANNOTATED_VIDEO_PATH}")


if __name__ == "__main__":
    ensure_dir(OUTPUT_DIR)
    ensure_dir(DEPTH_FRAMES_DIR)
    if not os.path.exists(LUT_PATH):
        create_linear_lut(LUT_PATH, LUT_MAX_DIST, LUT_MIN_DIST)
    process_video(VIDEO_PATH)
