#!/usr/bin/env python3
"""
Central-track gauge measurement (MPS accelerated) with terminal GAUGE logging.

Features:
- Multiprocessing for Canny+Hough (faster).
- Main-process smoothing & drawing -> stable blue gauge line.
- Live preview + saved output MP4.
- FPS meter overlay.
- GAUGE lines printed to terminal: timestamp, frame index, gauge_m.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np
import multiprocessing as mp
from flask import Flask, Response, jsonify
import threading

# --- FLASK GLOBALS ---
app = Flask(__name__)
dashboard_lock = threading.Lock()
video_lock = threading.Lock()
latest_data = {"gauge_m": 0.0, "timestamp": 0.0, "frame_idx": 0}

# Create a placeholder image immediately
placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.putText(placeholder, "WAITING FOR STREAM...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
_, ph_buf = cv2.imencode('.jpg', placeholder)
latest_jpeg = ph_buf.tobytes()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rail Gauge Monitor</title>
    <style>
        body { margin: 0; padding: 0; background-color: #121212; color: #00E5FF; font-family: 'Segoe UI', sans-serif; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
        header { background: #1e1e1e; padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #333; }
        h1 { margin: 0; font-size: 1.5rem; letter-spacing: 1px; color: white; }
        .container { flex: 1; display: flex; height: 100%; }
        .video-box { flex: 2; background: #000; display: flex; justify-content: center; align-items: center; position: relative; }
        .video-box img { max-width: 100%; max-height: 100%; }
        .data-panel { flex: 1; background: #181818; padding: 40px; border-left: 1px solid #333; display: flex; flex-direction: column; align-items: center; justify-content: center; }
        .metric-card { background: #222; padding: 20px; border-radius: 12px; width: 80%; text-align: center; margin-bottom: 30px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); border: 1px solid #333; }
        .metric-label { color: #888; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
        .metric-value { font-size: 3.5rem; font-weight: bold; color: #00E5FF; text-shadow: 0 0 20px rgba(0, 229, 255, 0.3); }
        .metric-unit { font-size: 1.2rem; color: #555; }
        .timestamp { font-family: 'Courier New', monospace; color: #aaa; margin-top: 10px; font-size: 1rem; }
    </style>
</head>
<body>
    <header>
        <h1>RAIL GAUGE MONITOR</h1>
        <div style="color: #00E676; font-weight: bold;">● LIVE</div>
    </header>
    <div class="container">
        <div class="video-box">
            <img src="/stream" alt="Live Feed" id="videoFeed">
        </div>
        <div class="data-panel">
            <div class="metric-card">
                <div class="metric-label">Gauge Distance</div>
                <div class="metric-value" id="gaugeVal">--</div>
                <div class="metric-unit">Meters</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Video Time</div>
                <div class="timestamp" id="timeVal">--</div>
                <div class="metric-unit">Minutes</div>
            </div>
        </div>
    </div>
    <script>
        const gEl = document.getElementById('gaugeVal');
        const tEl = document.getElementById('timeVal');
        
        setInterval(() => {
            fetch('/api/gauge').then(r => r.json()).then(d => {
                if (d.gauge_m > 0) {
                    gEl.innerText = d.gauge_m.toFixed(4);
                } else {
                    gEl.innerText = "--";
                }
                tEl.innerText = d.timestamp.toFixed(4);
            });
        }, 100);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/stream')
def video_feed():
    logging.info("Client connected to video stream")
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/gauge')
def api_gauge():
    with dashboard_lock:
        return jsonify(latest_data)

def gen_frames():
    while True:
        with video_lock:
            if latest_jpeg is None:
                time.sleep(0.01)
                continue
            data = latest_jpeg
        
        # Yield frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
        
        # Don't spin too fast, cap stream at ~30 FPS for browser stability
        time.sleep(0.03)

def run_flask():
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)


CALIB_FILE = str(Path(__file__).parent / "calib.json")

# keep OpenCV threads limited to avoid oversubscription
cv2.setUseOptimized(True)
cv2.setNumThreads(1)


# -------------------------------------------------------------------
# Detector (blue gauge line + smoothing)
# -------------------------------------------------------------------
class CentralTrackDetector:
    def __init__(
        self,
        pixels_per_meter: float,
        gauge_m: float,
        roi_y_ratio: float,
        center_x_rel: float,
        window_width_factor: float = 1.5,
        vertical_angle_min_deg: float = 45.0,
        smooth_alpha: float = 0.85,
    ):
        self.ppm = pixels_per_meter
        self.gauge_m = gauge_m
        self.roi_y_ratio = roi_y_ratio
        self.center_x_rel = center_x_rel
        self.window_width_factor = window_width_factor
        self.vertical_angle_min_deg = vertical_angle_min_deg
        self.alpha = smooth_alpha

        # smoothing state
        self.s_left_x: Optional[float] = None
        self.s_right_x: Optional[float] = None

    def process_frame_from_lines(
        self, frame: np.ndarray, lines: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, Optional[float]]:
        h, w = frame.shape[:2]
        roi_y = int(self.roi_y_ratio * h)
        center_x = self.center_x_rel * w
        gauge_px = self.ppm * self.gauge_m
        half_window = 0.5 * self.window_width_factor * gauge_px

        x_min = max(0, int(center_x - half_window))
        x_max = min(w - 1, int(center_x + half_window))

        annotated = frame.copy()

        # Yellow ROI guides
        cv2.line(annotated, (0, roi_y), (w, roi_y), (0, 255, 255), 1)
        cv2.rectangle(annotated, (x_min, 0), (x_max, h), (0, 255, 255), 1)

        if lines is None:
            cv2.putText(
                annotated,
                "No lines detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            return annotated, None

        candidates: List[Tuple[float, Tuple[int, int, int, int]]] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            angle = 90.0 if dx == 0 else float(np.degrees(np.arctan2(dy, dx)))

            if abs(angle) < self.vertical_angle_min_deg:
                continue
            if not (min(y1, y2) <= roi_y <= max(y1, y2)):
                continue
            if dy == 0:
                continue

            t = (roi_y - y1) / float(dy)
            x_at_roi = x1 + t * dx

            if x_min <= x_at_roi <= x_max:
                # show candidate lightly
                cx1, cy1, cx2, cy2 = map(int, (x1, y1, x2, y2))
                cv2.line(annotated, (cx1, cy1), (cx2, cy2), (150, 150, 150), 1)
                candidates.append((x_at_roi, (x1, y1, x2, y2)))

        if len(candidates) < 2:
            cv2.putText(
                annotated,
                "Not enough rails",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            return annotated, None

        expected_px = gauge_px
        best_pair = None
        best_diff = float("inf")
        for i in range(len(candidates)):
            xi, li = candidates[i]
            for j in range(i + 1, len(candidates)):
                xj, lj = candidates[j]
                dist_px = abs(xj - xi)
                diff = abs(dist_px - expected_px)
                if diff < best_diff:
                    best_diff = diff
                    best_pair = ((xi, li), (xj, lj))

        if best_pair is None:
            cv2.putText(
                annotated,
                "Gauge pair not found",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            return annotated, None

        (xL, _), (xR, _) = best_pair
        if xL > xR:
            xL, xR = xR, xL

        # smoothing (main thread, so it’s stable)
        if self.s_left_x is None:
            self.s_left_x = xL
        else:
            self.s_left_x = self.alpha * self.s_left_x + (1 - self.alpha) * xL

        if self.s_right_x is None:
            self.s_right_x = xR
        else:
            self.s_right_x = self.alpha * self.s_right_x + (1 - self.alpha) * xR

        sxL = int(round(self.s_left_x))
        sxR = int(round(self.s_right_x))

        # Blue gauge line + dots
        cv2.line(annotated, (sxL, roi_y), (sxR, roi_y), (255, 0, 0), 3)
        cv2.circle(annotated, (sxL, roi_y), 5, (255, 0, 0), -1)
        cv2.circle(annotated, (sxR, roi_y), 5, (255, 0, 0), -1)

        distance_px = abs(self.s_right_x - self.s_left_x)
        distance_m = distance_px / self.ppm

        cv2.putText(
            annotated,
            f"Gauge distance: {distance_m:.3f} m",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        return annotated, distance_m


# -------------------------------------------------------------------
# Calibration helpers
# -------------------------------------------------------------------
def calibrate_from_frame(frame: np.ndarray, gauge_m: float):
    img = frame.copy()
    points: List[Tuple[int, int]] = []
    h, w = img.shape[:2]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
            cv2.imshow("Calibration", img)

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", on_mouse)
    cv2.imshow("Calibration", img)
    cv2.waitKey(0)
    try:
        cv2.destroyWindow("Calibration")
    except:
        pass

    if len(points) < 2:
        logging.error("Need 2 clicks for calibration")
        return None

    (x1, y1), (x2, y2) = points[:2]
    dist_px = float(np.hypot(x2 - x1, y2 - y1))
    ppm = dist_px / gauge_m

    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)

    calib = {
        "pixels_per_meter": ppm,
        "gauge_m": gauge_m,
        "roi_y_ratio": center_y / h,
        "center_x_rel": center_x / w,
    }

    with open(CALIB_FILE, "w") as f:
        json.dump(calib, f, indent=4)

    logging.info(f"Saved calibration: {calib}")
    return calib


def calibrate_from_image(image_path: str, gauge_m: float):
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"Could not read image {image_path}")
        return None
    return calibrate_from_frame(img, gauge_m)


def load_calibration() -> dict:
    with open(CALIB_FILE, "r") as f:
        return json.load(f)


# -------------------------------------------------------------------
# Worker: heavy Hough part
# -------------------------------------------------------------------
def worker_hough(args):
    idx, frame = args
    cv2.setNumThreads(1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        60,
        minLineLength=60,
        maxLineGap=25,
    )
    return idx, lines


# -------------------------------------------------------------------
# Multiprocess video mode (terminal GAUGE logs + FPS overlay)
# -------------------------------------------------------------------
def mode_video_mps(video_path: str,
                   detector: CentralTrackDetector,
                   workers: int,
                   output_path: Optional[str],
                   no_save: bool = False,
                   max_fps: Optional[float] = None):
    global latest_jpeg

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Could not open video")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_fps = video_fps if 1.0 < video_fps < 200.0 else 25.0

    ui_delay = 1  # minimal delay for UI

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = None
    if not no_save:
        if output_path is None:
            output_path = Path(video_path).stem + "_gauge.mp4"

        out = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_fps,
            (w, h),
        )
        if not out.isOpened():
            logging.error(f"Could not open output {output_path}")
            cap.release()
            return

        logging.info(f"Writing output video: {output_path}")

    pool = mp.Pool(processes=workers)

    inflight_frames = {}
    inflight_results = {}
    next_idx = 0
    next_consume = 0
    max_flight = workers * 4
    finished = False

    fps_est = 0.0
    fps_est = 0.0
    last_t = time.time()
    last_loop_time = time.time()
    target_interval = 1.0 / max_fps if max_fps and max_fps > 0 else 0

    try:
        while True:
            # feed frames
            while not finished and len(inflight_frames) < max_flight:
                ret, frame = cap.read()
                if not ret:
                    finished = True
                    break

                # Skip some frames if we want to simulate "faster" reading or avoid buffer bloat
                # But for smoothing "stuck" behavior, usually simply consuming is enough.
                # However, if the USER means "stuck" as in "pauses to process", we need to pipeline better.
                # The MPS pool is already pipelining. 
                # Let's ensure we don't starve the display loop.
                
                idx = next_idx
                next_idx += 1

                inflight_frames[idx] = frame
                inflight_results[idx] = pool.apply_async(worker_hough, args=((idx, frame),))

            # consume in-order
            while next_consume in inflight_results and inflight_results[next_consume].ready():
                idx = next_consume
                
                # If we are too far behind real-time (optional heuristic), 
                # we could skip frames here, but let's just consume them.
                # To make it "smooth" if stuck, we might want to just process what is ready.
                
                next_consume += 1

                _, lines = inflight_results[idx].get()
                frame = inflight_frames.pop(idx)
                inflight_results.pop(idx)

                annotated, dist_m = detector.process_frame_from_lines(frame, lines)

                # --- DASHBOARD UPDATE (PROCESSED FRAME) ---
                ret, buffer = cv2.imencode('.jpg', annotated)
                if ret:
                    with video_lock:
                        latest_jpeg = buffer.tobytes()

                # FPS estimate
                now = time.time()
                dt = now - last_t
                last_t = now
                if dt > 0:
                    inst_fps = 1.0 / dt
                    fps_est = inst_fps if fps_est == 0 else 0.9 * fps_est + 0.1 * inst_fps

                # Calculate video time in minutes
                vid_time_min = (idx / video_fps) / 60.0

                fps_text = f"FPS: {fps_est:.1f}"
                cv2.putText(
                    annotated,
                    fps_text,
                    (w - 160, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

                # --- TERMINAL GAUGE LOG LINE ---
                # Only print when we have a valid distance
                if dist_m is not None:
                    # format: GAUGE <timestamp> <frame_index> <gauge_m>
                    print(f"GAUGE {now:.6f} {idx} {dist_m:.6f}", flush=True)
                    
                    with dashboard_lock:
                        latest_data['gauge_m'] = dist_m
                        latest_data['timestamp'] = vid_time_min
                        latest_data['frame_idx'] = idx

                if out is not None:
                    out.write(annotated)
                
                # Removed cv2.imshow to prevent server-side GUI blocking since we have a web dash
                # cv2.imshow("Gauge MPS", annotated)

                # Rate limiting
                wait_ms = ui_delay
                if target_interval > 0:
                    elapsed = time.time() - last_loop_time
                    remaining = target_interval - elapsed
                    if remaining > 0:
                        wait_ms = max(ui_delay, int(remaining * 1000))
                
                last_loop_time = time.time()

                # If processing is too fast, wait. If too slow (stuck), we don't wait extra.
                # This ensures we don't "run some time" then pause.
                # using sleep instead of waitKey since no window
                time.sleep(wait_ms / 1000.0)

            if finished and not inflight_results:
                break

            time.sleep(0.001)

    except KeyboardInterrupt:
        logging.info("Stopped by user.")

    cap.release()
    pool.terminate()
    pool.join()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    logging.info("Finished processing.")


# -------------------------------------------------------------------
# Modes
# -------------------------------------------------------------------
def mode_auto(video: str, gauge_m: float, workers: int, output: Optional[str], no_save: bool, max_fps: Optional[float]):
    if Path(CALIB_FILE).exists():
        calib = load_calibration()
    else:
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            logging.error("Cannot open video for calibration")
            return
        ret, frame = cap.read()
        cap.release()
        if not ret:
            logging.error("Cannot read first frame for calibration")
            return
        calib = calibrate_from_frame(frame, gauge_m)
        if calib is None:
            logging.error("Calibration failed")
            return

    detector = CentralTrackDetector(
        pixels_per_meter=float(calib["pixels_per_meter"]),
        gauge_m=float(calib.get("gauge_m", gauge_m)),
        roi_y_ratio=float(calib["roi_y_ratio"]),
        center_x_rel=float(calib["center_x_rel"]),
        window_width_factor=1.4,
        vertical_angle_min_deg=45.0,
        smooth_alpha=0.88,
    )
    mode_video_mps(video, detector, workers, output, no_save, max_fps)


def mode_video(video: str, gauge_m: float, workers: int, output: Optional[str], no_save: bool, max_fps: Optional[float]):
    if not Path(CALIB_FILE).exists():
        logging.error("calib.json not found. Run auto or calibrate first.")
        return
    calib = load_calibration()
    detector = CentralTrackDetector(
        pixels_per_meter=float(calib["pixels_per_meter"]),
        gauge_m=float(calib.get("gauge_m", gauge_m)),
        roi_y_ratio=float(calib["roi_y_ratio"]),
        center_x_rel=float(calib["center_x_rel"]),
        window_width_factor=1.4,
        vertical_angle_min_deg=45.0,
        smooth_alpha=0.88,
    )
    mode_video_mps(video, detector, workers, output, no_save, max_fps)


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Central-track gauge measurement with MPS and terminal logging")
    p.add_argument("--mode", required=True, choices=["auto", "calibrate", "video"])
    p.add_argument("--video", help="Video path (for auto/video)")
    p.add_argument("--image", help="Image path (for calibrate)")
    p.add_argument("--gauge", type=float, default=1.676, help="Gauge in meters")
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1),
                   help="Number of worker processes")
    p.add_argument("--output", help="Output video path (for auto/video)")
    p.add_argument("--no_save", action="store_true", help="Do not save output video")
    p.add_argument("--max_fps", type=float, help="Limit playback FPS (e.g. 90)")
    p.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main():
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    # Start Flask
    t = threading.Thread(target=run_flask, daemon=True)
    t.start()
    logging.info("Dashboard running at http://localhost:5001")

    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )

    if args.mode == "calibrate":
        if not args.image:
            raise SystemExit("--image is required for calibrate")
        calibrate_from_image(args.image, args.gauge)
    elif args.mode == "auto":
        if not args.video:
            raise SystemExit("--video is required for auto")
        mode_auto(args.video, args.gauge, args.workers, args.output, args.no_save, args.max_fps)
    elif args.mode == "video":
        if not args.video:
            raise SystemExit("--video is required for video")
        mode_video(args.video, args.gauge, args.workers, args.output, args.no_save, args.max_fps)


if __name__ == "__main__":
    main()
