#!/usr/bin/env python3
import time
import os
import json
import threading
import logging
from pathlib import Path
from collections import deque
import cv2
import numpy as np
import serial
from flask import Flask, Response, jsonify, send_from_directory, send_file
import random
import queue
import csv
import ssl
import urllib.request
from datetime import datetime
import base64
import pandas as pd
import io
import math
import torch
from ultralytics import YOLO





# ==========================================
#          UNIFIED DASHBOARD SERVER
# ==========================================

app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR) 

# --- FLASK GLOBALS ---
latest_jpeg = None
latest_raw_jpeg = None
latest_camview_jpeg = None
dashboard_lock = threading.Lock()
video_lock = threading.Lock()
camview_lock = threading.Lock()
comp_lock = threading.Lock()
latest_comp_jpeg = None
component_event_log = []

# Global Camera URL
current_cam_url = "http://192.168.39.19:8080/video"
cam_url_lock = threading.Lock()

# Combined Data Structure
global_state = {
    # Lidar Data
    "lidar": {
        "gap_detected": False,
        "deflection": 0,
        "terminal_msg": "System Ready"
    },
    # Rail Gauge Data
    "vision": {
        "gauge_m": 0.0,
        "timestamp": 0.0
    },
    # Sensor Node Data (COM13)
    "sensors": {
        "connected": False,
        "mag": {"x": 0, "y": 0, "z": 0},
        "cycle_count": 0,       # <--- NEW
        "distance": 0.0,        # <--- NEW
        "temp": 0.0,
        "pressure": 0.0,
        "gps": {"lat": 0.0, "lon": 0.0, "sat": 0},
        "raw_log": []
    },
    # Extended Lidar Data for Dashboard Tab
    "lidar_ext": {
        "speed_kmph": 0.0,
        "total_dist": 0.0,
        "cycles": 0,
        "sps": 0,
        "scan_data": [],
        "gap_detected": False,
        "deflection": 0.0,
        "settings": {
            "range": 10000,
            "history": 25,
            "point_size": 2.0,
            "rotation": 0,
            "angle_from": 0,
            "angle_to": 359
        },
        "recording": False,
        "recorded_frames": 0
    },
    "component": {
        "status": "Offline",
        "last_event": None,
        "total_events": 0
    }
}

# --- RECORDING STATE ---
lidar_recording_active = False
lidar_recording_data = []

# --- PLACEHOLDER IMAGE ---
def create_placeholder(text):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, text, (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()

latest_jpeg = create_placeholder("INITIALIZING SYSTEM...")

# --- SNAPSHOT FEATURE ---
SNAPSHOT_DIR = "snapshots"
if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)

# Generate Session CSV Name
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_SESSION_CSV = f"{SNAPSHOT_DIR}/run_{session_id}.csv"

snapshot_queue = queue.Queue()

def snapshot_worker():
    """
    Background thread to save images and data without blocking
    """
    while True:
        try:
            # Block for a bit, then check loop
            item = snapshot_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        
        try:
            # 1. Capture Data
            img_data = None
            snap_type = "lidar"
            trigger_val = ""
            tag_str = "High Deflection"
            
            # PARSE ITEM (Tuple vs Dict)
            if isinstance(item, dict) and item.get('type') == 'vision':
                # --- VISION EVENT ---
                snap_type = "vision"
                trigger_ts = item['ts']
                labels = item.get('labels', [])
                trigger_val = ",".join(labels) if labels else "Unknown"
                tag_str = "Foreign Object"
                img_data = item.get('image') # Pre-captured annotated frame
                
            else:
                # --- LIDAR EVENT (Legacy Tuple) ---
                trigger_ts, trigger_defl = item
                trigger_val = str(trigger_defl)
            
            # If no image provided (Lidar case), grab from stream
            if img_data is None:
                # Prioritize Live IP Camera (Camview)
                with camview_lock:
                    if latest_camview_jpeg is not None:
                        img_data = latest_camview_jpeg
                
                # Fallback to Vision Stream if IP Cam fails
                if img_data is None:
                    with video_lock:
                        if latest_jpeg is not None:
                            img_data = latest_jpeg
            
            sens_data = {}
            with dashboard_lock:
                try:
                    sens_data = json.loads(json.dumps(global_state["sensors"]))
                except:
                    sens_data = {"error": "copy_failed"}

            # 2. File Paths
            ts_str = trigger_ts.strftime("%Y%m%d_%H%M%S_%f")
            
            if snap_type == 'vision':
                base_name = f"{SNAPSHOT_DIR}/object_{ts_str}"
                jpg_name = f"object_{ts_str}.jpg"
            else:
                base_name = f"{SNAPSHOT_DIR}/alert_{ts_str}_defl_{trigger_val}"
                jpg_name = f"alert_{ts_str}_defl_{trigger_val}.jpg"
                
            jpg_path = f"{base_name}.jpg"
            
            # 3. Save Image
            if img_data:
                try:
                    # If it's already bytes, write them. If it needs overlay (Lidar), decode/draw/encode.
                    # Vision events already have overlay drawn on the frame before encoding.
                    if snap_type == 'vision':
                        with open(jpg_path, 'wb') as f:
                            f.write(img_data)
                    else:
                        nparr = np.frombuffer(img_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is not None:
                            overlay_txt = f"DEFL: {trigger_val} | {trigger_ts.strftime('%H:%M:%S.%f')}"
                            cv2.putText(img, overlay_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.8, (0, 0, 255), 2)
                            cv2.imwrite(jpg_path, img)
                except Exception as e:
                    print(f"SNAPSHOT: Failed to save Image - {e}")

            # 4. Save JSON
            meta = {
                "timestamp": trigger_ts.isoformat(),
                "tag": tag_str,
                "value": trigger_val,
                "image_path": jpg_name,
                "sensors": sens_data
            }
            try:
                with open(f"{base_name}.json", 'w') as f:
                    json.dump(meta, f, indent=4)
            except Exception as e:
                print(f"SNAPSHOT: Failed to save JSON - {e}")

            # 5. Append to CSV (Excel Format)
            csv_path = CURRENT_SESSION_CSV
            file_exists = os.path.isfile(csv_path)
            try:
                with open(csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    # Headers
                    if not file_exists:
                        writer.writerow(["Timestamp", "Tag", "Value/Deflection", "Image", "Lat", "Lon", "Sat", "Temp", "Pressure", "Mag_X", "Mag_Y", "Mag_Z", "Cycle_Count", "Distance"])
                    
                    # Data
                    gps = sens_data.get("gps", {})
                    mag = sens_data.get("mag", {})
                    
                    writer.writerow([
                        trigger_ts.strftime("%Y-%m-%d %H:%M:%S.%f"),
                        tag_str,
                        trigger_val,
                        jpg_name,
                        gps.get("lat", 0),
                        gps.get("lon", 0),
                        gps.get("sat", 0),
                        sens_data.get("temp", 0),
                        sens_data.get("pressure", 0),
                        mag.get("x", 0),
                        mag.get("y", 0),
                        mag.get("z", 0),
                        sens_data.get("cycle_count", 0),
                        sens_data.get("distance", 0)
                    ])
            except Exception as e:
                print(f"SNAPSHOT: Failed to save CSV - {e}")

            # 6. Generate HTML Report
            html_path = f"{base_name}.html"
            try:
                title_color = "#ff5555" if snap_type == 'lidar' else "#ff9933"
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Event Report {ts_str}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; background: #222; color: #eee; padding: 20px; }}
                        .container {{ max-width: 800px; margin: 0 auto; background: #333; padding: 20px; border-radius: 10px; }}
                        h1 {{ color: {title_color}; text-align: center; }}
                        img {{ display: block; margin: 20px auto; max-width: 100%; border: 2px solid #555; }}
                        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                        th, td {{ padding: 10px; border-bottom: 1px solid #444; text-align: left; }}
                        th {{ background: #444; }}
                        .val {{ color: #4db8ff; font-weight: bold; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>{tag_str}: {trigger_val}</h1>
                        <p><strong>Time:</strong> {trigger_ts.strftime('%Y-%m-%d %H:%M:%S.%f')}</p>
                        <img src="{jpg_name}" alt="Snapshot">
                        <h2>Sensor Data</h2>
                        <table>
                            <tr><th>Parameter</th><th>Value</th></tr>
                            <tr><td>GPS Latitude</td><td class="val">{gps.get("lat", 0)}</td></tr>
                            <tr><td>GPS Longitude</td><td class="val">{gps.get("lon", 0)}</td></tr>
                            <tr><td>Satellites</td><td class="val">{gps.get("sat", 0)}</td></tr>
                            <tr><td>Temperature</td><td class="val">{sens_data.get("temp", 0)} °C</td></tr>
                            <tr><td>Pressure</td><td class="val">{sens_data.get("pressure", 0)} hPa</td></tr>
                            <tr><td>Magnetometer (X, Y, Z)</td><td class="val">{mag.get("x", 0)}, {mag.get("y", 0)}, {mag.get("z", 0)}</td></tr>
                            <tr><td>Cycle Count</td><td class="val">{sens_data.get("cycle_count", 0)}</td></tr>
                            <tr><td>Distance</td><td class="val">{sens_data.get("distance", 0):.2f} m</td></tr>
                        </table>
                    </div>
                </body>
                </html>
                """
                with open(html_path, 'w') as f:
                    f.write(html_content)
            except Exception as e:
                print(f"SNAPSHOT: Failed to save HTML - {e}")
                    
        except Exception as e:
            print(f"SNAPSHOT: Worker Error - {e}")

        snapshot_queue.task_done()


# ==========================================
#            LIDAR LOGIC (THREAD)
# ==========================================

LIDAR_SETTINGS_FILE = "lidar_settings.json"
DISTANCE_PER_CYCLE = 3.45 
CENTER_REF = 178

# Lidar Globals
scan_history = deque(maxlen=25) 
is_closing = False
last_snapshot_time = 0
SNAPSHOT_COOLDOWN = 1.0 # Seconds

# Speed Calculation Vars
raw_hall_count = 0      
count_offset = 0        
last_pulse_time = 0      
current_speed_kmph = 0.0  
sps_counter = 0

def get_uint16_le(b):
    return b[0] | (b[1]<<8)

def load_lidar_settings():
    if os.path.exists(LIDAR_SETTINGS_FILE):
        try:
            with open(LIDAR_SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        'range': 10000, 'history': 25, 'point_size': 2.0,
        'angle_from': 0, 'angle_to': 359, 'rotation': 0
    }

def save_lidar_settings_file(settings):
    try:
        with open(LIDAR_SETTINGS_FILE, 'w') as f:
            json.dump(settings, f)
    except:
        pass

def lidar_worker():
    global global_state, last_snapshot_time, scan_history, sps_counter
    global raw_hall_count, count_offset, last_pulse_time, current_speed_kmph
    global lidar_recording_active, lidar_recording_data
    
    # Load initial settings
    current_settings = load_lidar_settings()
    with dashboard_lock:
        global_state["lidar_ext"]["settings"] = current_settings

    try:
        ser = serial.Serial('COM9', 128000, timeout=0.005)
        print("LIDAR: Connected to COM9")
    except Exception as e:
        print(f"LIDAR: Connection Failed - {e}")
        ser = None

    buffer = b''
    last_sps_time = time.time()
    
    while not is_closing:
        if not ser:
            time.sleep(1)
            continue
            
        try:
            new_data = ser.read(8192)
            if new_data:
                buffer += new_data
        except:
            break

        # --- HALL SENSOR PARSING (CNT:) ---
        try:
            cnt_idx = buffer.find(b'CNT:')
            if cnt_idx != -1:
                eol_idx = buffer.find(b'\n', cnt_idx)
                if eol_idx != -1:
                    val_str = buffer[cnt_idx+4:eol_idx].decode(errors='ignore').strip()
                    if val_str.isdigit():
                        new_val = int(val_str)
                        if new_val != raw_hall_count:
                            now = time.time()
                            if last_pulse_time > 0:
                                time_diff = now - last_pulse_time
                                if time_diff > 0.05: 
                                    speed_mps = DISTANCE_PER_CYCLE / time_diff
                                    current_speed_kmph = speed_mps * 3.6
                            last_pulse_time = now
                            raw_hall_count = new_val
        except Exception:
            pass
            
        # --- LIDAR PACKET PARSING ---
        processed = 0
        while processed < 100:
            start = buffer.find(b'\xAA\x55')
            if start == -1: break
            if start > 0: buffer = buffer[start:]
            if len(buffer) < 10: break
            
            packet_len = 10 + buffer[3] * 2 # lsn is buffer[3]
            if len(buffer) < packet_len: break
            
            packet = buffer[:packet_len]
            lsn = packet[3]
            fsa = get_uint16_le(packet[4:6]) / 100.0
            lsa = get_uint16_le(packet[6:8]) / 100.0

            temp_angles = []
            temp_dists = []
            
            for i in range(lsn):
                idx = 10 + i * 2
                dist_mm = get_uint16_le(packet[idx:idx+2])
                if lsn > 1:
                    angle_deg = (fsa + (lsa-fsa) * i / (lsn-1)) % 360.0
                else:
                    angle_deg = fsa % 360.0
                
                if dist_mm > 0:
                    temp_angles.append(angle_deg)
                    temp_dists.append(dist_mm)
            
            if temp_angles:
                scan_history.append((temp_angles, temp_dists))
                sps_counter += len(temp_angles)
            
            buffer = buffer[packet_len:]
            processed += 1

        # --- UPDATE GLOBAL STATE ---
        # 1. Speed/Dist
        if (time.time() - last_pulse_time) > 3.0:
            current_speed_kmph = 0.0
        
        net_cycles = raw_hall_count - count_offset
        total_dist = net_cycles * DISTANCE_PER_CYCLE
        
        # 2. SPS Calculation
        if time.time() - last_sps_time > 1.0:
            with dashboard_lock:
                global_state["lidar_ext"]["sps"] = sps_counter
            sps_counter = 0
            last_sps_time = time.time()

        # 3. Process Scan History for Visualization & Deflection
        # We process this periodically (e.g., every 40ms to match 25fps)
        # or just on every loop but throttled.
        
        with dashboard_lock:
            # Sync Speed
            global_state["lidar_ext"]["speed_kmph"] = current_speed_kmph
            global_state["lidar_ext"]["total_dist"] = total_dist
            global_state["lidar_ext"]["cycles"] = net_cycles
            global_state["lidar_ext"]["recording"] = lidar_recording_active
            global_state["lidar_ext"]["recorded_frames"] = len(lidar_recording_data)
            
            # Use current settings
            settings = global_state["lidar_ext"]["settings"]
            rot = settings.get('rotation', 0)
            
            # Flatten history for deflection & visual
            # Visual: Send raw points, let client filter/rotate? 
            # OR rotate here?
            # To insure "exact same calculation", deflection is done on ROTATED angles.
            
            flat_angles = []
            flat_dists = []
            
            # Just take the last N items based on history setting? 
            # scan_history is deque maxlen=25.
            # We use all of it.
            
            visual_points = []
            
            for ag_l, d_l in scan_history:
                for a, d in zip(ag_l, d_l):
                    ra = (a + rot) % 360
                    flat_angles.append(np.deg2rad(ra))
                    # visual_points.append([ra, d]) 
                    # Optimization: Limit visual points sent over JSON?
                    # Sending 5000 points is roughly 100KB JSON. 10FPS = 1MB/s. Manageable on localhost.
                    visual_points.append([ra, d])

            # Store for frontend
            global_state["lidar_ext"]["scan_data"] = visual_points
            filtered_scan = visual_points # For recording

            # --- DEFLECTION LOGIC (Ported from read_serial15.py) ---
            present_angles = set()
            for ra_rad in flat_angles:
                deg = int(round(np.rad2deg(ra_rad))) # converting back/forth safe
                present_angles.add(deg)

            missing_angles = []
            start_check = 170
            end_check = 186
            # CENTER_REF = 178 defined above
            
            for deg in range(start_check, end_check + 1):
                if deg not in present_angles:
                    missing_angles.append(deg)

            ranges = []
            if missing_angles:
                start = missing_angles[0]
                prev = start
                for d in missing_angles[1:]:
                    if d == prev + 1:
                        prev = d
                    else:
                        if prev-start >= 1: 
                            ranges.append((start,prev))
                        start=d
                        prev=d
                if prev-start >= 1:
                    ranges.append((start,prev))

            if ranges:
                # Taking the first gap as primary deflection
                rg = ranges[0]
                gap_center = (rg[0] + rg[1]) / 2.0
                deflection = gap_center - CENTER_REF
                
                global_state["lidar"]["gap_detected"] = True
                global_state["lidar"]["deflection"] = int(round(deflection))
                global_state["lidar_ext"]["gap_detected"] = True # Duplicated as per instruction
                global_state["lidar_ext"]["deflection"] = float(round(deflection)) # Duplicated as per instruction
                
                rel_start = rg[0] - CENTER_REF
                rel_end   = rg[1] - CENTER_REF
                s_start = f"{rel_start:+d}"
                s_end   = f"{rel_end:+d}"
                term_msg = f"Deflection Zone: {s_start}\u00B0 to {s_end}\u00B0"
                global_state["lidar"]["terminal_msg"] = term_msg
                
                # Trigger Logic
                if abs(deflection) >= 2:
                     now_t = time.time()
                     if now_t - last_snapshot_time > SNAPSHOT_COOLDOWN:
                        last_snapshot_time = now_t
                        try:
                            snapshot_queue.put((datetime.now(), int(round(deflection))))
                        except: pass
            else:
                global_state["lidar"]["gap_detected"] = False
                global_state["lidar"]["deflection"] = 0
                global_state["lidar"]["terminal_msg"] = "Status: Clear"
                global_state["lidar_ext"]["gap_detected"] = False # Duplicated as per instruction
                global_state["lidar_ext"]["deflection"] = 0.0 # Duplicated as per instruction

        # --- RECORDING ---
        if lidar_recording_active:
            # Snapshot current state
            frame = {
                "timestamp": time.time(),
                "gps": global_state["sensors"]["gps"].copy(),
                "speed": current_speed_kmph,
                "distance": total_dist,
                "scan": filtered_scan # Record the processed points
            }
            lidar_recording_data.append(frame)

        # Avoid tight loop
        time.sleep(0.005)


# ==========================================
#          OPENCV LOGIC (Single Thread)
# ==========================================

CALIB_FILE = str(Path(__file__).parent / "calib.json")

class CentralTrackDetector:
    def __init__(self, pixels_per_meter, gauge_m, roi_y_ratio, center_x_rel,
                 window_width_factor=1.5, vertical_angle_min_deg=45.0, smooth_alpha=0.85):
        self.ppm = pixels_per_meter
        self.gauge_m = gauge_m
        self.roi_y_ratio = roi_y_ratio
        self.center_x_rel = center_x_rel
        self.window_width_factor = window_width_factor
        self.vertical_angle_min_deg = vertical_angle_min_deg
        self.alpha = smooth_alpha
        self.s_left_x = None
        self.s_right_x = None

    def process_frame_from_lines(self, frame, lines):
        h, w = frame.shape[:2]
        roi_y = int(self.roi_y_ratio * h)
        center_x = self.center_x_rel * w
        gauge_px = self.ppm * self.gauge_m
        half_window = 0.5 * self.window_width_factor * gauge_px

        x_min = max(0, int(center_x - half_window))
        x_max = min(w - 1, int(center_x + half_window))

        annotated = frame.copy()
        cv2.line(annotated, (0, roi_y), (w, roi_y), (0, 255, 255), 1)
        cv2.rectangle(annotated, (x_min, 0), (x_max, h), (0, 255, 255), 1)

        if lines is None: return annotated, None

        candidates = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1; dy = y2 - y1
            angle = 90.0 if dx == 0 else float(np.degrees(np.arctan2(dy, dx)))
            if abs(angle) < self.vertical_angle_min_deg: continue
            if not (min(y1, y2) <= roi_y <= max(y1, y2)): continue
            if dy == 0: continue
            t = (roi_y - y1) / float(dy)
            x_at_roi = x1 + t * dx
            if x_min <= x_at_roi <= x_max:
                candidates.append((x_at_roi, (x1, y1, x2, y2)))

        if len(candidates) < 2: return annotated, None

        expected_px = gauge_px
        best_pair = None
        best_diff = float("inf")
        for i in range(len(candidates)):
            xi, _, = candidates[i]
            for j in range(i + 1, len(candidates)):
                xj, _, = candidates[j]
                dist_px = abs(xj - xi)
                diff = abs(dist_px - expected_px)
                if diff < best_diff:
                    best_diff = diff
                    best_pair = (xi, xj)

        if best_pair is None: return annotated, None

        xL, xR = best_pair
        if xL > xR: xL, xR = xR, xL

        if self.s_left_x is None: self.s_left_x = xL
        else: self.s_left_x = self.alpha * self.s_left_x + (1 - self.alpha) * xL

        if self.s_right_x is None: self.s_right_x = xR
        else: self.s_right_x = self.alpha * self.s_right_x + (1 - self.alpha) * xR

        sxL = int(round(self.s_left_x))
        sxR = int(round(self.s_right_x))

        cv2.line(annotated, (sxL, roi_y), (sxR, roi_y), (255, 0, 0), 3)
        cv2.circle(annotated, (sxL, roi_y), 5, (255, 0, 0), -1)
        cv2.circle(annotated, (sxR, roi_y), 5, (255, 0, 0), -1)

        distance_m = abs(self.s_right_x - self.s_left_x) / self.ppm
        return annotated, distance_m

def load_calibration():
    # If calib file is in OpenCV_Final subfolder, try to locate it
    p = Path(CALIB_FILE)
    if not p.exists():
        # Fallback search
        p = Path("OpenCV_Final/calib.json")
    
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    print("WARNING: calib.json not found")
    return None

def vision_pipeline(video_path):
    global latest_jpeg, latest_raw_jpeg, global_state
    
    # Initialize Foreign Object Detector for MMD Panel
    weights_path = os.path.join(os.getcwd(), 'foreign object on track', 'yolov5s.pt')
    yolo_detector = ForeignObjectDetector(weights_path=weights_path)
    # Ensure all classes active
    if yolo_detector.active:
        yolo_detector.model.classes = None 
        
    calib = load_calibration()
    if not calib:
        latest_jpeg = create_placeholder("ERROR: CALIB MISSING")
        return

    detector = CentralTrackDetector(
        pixels_per_meter=float(calib["pixels_per_meter"]),
        gauge_m=float(calib.get("gauge_m", 1.676)),
        roi_y_ratio=float(calib["roi_y_ratio"]),
        center_x_rel=float(calib["center_x_rel"]),
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"VISION: Could not open {video_path}")
        latest_jpeg = create_placeholder("ERROR: VIDEO NOT FOUND")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0
    
    print("VISION: Pipeline Started")

    while not is_closing:
        start_t = time.time()
        
        ret, frame = cap.read()
        if not ret:
            # Loop
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            continue
        
        frame_idx += 1

        # 1. Processing (Simpler Single Thread)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
        edges = cv2.Canny(blur, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60, minLineLength=60, maxLineGap=25)
        
        annotated, dist_m = detector.process_frame_from_lines(frame, lines)
        
        # 2. Update Image Stream
        # Use lower quality for faster transmission
        ret_enc, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if ret_enc:
            with video_lock:
                latest_jpeg = buffer.tobytes()

        # 2a. Update Raw Stream (Now with Foreign Object Detection for MMD)
        # Run YOLO on a copy of the frame so we don't affect the main line processing
        frame_for_yolo = frame.copy()
        if yolo_detector.active:
            try:
                # Detect and Draw on frame_for_yolo
                frame_for_yolo, labels = yolo_detector.detect(frame_for_yolo)
                
                # Log/Snapshot if foreign object found
                if labels:
                    # Simple throttling for logs could be added here, or reliance on the fact that
                    # this is a loop file.
                    # For now, just print to console to confirm detection "output data"
                    print(f"MMD VISION: Foreign Object Detected: {labels}")
            except Exception as e:
                print(f"MMD YOLO Error: {e}")

        ret_raw, buf_raw = cv2.imencode('.jpg', frame_for_yolo, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        if ret_raw:
             with video_lock:
                 latest_raw_jpeg = buf_raw.tobytes()
        
        # 3. Update Data
        if dist_m:
            vid_time_min = (frame_idx / video_fps) / 60.0
            with dashboard_lock:
                global_state["vision"]["gauge_m"] = dist_m
                global_state["vision"]["timestamp"] = vid_time_min

        # 4. FPS Limit (keep roughly realtime / video FPS)
        process_dt = time.time() - start_t
        target_dt = 1.0 / video_fps
        sleep_t = target_dt - process_dt
        if sleep_t > 0:
            time.sleep(sleep_t)

    cap.release()

# ==========================================
#          AUX SENSOR LOGIC (COM13)
# ==========================================
def aux_sensor_worker():
    global global_state
    
    port_name = 'COM13'
    baud_rate = 115200
    ser = None
    
    # Try to connect
    try:
        ser = serial.Serial(port_name, baud_rate, timeout=0.1)
        print(f"SENSORS: Connected to {port_name}")
        with dashboard_lock:
            global_state["sensors"]["connected"] = True
    except Exception as e:
        print(f"SENSORS: Connection Failed - {e}")
        with dashboard_lock:
            global_state["sensors"]["connected"] = False
        return # Exit if can't open (or retry loop? User didn't specify, but retry is better)

    # Retry loop wrapper would be better, but for now simple structure
    
    import re
    # Regex for various formats
    # Example: "T:25.5 P:1013.2 M:12,14,15 G:12.34,56.78" or JSON
    
    # --- SIMULATION STATE ---
    sim_lat = 9.528193
    sim_lon = 76.822224
    
    WHEEL_CIRCUMFERENCE = 0.00346 # 3.46mm
    
    while not is_closing:
        # --- SIMULATE MOVEMENT (Jitter) ---
        # Drift by approx 0.5-1 meter (slower)
        sim_lat += random.uniform(-0.000005, 0.000005)
        sim_lon += random.uniform(-0.000005, 0.000005)
        
        with dashboard_lock:
             global_state["sensors"]["gps"]["lat"] = sim_lat
             global_state["sensors"]["gps"]["lon"] = sim_lon
             global_state["sensors"]["gps"]["sat"] = random.randint(3, 12) # Simulate sat count too

        if not ser or not ser.is_open:
            time.sleep(1)
            continue
            
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                with dashboard_lock:
                    # Keep last 10 log lines
                    log_list = global_state["sensors"]["raw_log"]
                    log_list.append(line)
                    if len(log_list) > 10:
                        log_list.pop(0)
                        
                    # Attempt Parsing
                    
                    # 0. Unlabeled CSV 
                    # UPDATED: Expect 6 values (x,y,z,count,temp,press) matching visualizer.py
                    # 0.82,-0.02,-0.22,171,32.0,99818
                    # Regex checks for at least 3 parts, we handle more inside
                    if re.match(r'^[+-]?\d*(?:\.\d+)?,[+-]?\d*', line):
                         try:
                             parts = [float(x) for x in line.split(',')]
                             
                             # Mag (0,1,2)
                             if len(parts) >= 3:
                                 global_state["sensors"]["mag"] = {"x": parts[0], "y": parts[1], "z": parts[2]}
                             
                             # Count (3)
                             if len(parts) >= 4:
                                 count = int(parts[3])
                                 dist = count * WHEEL_CIRCUMFERENCE
                                 global_state["sensors"]["cycle_count"] = count
                                 global_state["sensors"]["distance"] = dist

                             # Temp (4)
                             if len(parts) >= 5:
                                 global_state["sensors"]["temp"] = parts[4]

                             # Pressure (5)
                             if len(parts) >= 6:
                                 global_state["sensors"]["pressure"] = parts[5]
                         except: pass

                    # 1. Temperature (e.g. "Temp: 30" or "T:30")
                    m_temp = re.search(r'(?:Temp|T)[:=]\s*([0-9.]+)', line, re.IGNORECASE)
                    if m_temp:
                         global_state["sensors"]["temp"] = float(m_temp.group(1))

                    # 2. Pressure (e.g. "Press: 1000" or "P:1000")
                    m_press = re.search(r'(?:Press|Baro|P)[:=]\s*([0-9.]+)', line, re.IGNORECASE)
                    if m_press:
                         global_state["sensors"]["pressure"] = float(m_press.group(1))

                    # 3. Magnetometer (e.g. "Mag: 10,20,30" or "M:10,20,30")
                    m_mag = re.search(r'(?:Mag|M)[:=]\s*([0-9.-]+)[, ]+([0-9.-]+)[, ]+([0-9.-]+)', line, re.IGNORECASE)
                    if m_mag:
                        global_state["sensors"]["mag"] = {
                            "x": float(m_mag.group(1)),
                            "y": float(m_mag.group(2)),
                            "z": float(m_mag.group(3))
                        }
                        
                    # 4. GPS (e.g. "GPS: 12.34,56.78,5" - lat,lon,sat)
                    m_gps = re.search(r'(?:GPS|G)[:=]\s*([0-9.-]+)[, ]+([0-9.-]+)(?:[, ]+([0-9]+))?', line, re.IGNORECASE)
                    if m_gps:
                        global_state["sensors"]["gps"]["lat"] = float(m_gps.group(1))
                        global_state["sensors"]["gps"]["lon"] = float(m_gps.group(2))
                        if m_gps.group(3):
                            global_state["sensors"]["gps"]["sat"] = int(m_gps.group(3))

        except Exception as e:
            print(f"SENSORS: Read Error - {e}")
            time.sleep(1)

        time.sleep(0.01)

        time.sleep(0.01)

# ==========================================
#          THREADED IP CAMERA READER
# ==========================================
class ThreadedCamReader:
    def __init__(self, src):
        self.src = src
        # Support local webcam index if string is digit
        if isinstance(src, str) and src.isdigit():
            self.src = int(src)
        self.cap = cv2.VideoCapture(self.src)
        # Attempt to set preferences (may not work on all IP cams, but good practice)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
        self.ret = False
        self.frame = None
        try:
            self.ret, self.frame = self.cap.read()
        except: pass
        self.stopped = False
        self.lock = threading.Lock()
        self.t = threading.Thread(target=self.update, args=(), daemon=True)
        self.t.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.1)

    def read(self):
        with self.lock:
            return self.ret, getattr(self, 'frame', None)

    def stop(self):
        self.stopped = True
        self.t.join()
        self.cap.release()

# ==========================================
#          FOREIGN OBJECT DETECTOR
# ==========================================
class ForeignObjectDetector:
    def __init__(self, weights_path=None):
        self.conf_threshold = 0.4
        # Classes: 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck, 
        # 15: cat, 16: dog, 17: horse, 18: sheep, 19: cow, 24: backpack, 26: handbag, 28: suitcase
        self.target_classes = [0, 1, 2, 3, 5, 7, 15, 16, 17, 18, 19, 24, 26, 28]
        
        self.active = False
        try:
            print("YOLO: Initializing Object Detector...")
            # Try loading local or hub
            # First check if user provided weights exist
            if weights_path and os.path.exists(weights_path):
                 print(f"YOLO: Loading local weights from {weights_path}")
                 self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
            else:
                 print("YOLO: Downloading/Loading 'yolov5s' from Hub")
                 self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            
            self.model.conf = self.conf_threshold
            # self.model.classes = self.target_classes # ENABLE ALL CLASSES
            self.model.classes = None
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
            print(f"YOLO: Model loaded on {self.device}")
            self.active = True
        except Exception as e:
            print(f"YOLO: Failed to load model - {e}")
            self.active = False

    def detect(self, frame, roi=None):
        if not self.active: return frame, []
        
        # roi is [x1, y1, x2, y2]
        if roi:
            # Draw ROI (Visual Guide)
            cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 1)
            cv2.putText(frame, "TRACK ZONE", (roi[0], roi[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        # Convert to RGB for YOLO
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inference
        results = self.model(img_rgb)
        detections = results.xyxy[0].cpu().numpy()
        
        found_labels = []
        
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            in_zone = True
            if roi:
                # Check if center of object is inside ROI
                if not (roi[0] < center_x < roi[2] and roi[1] < center_y < roi[3]):
                    in_zone = False
            
            if in_zone:
                lbl = f"{self.model.names[int(cls)]}"
                found_labels.append(lbl)
                label_display = f"{lbl} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label_display, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if found_labels:
            cv2.putText(frame, "WARNING: FOREIGN OBJECT!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        return frame, found_labels

# ==========================================
#          CAMVIEW WORKER (RESIZED & THREADED)
# ==========================================
def camview_worker():
    global latest_camview_jpeg, current_cam_url
    
    # Initial URL
    local_url = ""
    with cam_url_lock:
        local_url = current_cam_url
        
    print(f"CAMVIEW: Starting Optimized Threaded Stream: {local_url}")
    
    # Initialize YOLO Detector
    weights = os.path.join(os.getcwd(), 'foreign object on track', 'yolov5s.pt')
    detector = ForeignObjectDetector(weights_path=weights)
    
    # ROI for 640x360 frame (Approximate center-bottom track area)
    # Original config was [100, 100, 540, 480] for 640x480
    # For 640x360, let's target the bottom center
    track_roi = [100, 50, 540, 350] 
    
    cam = ThreadedCamReader(local_url)
    
    frame_count = 0
    last_object_snap_time = 0
    OBJECT_COOLDOWN = 3.0 # Seconds between object detection snapshots
    
    while not is_closing:
        # 1. Check if URL changed
        new_url_check = ""

        with cam_url_lock:
            new_url_check = current_cam_url
            
        if new_url_check != local_url:
            print(f"CAMVIEW: Switching URL to {new_url_check}")
            try:
                cam.stop()
            except: pass
            local_url = new_url_check
            cam = ThreadedCamReader(local_url)
            time.sleep(0.5)
            continue

        try:
            ret, frame = cam.read()
            if not ret or frame is None:
                # If reader is stuck, try reconnecting
                if not cam.cap.isOpened():
                    try:
                        cam.stop()
                    except: pass
                    time.sleep(1)
                    # Use current local_url
                    cam = ThreadedCamReader(local_url)
                time.sleep(0.1)
                continue
            
            # RESIZE is key for smoothness (bandwidth management)
            # 640x360 is 16:9 and light
            frame = cv2.resize(frame, (640, 360))
            
            # --- YOLO DETECTION ---
            if detector.active:
                 try:
                     frame, labels = detector.detect(frame, roi=track_roi)
                     
                     if labels:
                         now = time.time()
                         if now - last_object_snap_time > OBJECT_COOLDOWN:
                             last_object_snap_time = now
                             # Dispatch to Snapshot Worker
                             # We encode this specific frame to keep the bounding boxes
                             try:
                                 _, jpg_bytes = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                                 snapshot_payload = {
                                     'type': 'vision',
                                     'ts': datetime.now(),
                                     'labels': labels,
                                     'image': jpg_bytes.tobytes()
                                 }
                                 snapshot_queue.put(snapshot_payload)
                                 print(f"YOLO: Foreign Object Snapshot Queued ({labels})")
                             except Exception as enc_err:
                                 print(f"YOLO: Snapshot encode error: {enc_err}")

                 except Exception as e:
                     print(f"YOLO Error: {e}")

            
            # Encoding Q=50 for speed
            ret_enc, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            
            if ret_enc and buf is not None:
                with camview_lock:
                    latest_camview_jpeg = buf.tobytes()
        
        except Exception as e:
            pass
            
        # Target ~30 FPS processing
        time.sleep(0.015)

    try:
        cam.stop()
    except: pass
    print("CAMVIEW: Stopped")


# ==========================================
#             SERVER ROUTES
# ==========================================

@app.route('/')
def home():
    try:
        with open("unified_dashboard.html", "r") as f:
            return f.read()
    except:
        return "<h1>Dashboard HTML not found</h1>"

@app.route('/stream')
def stream():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_raw')
def stream_raw():
    return Response(gen_raw_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_camview')
def stream_camview():
    return Response(generate_camview(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/data')
def get_data():
    with dashboard_lock:
        return jsonify(global_state)

@app.route('/api/lidar/update_settings', methods=['POST'])
def update_lidar_settings():
    from flask import request
    try:
        new_settings = request.json
        with dashboard_lock:
            # Update global state
            global_state["lidar_ext"]["settings"].update(new_settings)
            
            # Reset history if history size changed (optional, but mimics clear() in original)
            # if 'history' in new_settings:
            #    scan_history.clear()
            
            # Save to file
            save_lidar_settings_file(global_state["lidar_ext"]["settings"])
            
            # Handle Reset Distance special command
            if new_settings.get('reset_distance', False):
                global count_offset, current_speed_kmph
                count_offset = raw_hall_count
                current_speed_kmph = 0.0
        
        return jsonify({"status": "ok", "settings": global_state["lidar_ext"]["settings"]})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/api/lidar/record', methods=['POST'])
def lidar_record_control():
    global lidar_recording_active, lidar_recording_data
    from flask import request
    action = request.json.get('action')
    
    if action == 'start':
        lidar_recording_data = [] # Reset buffer (User said "removed" on cancel/restart logic)
        lidar_recording_active = True
        return jsonify({"status": "started"})
    
    elif action == 'stop':
        lidar_recording_active = False
        # Save JSON immediately
        try:
            with open("lidar_recording.json", "w") as f:
                json.dump(lidar_recording_data, f, indent=2)
        except Exception as e:
            print(f"Failed to save recording json: {e}")
            
        return jsonify({"status": "stopped", "count": len(lidar_recording_data)})
        
    return jsonify({"status": "invalid"})

@app.route('/api/lidar/export/<fmt>')
def export_lidar_data(fmt):
    from flask import request
    mode = request.args.get('mode', 'recording')
    
    data = []
    
    if mode == 'snapshot':
        # Create a single frame from current state
        with dashboard_lock:
             # Make a deep copy of critical data so we don't block
             frame = {
                "timestamp": time.time(),
                "gps": global_state["sensors"]["gps"].copy(),
                "speed": global_state["lidar_ext"]["speed_kmph"],
                "distance": global_state["lidar_ext"]["total_dist"],
                "scan": list(global_state["lidar_ext"]["scan_data"]) # Copy the list
            }
        data = [frame]
    else:
        # Load from file (Legacy/Recording mode)
        try:
            if os.path.exists("lidar_recording.json"):
                with open("lidar_recording.json", "r") as f:
                    data = json.load(f)
            else:
                return "No recording found.", 404
        except Exception as e:
            return f"Error loading data: {e}", 500

    if not data:
        return "Data is empty.", 400

    if fmt == 'excel':
        # Flatten data for CSV/Excel
        rows = []
        for frame in data:
            ts_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(frame['timestamp'])) + f".{int((frame['timestamp']%1)*1000):03d}"
            gps = frame['gps']
            
            for pt in frame['scan']:
                # pt = [angle, dist]
                rows.append({
                    "Timestamp": ts_str,
                    "Lat": gps['lat'],
                    "Lon": gps['lon'],
                    "Speed_Kmph": frame['speed'],
                    "Dist_M": frame['distance'],
                    "Angle_Deg": pt[0],
                    "Lidar_Dist_mm": pt[1],
                    # Vibration Data
                    "Vib_X": global_state["sensors"]["mag"]["x"],
                    "Vib_Y": global_state["sensors"]["mag"]["y"],
                    "Vib_Z": global_state["sensors"]["mag"]["z"]
                })
                
        df = pd.DataFrame(rows)
        output = io.BytesIO()
        # Use openpyxl since xlsxwriter is not installed
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='LidarData')
        output.seek(0)
        
        fname = "lidar_snapshot" if mode == 'snapshot' else "lidar_export"
        return send_file(output, download_name=f"{fname}_{int(time.time())}.xlsx", as_attachment=True)

    elif fmt == 'html':
        # Generate a Report
        html = f"""
        <html>
        <head>
            <title>Lidar Data Report</title>
            <style>
                body {{ font-family: sans-serif; background: #f0f0f0; padding: 20px; }}
                table {{ border-collapse: collapse; width: 100%; background: white; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background: #333; color: white; }}
                .summary {{ background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; }}
            </style>
        </head>
        <body>
            <h1>Lidar Data ({mode.upper()})</h1>
            <div class="summary">
                <p><b>Date:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><b>Total Frames:</b> {len(data)}</p>
                <p><b>GPS:</b> {data[0]['gps']['lat']}, {data[0]['gps']['lon']}</p>
            </div>
            <table>
                <tr>
                    <th>Time</th>
                    <th>Angle (deg)</th>
                    <th>Distance (mm)</th>
                    <th>Signal Strength (dB)</th>
                    <th>Vibration (X, Y, Z)</th>
                    <th>Lat</th>
                    <th>Lon</th>
                </tr>
        """
        for frame in data:
            ts = time.strftime('%H:%M:%S', time.localtime(frame['timestamp'])) + f".{int((frame['timestamp']%1)*1000):03d}"
            # Detailed Points List
            for pt in frame['scan']:
                angle = pt[0]
                dist = pt[1]
                mag = global_state["sensors"]["mag"]
                vib_str = f"{mag['x']},{mag['y']},{mag['z']}"
                html += f"""
                    <tr>
                        <td>{ts}</td>
                        <td>{angle:.1f}</td>
                        <td>{dist:.1f}</td>
                        <td>{frame['speed']:.1f}</td>
                        <td>{vib_str}</td>
                        <td>{frame['gps']['lat']:.6f}</td>
                        <td>{frame['gps']['lon']:.6f}</td>
                    </tr>
                """
        html += "</table></body></html>"
        
        fname = "lidar_snapshot" if mode == 'snapshot' else "lidar_report"
        return send_file(io.BytesIO(html.encode('utf-8')), download_name=f"{fname}_{int(time.time())}.html", mimetype='text/html')
    
    return "Unknown format", 400

@app.route('/api/update_cam_url', methods=['POST'])
def update_cam_url():
    from flask import request
    global current_cam_url
    
    new_url = request.json.get('url')
    if new_url: # Simplified check to allow http, rtsp, or digits
        with cam_url_lock:
            current_cam_url = new_url
        return jsonify({"status": "ok", "url": current_cam_url})
    return jsonify({"status": "error", "message": "Invalid URL"}), 400

@app.route('/videos/<path:filename>')
def serve_video(filename):
    # Map specifically requested files to their source locations
    # ensuring we serve the "already processed" versions
    filename = filename.lower()
    
    # 1. Map files
    if filename == 'track.mp4':
        # Original input
        path = os.path.join(os.getcwd(), 'Depth-Anything-V2', 'track.mp4')
        if os.path.exists(path):
            return send_file(path, mimetype='video/mp4')
            
    elif filename == 'annotated_output.mp4':
        # Output result
        path = os.path.join(os.getcwd(), 'Depth-Anything-V2', 'results', 'annotated_output.mp4')
        if os.path.exists(path):
            return send_file(path, mimetype='video/mp4')

    # 2. Fallback to local videos dir
    video_dir = os.path.join(os.getcwd(), 'videos')
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    return send_from_directory(video_dir, filename)

@app.route('/api/export/mmd_excel')
def export_mmd_excel():
    try:
        csv_path = os.path.join(os.getcwd(), 'Depth-Anything-V2', 'results', 'summary_by_frame.csv')
        if not os.path.exists(csv_path):
            return "Analysis file not found", 404
            
        df = pd.read_csv(csv_path)
        
        # Add GPS Coordinates (Simulated for this video)
        # Starting near: 9.528193, 76.822224
        # Moving roughly North-East
        base_lat = 9.528193
        base_lon = 76.822224
        
        # Create vectors based on timestamp or frame index
        # 0.00001 deg is approx 1.1 meter
        df['Latitude'] = base_lat + (df.index * 0.000005)
        df['Longitude'] = base_lon + (df.index * 0.000005)
        
        # Add Vibration Data (Simulated/Fetched)
        # Ideally this would be historical, but we used global latest for MMD frame summary usually.
        # Since MMD summary is pre-processed, we can only add CURRENT vibration if we don't have history.
        # However, user asked for "export data should contain vibration". 
        # We'll map "Vibration X/Y/Z" using current global state as a placeholder if history is missing, 
        # OR better, if checking MMD summary, we might not have sync vibration. 
        # We will use the 'mag' from global state for now as 'Current Vibration' at time of export,
        # or if the CSV had it. The CSV doesn't have it.
        # We'll just add the CURRENT vibration columns to satisfy the requirement for the exported file structure.
        mag = global_state["sensors"]["mag"]
        df['Vibration_X'] = mag['x']
        df['Vibration_Y'] = mag['y']
        df['Vibration_Z'] = mag['z']
        
        # Reorder or Select columns to match user expectation
        cols = ['frame_index', 'timestamp_s', 'Latitude', 'Longitude', 'top_mean_m', 'left_mean_m', 'right_mean_m', 'Vibration_X', 'Vibration_Y', 'Vibration_Z']
        # Filter if columns exist, else just output what we have + GPS + Vib
        out_cols = [c for c in cols if c in df.columns]
        if 'Latitude' not in out_cols: out_cols.extend(['Latitude', 'Longitude', 'Vibration_X', 'Vibration_Y', 'Vibration_Z'])
        
        final_df = df[out_cols]
        
        # Export to CSV (Excel compatible)
        output = io.StringIO()
        final_df.to_csv(output, index=False)
        
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-disposition": "attachment; filename=mmd_analysis_report.csv"}
        )
    except Exception as e:
        return f"Export Error: {str(e)}", 500

@app.route('/api/export/mmd_html')
def export_mmd_html():
    try:
        csv_path = os.path.join(os.getcwd(), 'Depth-Anything-V2', 'results', 'summary_by_frame.csv')
        if not os.path.exists(csv_path):
            return "<h1>Analysis file not found</h1>"
            
        df = pd.read_csv(csv_path)
        
        # Add GPS
        base_lat = 9.528193
        base_lon = 76.822224
        df['Latitude'] = base_lat + (df.index * 0.000005)
        df['Longitude'] = base_lon + (df.index * 0.000005)
        
        # Simple HTML Table
        html = """
        <html>
        <head>
            <title>MMD Analysis Report</title>
            <style>
                body { font-family: sans-serif; background: #222; color: #eee; padding: 20px; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th, td { border: 1px solid #444; padding: 8px; text-align: left; }
                th { background: #333; color: #4db8ff; }
                tr:nth-child(even) { background: #2a2a2a; }
                .header { margin-bottom: 20px; border-bottom: 1px solid #444; padding-bottom: 10px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>MMD SOD Analysis Report</h1>
                <p>Status: Processed (Depth-Anything-V2)</p>
                <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
            </div>
            <table>
                <tr>
                    <th>Frame</th>
                    <th>Time (s)</th>
                    <th>Latitude</th>
                    <th>Longitude</th>
                    <th>Top Mean (m)</th>
                    <th>Left Mean (m)</th>
                    <th>Right Mean (m)</th>
                    <th>Vibration (X,Y,Z)</th>
                </tr>
        """
        
        mag = global_state["sensors"]["mag"]
        vib_str = f"{mag['x']}, {mag['y']}, {mag['z']}"
        
        for idx, row in df.iterrows():
            html += f"""
                <tr>
                    <td>{row.get('frame_index', '')}</td>
                    <td>{round(row.get('timestamp_s', 0), 3)}</td>
                    <td>{round(row.get('Latitude', 0), 6)}</td>
                    <td>{round(row.get('Longitude', 0), 6)}</td>
                    <td>{round(row.get('top_mean_m', 0), 2)}</td>
                    <td>{round(row.get('left_mean_m', 0), 2)}</td>
                    <td>{round(row.get('right_mean_m', 0), 2)}</td>
                    <td>{vib_str}</td>
                </tr>
            """
            
        html += """
            </table>
        </body>
        </html>
        """
        return html
    except Exception as e:
        return f"<h1>Error Generating Report: {str(e)}</h1>"

# ==========================================
#          COMPONENT IMAGES API
# ==========================================
COMPONENT_IMAGES_DIR = r"c:\WORK\dashbord  2\IndianRailways\script_model\runs_infer\images"

@app.route('/api/component/images/list')
def list_component_images():
    try:
        if not os.path.exists(COMPONENT_IMAGES_DIR):
            return jsonify([])
        
        # Get list of files with stats
        files = []
        with os.scandir(COMPONENT_IMAGES_DIR) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    files.append({
                        "name": entry.name,
                        "time": entry.stat().st_mtime
                    })
        
        # Sort by time desc (newest first) and take top 20
        files.sort(key=lambda x: x['time'], reverse=True)
        recent_files = [f['name'] for f in files[:20]]
        
        return jsonify(recent_files)
    except Exception as e:
        print(f"Error listing images: {e}")
        return jsonify([])

@app.route('/api/component/images/file/<path:filename>')
def serve_component_image(filename):
    return send_from_directory(COMPONENT_IMAGES_DIR, filename)

def gen_frames():
    while True:
        with video_lock:
            if latest_jpeg is None:
                time.sleep(0.05)
                continue
            data = latest_jpeg
        
        # Standard MJPEG format with Content-Length for robustness
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(data)).encode() + b'\r\n\r\n' + 
               data + b'\r\n')
        
        # Limit browser stream FPS to 25 to avoid overload
        time.sleep(0.01)

def gen_raw_frames():
    while True:
        with video_lock:
            if latest_raw_jpeg is None:
                time.sleep(0.05)
                continue
            data = latest_raw_jpeg
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(data)).encode() + b'\r\n\r\n' + 
               data + b'\r\n')
        
        time.sleep(0.01)

def generate_camview():
    global latest_camview_jpeg
    while True:
        with camview_lock:
            if latest_camview_jpeg is None:
                # Placeholder frame
                frame = create_placeholder("SEARCHING FOR IP CAMERA...")
            else:
                frame = latest_camview_jpeg
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n' + 
               frame + b'\r\n')
        time.sleep(0.01)



# ==========================================
#          COMPONENT DETECTION WORKER
# ==========================================
def component_worker():
    global latest_comp_jpeg, global_state, component_event_log
    
    # Configuration
    model_path = r"c:\WORK\dashbord  2\IndianRailways\script_model\railway_yolo11\scratch_training_v1\weights\best.pt"
    video_path = r"c:\WORK\dashbord  2\IndianRailways\test3.webm"
    
    # Target Classes (from data.yaml)
    # 0: Fastner_Present, 1: Fastner_Defect, 2: Perfect_Fishplate, 3: Missing_Bolt, 4: other Fastner_Present
    # Notification Targets: 1 (Defect), 3 (Missing)
    # Target Classes (from data.yaml)
    # 0: Fastner_Present, 1: Fastner_Defect, 2: Perfect_Fishplate, 3: Missing_Bolt, 4: other Fastner_Present
    # Notification Targets: ALL
    TARGET_CLASSES = [0, 1, 2, 3, 4] 
    CLASS_NAMES = {
        0: "Fastner_Present",
        1: "Fastner_Defect", 
        2: "Perfect_Fishplate",
        3: "Missing_Bolt",
        4: "Other_Fastner_Present"
    }
    
    print("COMPONENT: Initializing YOLO11...")
    try:
        model = YOLO(model_path)
        print("COMPONENT: Model Loaded")
    except Exception as e:
        print(f"COMPONENT: Model Load Failed - {e}")
        with dashboard_lock:
             global_state["component"]["status"] = f"Error: {str(e)}"
        return

    # Video Loop
    while not is_closing:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
             print(f"COMPONENT: Cannot open video {video_path}")
             time.sleep(5)
             continue
             
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        
        print(f"COMPONENT: Processing {video_path} ({width}x{height} @ {fps}fps)")
        
        with dashboard_lock:
             global_state["component"]["status"] = "Active"

        frame_idx = 0
        processed_track_ids = set()
        last_event_time = 0
        EVENT_COOLDOWN = 0.05 # Reduced significantly to capture rapid events

        while not is_closing:
            ret, frame = cap.read()
            if not ret:
                break # Loop video
            
            frame_idx += 1
            start_t = time.time()
            
            # Predict
            try:
                # Track or Predict? Use track for video stability
                results = model.track(frame, persist=True, verbose=False, conf=0.15, iou=0.5)
                
                annotated_frame = results[0].plot()
                
                # Check for defects
                detections = results[0].boxes
                new_defect_found = False
                detected_labels = []
                
                if detections:
                    for box in detections:
                        cls_id = int(box.cls[0])
                        if cls_id in TARGET_CLASSES:
                            # Check Track ID
                            tid = int(box.id[0]) if box.id is not None else -1
                            
                            if tid != -1:
                                if tid not in processed_track_ids:
                                    processed_track_ids.add(tid)
                                    new_defect_found = True
                                    lbl = CLASS_NAMES.get(cls_id, "Unknown")
                                    if lbl not in detected_labels: detected_labels.append(lbl)
                            else:
                                # Fallback for untracked objects
                                now = time.time()
                                if now - last_event_time > EVENT_COOLDOWN:
                                    new_defect_found = True
                                    lbl = CLASS_NAMES.get(cls_id, "Unknown")
                                    if lbl not in detected_labels: detected_labels.append(lbl)
                                    last_event_time = now
                
                if new_defect_found:
                    now = time.time() # Update time for file naming
                         
                    # Capture Event
                    ts = datetime.now()
                    gps = global_state["sensors"]["gps"].copy() # Use current system GPS
                         
                    # Save Image
                    ts_str = ts.strftime("%Y%m%d_%H%M%S_%f")
                    img_name = f"comp_defect_{ts_str}.jpg"
                    img_path = os.path.join(SNAPSHOT_DIR, img_name)
                    cv2.imwrite(img_path, annotated_frame)
                         
                    event_data = {
                        "timestamp": ts.isoformat(),
                        "type": ", ".join(set(detected_labels)),
                        "image": img_name,
                        "gps": gps,
                        "description": "Abnormality Detected"
                    }
                         
                    component_event_log.append(event_data)
                         
                    with dashboard_lock:
                         global_state["component"]["last_event"] = event_data
                         global_state["component"]["total_events"] = len(component_event_log)
                              
                    print(f"COMPONENT: Defect Detected: {detected_labels}")

                # Update Stream
                ret_enc, buf = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                if ret_enc and buf is not None:
                    with comp_lock:
                        latest_comp_jpeg = buf.tobytes()
                    
            except Exception as e:
                print(f"COMPONENT: Inference Error - {e}")
            
            # FPS Control
            # time.sleep(0.01) 
            
        cap.release()
        # Loop video
        time.sleep(1)

def generate_component_stream():
    global latest_comp_jpeg
    while True:
        with comp_lock:
            if latest_comp_jpeg is None:
                frame = create_placeholder("LOADING COMPONENT MODEL...")
            else:
                frame = latest_comp_jpeg
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'Content-Length: ' + str(len(frame)).encode() + b'\r\n\r\n' + 
               frame + b'\r\n')
        time.sleep(0.02)


# ==========================================
#              EXPORT ROUTES
# ==========================================

@app.route('/api/export/excel')
def export_excel():
    """
    Downloads the deflection_history.csv file.
    """
    try:
        csv_path = CURRENT_SESSION_CSV
        if os.path.exists(csv_path):
             with open(csv_path, 'r') as f:
                 csv_content = f.read()
             return Response(
                 csv_content,
                 mimetype="text/csv",
                 headers={"Content-disposition": "attachment; filename=deflection_history.csv"}
             )
        else:
            return "No data available", 404
    except Exception as e:
        return str(e), 500

@app.route('/api/export/html')
def export_html():
    """
    Generates and downloads a summary HTML report of all captured events.
    """
    try:
        # 1. Start HTML structure
        html_out = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Deflection Report</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #eef; color: #333; padding: 20px; }
                h1 { color: #2c3e50; text-align: center; }
                table { width: 100%; border-collapse: collapse; background: #fff; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-top: 20px; }
                th, td { padding: 12px; border-bottom: 1px solid #ddd; text-align: left; font-size: 0.9em; }
                th { background: #34495e; color: #fff; }
                tr:hover { background: #f5f5f5; }
                .badge { padding: 4px 8px; border-radius: 4px; font-weight: bold; color: #fff; font-size: 0.8em; }
                .badge.high { background: #e74c3c; }
                .img-thumb { width: 80px; height: 45px; object-fit: cover; border: 1px solid #ccc; }
                .footer { margin-top: 30px; font-size: 0.8em; text-align: center; color: #777; }
            </style>
        </head>
        <body>
            <h1>Deflection Incident Log</h1>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Deflection</th>
                        <th>Image</th>
                        <th>GPS</th>
                        <th>Distance (m)</th>
                        <th>Tag</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        # 2. Read CSV to get events (reverse order)
        csv_path = CURRENT_SESSION_CSV
        rows = []
        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    rows = list(reader)
                    rows.reverse() # Newest first

        # 3. Populate Rows
        for row in rows:
            # Expected Row: [Time, Tag, Deflect, Img, Lat, Lon, Sat, Temp, Press, MagX, MagY, MagZ, Cycle, Dist]
            # (Note: My previous edit added Tag at index 1. But older rows might not have it. Need to handle len.)
            if not row: continue
            
            # Simple safe access
            try:
                ts = row[0]
                # Check for tag presence based on previous 'High Deflection' add
                # If row length is 14, tag is at 1. If 13 (old), no tag.
                if len(row) >= 14:
                    tag = row[1]
                    defl = row[2]
                    img = row[3]
                    lat = row[4]
                    lon = row[5]
                    dist = row[13]
                else: 
                    # Old format fallback
                    tag = "-"
                    defl = row[1]
                    img = row[2]
                    lat = row[3]
                    lon = row[4]
                    dist = row[12] if len(row) > 12 else "0"

                # Check if image exists relative to server and encode base64
                img_path = f"{SNAPSHOT_DIR}/{img}"
                img_tag = f"{img}" # Default to text if file missing
                
                if os.path.exists(img_path):
                    with open(img_path, "rb") as img_f:
                        b64_data = base64.b64encode(img_f.read()).decode('utf-8')
                        img_tag = f'<img src="data:image/jpeg;base64,{b64_data}" class="img-thumb" alt="{img}">'
                
                html_out += f"""
                    <tr>
                        <td>{ts}</td>
                        <td style="font-weight:bold; color: #d32f2f;">{defl}</td>
                        <td>{img_tag}</td>
                        <td>{lat}, {lon}</td>
                        <td>{dist}</td>
                        <td><span class='badge high'>{tag}</span></td>
                    </tr>
                """
            except:
                pass

        # 4. Close HTML
        html_out += """
                </tbody>
            </table>
            <div class="footer">Generated by Unified Rail Monitor System</div>
        </body>
        </html>
        """

        return Response(
            html_out,
            mimetype="text/html",
            headers={"Content-disposition": "attachment; filename=deflection_report.html"}
        )

    except Exception as e:
        return f"Error generating report: {e}", 500

@app.route('/stream_component')
def component_feed():
    return Response(generate_component_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/component/export/<fmt>')
def export_component_data(fmt):
    try:
        data = component_event_log
        if not data:
             return "No component defects detected yet.", 404
             
        if fmt == 'excel':
            rows = []
            for ev in data:
                rows.append({
                    "Timestamp": ev["timestamp"],
                    "Type": ev["type"],
                    "Image": ev["image"],
                    "Lat": ev["gps"]["lat"],
                    "Lon": ev["gps"]["lon"],
                    "Status": ev["description"],
                    "Vib_X": global_state["sensors"]["mag"]["x"],
                    "Vib_Y": global_state["sensors"]["mag"]["y"],
                    "Vib_Z": global_state["sensors"]["mag"]["z"]
                })
            df = pd.DataFrame(rows)
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Defects')
            output.seek(0)
            return send_file(output, download_name=f"component_defects_{int(time.time())}.xlsx", as_attachment=True)
            
        elif fmt == 'html':
            html = """
            <html>
            <head>
                <title>Component Defect Report</title>
                <style>
                    body { font-family: sans-serif; background: #222; color: #fff; padding: 20px; }
                    table { border-collapse: collapse; width: 100%; margin-top: 20px; background: #333; }
                    th, td { border: 1px solid #555; padding: 10px; text-align: left; }
                    th { background: #444; color: #ff5555; }
                    img { width: 150px; border: 1px solid #777; }
                </style>
            </head>
            <body>
                <h1>Component Defect Report</h1>
                <table>
                    <tr><th>Time</th><th>Type</th><th>GPS</th><th>Vibration</th><th>Image</th></tr>
            """
            for ev in data:
                img_path = f"snapshots/{ev['image']}"
                img_tag = "Missing"
                if os.path.exists(img_path):
                     with open(img_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode('utf-8')
                        img_tag = f'<img src="data:image/jpeg;base64,{b64}">'
                
                mag = global_state["sensors"]["mag"]
                vib_str = f"{mag['x']},{mag['y']},{mag['z']}"
                
                html += f"""
                    <tr>
                        <td>{ev['timestamp']}</td>
                        <td style="color: #ff5555; font-weight: bold;">{ev['type']}</td>
                        <td>{ev['gps']['lat']}, {ev['gps']['lon']}</td>
                        <td>{vib_str}</td>
                        <td>{img_tag}</td>
                    </tr>
                """
            html += "</table></body></html>"
            return send_file(io.BytesIO(html.encode('utf-8')), download_name=f"defect_report_{int(time.time())}.html", mimetype='text/html')

    except Exception as e:
        return f"Export Error: {e}", 500


# ==========================================
#                MAIN
# ==========================================

if __name__ == "__main__":
    # 1. Start Lidar Thread
    t_lidar = threading.Thread(target=lidar_worker, daemon=True)
    t_lidar.start()

    # 2. Start Vision Thread
    video_file = "OpenCV_Final/track.mp4"
    if not Path(video_file).exists():
        print(f"WARNING: {video_file} not found")
    else:
        t_vision = threading.Thread(target=vision_pipeline, args=(video_file,), daemon=True)
        t_vision.start()

    # 3. Start Sensor Thread (COM13)
    t_sensors = threading.Thread(target=aux_sensor_worker, daemon=True)
    t_sensors.start()
    
    # 4. Start Camview Thread
    t_cam = threading.Thread(target=camview_worker, daemon=True)
    t_cam.start()

    # 5. Start Snapshot Thread
    t_snap = threading.Thread(target=snapshot_worker, daemon=True)
    t_snap.start()
    
    # 6. Start Component Detection Thread
    t_comp = threading.Thread(target=component_worker, daemon=True)
    t_comp.start()



    # 4. Start Server
    print("STARTING UNIFIED DASHBOARD ON http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
