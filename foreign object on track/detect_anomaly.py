import torch
import cv2
import time
import os
import numpy as np
from datetime import datetime

# --- CONFIGURATION ---
# Region of Interest (ROI) for the track. 
# Format: [x1, y1, x2, y2] (Top-Left and Bottom-Right coordinates)
# Adjust these values based on your camera view to focus ONLY on the track.
# For a 640x480 feed, this example covers the central bottom part.
TRACK_ROI = [100, 100, 540, 480] 

# Calibration constant: How many pixels represent 1 meter at the detection distance?
# You MUST calibrate this for your specific camera setup and distance.
# Example: If a 1 meter wide track appears as 200 pixels wide on screen.
PIXELS_PER_METER = 200.0 

# Confidence threshold for detection
CONF_THRESHOLD = 0.4

# Classes to consider as "Foreign Objects" (COCO class IDs)
# 0: person, 1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck, 
# 15: cat, 16: dog, 17: horse, 18: sheep, 19: cow, 24: backpack, 26: handbag, 28: suitcase
TARGET_CLASSES = [0, 1, 2, 3, 5, 7, 15, 16, 17, 18, 19, 24, 26, 28]

# Snapshot directory
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

def detect_anomaly():
    print("Loading YOLOv5 model...")
    # Load model from PyTorch Hub (uses 'ultralytics/yolov5', 'yolov5s' is the small, fast model)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    
    # Configure model settings
    model.conf = CONF_THRESHOLD  # Confidence threshold
    model.classes = TARGET_CLASSES # Filter specific classes

    print("Starting video stream...")
    # Open default camera (0). Replace with video file path 'video.mp4' if needed.
    cap = cv2.VideoCapture(0) 

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Set resolution (Increased for better sampling/detail)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    last_snapshot_time = 0
    snapshot_cooldown = 0.5 # Reduced cooldown for higher sampling frequency (0.5s)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # 1. Draw ROI on the frame (Visual guide for the track area)
        # We draw it in Blue
        cv2.rectangle(frame, (TRACK_ROI[0], TRACK_ROI[1]), (TRACK_ROI[2], TRACK_ROI[3]), (255, 0, 0), 2)
        cv2.putText(frame, "TRACK ZONE", (TRACK_ROI[0], TRACK_ROI[1] - 10), font, 0.5, (255, 0, 0), 1)

        # 2. Perform Detection
        # Convert frame to RGB (YOLO expects RGB)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)

        # 3. Process Detections
        # results.xyxy[0] is a tensor: [x1, y1, x2, y2, confidence, class]
        detections = results.xyxy[0].cpu().numpy()

        anomaly_present = False

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Calculate center of the detected object
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Check if the object's center is inside the Track ROI
            if (TRACK_ROI[0] < center_x < TRACK_ROI[2]) and (TRACK_ROI[1] < center_y < TRACK_ROI[3]):
                anomaly_present = True
                
                # --- Dimension Estimation ---
                width_px = x2 - x1
                height_px = y2 - y1
                
                width_m = width_px / PIXELS_PER_METER
                height_m = height_px / PIXELS_PER_METER
                
                label = f"{model.names[int(cls)]} {conf:.2f}"
                dim_label = f"{width_m:.2f}m x {height_m:.2f}m"

                # Draw Bounding Box (Red for Anomaly)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 25), font, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, dim_label, (x1, y1 - 10), font, 0.5, (0, 255, 255), 1)
            else:
                # Object detected but outside track zone (Green box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # 4. Alert and Snapshot
        if anomaly_present:
            cv2.putText(frame, "WARNING: FOREIGN OBJECT ON TRACK!", (50, 50), font, 1.0, (0, 0, 255), 3)
            
            current_time = time.time()
            if current_time - last_snapshot_time > snapshot_cooldown:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{SNAPSHOT_DIR}/anomaly_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Snapshot saved: {filename}")
                last_snapshot_time = current_time

        # Show the frame
        cv2.imshow('Railway Foreign Object Detection', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_anomaly()
