# railway_model_pipeline.py
# Industry-style pipeline for YOLO11 railway fault detection
# - Inference on images / folders / video
# - Evaluation on val/test split
# - Export for deployment (ONNX, TorchScript)

from ultralytics import YOLO
from pathlib import Path

import cv2

# -----------------------
# CONFIGURATION
# -----------------------
# Base project directory (where this script and data.yaml live)
BASE_DIR = Path(__file__).resolve().parent

# Trained model path (from training script)
MODEL_PATH = r"D:\IndianRailways\script_model\railway_yolo11\scratch_training_v1\weights\best.pt"


# Data config path
DATA_YAML = r"D:\IndianRailways\script_model\data.yaml"

# Default folder for manual testing images
TEST_IMAGES_DIR = r"D:\IndianRailways\dataset\images\test"


# -----------------------
# 1. LOAD MODEL
# -----------------------
def load_model():
    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))
    return model


# -----------------------
# 2. INFERENCE FUNCTIONS
# -----------------------
def infer_on_folder(folder_path: Path, conf: float = 0.25):
    """
    Run inference on all images in a folder and save annotated results.
    """
    model = load_model()
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    print(f"[INFO] Running inference on folder: {folder_path}")
    results = model.predict(
        source=str(folder_path),
        imgsz=640,
        conf=conf,
        save=True,       # saves annotated images
        project=str(BASE_DIR / "runs_infer"),
        name="images"
    )

    print(f"[INFO] Inference complete. "
          f"Annotated images saved under: {BASE_DIR / 'runs_infer' / 'images'}")
    return results


def infer_on_image(image_path: Path, conf: float = 0.25):
    """
    Run inference on a single image.
    """
    model = load_model()
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"[INFO] Running inference on image: {image_path}")
    results = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=conf,
        save=True,
        project=str(BASE_DIR / "runs_infer"),
        name="single_image"
    )

    print(f"[INFO] Result saved under: {BASE_DIR / 'runs_infer' / 'single_image'}")
    return results


def infer_on_video(video_path: Path, conf: float = 0.25):
    """
    Safe + optimized rail-video inference with YOLO11 and tracking (BoT-SORT).
    Handles:
        - Motion blur
        - High FPS fast trains
        - High-resolution videos (4K, 2K)
        - Corrupted frames
        - GPU memory limits
    Saves a fully annotated video with TRACK IDs.
    """

    model = load_model()
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"[INFO] Starting YOLO11 tracking on video: {video_path}")

    # -----------------------------
    # Open video safely with OpenCV
    # -----------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"[ERROR] Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Original resolution: {width}x{height}")
    print(f"[INFO] FPS: {original_fps}")

    # Output folder
    output_dir = BASE_DIR / "runs_infer" / "video_tracking"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{video_path.stem}_tracked.mp4"

    # FORCE SAFE OUTPUT SIZE FOR FAST TRAINS
    out_w, out_h = 1280, 720

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, original_fps, (out_w, out_h))

    print(f"[INFO] Output will be saved at: {output_path}")

    frame_no = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video reached.")
            break

        frame_no += 1
        if frame is None:
            print(f"[WARN] Skipped corrupted frame #{frame_no}")
            continue

        # Resize frame to avoid memory errors
        frame = cv2.resize(frame, (out_w, out_h))

        # ---------------------------------------
        # YOLO11 + BoT-SORT Tracking on 1 frame
        # ---------------------------------------
        try:
            results = model.track(
                source=frame,              # single frame
                imgsz=640,
                conf=conf,
                iou=0.5,
                stream=False,
                verbose=False,
                tracker="botsort.yaml",   # real tracker
                device=0
            )
        except Exception as e:
            print(f"[WARN] Frame #{frame_no} skipped due to error: {e}")
            continue

        # Draw detections
        annotated_frame = results[0].plot()

        # Save annotated frame
        out.write(annotated_frame)

        if frame_no % 50 == 0:
            print(f"[INFO] Processed {frame_no} frames...")

    # Cleanup
    cap.release()
    out.release()

    print("[INFO] Tracking complete.")
    print(f"[INFO] Video saved at: {output_path}")

# -----------------------
# 3. EVALUATION FUNCTIONS
# -----------------------
def evaluate_on_split(split: str = "val"):
    """
    Evaluate model on a dataset split: 'val' or 'test'.
    Uses data.yaml definitions.
    """
    assert split in ("val", "test"), "split must be 'val' or 'test'"

    model = load_model()
    print(f"[INFO] Evaluating on {split} split using {DATA_YAML}")

    metrics = model.val(
        data=str(DATA_YAML),
        split=split,       # "val" or "test"
        plots=True         # saves PR, F1, confusion matrix etc.
    )

    print("\n[INFO] Evaluation complete.")
    print(f"      mAP50:     {metrics.box.map50:.3f}")
    print(f"      mAP50-95:  {metrics.box.map:.3f}")
    print(f"      Results & plots saved in: {model.results_dir}")
    return metrics


# -----------------------
# 4. EXPORT FUNCTIONS
# -----------------------
def export_model(formats=("onnx", "torchscript")):
    """
    Export trained model to multiple formats for deployment.
    """
    model = load_model()

    for fmt in formats:
        print(f"[INFO] Exporting model to {fmt}...")
        model.export(format=fmt)
        # exports to BASE_DIR / "railway_yolo11" / "scratch_training_v1" / "weights" / model.<fmt>
    print("[INFO] Export complete. Check weights folder for exported files.")


# -----------------------
# 5. MAIN MENU
# -----------------------
def main():
    # Simple CLI-style menu (you can also call functions directly)
    print("\n==== Railway Fault YOLO11 Pipeline ====")
    print("1. Inference on test_images folder")
    print("2. Inference on a single image")
    print("3. Inference on a video")
    print("4. Evaluate on validation set")
    print("5. Evaluate on test set")
    print("6. Export model (ONNX + TorchScript)")
    print("0. Exit")

    choice = input("Select an option: ").strip()

    if choice == "1":
        # Ensure test_images folder exists and has images
        infer_on_folder(TEST_IMAGES_DIR, conf=0.25)

    elif choice == "2":
        path = input("Enter path to image: ").strip().strip('"')
        infer_on_image(Path(path), conf=0.25)

    elif choice == "3":
        path = input("Enter path to video: ").strip().strip('"')
        infer_on_video(Path(path), conf=0.25)
        
    elif choice == "4":
        evaluate_on_split(split="val")

    elif choice == "5":
        evaluate_on_split(split="test")

    elif choice == "6":
        export_model()

    else:
        print("Exiting.")


if __name__ == "__main__":
    main()
