# train_yolo11_railway.py

from ultralytics import YOLO


def main():
    # ---------------------------------------------
    # 1. Build YOLO11 *medium* model from YAML only
    #    -> this uses RANDOM WEIGHTS (no pretrained .pt)
    # ---------------------------------------------
    model = YOLO("yolo11m.yaml")   # change to 'yolo11n.yaml' if GPU is weak

    # ---------------------------------------------
    # 2. Train from scratch on your railway dataset
    # ---------------------------------------------
    model.train(
        data="data.yaml",          # must be in the same folder as this script
        epochs=400,                # upper limit; early stopping may stop earlier
        patience=40,               # stop if no improvement for 40 epochs
        imgsz=640,                 # your images are 640x640
        batch=4,                   # good start for RTX 3050 75W
        device=0,                  # GPU 0; use 'cpu' if no GPU
        project="railway_yolo11",  # top-level results folder
        name="scratch_training_v1" # experiment name inside project
    )

    print("\nTraining finished.")
    print("Best model should be here:")
    print(r"railway_yolo11\scratch_training_v1\weights\best.pt")


if __name__ == "__main__":
    main()
