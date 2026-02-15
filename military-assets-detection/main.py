import os
import argparse
import random
import shutil
from pathlib import Path
from ultralytics import YOLO

DATA_ROOT = "data/amad-5"  # adjust if needed

def train():
    """Train YOLO11n on AMAD-5 dataset."""
    data_yaml = os.path.join(DATA_ROOT, "dataset.yaml")
    if not os.path.exists(data_yaml):
        print(f"dataset.yaml not found at {data_yaml}. Please ensure the dataset is placed correctly.")
        return

    model = YOLO("yolo11n.pt")
    model.train(
        data=data_yaml,
        epochs=20,
        imgsz=1024,
        batch=24,
        lr0=0.01,
        patience=25,
        project="military_training",
        name="yolo11n_iter1",
        save=True,
        save_period=5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1
    )
    print("Training completed. Best weights saved to military_training/yolo11n_iter1/weights/best.pt")

def evaluate():
    """Evaluate on test set."""
    model_path = "military_training/yolo11n_iter1/weights/best.pt"
    if not os.path.exists(model_path):
        print("Trained model not found. Please train first.")
        return

    test_images = os.path.join(DATA_ROOT, "test/images")
    if not os.path.exists(test_images):
        print("Test images not found.")
        return

    model = YOLO(model_path)
    metrics = model.val(data=os.path.join(DATA_ROOT, "dataset.yaml"), split="test")
    print(metrics)

def predict(source):
    """Run inference on an image or folder."""
    model_path = "military_training/yolo11n_iter1/weights/best.pt"
    if not os.path.exists(model_path):
        print("Trained model not found. Please train first.")
        return

    model = YOLO(model_path)
    model.predict(source=source, conf=0.45, save=True)
    print(f"Predictions saved to runs/detect/predict/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "evaluate", "predict"], required=True)
    parser.add_argument("--source", type=str, help="Image or folder for prediction")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "evaluate":
        evaluate()
    elif args.mode == "predict":
        if not args.source:
            print("Please provide --source argument")
        else:
            predict(args.source)
