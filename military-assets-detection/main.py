import os
import argparse
import random
import shutil
from pathlib import Path
from ultralytics import YOLO

# hardcoding this for now , if the dataset moves I will just change it here instead of hunting it down everywhere
DATA_ROOT = "data/amad-5"


def train():
    # making sure the yaml is actually there
    data_yaml = os.path.join(DATA_ROOT, "dataset.yaml")
    if not os.path.exists(data_yaml):
        print(f"dataset.yaml not found at {data_yaml}. Please ensure the dataset is placed correctly.")
        return

    # yolo11n , going nano first, always. validate the pipeline works before scaling up
    model = YOLO("yolo11n.pt")

    model.train(
        data=data_yaml,
        epochs=20,           # 20 to start, not trying to overfit on iteration 1
        imgsz=1024,          # keeping res high , missing small targets at 640 last time
        batch=24,            # 24 fits on my gpu without OOM, don't touch this
        lr0=0.01,            # default lr, not overthinking it yet
        patience=25,         # giving it room to breathe before early stop kicks in
        project="military_training",
        name="yolo11n_iter1",  # iter1 so i know this isn't the final model
        save=True,
        save_period=5,       # checkpoint every 5 epochs , learned this the hard way after a crash
        mosaic=1.0,          # full mosaic, good for small object detection
        mixup=0.1,           # light mixup, just enough to help generalization
        copy_paste=0.1       # small object augmentation , helpful for this dataset specifically
    )

    # reminder to myself where the weights actually end up
    print("Training completed. Best weights saved to military_training/yolo11n_iter1/weights/best.pt")


def evaluate():
    # don't even try running this without training first , it will just error out
    model_path = "military_training/yolo11n_iter1/weights/best.pt"
    if not os.path.exists(model_path):
        print("Trained model not found. Please train first.")
        return

    # same check for test images , if the split isn't there the val will fail silently in weird ways
    test_images = os.path.join(DATA_ROOT, "test/images")
    if not os.path.exists(test_images):
        print("Test images not found.")
        return

    model = YOLO(model_path)

    # explicitly passing split="test" so it doesn't accidentally evaluate on val
    metrics = model.val(data=os.path.join(DATA_ROOT, "dataset.yaml"), split="test")
    print(metrics)


def predict(source):
    # same guard as evaluate , no model, no point
    model_path = "military_training/yolo11n_iter1/weights/best.pt"
    if not os.path.exists(model_path):
        print("Trained model not found. Please train first.")
        return

    model = YOLO(model_path)

    # conf=0.45 , tuned this manually, below 0.45 gets noisy with false positives on this class set
    model.predict(source=source, conf=0.45, save=True)

    # results go here by default, ultralytics doesn't make this obvious
    print(f"Predictions saved to runs/detect/predict/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "evaluate", "predict"], required=True)
    parser.add_argument("--source", type=str, help="Image or folder for prediction")
    args = parser.parse_args()

    # straightforward dispatch , keeping this flat, no need to over-engineer the entrypoint
    if args.mode == "train":
        train()
    elif args.mode == "evaluate":
        evaluate()
    elif args.mode == "predict":
        # catching the missing source here instead of letting it fail inside predict()
        if not args.source:
            print("Please provide --source argument")
        else:
            predict(args.source)
