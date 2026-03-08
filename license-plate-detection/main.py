import os
import argparse
import cv2
import torch
from ultralytics import YOLO


def train():
    # writing the yaml inline instead of maintaining a separate file — one less thing to lose track of
    # nc=1 because we only care about the plate itself, not the characters on it
    data_yaml = """
train: data/license-plate-dataset/images/train
val: data/license-plate-dataset/images/val
nc: 1
names: ['license_plate']
"""
    # this overwrites every time — intentional, keeps it in sync with whatever's hardcoded above
    with open("data.yaml", "w") as f:
        f.write(data_yaml)

    # nano model — fast to iterate, can always swap to yolo11s or yolo11m later if precision suffers
    model = YOLO("yolo11n.pt")

    model.train(
        data="data.yaml",
        epochs=50,           # 50 is reasonable for a single-class detector, not trying to overdo it
        imgsz=640,           # plates are usually readable at 640, no need to go higher
        batch=32,            # bumped from 24 — single class, smaller model, can afford it
        lr0=0.0005,          # lower starting lr than default — helps with fine-tuning on specific domains
        lrf=0.1,             # cosine decay down to lr0 * lrf at the end
        augment=True         # letting ultralytics handle augmentation, good enough here
    )

    # weights end up here by default — just a reminder so i don't go hunting for them
    print("Training completed. Best weights saved to runs/detect/train/weights/best.pt")


def predict(video_path):
    # same guard as always — running predict before train is a common mistake
    model_path = "runs/detect/train/weights/best.pt"
    if not os.path.exists(model_path):
        print("Trained model not found. Please train first.")
        return

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)

    # pulling these from the source video so the output matches exactly — don't hardcode dimensions
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # mp4v is the safe cross-platform choice, avc1 would be better quality but more finicky
    out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # either end of video or a bad read — either way, stop
            break

        # verbose=False keeps the console clean during video processing
        results = model(frame, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()

                # green box — high contrast on most road/parking lot backgrounds
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # label above the box — y1-10 so it doesn't overlap the detection
                cv2.putText(frame, f"Plate {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

    # always release both — forgetting out.release() corrupts the file
    cap.release()
    out.release()
    print(f"Output video saved to output_video.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"], required=True)
    parser.add_argument("--video", type=str, help="Path to input video for prediction")
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "predict":
        # catching missing --video here instead of crashing inside predict()
        if not args.video:
            print("Please provide --video argument")
        else:
            predict(args.video)
