import os
import argparse
import cv2
import torch
from ultralytics import YOLO

def train():
    """Train YOLO11n on license plate dataset."""
    # Create data.yaml content (adjust paths to your dataset location)
    data_yaml = """
train: data/license-plate-dataset/images/train
val: data/license-plate-dataset/images/val
nc: 1
names: ['license_plate']
"""
    with open("data.yaml", "w") as f:
        f.write(data_yaml)

    model = YOLO("yolo11n.pt")  # load pretrained
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=32,
        lr0=0.0005,
        lrf=0.1,
        augment=True
    )
    print("Training completed. Best weights saved to runs/detect/train/weights/best.pt")

def predict(video_path):
    """Run inference on a video using the trained model."""
    # Load trained model
    model_path = "runs/detect/train/weights/best.pt"
    if not os.path.exists(model_path):
        print("Trained model not found. Please train first.")
        return
    model = YOLO(model_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame, verbose=False)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"Plate {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        out.write(frame)
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
        if not args.video:
            print("Please provide --video argument")
        else:
            predict(args.video)
