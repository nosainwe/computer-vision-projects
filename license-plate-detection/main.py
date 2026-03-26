import os
import argparse
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

# easyocr is lazily imported inside predict() — no penalty if you're only training
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


# =============================================================================
# CONFIG
# =============================================================================

MODEL_PATH  = "runs/detect/train/weights/best.pt"
DATA_YAML   = "data.yaml"
CROPS_DIR   = "detected_plates"
CONF_THRESH = 0.45   # default confidence — override with --conf at runtime


# =============================================================================
# TRAIN
# =============================================================================

def train(epochs: int = 50, batch: int = 32, imgsz: int = 640):
    # writing yaml inline — one less file to maintain, always in sync with these values
    data_yaml = """
train: data/license-plate-dataset/images/train
val: data/license-plate-dataset/images/val
nc: 1
names: ['license_plate']
"""
    with open(DATA_YAML, "w") as f:
        f.write(data_yaml)

    # nano model — fast to iterate, swap to yolo11s/m if precision needs a bump
    model = YOLO("yolo11n.pt")
    model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=0.0005,    # lower than default — helps on specific domains like plates
        lrf=0.1,       # cosine decay: final lr = lr0 * lrf
        augment=True,
    )
    print(f"Training done. Best weights at {MODEL_PATH}")


# =============================================================================
# OCR HELPER
# =============================================================================

def load_ocr_reader():
    if not OCR_AVAILABLE:
        print("easyocr not installed. Run: pip install easyocr")
        print("Continuing without OCR — crops will still be saved.")
        return None
    # gpu=True if cuda is available, falls back to cpu silently
    return easyocr.Reader(["en"], gpu=torch.cuda.is_available())


def read_plate_text(reader, crop: "np.ndarray") -> str:
    """Run OCR on a single plate crop. Returns best candidate string or empty."""
    if reader is None:
        return ""
    results = reader.readtext(crop, detail=0, paragraph=True)
    # join fragments and strip whitespace — plates often read as multiple chunks
    text = " ".join(results).strip().upper()
    return text


# =============================================================================
# PREDICT
# =============================================================================

def predict(video_path: str, conf: float = CONF_THRESH, save_crops: bool = True):
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Train first.")
        return

    model  = YOLO(MODEL_PATH)
    reader = load_ocr_reader()

    # crops directory — one subfolder per run so nothing gets overwritten
    if save_crops:
        crops_path = Path(CROPS_DIR)
        crops_path.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    # read properties from source so output dimensions match exactly
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 25   # fallback if fps reads as 0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # mp4v is the safe cross-platform choice — avc1 has better quality but is finicky on Windows
    out = cv2.VideoWriter(
        "output_video.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    frame_idx  = 0
    crop_idx   = 0
    ocr_log    = []   # accumulate (frame, crop_file, text) for summary at the end

    print(f"Processing {total} frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf, verbose=False)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()

                # --- save crop ---
                crop_file = ""
                if save_crops:
                    plate_crop = frame[y1:y2, x1:x2]
                    if plate_crop.size > 0:   # skip degenerate boxes
                        crop_file = str(crops_path / f"plate_{crop_idx:05d}.jpg")
                        cv2.imwrite(crop_file, plate_crop)
                        crop_idx += 1

                # --- OCR ---
                plate_text = ""
                if reader is not None and crop_file:
                    plate_text = read_plate_text(reader, cv2.imread(crop_file))
                    if plate_text:
                        ocr_log.append((frame_idx, crop_file, plate_text))

                # --- draw ---
                label = f"{plate_text}  {confidence:.2f}" if plate_text else f"Plate {confidence:.2f}"
                # green box — high contrast on road/parking backgrounds
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # label sits above the box — y1-10 keeps it from overlapping the detection
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        out.write(frame)
        frame_idx += 1

    # always release both — forgetting out.release() silently corrupts the output file
    cap.release()
    out.release()

    print(f"\nOutput saved: output_video.mp4")
    if save_crops:
        print(f"Plate crops saved: {crops_path}/ ({crop_idx} total)")
    if ocr_log:
        print(f"\nOCR results ({len(ocr_log)} reads):")
        for f_idx, fname, text in ocr_log:
            print(f"  frame {f_idx:05d} | {Path(fname).name} | {text}")


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",    choices=["train", "predict"], required=True)
    parser.add_argument("--video",   type=str,  help="Path to input video")
    parser.add_argument("--conf",    type=float, default=CONF_THRESH,
                        help=f"Detection confidence threshold (default: {CONF_THRESH})")
    parser.add_argument("--no-crops", action="store_true",
                        help="Skip saving plate crops")
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--imgsz",   type=int, default=640)
    args = parser.parse_args()

    if args.mode == "train":
        train(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
    elif args.mode == "predict":
        if not args.video:
            print("Please provide --video argument")
        else:
            predict(
                video_path=args.video,
                conf=args.conf,
                save_crops=not args.no_crops,
            )
