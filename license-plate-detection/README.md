# 🚗 License Plate Detection + OCR (YOLO11n)

Detect **vehicle license plates** in videos using a fine-tuned **YOLO11n** model (Ultralytics), with optional **EasyOCR** to read the plate text directly from detections.

> 🔒 **Privacy note**
> License plates are personal data in many places. Use this project responsibly, and only on footage you have the right to process.

---

## 📦 Dataset

This project uses the Kaggle dataset:
- **Car License Plate Detection** (Andrew Mvd):  
  https://www.kaggle.com/datasets/andrewmvd/car-plate-detection

The dataset contains images with bounding-box annotations for plates (VOC-style).

### Recommended dataset layout (YOLO format)

```text
data/license-plate-dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

✅ If the dataset downloads with XML annotations, you have two options:
1. **Convert VOC → YOLO** (recommended)
2. Update `license_plate.py` to read VOC XML directly (less common for Ultralytics)

> Tip: if you want a dataset already in YOLO format, search Kaggle for "car licence plate detection YOLO".

---

## ⚙️ Setup

### 1) Create a virtual environment (recommended)

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

For OCR support, also install:

```bash
pip install easyocr
```

> EasyOCR is optional — detection and crop saving work without it.

### 3) Download and place the dataset

Download from Kaggle (link above), then extract into:

```text
data/license-plate-dataset/
```

---

## 🏋️ Training

```bash
python license_plate.py --mode train
```

Optional overrides:

```bash
python license_plate.py --mode train --epochs 100 --batch 16 --imgsz 640
```

Trains a pretrained YOLO11n checkpoint for 50 epochs by default. Weights saved to:

```text
runs/detect/train/weights/
├── best.pt
└── last.pt
```

---

## 🎥 Inference on Video

Basic detection:

```bash
python license_plate.py --mode predict --video path/to/video.mp4
```

With a custom confidence threshold:

```bash
python license_plate.py --mode predict --video path/to/video.mp4 --conf 0.5
```

Skip saving plate crops:

```bash
python license_plate.py --mode predict --video path/to/video.mp4 --no-crops
```

### Outputs

| Output | Description |
|--------|-------------|
| `output_video.mp4` | Annotated video with bounding boxes and labels |
| `detected_plates/plate_00000.jpg` | Cropped plate images (one per detection) |
| Console OCR log | Frame index, crop filename, and read plate text |

If EasyOCR is installed, detected plate text is overlaid in the bounding box label and printed as a summary at the end of the run.

---

## 🔍 CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | required | `train` or `predict` |
| `--video` | — | Path to input video (predict mode) |
| `--conf` | `0.45` | Detection confidence threshold |
| `--no-crops` | off | Disable saving plate crops |
| `--epochs` | `50` | Training epochs |
| `--batch` | `32` | Batch size |
| `--imgsz` | `640` | Input image size |

---

## 📊 Results

Performance varies depending on split quality, label conversion, and training settings. If you're seeing inflated numbers (94%+) on a small dataset, sanity-check for:
- data leakage (same scenes in train/val)
- duplicate images across splits
- label normalisation errors from VOC → YOLO conversion

---

## 🧰 Further improvements

- Use a larger backbone (`yolo11s`, `yolo11m`) if your GPU allows
- Add motion blur, glare, and low-light augmentations for real-world robustness
- Fine-tune OCR with a plate-specific EasyOCR model for higher read accuracy
- Evaluate on new, unseen video footage — not just the dataset split

---

## 🙏 Acknowledgements

- **Ultralytics YOLO**: https://github.com/ultralytics/ultralytics
- Dataset: **Andrew Mvd**, Kaggle — https://www.kaggle.com/datasets/andrewmvd/car-plate-detection
- OCR: **EasyOCR** — https://github.com/JaidedAI/EasyOCR
