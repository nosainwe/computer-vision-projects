# 🚗 License Plate Detection (YOLO11n)

Detect **vehicle license plates** in **images and videos** using a fine‑tuned **YOLO11n** model (Ultralytics).

> 🔒 **Privacy note**
> License plates are personal data in many places. Use this project responsibly, and only on footage you have the right to process.

---

## 📦 Dataset

This project uses the Kaggle dataset:
- **Car License Plate Detection** (Andrew Mvd):  
  https://www.kaggle.com/datasets/andrewmvd/car-plate-detection

The dataset contains images with bounding‑box annotations for plates (VOC-style). citeturn0search0

### Recommended dataset layout (YOLO format)

If your training code expects YOLO-style labels (`.txt`) split into train/val, organise it like this:

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
2. Update `main.py` to read VOC XML directly (less common for Ultralytics)

> Tip: if you want a dataset that is *already* in YOLO format, search Kaggle for “car licence plate detection YOLO” or use a YOLO-converted version. (Your repo can support either approach as long as the loader matches.)

---

## ⚙️ Setup

### 1) Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Download and place the dataset

Download from Kaggle (link above), then extract into:

```text
data/license-plate-dataset/
```

---

## 🏋️ Training

Run training:

```bash
python main.py --mode train
```

Typical training behaviour (adjust in `main.py` if needed):
- loads a pretrained **YOLO11n** checkpoint
- trains for **50 epochs** (solid baseline for a small dataset)
- saves weights to:

```text
runs/detect/train/weights/
├── best.pt
└── last.pt
```

---

## ✅ Evaluation (optional but recommended)

If your `main.py` supports evaluation:

```bash
python main.py --mode evaluate
```

This should output metrics like:
- **mAP@0.5**
- **mAP@0.5:0.95**

…and optionally save sample predictions.

---

## 🎥 Inference on Video

Run inference on a video:

```bash
python main.py --mode predict --video path/to/video.mp4
```

Expected output:
- An output video with bounding boxes (e.g., `output_video.mp4`)
- Or results saved under:

```text
runs/detect/predict/
```

(Exact output location depends on how `main.py` is implemented.)

---

## 📸 Inference on Images

Single image:

```bash
python main.py --mode predict --source path/to/image.jpg
```

Folder:

```bash
python main.py --mode predict --source path/to/images/
```

---

## 📊 Results (example)

Performance varies depending on split, label quality, and training settings.


If you’re seeing inflated numbers (like 94%+) on a small dataset, sanity‑check:
- data leakage (same scenes in train/val)
- duplicate images across splits
- label conversion errors (bad normalisation)

---

## 🧰 Practical improvements

- ✅ Increase robustness with augmentations: motion blur, glare, low-light, rain, occlusion  
- ✅ Use a larger model (`yolo11s`) if your GPU allows  
- ✅ Add a second stage: **plate text recognition** (EasyOCR / Tesseract)  
- ✅ Evaluate on *new videos* (not just the dataset split)

---

## 🙏 Acknowledgements

- **Ultralytics YOLO**: https://github.com/ultralytics/ultralytics  
- Dataset: **Andrew Mvd**, Kaggle — https://www.kaggle.com/datasets/andrewmvd/car-plate-detection
