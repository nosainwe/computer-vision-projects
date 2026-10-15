# 🛰️ Aerial Military Asset Detection (YOLO11n)

Detect objects such as **vehicles, personnel, and equipment** in aerial / surveillance images using a **fine‑tuned YOLO11n** model (Ultralytics).

> ⚠️ **Responsible use**
> This repository is for **computer‑vision learning and defensive/security research**. Do not use it to support harm, targeting, or wrongdoing.

---

## 📦 Dataset

✅ Use this active Kaggle dataset instead (YOLO format):
- **Military Assets Dataset (12 Classes – YOLO format)**:  
 [ https://www.kaggle.com/datasets/rawsi18/military-assets-dataset-12-classes-yolo8-format](https://www.kaggle.com/datasets/rupankarmajumdar/amad-5-aerial-military-asset-detection-dataset)

### Expected folder layout

After downloading and extracting, place the dataset here:

```text
data/military-assets/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── dataset.yaml
```

Example `dataset.yaml` (edit paths/classes to match your dataset):

```yaml
path: data/military-assets
train: train/images
val: val/images
test: test/images

names:
  0: soldier
  1: vehicle
  2: artillery
  3: helicopter
  4: tank
  5: ship
  6: aircraft
  7: drone
  8: weapon
  9: radar
  10: missile
  11: other
```

> Note: class names vary by dataset version. Confirm yours by checking the provided `dataset.yaml`.

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

---

## 🏋️ Training

Run training:

```bash
python main.py --mode train
```

Typical training defaults (adjust in `main.py` if needed):
- Pretrained **YOLO11n** weights as a starting point
- ~20 epochs (good baseline while learning)
- 1024×1024 image size (helps with small objects in aerial views)

📁 Outputs are saved under:

```text
runs/detect/train/
└── weights/
    ├── best.pt
    └── last.pt
```

---

## ✅ Evaluation

Evaluate on the test split:

```bash
python main.py --mode evaluate
```

This should:
- print mAP metrics (e.g., mAP@0.5, mAP@0.5:0.95)
- save prediction visualisations to a `runs/` folder

---

## 🔎 Inference

Run inference on a single image:

```bash
python main.py --mode predict --source path/to/image.jpg
```

Run inference on a folder:

```bash
python main.py --mode predict --source path/to/images/
```

Results will be saved to:

```text
runs/detect/predict/
```

---

## 📊 Results (example)

After ~20 epochs, a typical baseline can reach strong validation performance depending on class balance and image quality.

> Replace this with your real numbers after you run training:
- **mAP@0.5 (val):** `__`
- **mAP@0.5:0.95 (val):** `__`

---

## 🧰 Future plans

- ✅ Use **class rebalancing** if a few classes dominate
- ✅ Try `yolo11s` or `yolo11m` if your GPU can handle it
- ✅ Add augmentation for aerial views: rotation, scale, blur, haze
- ✅ Check label quality - noisy labels destroy mAP faster than anything

---

## 🙏 Acknowledgements

- **Ultralytics YOLO** (training/inference framework): https://github.com/ultralytics/ultralytics  
- Dataset source (Kaggle): [https://www.kaggle.com/datasets/rawsi18/military-assets-dataset-12-classes-yolo8-format
](https://www.kaggle.com/datasets/rupankarmajumdar/amad-5-aerial-military-asset-detection-dataset)
