# 🎥 CCTV Weapon Presence Classification (Transfer Learning)

A **binary image classifier** that predicts whether a CCTV / surveillance frame contains a **visible weapon** (e.g., gun/knife).

Built with **TensorFlow / Keras** using **transfer learning** (MobileNetV2).

> ⚠️ **Responsible use**
> This project is for **computer‑vision learning and defensive/security research**. Don’t use it to support harm, targeting, or illegal surveillance.

---

## 📌 Project overview

**Goal:** Given a CCTV frame, predict **weapon** vs **no weapon**.

**Dataset:** Synthetic CCTV images with **YOLO-format bounding boxes** for:
- `person`
- `weapon`

**Key idea (how labels are created):**
1. Each image has a YOLO label file (one line per box).
2. If any box belongs to the `weapon` class, the image gets the binary label **weapon=1**.
3. Otherwise the image label is **weapon=0**.

**Model approach:**
- Use **MobileNetV2** pretrained on ImageNet as a feature extractor.
- Train a small classification head on top.
- Evaluate using accuracy + confusion matrix + sample predictions.

---

## 📦 Dataset

Kaggle dataset (active):
- **CCTV Weapon Detection dataset** by Simuletic  
  https://www.kaggle.com/datasets/simuletic/cctv-weapon-dataset

### Expected folder structure

After downloading and extracting, place the files here:

```text
data/cctv-weapon-dataset/
├── images/          # .jpg/.png images
└── labels/          # .txt YOLO annotation files with same base names as images
```

Example:

```text
data/cctv-weapon-dataset/images/000123.jpg
data/cctv-weapon-dataset/labels/000123.txt
```

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

### 3) Get the dataset

Download from Kaggle (link above), then extract into:

```text
data/cctv-weapon-dataset/
```

---

## 🚀 Run the project

### Option A — Run the notebook (recommended for learning)

1. Launch Jupyter
2. Open: `cctv_weapon_classification.ipynb`
3. Run all cells

The notebook should:
- parse YOLO labels → build binary targets
- train MobileNetV2-based classifier
- output accuracy + confusion matrix
- show sample predictions


## 📊 Results (example)

Your accuracy will vary depending on:
- how the dataset is split
- how many “weapon” vs “no weapon” images are used
- augmentation and training settings

### Sanity checks (so your results are real)
If you see very high accuracy (e.g., 90%+), double‑check:
- **data leakage** (same or near-duplicate scenes in train/test)
- **class imbalance** (model predicting the majority class)
- **label parsing** (weapon class id correct?)

---

## 🛠️ Future improvements

- Train an **object detector** (YOLOv8/YOLO11) to *localise* weapons, not just classify presence.
- Add stronger **data augmentation** (motion blur, low light, compression artifacts).
- Try other backbones: **EfficientNet**, **ResNet50**, **ConvNeXt**.
- Add **Grad‑CAM** visualisations to show what the model is focusing on.
- Evaluate on a small set of **real CCTV frames** (if you have permission) to test generalisation beyond synthetic data.

---

## 🙏 Acknowledgements

- Dataset: **Simuletic** (Kaggle) — https://www.kaggle.com/datasets/simuletic/cctv-weapon-dataset  
- Transfer learning references: general Keras / TensorFlow examples and community tutorials.
