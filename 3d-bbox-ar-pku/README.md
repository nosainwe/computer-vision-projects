# 🚘 3D Bounding Boxes + Augmented Reality Overlay (PKU/Baidu)

Visualise **3D car poses** by projecting:
- **3D bounding boxes** into the image, and
- an optional **CAD mesh overlay** (basic “augmented reality” effect)

using the **PKU/Baidu Autonomous Driving** Kaggle competition data (intrinsics + pose labels + car CAD models).

> ✅ Portfolio angle: this shows you understand camera geometry, projection, and working with real dataset formats - not just running a detector.

---

## 📦 Dataset

This project uses the Kaggle competition data:
- **Peking University/Baidu - Autonomous Driving**

You’ll need:
- `train.csv` (pose labels + PredictionString)
- `train_images/` (images)
- `car_models_json/` (CAD meshes)

### Expected folder layout

Download the competition data and place it like this:

```text
data/pku-autonomous-driving/
├── train.csv
├── train_images/
│   ├── 0000c7b2b8d5a4b0.jpg
│   └── ...
└── car_models_json/
    ├── dazhongmaiteng.json
    └── ...
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

---

## 🚀 Run it

### Option A — Notebook (best for learning)

Open and run:

```text
notebooks/3d_bounding_box_and_augmented_reality_pku.ipynb
```

✅ This notebook is the Kaggle version adapted to **local paths** (`data/pku-autonomous-driving/...`).

### Option B — CLI demo 

Render 3D bounding boxes for one row from `train.csv`:

```bash
python -m src.visualize_3d_bbox --data_dir data/pku-autonomous-driving --row 4002 --out assets/output_row4002.jpg
```

Optional: also try drawing the mesh overlay for a single model:

```bash
python -m src.visualize_3d_bbox --data_dir data/pku-autonomous-driving --row 4002 --mesh_model dazhongmaiteng --out assets/output_row4002_mesh.jpg
```

> ⚠️ Mesh overlay only works cleanly when the car type in the label matches the chosen mesh model.

---

## 🧠 How it works (plain English)

1. **Read pose labels** from `PredictionString` in `train.csv`  
   Each car instance has:
   - rotation (yaw, pitch, roll)
   - translation (x, y, z)
   - model id / type

2. Convert Euler angles → **rotation matrix**.

3. Build 3D box corners in the car coordinate frame.

4. Transform corners into camera coordinates and project onto the image using the **camera intrinsics** matrix.

5. Draw the 2D lines that represent the 3D box.  
   (Optional) load CAD vertices/triangles from `car_models_json/` and project them too.

---

## 📌 Notes / limitations 

- This is a **visualisation** pipeline, not a full detection model.
- Results depend heavily on:
  - correct intrinsics
  - the label format
  - picking matching car meshes for AR overlay
- Mesh overlay is intentionally “simple” (triangle fill). It’s meant to show the idea.

---

## 🙏 Acknowledgements

- Dataset and CAD models: Kaggle competition — **Peking University/Baidu - Autonomous Driving**
