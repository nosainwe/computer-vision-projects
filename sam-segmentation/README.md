# 🧩 Segment Anything Model (SAM) - Zero‑Shot Interactive Segmentation

A practical, notebook-based implementation of **Meta AI’s Segment Anything Model (SAM)** - a promptable foundation model that can segment objects in images **without retraining**.

This project demonstrates how to use SAM with:
- ✅ **point prompts** (clicks)
- ✅ **bounding box prompts**
- ✅ **automatic mask generation**

---

## 📌 Project overview

**Goal:** Perform **zero‑shot** image segmentation using user-provided prompts (points, boxes, or coarse masks).

**Why SAM is useful:**
- Works across many object types and environments
- Doesn’t require task-specific training for basic segmentation
- Can be used as a “segmentation backbone” inside larger systems (labelling tools, AR, medical, robotics)

---

## 🧠 How SAM works 

SAM has three main parts:

### 1) Image encoder
A Vision Transformer (ViT) processes the image and produces a dense feature map.

### 2) Prompt encoder
Encodes your prompt into embeddings:
- **Sparse prompts** (points, boxes) → token embeddings
- **Dense prompts** (masks) → feature-map style embeddings

### 3) Mask decoder
A lightweight transformer combines:
- image features
- prompt embeddings
- learnable output tokens

…then produces:
- up to **3 candidate masks** per prompt
- an **IoU quality score** for each mask (to help pick the best one)

### Training (context)
SAM was trained on the **SA‑1B** dataset (large-scale segmentation masks) using an annotation pipeline that moves from manual labelling toward automated mask generation.

---

## 🗂️ Repository structure 
```text
sam-segmentation/
├── sam_segmentation.ipynb
├── requirements.txt
└── README.md
```

> If this lives inside a multi-project repo, keep it under:
> `computer-vision/sam-segmentation/`

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

## 🚀 Run the notebook

1. Launch Jupyter
2. Open: `sam_segmentation.ipynb`
3. Run all cells

The notebook should:
- load a pre‑trained SAM model
- load an image (sample or your own)
- run segmentation using:
  - point prompts
  - bounding box prompts
  - automatic mask generation
- visualise masks over the image

---

## 📊 Results (what to expect)

SAM typically produces strong masks with minimal prompting, especially for clear foreground objects.

> Replace this with your own example output (recommended):
- Save a sample to `assets/` and embed it in README.md
- Mention which prompt you used and the model type (e.g., ViT‑B / ViT‑L)

---

## 🔧 Future improvements

- Fine‑tune SAM for a domain (e.g., medical, satellite imagery).
- Combine SAM + text (e.g., CLIP) for text-guided segmentation.
- Export to ONNX for faster inference.
- Build an interactive demo with **Gradio** or **Streamlit**.

---

## 🙏 Acknowledgements

- Paper: **Segment Anything** - Kirillov et al. (2023)
- SAM model + by Meta AI
- Inspiration: Mayank Pratap Singh and Sreedath Panat (article/tutorial)

---
