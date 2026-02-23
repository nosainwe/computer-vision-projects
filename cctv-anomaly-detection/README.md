# 🕵️ CCTV Anomaly Detection (ConvLSTM Autoencoder)

Unsupervised anomaly detection for surveillance videos using a **ConvLSTM Autoencoder**.

The model learns to **reconstruct normal video clips**. When something unusual happens, reconstruction gets worse, and the **reconstruction error spikes** those high-error frames are flagged as anomalies.

> ⚠️ **Responsible use**
> This is for learning and defensive/security research. Make sure you have permission to use any footage, and follow privacy laws in your location.

---

## 📦 Dataset
- **Real Time Anomaly Detection in CCTV Surveillance** (Kaggle dataset)  
  https://www.kaggle.com/datasets/webadvisor/real-time-anomaly-detection-in-cctv-surveillance

This dataset contains videos across multiple categories (normal events + different anomaly classes). 

### Expected folder structure

After downloading and extracting, place the dataset folder inside this project as `data/` (keep the dataset’s original class folders):

```text
data/
├── roadaccidents/
├── assault/
├── burglary/
├── fighting/
├── explosion/
└── ...
```

> The exact class folder names may differ slightly depending on the dataset version you download.

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

### 3) Download the dataset

Download from Kaggle (link above), then extract into the project root as:

```text
./data/
```

---

## 🚀 Usage

Run the main script:

```bash
python main.py
```

### What the script does

A typical baseline flow looks like this:

1. **Load a “normal” video clip** (for example from `roadaccidents/` or a “normal” category).
2. Extract a short clip of **N frames** (e.g., 16 frames) and resize them (e.g., **128×128**).
3. Train a **ConvLSTM Autoencoder** to reconstruct those frames (self-reconstruction).
4. Load a **test video** (e.g., a different category such as `assault/`).
5. Compute per-frame **reconstruction error** (e.g., MSE).
6. Plot the error curve and flag frames above a **threshold** as anomalies.

> You can change the train/test video paths inside `main.py` to try different categories.

---

## 🧠 How it works (simple explanation)

1. **Frame extraction:** sample a fixed-length clip (e.g., 16 frames).
2. **Autoencoder training:** ConvLSTM learns the spatiotemporal pattern of “normal”.
3. **Reconstruction error:** for each frame (or clip), compute MSE:
   - low error → looks like what the model learned
   - high error → looks unusual (potential anomaly)
4. **Thresholding:** flag frames where error > threshold.

---

## 📊 Results (what you should expect)

This method usually behaves like this:
- It reconstructs the training “normal” scene well (low error).
- When the test clip contains different motion/behaviour, error increases.


### Important limitations
This simple baseline can trigger “false anomalies” when:
- lighting changes suddenly (night/day, flicker)
- camera shakes / zoom changes
- heavy compression artifacts
- the “normal” clip is too short or too specific (overfitting)

If your script trains on a **single clip**, expect brittle behaviour, it’s great for learning, but not production-ready.

---

## 🛠️ Ideas to improve it

- Train on **many normal clips**, not just one.
- Use a **regularity score** rather than a hard threshold.
- Try **prediction-based** models (predict future frames instead of reconstructing).
- Add evaluation: AUC / ROC using labelled anomaly intervals (if available).
- Use a stronger backbone (3D CNN encoder + ConvLSTM, or Transformers).

---

## 🙏 Acknowledgements

- Dataset: Kaggle - **Real Time Anomaly Detection in CCTV Surveillance** 
- Inspired by classic video anomaly detection work using reconstruction error + spatiotemporal models.
