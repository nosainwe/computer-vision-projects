# 🕵️ CCTV Anomaly Detection, notebook version

This project rebuilds the original ConvLSTM autoencoder script as a **teaching-first Jupyter notebook**.

It does not just run the model. It walks through the working idea from the ground up:

- inspect the dataset and folder structure
- read and sample video frames
- explain tensor shapes in plain English
- train a ConvLSTM autoencoder on normal clips
- compute reconstruction error on a test clip
- plot anomaly scores
- save output figures for GitHub

## Why this version is better than the old script

The old `main.py` is useful as a quick demo, but it has a few weak points:

- it hard-codes one training video and one test video
- it assumes `roadaccidents` can act as a normal class
- it uses a hand-picked threshold without learning it from normal data
- it hides too much of the data handling from view

This notebook fixes that.

It is built for learning first, then extension.

## Recommended folder structure

```text
cctv-anomaly-detection/
├── cctv_anomaly_detection_teaching_notebook.ipynb
├── README.md
├── requirements.txt
├── data/
│   ├── normal/
│   │   ├── normal_video_01.mp4
│   │   ├── normal_video_02.mp4
│   │   └── ...
│   ├── assault/
│   ├── burglary/
│   ├── fighting/
│   ├── explosion/
│   └── ...
└── outputs/
    ├── plots/
    ├── frames/
    ├── gifs/
    └── models/
```

## Dataset

Source dataset used in the original project:

**Real Time Anomaly Detection in CCTV Surveillance**  
Kaggle: `https://www.kaggle.com/datasets/webadvisor/real-time-anomaly-detection-in-cctv-surveillance`

After downloading, place the files inside `data/`.

### Important note about “normal” data

Do not blindly train on a folder like `roadaccidents/` and call it normal.

For anomaly detection, define a separate `data/normal/` folder and place clips there that represent ordinary behaviour.  
That gives the model a fair target to learn.

## What the notebook covers

### 1. Dataset inspection
The notebook scans the `data/` folder, counts video files, and shows sample names.

### 2. Video loading
It reads videos with OpenCV, resizes frames, converts BGR to RGB, and normalises pixel values to `[0, 1]`.

### 3. Frame visualisation
It shows sampled frames before training so you can inspect what the model is actually learning from.

### 4. ConvLSTM autoencoder
The notebook builds a small ConvLSTM autoencoder that learns spatiotemporal patterns in normal clips.

### 5. Training
It stacks clips into a training tensor shaped like:

```python
(batch, time, height, width, channels)
```

Example:

```python
(8, 16, 128, 128, 3)
```

### 6. Reconstruction error
For a test clip, the model reconstructs each frame and computes frame-wise MSE.

Low MSE means the frame looks familiar.  
High MSE means the model struggled, which may indicate unusual behaviour.

### 7. Thresholding
Instead of relying only on a hand-tuned threshold, the notebook estimates one from the distribution of reconstruction errors on normal training clips.

### 8. Saved outputs
The notebook saves:

- training loss plot
- anomaly score plot
- selected test frames
- trained model file

This makes it easier to upload proof of results to GitHub.

## Run it

Launch Jupyter and open:

```text
cctv_anomaly_detection_teaching_notebook.ipynb
```

Then run the notebook top to bottom.

## Good GitHub practice

When you push this project, include:

- the notebook
- the updated README
- 2 to 4 output images under `outputs/plots/` or `outputs/frames/`
- a short note on which videos were used for training and testing

That matters. Otherwise it just looks like untested code.

## Limits of this baseline

This is a learning baseline, not a production anomaly detector.

It can break when:

- lighting changes hard
- the camera shakes
- compression artefacts are strong
- the model sees too few normal clips
- the threshold is chosen badly

## What to build next

The next sensible upgrades are:

- sliding-window clip extraction
- training on many normal videos
- a normal-only validation split for threshold tuning
- ROC / AUC if labels exist
- stronger video backbones

## Future reuse

You can reuse this same notebook structure for:

- MRI anomaly detection
- X-ray classification notebooks
- microscopy pipelines
- industrial inspection videos

Only the data loader and model head change. The teaching structure should stay almost the same.
