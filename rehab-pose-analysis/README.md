# 🏋️ Arm Raise Rehabilitation Analysis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.30-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-Video%20Processing-green)
![Status](https://img.shields.io/badge/status-learning%20project-informational)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

A notebook-based computer vision project for analysing **arm raise rehabilitation exercises** from video.

This project uses **MediaPipe Pose** to extract upper-limb landmarks frame by frame, then turns those landmarks into simple movement signals such as **shoulder angle**, **elbow angle**, **wrist trajectory**, and **repetition structure**.

I built it as a small, focused movement-analysis pipeline, not as a generic pose demo. The point is to move from raw video to something I can inspect and talk about properly: movement range, path, repetition, and visible differences between **correct** and **incorrect** exercise patterns.

---

## 🎯 Project goal

A training system can show a movement. That is only half the job.

The harder part is this: **how do I measure what the person actually did?**

This notebook is my first answer to that problem. Using arm raise videos, I test whether a lightweight vision pipeline can extract movement features that help distinguish **better** and **worse** exercise execution.

So this repo is not about drawing a skeleton on a person and calling it done. It is about building a first behavioural analysis layer from video.

---

## 🧠 What the project does

The notebook follows this pipeline:

1. Load a small set of arm raise videos
2. Run **MediaPipe Pose** on each frame
3. Extract upper-limb landmarks such as:
   - shoulder
   - elbow
   - wrist
   - hip
4. Save the landmark coordinates into a structured table
5. Compute movement features from those landmarks
6. Compare **correct** and **incorrect** arm raise videos

### Movement features used

This baseline version computes:

- 📈 **Shoulder angle over time**
- 📉 **Elbow angle over time**
- ✋ **Wrist trajectory**
- 📏 **Wrist path length**
- 🔄 **Simple repetition counting from angle peaks**
- 📊 **Per-video summary features** for group comparison

These are not clinical measurements. They are low-cost movement proxies from monocular RGB video.

---

## 📦 Dataset

This project uses a small subset of the **WLU Rehabilitation Posture** dataset from Kaggle.

For this version, I only used:

- **5 videos** from `Arm Raise Correct`
- **5 videos** from `Arm Raise Incorrect`

That narrow scope is deliberate. It keeps the notebook readable, the outputs easier to inspect, and the movement comparison easier to explain.

### Expected folder structure

```text
rehab-pose-analysis/
├── data/
│   └── wlu-rehabilitation-posture/
│       └── Blurred/
│           ├── Arm Raise Correct/
│           │   ├── 11.mp4
│           │   ├── 110.mp4
│           │   ├── 111.mp4
│           │   ├── 112.mp4
│           │   └── 113.mp4
│           └── Arm Raise Incorrect/
│               ├── 01.mp4
│               ├── 010.mp4
│               ├── 011.mp4
│               ├── 012.mp4
│               └── 013.mp4
├── notebooks/
│   └── arm_raise_rehab_analysis.ipynb
├── outputs/
│   ├── csv/
│   ├── frames/
│   ├── plots/
│   └── videos/
├── README.md
└── requirements.txt
