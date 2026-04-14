# Arm Raise Rehabilitation Analysis

A notebook-based computer vision project for analysing **arm raise exercises** from blurred rehabilitation videos.

This project uses **MediaPipe Pose** to extract upper-limb landmarks from video, then turns those landmarks into simple movement signals such as shoulder angle, elbow angle, wrist trajectory, and repetition structure.

I built it as a small, focused movement-analysis pipeline, not as a generic pose demo. The point is to move from raw video to something I can actually inspect and talk about: range, path, repetition, and differences between more correct and less correct movement.

## Why I built this

A training system can show an exercise. That part is easy to say and harder to do well. The bigger question comes after that:

**How do I measure what the person actually did?**

That is the problem this notebook starts to address.

Using arm raise videos, I wanted to check whether a lightweight vision pipeline could pull out simple movement features that might help separate **correct** and **incorrect** exercise patterns.

That makes this project useful beyond a pose-estimation screenshot. It starts to act like a behavioural analysis layer.

## Dataset

This project uses a small subset of the **WLU Rehabilitation Posture** dataset from Kaggle.

Dataset page:
`WLU Rehabilitation Posture`

For this version, I only used:

- **5 videos** from `Arm Raise Correct`
- **5 videos** from `Arm Raise Incorrect`

That small scope is deliberate. It keeps the notebook readable, the outputs easier to inspect, and the movement comparison easier to explain.

### Folder layout

Place the dataset like this:

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
