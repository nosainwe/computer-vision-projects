# 😠😄😨 Facial Emotion Recognition (CNN + FER-2013)

Custom CNN trained from scratch to classify facial expressions into 7 emotion categories using the FER-2013 benchmark dataset.

**Achieves ~66–67% validation accuracy** - consistent with published results for CNN baselines on this dataset.

## Classes

| Label | Emotion |
|-------|---------|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Neutral |
| 5 | Sad |
| 6 | Surprise |

## Dataset

**FER-2013** - Facial Expression Recognition 2013  
https://www.kaggle.com/datasets/msambare/fer2013

- 48×48 grayscale images
- 28,709 training samples / 3,589 test samples
- Class-imbalanced (Disgust is severely underrepresented)

Expected layout after download:

```
fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    └── (same structure)
```

## Architecture

4-block CNN with progressive feature extraction:

```
Conv(1→64) → ReLU → BN → MaxPool → Dropout(0.25)
Conv(64→128, 5×5) → ReLU → BN → MaxPool → Dropout(0.25)
Conv(128→256) → ReLU → BN → MaxPool → Dropout(0.25)
Conv(256→512) → ReLU → BN → MaxPool → Dropout(0.25)
→ Flatten → Linear(4608, 256) → Linear(256, 512) → Linear(512, 7)
```

- **Loss:** CrossEntropyLoss with inverse-frequency class weights
- **Optimizer:** Adam (lr=5e-4)
- **Scheduler:** ReduceLROnPlateau - halves lr after 3 epochs without val accuracy improvement

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**Train:**
```bash
python emotion_detector.py --mode train --data_dir path/to/fer2013
```

**Evaluate (confusion matrix + classification report):**
```bash
python emotion_detector.py --mode evaluate --data_dir path/to/fer2013
```

**Prediction gallery (10 sample predictions):**
```bash
python emotion_detector.py --mode gallery --data_dir path/to/fer2013
```

## Outputs

| File | Description |
|------|-------------|
| `best_fer_cnn.pt` | Best model checkpoint by val accuracy |
| `training_curves.png` | Accuracy and loss curves side by side |
| `confusion_matrix.png` | Per-class confusion heatmap |
| `prediction_gallery.png` | 10 sample predictions (green = correct, red = wrong) |

## Notes

- Val accuracy plateaus around 66–67% - this is expected for a plain CNN on FER-2013. The dataset has noisy labels and very low resolution (48×48), which limits the ceiling for single-model approaches.
- The `Disgust` class consistently has the lowest recall due to severe underrepresentation (~500 samples vs ~7000 for Happy). Class weighting partially compensates but doesn't fully close the gap.
- For higher accuracy, consider transfer learning from a pretrained face model or a Vision Transformer backbone.

## Acknowledgements

- FER-2013 dataset: originally from the ICML 2013 Challenges in Representation Learning workshop
- Kaggle mirror: https://www.kaggle.com/datasets/msambare/fer2013
