# 😠😄😨 Facial Emotion Recognition (CNN + FER-2013)

Custom CNN trained from scratch to classify facial expressions into 7
emotion categories using the FER-2013 benchmark dataset.

**\~66--67% validation accuracy**, aligned with standard CNN baselines
on this dataset.

------------------------------------------------------------------------

## 🔥 Features

-   🧠 Custom CNN (no transfer learning)
-   🎯 Class imbalance handling (weighted loss)
-   📊 Evaluation: confusion matrix + classification report\
-   🖼️ Grad-CAM visual explanations (model interpretability)
-   🎥 Real-time webcam emotion detection
-   📉 Training curves (accuracy + loss)

------------------------------------------------------------------------

## 🎭 Classes

  Label   Emotion
  ------- ----------
  0       Angry
  1       Disgust
  2       Fear
  3       Happy
  4       Neutral
  5       Sad
  6       Surprise

------------------------------------------------------------------------

## 📂 Dataset

**FER-2013 --- Facial Expression Recognition 2013**\
https://www.kaggle.com/datasets/msambare/fer2013

-   48×48 grayscale images\
-   28,709 training / 3,589 test\
-   Highly imbalanced (Disgust is rare)

### Expected structure

fer2013/ ├── train/ │ ├── angry/ │ ├── disgust/ │ ├── fear/ │ ├── happy/
│ ├── neutral/ │ ├── sad/ │ └── surprise/ └── test/ └── (same structure)

------------------------------------------------------------------------

## 🧱 Architecture

4-stage CNN feature extractor:

Conv(1→64) → ReLU → BN → MaxPool → Dropout\
Conv(64→128, 5×5) → ReLU → BN → MaxPool → Dropout\
Conv(128→256) → ReLU → BN → MaxPool → Dropout\
Conv(256→512) → ReLU → BN → MaxPool → Dropout\
→ Flatten → FC → FC → FC(7 classes)

**Training setup** - Loss: CrossEntropy (with class weights) -
Optimizer: Adam (lr = 5e-4) - Scheduler: ReduceLROnPlateau

------------------------------------------------------------------------

## ⚙️ Setup

python -m venv .venv\
source .venv/bin/activate \# Windows:
.venv`\Scripts`{=tex}`\activate  `{=tex} pip install -r requirements.txt

------------------------------------------------------------------------

## 🚀 Usage

Train: python emotion_detector.py --mode train --data_dir
path/to/fer2013

Evaluate: python emotion_detector.py --mode evaluate --data_dir
path/to/fer2013

Grad-CAM: python emotion_detector.py --mode gradcam --data_dir
path/to/fer2013

Webcam: python emotion_detector.py --mode webcam --data_dir
path/to/fer2013

Press 'q' to quit webcam.

------------------------------------------------------------------------

## 📊 Outputs

-   best_fer_cnn.pt --- Best model checkpoint\
-   training_curves.png --- Accuracy + loss curves\
-   confusion_matrix.png --- Per-class performance

------------------------------------------------------------------------

## 🎯 Notes

-   Accuracy stabilises around 66--67%\
-   Disgust class has lowest recall\
-   Dataset is noisy and low resolution

------------------------------------------------------------------------

## 🙏 Acknowledgements

FER-2013 dataset (ICML 2013 workshop)\
https://www.kaggle.com/datasets/msambare/fer2013
