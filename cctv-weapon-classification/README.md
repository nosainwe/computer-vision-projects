# CCTV Weapon Detection using Transfer Learning

A binary image classifier that detects whether a surveillance image contains a weapon (gun/knife).  
Built with TensorFlow/Keras using transfer learning (MobileNetV2) and the synthetic [CCTV Weapon Detection dataset](https://www.kaggle.com/datasets/simuletic/cctv-weapon-dataset) by Simuletic.

## 📌 Project Overview

- **Goal**: Given a CCTV frame, predict if a weapon is present.
- **Dataset**: 1,000+ synthetic images with YOLO‑format bounding boxes for `person` and `weapon`.
- **Approach**: 
  1. Parse YOLO labels to assign a binary label (`weapon` / `no weapon`) to each image.
  2. Use transfer learning with MobileNetV2 (pretrained on ImageNet) as feature extractor.
  3. Train a classification head on top.
  4. Evaluate accuracy, confusion matrix, and sample predictions.
- **Results**: ~94% test accuracy (can be improved with more data / hyperparameter tuning).

## 🚀 Getting Started

### 1. Clone the repository
    
    git clone https://github.com/nosainwe/computer-vision-projects.git
cd computer-vision-projects/cctv-weapon-classification
2. Download the dataset
Go to CCTV Weapon Dataset on Kaggle

Download and extract the files into data/cctv-weapon-dataset/ so you have:

text
data/cctv-weapon-dataset/
├── images/          # all .jpg/.png files
└── labels/          # corresponding .txt files (YOLO format)
3. Install dependencies
bash
pip install -r requirements.txt
4. Run the notebook
Launch Jupyter and open cctv_weapon_classification.ipynb. Run all cells.

📊 Results
Test Accuracy: ~94%

Confusion Matrix and sample predictions are generated inside the notebook.

🛠️ Future Improvements
Train an object detector (YOLOv8) to localise weapons.

Use data augmentation to improve generalisation.

Experiment with other backbones (EfficientNet, ResNet).

🙏 Acknowledgements
Dataset by Simuletic

Inspired by transfer learning tutorials from Ghassen Khaled and Jocelyn Dumlao.
