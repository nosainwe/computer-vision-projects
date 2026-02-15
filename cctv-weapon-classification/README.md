# CCTV Weapon Image Classification

[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Dataset-blue)](https://www.kaggle.com/datasets/simuletic/cctv-weapon-dataset)

A binary image classifier that detects whether a CCTV frame contains a weapon (gun/knife).  
Built with TensorFlow/Keras using transfer learning (MobileNetV2) and the synthetic [CCTV Weapon Detection dataset](https://www.kaggle.com/datasets/simuletic/cctv-weapon-dataset) by Simuletic.

## 📌 Project Overview

- **Goal**: Given a surveillance image, predict if a weapon is present.
- **Dataset**: 1,000+ synthetic images with YOLO‑format bounding boxes for `person` and `weapon`.
- **Approach**: 
  1. Parse YOLO labels to assign a binary label (`weapon` / `no weapon`) to each image.
  2. Use transfer learning with MobileNetV2 (pretrained on ImageNet) as feature extractor.
  3. Train a classification head on top.
  4. Evaluate accuracy, confusion matrix, and sample predictions.
- **Results**: ~94% test accuracy (can be improved with more data / hyperparameter tuning).

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/computer-vision-projects.git
cd computer-vision-projects/cctv-weapon-classification
