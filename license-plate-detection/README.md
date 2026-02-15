# License Plate Detection with YOLO11n

Detect vehicle license plates in images and videos using a fine‑tuned YOLO11n model.

## Dataset
This project uses the [License Plate Dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection) from Kaggle.  
After downloading, place it in `data/license-plate-dataset/` with the following structure:
data/license-plate-dataset/
├── images/
│ ├── train/
│ └── val/
└── labels/
├── train/
└── val/

text

## Setup
1. Clone this repository and navigate to the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
