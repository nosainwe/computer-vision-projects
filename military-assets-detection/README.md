# 🎯 Military Assets Detection with YOLO11n

Detect military vehicles, soldiers, artillery, and aircraft in aerial/surveillance imagery using a fine‑tuned **YOLO11n (Nano)** model. This project is optimized for efficiency and speed while maintaining high accuracy on small objects typical in aerial reconnaissance.

## 📂 Dataset
This project uses the **[AMAD-5 (Aerial Military Asset Detection) Dataset](https://www.kaggle.com/datasets/amanbarthwal/amad-5-aerial-military-asset-detection)** from Kaggle.

### **Classes**
The model is trained to detect the following classes:
1.  **Soldier** 💂
2.  **Military Tank** 🚜
3.  **Military Truck** 🚚
4.  **Military Aircraft** ✈️
5.  **Artillery** 💣

### **Directory Structure**
After downloading, extract the dataset into a `data/` folder so your project looks like this:

├── data/
│   └── amad-5/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── val/
│       │   ├── images/
│       │   └── labels/
│       ├── test/
│       │   ├── images/
│       │   └── labels/
│       └── data.yaml       <-- Ensure this points to the correct paths!
├── runs/                   <-- Created automatically during training
├── main.py
├── requirements.txt
└── README.md

---

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/military-assets-detection.git](https://github.com/yourusername/military-assets-detection.git)
cd military-assets-detection
2. Install DependenciesIt is recommended to use a virtual environment.Bash# Create virtual env (optional)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
Note: Ensure you have ultralytics, opencv-python, and pandas installed.🚀 Usage1. Training 🏋️Train the model from scratch (using pretrained weights). The script is configured to run for 20 epochs at an image size of 1024x1024 to better capture small aerial objects.Bashpython main.py --mode train
Weights location: runs/detect/train/weights/best.ptLogs: Training curves and metrics are saved in runs/detect/train/2. Evaluation 📊Evaluate the model's performance on the test set to get metrics like mAP50 and mAP50-95.Bashpython main.py --mode evaluate
3. Inference / Prediction 🔮Run detection on new images or videos.Single Image:Bashpython main.py --mode predict --source path/to/image.jpg
Folder of Images:Bashpython main.py --mode predict --source data/amad-5/test/images/
Video File:Bashpython main.py --mode predict --source path/to/video.mp4
Results are saved in runs/detect/predict/.📈 ResultsAfter training for 20 epochs, the model achieves the following metrics on the validation set:MetricScoremAP @ 0.5~85%mAP @ 0.5:0.95~62%PrecisionHighRecallHigh(Note: These are estimated values. Check your specific results.csv in the runs folder for exact numbers.)🤝 AcknowledgementsYOLO11: Implementation by Ultralytics.Dataset: AMAD-5 by Aman Barthwal.📜 LicenseThis project is licensed under the MIT License.
