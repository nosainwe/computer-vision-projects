# Military Assets Detection with YOLO11n

Detect military vehicles, soldiers, artillery, and more in aerial/surveillance imagery using a fine‑tuned YOLO11n model.

## Dataset
This project uses the [AMAD-5 Aerial Military Asset Detection Dataset](https://www.kaggle.com/datasets/amanbarthwal/amad-5-aerial-military-asset-detection) from Kaggle.  
After downloading, place it in `data/amad-5/` with the structure:
data/amad-5/
├── train/
│ ├── images/
│ └── labels/
├── val/
│ ├── images/
│ └── labels/
├── test/
│ ├── images/
│ └── labels/
└── dataset.yaml

text

## Setup
1. Clone this repository and navigate to the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
Download the dataset (link above) and extract it into data/.

Training
Run the training script:

bash
python main.py --mode train
The script uses a pretrained YOLO11n model.

Trains for 20 epochs at 1024×1024 resolution (to capture small objects).

Best weights are saved to runs/detect/train/weights/best.pt.

Evaluation
To evaluate on the test set:

bash
python main.py --mode evaluate
This will output mAP metrics and save prediction visualizations.

Inference on Images
Run inference on a single image or a folder:

bash
python main.py --mode predict --source path/to/image_or_folder
Results are saved in runs/detect/predict/.

Results
After 20 epochs, the model achieves ~85% mAP@0.5 on the validation set.
Acknowledgements
YOLO implementation by Ultralytics.

Dataset by Aman Barthwal.
