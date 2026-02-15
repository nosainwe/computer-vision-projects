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
Download the dataset (link above) and extract it into data/.

Training
Run the training script:

    ```bash
    python main.py --mode train
The script will:

Load a pretrained YOLO11n model.

Train for 50 epochs on the license plate dataset.

Save the best weights to runs/detect/train/weights/best.pt.

Inference on a Video
After training, run inference on a video:

     ```bash
     python main.py --mode predict --video path/to/video.mp4


The output video with detected plates will be saved as output_video.mp4.

Results
Test accuracy: ~94% mAP@0.5 (varies with dataset split).



Acknowledgements
YOLO implementation by Ultralytics.

Dataset by Andrew Mvd.
