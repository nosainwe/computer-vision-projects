# CCTV Anomaly Detection using ConvLSTM Autoencoder

Unsupervised anomaly detection in surveillance videos. The model learns to reconstruct normal video clips; anomalous frames yield high reconstruction error.

## Dataset
This project uses the [Real‑time Anomaly Detection in CCTV Surveillance](https://www.kaggle.com/datasets/marwaelsheikh/real-time-anomaly-detection-in-cctv-surveillance) dataset from Kaggle.  
After downloading, place the `data` folder inside this project as `data/` with the original structure:
data/
├── roadaccidents/
├── assault/
├── ...

text

## Setup
1. Clone this repository and navigate to the project folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
Download the dataset (link above) and extract it into data/.

Usage
Run the main script:

bash
python main.py
This will:

Load a normal video (roadaccidents/RoadAccidents009_x264.mp4) and extract 16 frames.

Train a ConvLSTM autoencoder on this single clip (self‑reconstruction).

Load an anomalous video (assault/Assault009_x264.mp4) and compute per‑frame reconstruction error.

Plot the error and flag frames above a threshold as anomalies.

You can change the video paths inside the script.

Results
The autoencoder learns to reconstruct normal scenes. Anomalous frames (e.g., assault) produce higher MSE, allowing detection.



How it works
Extract 16 frames from a normal video, resize to 128×128.

Train a ConvLSTM autoencoder to reconstruct these frames.

For a test video, compute reconstruction MSE per frame.

Frames with MSE > threshold are flagged as anomalous.

Acknowledgements
Dataset by Marwa Elsheikh.

Inspired by video anomaly detection literature.
