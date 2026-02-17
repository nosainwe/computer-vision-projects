# Computer Vision Projects

A collection of computer vision projects I have worked on and learned from, demonstrating detection, classification, anomaly detection, and geometry-based visualisation. Each project is self-contained and includes setup + run instructions.

## Projects

| Project | Description | Technologies |
|---------|-------------|--------------|
| [🚘 License Plate Detection](./license-plate-detection/) | Detect vehicle license plates in images/videos using a fine-tuned YOLO11n model. | Ultralytics YOLO, OpenCV, Python |
| [🚁 Military Assets Detection](./military-assets-detection/) | Detect objects in aerial imagery using a YOLO11n model (dataset-driven classes). | Ultralytics YOLO, OpenCV, Python |
| [🎥 CCTV Anomaly Detection](./cctv-anomaly-detection/) | Unsupervised anomaly detection in surveillance videos using a ConvLSTM autoencoder + reconstruction error. | TensorFlow/Keras, OpenCV, Python |
| [🔫 CCTV Weapon Presence Classification](./cctv-weapon-classification/) | Binary classification of CCTV frames to predict weapon presence using transfer learning (MobileNetV2). | TensorFlow/Keras, OpenCV, Python |
| [🕶️ OpenAR Vision Bridge (Virtual OLED HUD)](./openar-vision-bridge/) | Runs YOLO on a webcam feed, then distils detections into a **128×64 monochrome HUD** (virtual OLED) for embedded-style AR glasses. Includes a script to export a GIF/MP4 demo. | Ultralytics YOLO, OpenCV, NumPy |
| [🧊 3D Bounding Boxes + AR Overlay (PKU/Baidu)](./3d-bbox-ar-pku/) | Projects **3D bounding boxes** (and optional CAD mesh) onto images using camera intrinsics + pose labels from the PKU/Baidu autonomous driving dataset. | NumPy, OpenCV, Pandas, Camera Geometry |

---

## What’s inside each project folder

Each folder typically contains:
- `README.md` — project overview, setup, usage, and notes
- `requirements.txt` — Python dependencies
- `main.py` and/or `.ipynb` — the entry point (script or notebook)
- `src/` — helper code (if the project has a CLI/demo script)
- `assets/` — optional outputs (GIFs/images), usually gitignored unless small and intentional

Click any project above to see the details.

---

If this is aimed at recruiters, consider renaming any project titles that sound “military/weapon-first” to something more neutral (you can keep the exact same code and dataset inside). It reduces unnecessary friction while still showing the technical work.
