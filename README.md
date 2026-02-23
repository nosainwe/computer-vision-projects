# Computer Vision Projects

A collection of computer vision projects I’ve worked on (and learned from), covering **detection**, **classification**, **segmentation**, **anomaly detection**, and **geometry-based visualisation**.  
Each project is self-contained and includes setup + run instructions.

## Projects

| Project | Description | Technologies |
|---------|-------------|--------------|
| [🕶️ OpenAR Vision Bridge (Virtual OLED HUD)](./openar-vision-bridge/) | Runs object detection on a webcam feed, then distils results into a **128×64 monochrome HUD** (virtual OLED) to simulate embedded AR glasses output. Includes a script to export a **GIF/MP4** demo. | Ultralytics YOLO, OpenCV, NumPy |
| [🧊 3D Bounding Boxes + AR Overlay (PKU/Baidu)](./3d-bbox-ar-pku/) | Projects **3D bounding boxes** (and optional CAD mesh) onto images using camera intrinsics + pose labels from the PKU/Baidu autonomous driving dataset. | NumPy, OpenCV, Pandas, Camera Geometry |
| [🧩 Segment Anything (SAM) - Zero-Shot Segmentation](./sam-segmentation/) | Zero-shot interactive segmentation using **SAM** with point prompts, box prompts, and automatic mask generation. | PyTorch, Segment Anything, OpenCV/Matplotlib |
| [🧠 Object Detection with DETR](./detr-object-detection/) | End-to-end object detection using **DETR** (transformer set prediction) with pre-trained COCO weights and inference on custom images. | PyTorch, Hugging Face Transformers, OpenCV/Matplotlib |
| [🎥 CCTV Anomaly Detection](./cctv-anomaly-detection/) | Unsupervised anomaly detection in surveillance videos using a ConvLSTM autoencoder + reconstruction error thresholding. | TensorFlow/Keras, OpenCV, Python |
| [🚘 License Plate Detection](./license-plate-detection/) | Detect vehicle license plates in images/videos using a fine-tuned YOLO model. | Ultralytics YOLO, OpenCV, Python |
| [🛰️ Aerial Object Detection](./military-assets-detection/) | Object detection in aerial imagery using a YOLO model (dataset-driven classes). | Ultralytics YOLO, OpenCV, Python |
| [🧪 CCTV Threat-Object Classification](./cctv-weapon-classification/) | Binary classification of CCTV frames using transfer learning (MobileNetV2) to predict whether a threat object is present. | TensorFlow/Keras, OpenCV, Python |

---

## What’s inside each project folder

Each folder typically contains:
- `README.md` - project overview, setup, usage, and notes
- `requirements.txt` - Python dependencies
- `main.py` and/or `.ipynb` - the entry point (script or notebook)
- `src/` - helper code (if the project has a CLI/demo script)
- `assets/` - optional outputs (GIFs/images), usually gitignored unless small and intentional

Click any project above to see the details.

---

### Small pushback (so this lands better on GitHub)
If you’re using this to apply for roles, don’t lead with “military/weapon” language on the front page. The code can stay the same, but neutral titles reduce friction and keep reviewers focused on your skills.
