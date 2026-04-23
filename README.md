# Computer Vision Projects

A running collection of the computer vision projects I’ve built, tested, broken, fixed, and learned from.

The work here cuts across:
- object detection
- image classification
- segmentation
- video anomaly detection
- pose and movement analysis
- camera geometry and AR-style visualisation

Each project lives in its own folder and is meant to stand on its own, with its own setup, notes, and entry point.

---

## Project list

| Project | Description | Technologies |
|---------|-------------|--------------|
| [🔍 Vision Transformer Image Classifier](./vit-image-classifier/) | Interactive web app that classifies an image using Google’s ViT model. Supports drag and drop, paste, and camera input. Includes a patch-grid overlay so the tokenisation step is visible instead of hidden. **[Live Demo](https://nosainwe.github.io/computer-vision-projects/vit-image-classifier/)** | Vision Transformer, Hugging Face Inference API, Vanilla JS |
| [🕶️ OpenAR Vision Bridge](./openar-vision-bridge/) | Webcam object detection distilled into a **128×64 monochrome HUD** to mimic the output constraints of embedded AR glasses. Includes export support for short demo GIFs and MP4s. | Ultralytics YOLO, OpenCV, NumPy |
| [🧊 3D Bounding Boxes + AR Overlay (PKU/Baidu)](./3d-bbox-ar-pku/) | Projects 3D bounding boxes, and optionally CAD meshes, onto driving images using camera intrinsics and pose labels from the PKU/Baidu autonomous driving dataset. | NumPy, OpenCV, Pandas, Camera Geometry |
| [🧩 Segment Anything (SAM)](./sam-segmentation/) | Interactive zero-shot segmentation with point prompts, box prompts, and automatic mask generation. Built to make the prompting behaviour visible rather than treating SAM like a black box. | PyTorch, Segment Anything, OpenCV, Matplotlib |
| [🧠 DETR Object Detection](./detr-object-detection/) | End-to-end object detection with DETR using pre-trained COCO weights and inference on custom images. Good project for understanding transformer-based detection without anchors or NMS-heavy logic. | PyTorch, Hugging Face Transformers, OpenCV, Matplotlib |
| [🎥 CCTV Anomaly Detection](./cctv-anomaly-detection/) | Reconstruction-based anomaly detection for surveillance video using a ConvLSTM autoencoder and frame-wise reconstruction error. Reworked into a notebook-style learning project rather than a bare script. | TensorFlow/Keras, OpenCV, Python |
| [🔫 CCTV Threat-Object Classification](./cctv-weapon-classification/) | Binary classification pipeline for CCTV frames using transfer learning to predict whether a threat object is present. Built around image-level classification rather than detection. | TensorFlow/Keras, OpenCV, Python |
| [🚘 License Plate Detection](./license-plate-detection/) | Detects vehicle number plates in images and video using a fine-tuned YOLO detector. | Ultralytics YOLO, OpenCV, Python |
| [🛰️ Military Assets Detection](./military-assets-detection/) | Object detection in aerial or overhead imagery using a YOLO-based workflow trained on military-asset classes from the chosen dataset. | Ultralytics YOLO, OpenCV, Python |
| [🔗 CLIP from Scratch](./clip-from-scratch/) | Ground-up CLIP-style training setup using a ViT vision encoder and a causal language model aligned into a shared embedding space. Built to understand the mechanics, not just call a wrapper. | PyTorch, PEFT, bitsandbytes, Hugging Face Transformers |
| [😊 CNN-based Facial Emotion Recognition](./CNN-based%20facial%20emotion%20recognition/) | End-to-end facial emotion recognition pipeline built on a custom 4-block CNN. Includes augmentation, weighted loss for class imbalance, confusion matrices, and qualitative prediction review. | PyTorch, Torchvision, Matplotlib, scikit-learn |
| [🏋️ Rehab Pose Analysis](./rehab-pose-analysis/) | Upper-limb movement analysis from rehabilitation arm-raise videos using MediaPipe pose landmarks. Extracts angles, trajectories, repetition structure, and simple summary features to compare correct and incorrect motion patterns. | MediaPipe Tasks, OpenCV, Pandas, Matplotlib |

---

## What this repository is for

This repo is not a polished product suite. It is a working portfolio.

Some projects are closer to deployment-style demos. Some are notebook-led learning builds. Some are there because I wanted to understand the method properly instead of pretending I did after one tutorial.

That mix is intentional.

---

## What’s usually inside each project folder

Most folders contain some combination of:

- `README.md` — project overview, setup, usage, and notes
- `requirements.txt` — project-specific dependencies
- `main.py` or `.ipynb` — main script or notebook entry point
- `src/` — helper code for larger projects
- `assets/` or `outputs/` — sample images, plots, GIFs, or saved results
- model/config files where needed

Not every project has the exact same shape. They shouldn’t. A browser demo, a training notebook, and a geometry pipeline do not need identical folder structure.

---

## Themes across the repo

A few patterns show up again and again in this work:

- learning by rebuilding things properly
- making outputs visible, not just printing metrics
- treating file handling and data layout as part of the real work
- preferring small working pipelines over inflated claims
- using projects to understand the method, not only the library call

---

## How to use this repository

Pick a project folder and start there.

Each project should explain:
- what it does
- how to install dependencies
- how to run it
- what outputs to expect
- where the rough edges still are

If something looks unfinished, that usually means it is. I’d rather leave the rough edges visible than cover them with fake polish.

---

## Suggested starting points

If you want the fastest overview of the repo, start with:

- `vit-image-classifier` for a browser-based interactive demo
- `cctv-anomaly-detection` for a notebook-led video pipeline
- `rehab-pose-analysis` for movement analysis from video
- `3d-bbox-ar-pku` for camera geometry and projection work
- `clip-from-scratch` for representation-learning depth

---

## Notes

- Some projects depend on external datasets or model checkpoints that are not stored in the repo.
- Some outputs are intentionally not committed if they are too large.
- Read each project’s own README before trying to run it.

---

## Contact

- GitHub: [nosainwe](https://github.com/nosainwe)
- LinkedIn: [nosa-inwe](https://www.linkedin.com/in/nosa-inwe/)
