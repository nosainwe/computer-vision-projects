# Computer Vision Projects

A collection of computer vision projects I've worked on (and learned from), covering **detection**, **classification**, **segmentation**, **anomaly detection**, and **geometry-based visualisation**.  

Each project is self-contained and includes setup + run instructions.

## Projects

| Project | Description | Technologies |
|---------|-------------|--------------|
| [🔍 Vision Transformer Image Classifier](./vit-image-classifier/) | Interactive web app that classifies any image using **Google's ViT** (Vision Transformer) model. Drag & drop, paste, or use your camera. Includes a live patch-grid overlay that visualises how ViT slices images into tokens. **[→ Live Demo](https://nosainwe.github.io/computer-vision-projects/vit-image-classifier/)** | Vision Transformer, HuggingFace Inference API, Vanilla JS |
| [🕶️ OpenAR Vision Bridge (Virtual OLED HUD)](./openar-vision-bridge/) | Runs object detection on a webcam feed, then distils results into a **128×64 monochrome HUD** (virtual OLED) to simulate embedded AR glasses output. Includes a script to export a **GIF/MP4** demo. | Ultralytics YOLO, OpenCV, NumPy |
| [🧊 3D Bounding Boxes + AR Overlay (PKU/Baidu)](./3d-bbox-ar-pku/) | Projects **3D bounding boxes** (and optional CAD mesh) onto images using camera intrinsics + pose labels from the PKU/Baidu autonomous driving dataset. | NumPy, OpenCV, Pandas, Camera Geometry |
| [🧩 Segment Anything (SAM) - Zero-Shot Segmentation](./sam-segmentation/) | Zero-shot interactive segmentation using **SAM** with point prompts, box prompts, and automatic mask generation. | PyTorch, Segment Anything, OpenCV/Matplotlib |
| [🧠 Object Detection with DETR](./detr-object-detection/) | End-to-end object detection using **DETR** (transformer set prediction) with pre-trained COCO weights and inference on custom images. | PyTorch, Hugging Face Transformers, OpenCV/Matplotlib |
| [🎥 CCTV Anomaly Detection](./cctv-anomaly-detection/) | Unsupervised anomaly detection in surveillance videos using a ConvLSTM autoencoder + reconstruction error thresholding. | TensorFlow/Keras, OpenCV, Python |
| [🚘 License Plate Detection](./license-plate-detection/) | Detect vehicle license plates in images/videos using a fine-tuned YOLO model. | Ultralytics YOLO, OpenCV, Python |
| [🛰️ Aerial Object Detection](./military-assets-detection/) | Object detection in aerial imagery using a YOLO model (dataset-driven classes). | Ultralytics YOLO, OpenCV, Python |
| [🧪 CCTV Threat-Object Classification](./cctv-weapon-classification/) | Binary classification of CCTV frames using transfer learning (MobileNetV2) to predict whether a threat object is present. | TensorFlow/Keras, OpenCV, Python |
| [🔗 CLIP from Scratch](./clip-from-scratch/) | Ground-up implementation of **CLIP** — no OpenAI weights, no HuggingFace wrappers. ViT vision encoder (random init) + 1.5B causal LM (QLoRA 4-bit) contrastively trained on Flickr30k into a shared 512-dim embedding space. | PyTorch, PEFT, bitsandbytes, HuggingFace Transformers |
| [😊 Facial Emotion Recognition (FER-2013 CNN)](https://github.com/nosainwe/computer-vision-projects/tree/main/CNN-based%20facial%20emotion%20recognition) | End-to-end deep learning pipeline for facial emotion recognition using a custom 4-block CNN. Handles class imbalance with weighted loss, applies data augmentation, and provides evaluation via classification reports and confusion matrices, plus a prediction gallery for qualitative analysis. | PyTorch, Torchvision, Matplotlib, Seaborn, scikit-learn |

---

## What's inside each project folder

Each folder typically contains:

- `README.md` - project overview, setup, usage, and notes  
- `requirements.txt` - Python dependencies  
- `main.py` and/or `.ipynb` - the entry point (script or notebook)  
- `src/` - helper code (if the project has a CLI/demo script)  
- `assets/` - optional outputs (GIFs/images), usually gitignored unless small and useful  

Click any project above to see the details.
