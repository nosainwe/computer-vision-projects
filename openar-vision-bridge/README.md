# 👓 OpenAR Vision Bridge (Virtual OLED HUD)
### YOLO + 128×64 "Glasses View"

---

## 🔍 What This Is
This project demonstrates a practical "bridge" between modern computer vision and embedded display constraints. 

Instead of showing full-resolution bounding boxes (which tiny wearable displays cannot render), the script:
1. 📷 **Runs object detection** on a webcam feed (YOLO).
2. 📉 **Reduces the output** to a compact, readable HUD.
3. ⬛ **Renders that HUD** onto a simulated 128×64 monochrome OLED.
4. 🔎 **Upscales it** with nearest-neighbour interpolation so you can preview the pixel look on a laptop.

---

## 💡 Why This Is Useful
Small OLEDs used in hobby AR and wearable projects (like the SSD1306) are often limited to 128×64 pixels. This means you cannot simply "stream vision." Instead, you must distill information into:
* 🏷️ A short label
* 📊 A confidence indicator
* 🎯 Simple shapes (crosshairs, reticles, bars, icons)

This repository demonstrates a practical understanding of:
* 🎨 **Low-resolution UI design**: Crafting interfaces for extremely constrained displays.
* 🗜️ **Data reduction**: Filtering data for microcontroller-class hardware.
* 💻 **Rapid prototyping**: Designing and testing a display layout on a PC before touching any physical hardware.

---

## 👀 What You'll See When You Run It
Two windows will appear on your screen:
1. 🖥️ **"Computer Vision (Backend)"** - A normal webcam view with a single best-detection box drawn.
2. 🥽 **"OpenAR Glasses View (128x64)"** - A pixelated black-and-white HUD rendered on a 128×64 canvas.

---

## 🎮 Controls
* 🛑 Press `q` to quit the application.

---

## ⚙️ Requirements
* 🐍 **Python 3.9+** (recommended)
* 📹 **A connected webcam**
* 🧠 **Ultralytics YOLO weights** (download will happen automatically on the first run; an internet connection is required once).

---

## 🚀 Quickstart

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Run the notebook:**
Open the provided notebook file and run all cells:
`openar_vision_bridge_virtual_oled.ipynb`

---

## 🎥 Record a Demo for Your GitHub README (MP4 + GIF)
You can easily generate a short demo clip without needing external screen-recording software.

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Run the recording script:**
```bash
python record_demo.py --seconds 8 --fps 12
```

**Outputs:**
* 🎞️ `demo_oled.mp4` (Ideal for uploading to GitHub releases or attaching elsewhere)
* 🖼️ `demo_oled.gif` (Perfect for embedding directly in a README)

---

## 💡 Pro Tips
* 📸 If your webcam isn't on index `0`, try adding the flag `--camera 1`.
* ⚡ If YOLO feels slow, try reducing the FPS or the total seconds recorded.

---

## 🛠️ Notes & Tweaks
* 📏 OLED size is defined by the `OLED_WIDTH` and `OLED_HEIGHT` variables.
* 🔍 `SCALE_FACTOR` controls how large the virtual OLED appears on your screen.
* 📐 The HUD layout is deliberately simple (header, target label, confidence, crosshair).
* 🎯 If you want more stable output, you can tweak the code to track detections across frames or average the confidence scores.
