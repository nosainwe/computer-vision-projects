OpenAR Vision Bridge (Virtual OLED HUD) — YOLO + 128×64 “Glasses View”
==========================================================================

What this is
------------
This project demonstrates a practical “bridge” between modern computer vision and embedded display constraints.

Instead of showing full-resolution bounding boxes (which tiny wearable displays cannot render), the script:
1) runs object detection on a webcam feed (YOLO),
2) reduces the output to a compact, readable HUD,
3) renders that HUD onto a simulated 128×64 monochrome OLED,
4) upscales it with nearest-neighbour interpolation so you can preview the pixel look on a laptop.

Why this is useful
------------------
Small OLEDs used in hobby AR / wearable projects (SSD1306-style) are often 128×64 pixels.
That means you cannot “stream vision”, you must distil information into:
- a short label,
- a confidence indicator,
- simple shapes (crosshair/reticle, bars, icons).

This repo shows that you understand:
- low-resolution UI design,
- data reduction for microcontroller-class hardware,
- how to prototype a display layout on a PC before touching any hardware.

What you’ll see when you run it
-------------------------------
Two windows appear:
1) “Computer Vision (Backend)” — normal webcam view with a single best detection box drawn.
2) “OpenAR Glasses View (128x64)” — a pixelated black/white HUD rendered on a 128×64 canvas.

Controls
--------
- Press 'q' to quit.

Requirements
------------
- Python 3.9+ recommended
- A webcam
- Ultralytics YOLO weights download will happen automatically on first run (internet required once).

Quickstart
----------
1) Install dependencies:

   pip install -r requirements.txt

2) Run the notebook:
   Open the notebook file and run all cells:
   openar_vision_bridge_virtual_oled.ipynb

Record a demo for your GitHub README (MP4 + GIF)
------------------------------------------------
You can generate a short demo clip without screen-recording.

1) Install dependencies:

   pip install -r requirements.txt

2) Run:

   python record_demo.py --seconds 8 --fps 12

Outputs:
- demo_oled.mp4  (easy to upload to GitHub releases / attach elsewhere)
- demo_oled.gif  (perfect for embedding in a README)

Tip:
- If your webcam isn’t on index 0, try --camera 1
- If YOLO feels slow, reduce FPS or seconds.

Notes / tweaks
--------------
- OLED size is defined by OLED_WIDTH and OLED_HEIGHT.
- SCALE_FACTOR controls how large the virtual OLED appears on your screen.
- The HUD layout is deliberately simple (header, target label, confidence, crosshair).
- If you want more stable output, you can track detections across frames or average confidence.

