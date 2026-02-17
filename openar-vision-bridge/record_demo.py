"""
record_demo.py
--------------
Generates a short demo of the virtual 128×64 OLED HUD as:
- demo_oled.mp4
- demo_oled.gif

Usage:
  python record_demo.py --seconds 8 --fps 12
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import imageio
import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=int, default=8, help="Length of demo recording.")
    p.add_argument("--fps", type=int, default=12, help="Frames per second to save.")
    p.add_argument("--camera", type=int, default=0, help="Webcam index (0, 1, ...).")
    p.add_argument("--weights", type=str, default="yolov8n.pt", help="YOLO weights.")
    p.add_argument("--scale", type=int, default=6, help="Upscale factor for OLED preview.")
    p.add_argument("--out_mp4", type=str, default="demo_oled.mp4")
    p.add_argument("--out_gif", type=str, default="demo_oled.gif")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    OLED_WIDTH = 128
    OLED_HEIGHT = 64
    SCALE_FACTOR = max(1, args.scale)
    font = cv2.FONT_HERSHEY_SIMPLEX

    out_mp4 = Path(args.out_mp4)
    out_gif = Path(args.out_gif)

    model = YOLO(args.weights)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam index {args.camera}. "
            f"Try --camera 1 or close other apps using the camera."
        )

    # VideoWriter for MP4 (scaled OLED frames)
    w = OLED_WIDTH * SCALE_FACTOR
    h = OLED_HEIGHT * SCALE_FACTOR
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_mp4), fourcc, args.fps, (w, h), isColor=True)

    gif_frames_rgb: list[np.ndarray] = []

    frame_count = args.seconds * args.fps
    print(f"Recording {frame_count} frames (~{args.seconds}s @ {args.fps}fps)...")

    # Pace the loop to the requested FPS
    frame_interval = 1.0 / max(1, args.fps)
    next_t = time.time()

    try:
        for _ in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO inference
            results = model.predict(frame, conf=args.conf, verbose=False)

            detection_text = "SCANNING..."
            conf_score = ""

            if results and results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                best = max(boxes, key=lambda x: float(x.conf[0]))
                cls = int(best.cls[0])
                label = model.names.get(cls, str(cls))
                conf = float(best.conf[0])

                detection_text = str(label).upper()
                conf_score = f"{int(conf * 100)}%"

            # Build OLED canvas (monochrome)
            oled = np.zeros((OLED_HEIGHT, OLED_WIDTH), dtype=np.uint8)

            # Header bar (white)
            cv2.rectangle(oled, (0, 0), (OLED_WIDTH, 12), 255, -1)
            cv2.putText(oled, "OPEN-AR HUD", (2, 9), font, 0.3, 0, 1)

            # Body text
            cv2.putText(oled, "TARGET:", (5, 25), font, 0.3, 255, 1)
            cv2.putText(oled, detection_text, (5, 45), font, 0.6, 255, 1)

            if conf_score:
                cv2.putText(oled, conf_score, (90, 60), font, 0.3, 255, 1)

            # Reticle
            cx, cy = OLED_WIDTH // 2, OLED_HEIGHT // 2
            cv2.line(oled, (cx - 5, cy), (cx + 5, cy), 255, 1)
            cv2.line(oled, (cx, cy - 5), (cx, cy + 5), 255, 1)

            # Upscale with nearest-neighbour to preserve pixel look
            oled_scaled = cv2.resize(
                oled, (w, h), interpolation=cv2.INTER_NEAREST
            )

            # Convert to 3-channel BGR for VideoWriter
            oled_bgr = cv2.cvtColor(oled_scaled, cv2.COLOR_GRAY2BGR)

            vw.write(oled_bgr)

            # Save RGB frame for GIF
            oled_rgb = cv2.cvtColor(oled_bgr, cv2.COLOR_BGR2RGB)
            gif_frames_rgb.append(oled_rgb)

            # Pace to fps
            next_t += frame_interval
            sleep_for = next_t - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)

    finally:
        cap.release()
        vw.release()

    if len(gif_frames_rgb) == 0:
        raise RuntimeError("No frames captured. Check your webcam.")

    # Write GIF (duration per frame in seconds)
    imageio.mimsave(out_gif, gif_frames_rgb, duration=1.0 / max(1, args.fps))

    print(f"Saved: {out_mp4.resolve()}")
    print(f"Saved: {out_gif.resolve()}")


if __name__ == "__main__":
    main()
