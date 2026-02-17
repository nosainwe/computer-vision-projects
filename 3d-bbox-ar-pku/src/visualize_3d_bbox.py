from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from .utils import (
    K_DEFAULT,
    euler_to_rot,
    load_mesh,
    make_3d_box,
    parse_prediction_string,
    project_points,
    transform_points,
)


BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # left face
    (4, 5), (5, 6), (6, 7), (7, 4),  # right face
    (0, 4), (1, 5), (2, 6), (3, 7),  # connections
]


def draw_box(img: np.ndarray, pts2d: np.ndarray, color=(255, 0, 255), thickness=2) -> None:
    pts = pts2d.astype(int)
    for a, b in BOX_EDGES:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, thickness, cv2.LINE_AA)


def draw_mesh(img: np.ndarray, verts2d: np.ndarray, triangles: np.ndarray) -> None:
    """
    Simple triangle fill on the image. This is intentionally basic.
    """
    h, w = img.shape[:2]
    for tri in triangles:
        poly = verts2d[tri].astype(np.int32)
        # skip triangles that are way out of bounds
        if np.any(poly[:, 0] < -w) or np.any(poly[:, 0] > 2 * w) or np.any(poly[:, 1] < -h) or np.any(poly[:, 1] > 2 * h):
            continue
        cv2.fillConvexPoly(img, poly, (0, 255, 0), lineType=cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render 3D bounding boxes (and optional mesh) on PKU/Baidu images.")
    p.add_argument("--data_dir", type=str, required=True, help="Path to data/pku-autonomous-driving/")
    p.add_argument("--row", type=int, default=0, help="Row index in train.csv to visualise.")
    p.add_argument("--out", type=str, default="assets/output.jpg", help="Output image path.")
    p.add_argument("--imgsz", type=int, default=2, help="Line thickness scale (simple).")
    p.add_argument("--mesh_model", type=str, default=None, help="Optional car model JSON name for mesh overlay (e.g. dazhongmaiteng).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    train_csv = data_dir / "train.csv"
    train_images = data_dir / "train_images"
    car_models = data_dir / "car_models_json"

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing {train_csv}. Did you download the Kaggle competition data?")
    if not train_images.exists():
        raise FileNotFoundError(f"Missing {train_images}. Expected train_images/ folder.")

    df = pd.read_csv(train_csv)
    if args.row < 0 or args.row >= len(df):
        raise ValueError(f"--row out of range. train.csv has {len(df)} rows.")

    image_id = df.loc[args.row, "ImageId"]
    pred_str = df.loc[args.row, "PredictionString"]

    img_path = train_images / f"{image_id}.jpg"
    if not img_path.exists():
        # sometimes png in some mirrors, try png
        alt = train_images / f"{image_id}.png"
        if alt.exists():
            img_path = alt
        else:
            raise FileNotFoundError(f"Image not found: {img_path}")

    img = cv2.imread(str(img_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {img_path}")

    poses = parse_prediction_string(str(pred_str))

    # generic car box size used in many baseline visualisations (approx)
    box_corners = make_3d_box((1.8, 1.6, 4.0))  # (x,y,z) rough proportions

    for pose in poses:
        R = euler_to_rot(pose.yaw, pose.pitch, pose.roll)
        t = np.array([pose.x, pose.y, pose.z], dtype=np.float32)

        corners_cam = transform_points(box_corners, R, t)
        corners_2d = project_points(corners_cam, K_DEFAULT)

        draw_box(img, corners_2d, thickness=max(1, args.imgsz))

        # optional mesh overlay
        if args.mesh_model:
            verts, tris = load_mesh(car_models, args.mesh_model)
            verts_cam = transform_points(verts, R, t)
            verts_2d = project_points(verts_cam, K_DEFAULT)
            draw_mesh(img, verts_2d, tris)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
