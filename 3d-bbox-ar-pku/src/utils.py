from __future__ import annotations

import json
from dataclasses import dataclass
from math import sin, cos
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np


# Default camera intrinsics used in many PKU/Baidu visualisation notebooks
# (This dataset uses a fixed intrinsics matrix for the main camera.)
K_DEFAULT = np.array(
    [
        [2304.5479, 0.0, 1686.2379],
        [0.0, 2305.8757, 1354.9849],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


@dataclass
class Pose:
    model_type: str
    yaw: float
    pitch: float
    roll: float
    x: float
    y: float
    z: float


def parse_prediction_string(pred: str) -> List[Pose]:
    """
    PredictionString format in train.csv:
    model_type yaw pitch roll x y z (repeated...)
    """
    items = pred.strip().split()
    if len(items) % 7 != 0:
        raise ValueError(f"PredictionString length not divisible by 7. Got {len(items)} items.")

    poses: List[Pose] = []
    for i in range(0, len(items), 7):
        model_type = items[i]
        yaw, pitch, roll, x, y, z = map(float, items[i + 1 : i + 7])
        poses.append(Pose(model_type, yaw, pitch, roll, x, y, z))
    return poses


def euler_to_rot(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Yaw (Y axis), pitch (X axis), roll (Z axis) -> rotation matrix."""
    Y = np.array(
        [[cos(yaw), 0, sin(yaw)], [0, 1, 0], [-sin(yaw), 0, cos(yaw)]],
        dtype=np.float32,
    )
    P = np.array(
        [[1, 0, 0], [0, cos(pitch), -sin(pitch)], [0, sin(pitch), cos(pitch)]],
        dtype=np.float32,
    )
    R = np.array(
        [[cos(roll), -sin(roll), 0], [sin(roll), cos(roll), 0], [0, 0, 1]],
        dtype=np.float32,
    )
    return R @ P @ Y


def project_points(points_3d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Project 3D camera-coordinate points (N,3) -> image points (N,2).
    """
    pts = points_3d.copy()
    zs = pts[:, 2:3]
    zs = np.clip(zs, 1e-6, None)

    pts_norm = pts[:, :2] / zs
    pts_h = np.concatenate([pts_norm, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    proj = (K @ pts_h.T).T
    return proj[:, :2]


def make_3d_box(size_xyz: Tuple[float, float, float]) -> np.ndarray:
    """
    Return 8 corners of a 3D box centered at origin in object frame.
    size_xyz: (length_x, width_y, height_z) roughly.
    """
    lx, ly, lz = size_xyz
    x = lx / 2
    y = ly / 2
    z = lz / 2

    corners = np.array(
        [
            [-x, -y, -z],
            [-x, -y,  z],
            [-x,  y,  z],
            [-x,  y, -z],
            [ x, -y, -z],
            [ x, -y,  z],
            [ x,  y,  z],
            [ x,  y, -z],
        ],
        dtype=np.float32,
    )
    return corners


def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Object frame -> camera frame."""
    return (R @ points.T).T + t.reshape(1, 3)


def load_mesh(car_models_dir: Path, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load CAD mesh from car_models_json/<model_name>.json
    Returns (vertices Nx3, triangles Mx3).
    """
    p = car_models_dir / f"{model_name}.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    vertices = np.array(data["vertices"], dtype=np.float32)
    triangles = np.array(data["triangles"], dtype=np.int32)

    # Many visualisation notebooks flip Y to match camera coords
    vertices[:, 1] = -vertices[:, 1]
    return vertices, triangles
