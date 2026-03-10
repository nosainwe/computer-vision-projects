from __future__ import annotations
import json
from dataclasses import dataclass
from math import sin, cos
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

# fixed intrinsics from the PKU/Baidu competition - this camera doesn't change across the dataset
# fx, fy are the focal lengths, cx/cy are the principal point - pulled from the competition notebooks
K_DEFAULT = np.array(
    [
        [2304.5479, 0.0, 1686.2379],
        [0.0, 2305.8757, 1354.9849],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


# one pose = one car in the scene - model_type tells us which CAD model to use if we're rendering mesh
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
    # format is flat: model_type yaw pitch roll x y z, repeated for every car in the image
    # divisiblity check catches corrupted or truncated rows before they cause confusing index errors
    items = pred.strip().split()
    if len(items) % 7 != 0:
        raise ValueError(f"PredictionString length not divisible by 7. Got {len(items)} items.")

    poses: List[Pose] = []
    for i in range(0, len(items), 7):
        model_type = items[i]
        yaw, pitch, roll, x, y, z = map(float, items[i + 1: i + 7])
        poses.append(Pose(model_type, yaw, pitch, roll, x, y, z))
    return poses


def euler_to_rot(yaw: float, pitch: float, roll: float) -> np.ndarray:
    # yaw rotates around Y, pitch around X, roll around Z
    # application order matters - R @ P @ Y means yaw is applied first
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
    # perspective divide - dividing x,y by z gives normalized image coords
    # clipping z to 1e-6 avoids division by zero for points exactly on the camera plane
    pts = points_3d.copy()
    zs = pts[:, 2:3]
    zs = np.clip(zs, 1e-6, None)
    pts_norm = pts[:, :2] / zs

    # homogeneous form so we can apply K with a single matmul
    pts_h = np.concatenate([pts_norm, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    proj = (K @ pts_h.T).T

    # dropping the homogeneous coordinate - we only need pixel (u, v)
    return proj[:, :2]


def make_3d_box(size_xyz: Tuple[float, float, float]) -> np.ndarray:
    # building the 8 corners of an axis-aligned box centered at origin
    # left face = corners 0-3, right face = corners 4-7 - matches BOX_EDGES in visualize.py
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
    # rotating then translating - order matters, this puts us in camera frame
    return (R @ points.T).T + t.reshape(1, 3)


def load_mesh(car_models_dir: Path, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
    # json files live in car_models_json/ - each file is one car body
    p = car_models_dir / f"{model_name}.json"
    data = json.loads(p.read_text(encoding="utf-8"))

    vertices = np.array(data["vertices"], dtype=np.float32)
    triangles = np.array(data["triangles"], dtype=np.int32)

    # Y flip is load-bearing - the CAD models use a different handedness than the camera coordinate system
    # skipping this makes every car look upside down
    vertices[:, 1] = -vertices[:, 1]

    return vertices, triangles
