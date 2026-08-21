from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoseBundle:
    ego_to_world: np.ndarray
    camera_to_ego: np.ndarray

    @property
    def camera_to_world(self) -> np.ndarray:
        return self.ego_to_world @ self.camera_to_ego

    @property
    def world_to_camera(self) -> np.ndarray:
        return np.linalg.inv(self.camera_to_world)

    @property
    def camera_center_world(self) -> np.ndarray:
        return self.camera_to_world[:3, 3].copy()


def quat_xyzw_to_matrix(q: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n <= 1.0e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float64)], axis=1)
    return (homo @ transform.T)[:, :3]


def project_points(points_cam: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_cam, dtype=np.float64)
    z = np.maximum(pts[:, 2], 1.0e-6)
    u = intrinsic[0, 0] * pts[:, 0] / z + intrinsic[0, 2]
    v = intrinsic[1, 1] * pts[:, 1] / z + intrinsic[1, 2]
    return np.stack([u, v], axis=1)


def axes_from_camera_pose(camera_to_world: np.ndarray) -> np.ndarray:
    length_axis = camera_to_world[:3, :3] @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    up_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    length_axis = length_axis - np.dot(length_axis, up_axis) * up_axis
    if np.linalg.norm(length_axis) < 1.0e-6:
        length_axis = camera_to_world[:3, :3] @ np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        length_axis = length_axis - np.dot(length_axis, up_axis) * up_axis
    length_axis = length_axis / max(float(np.linalg.norm(length_axis)), 1.0e-9)
    width_axis = np.cross(up_axis, length_axis)
    width_axis = width_axis / max(float(np.linalg.norm(width_axis)), 1.0e-9)
    return np.stack([length_axis, width_axis, up_axis], axis=0)


def car_axes_from_orientation_camera(camera_to_car: np.ndarray) -> np.ndarray:
    length_axis = camera_to_car[:3, :3] @ np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    up_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    length_axis = length_axis - float(np.dot(length_axis, up_axis)) * up_axis
    if float(np.linalg.norm(length_axis)) < 1.0e-6:
        length_axis = camera_to_car[:3, :3] @ np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        length_axis = length_axis - float(np.dot(length_axis, up_axis)) * up_axis
    length_axis = length_axis / max(float(np.linalg.norm(length_axis)), 1.0e-9)
    width_axis = np.cross(up_axis, length_axis)
    width_axis = width_axis / max(float(np.linalg.norm(width_axis)), 1.0e-9)
    return np.stack([length_axis, width_axis, up_axis], axis=0)
