from __future__ import annotations

import math

import numpy as np


CAMERA_BOX_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def camera_box_corners(
    center: np.ndarray,
    length: float,
    width: float,
    height: float,
    yaw: float,
) -> np.ndarray:
    """Return 8 camera-frame corners for box=[cx,cy,cz,length,width,height,yaw].

    Camera frame:
    - x: image right
    - y: image down
    - z: camera forward/depth

    Axis convention follows the requested debug definition:
    - length axis: [sin(yaw), 0, cos(yaw)]
    - width axis:  [cos(yaw), 0, -sin(yaw)]
    - height up:   [0, -1, 0]

    Corner order:
    0: (-l, -w, -h)
    1: (+l, -w, -h)
    2: (+l, +w, -h)
    3: (-l, +w, -h)
    4: (-l, -w, +h)
    5: (+l, -w, +h)
    6: (+l, +w, +h)
    7: (-l, +w, +h)
    """
    c = np.asarray(center, dtype=np.float64)
    cos_y, sin_y = math.cos(float(yaw)), math.sin(float(yaw))
    x_axis = np.asarray([sin_y, 0.0, cos_y], dtype=np.float64)
    z_axis = np.asarray([cos_y, 0.0, -sin_y], dtype=np.float64)
    y_axis = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
    corners = []
    for sl, sw, sh in [
        (-1, -1, -1),
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, 1, 1),
    ]:
        corners.append(
            c
            + 0.5 * sl * float(length) * x_axis
            + 0.5 * sw * float(width) * z_axis
            + 0.5 * sh * float(height) * y_axis
        )
    return np.asarray(corners, dtype=np.float64)


def project_camera_points(points_cam: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_cam, dtype=np.float64)
    z = np.maximum(pts[:, 2], 1.0e-6)
    u = intrinsic[0, 0] * pts[:, 0] / z + intrinsic[0, 2]
    v = intrinsic[1, 1] * pts[:, 1] / z + intrinsic[1, 2]
    return np.stack([u, v], axis=1)


def projected_bbox(points_px: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            float(np.min(points_px[:, 0])),
            float(np.min(points_px[:, 1])),
            float(np.max(points_px[:, 0])),
            float(np.max(points_px[:, 1])),
        ],
        dtype=np.float64,
    )
