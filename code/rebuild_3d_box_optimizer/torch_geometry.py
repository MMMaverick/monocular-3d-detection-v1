from __future__ import annotations

import torch


def camera_box_corners(
    centers: torch.Tensor,
    size: torch.Tensor,
    yaw: torch.Tensor,
    axes_cam: torch.Tensor | None = None,
) -> torch.Tensor:
    if size.ndim == 1:
        length = size[0].clamp_min(0.1).expand(centers.shape[0])
        width = size[1].clamp_min(0.1).expand(centers.shape[0])
        height = size[2].clamp_min(0.1).expand(centers.shape[0])
    else:
        length = size[:, 0].clamp_min(0.1)
        width = size[:, 1].clamp_min(0.1)
        height = size[:, 2].clamp_min(0.1)
    if axes_cam is None:
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)
        x_axis = torch.stack([sin_y, torch.zeros_like(yaw), cos_y], dim=1)
        z_axis = torch.stack([cos_y, torch.zeros_like(yaw), -sin_y], dim=1)
        y_axis = torch.zeros_like(x_axis)
        y_axis[:, 1] = -1.0
        axes = torch.stack([x_axis, z_axis, y_axis], dim=1)
    else:
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)
        length_axis = cos_y[:, None] * axes_cam[:, 0, :] + sin_y[:, None] * axes_cam[:, 1, :]
        width_axis = -sin_y[:, None] * axes_cam[:, 0, :] + cos_y[:, None] * axes_cam[:, 1, :]
        height_axis = axes_cam[:, 2, :]
        axes = torch.stack([length_axis, width_axis, height_axis], dim=1)
    coeffs = torch.tensor(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=centers.dtype,
        device=centers.device,
    )
    dims = torch.stack([length, width, height], dim=1)
    local = coeffs[None, :, :] * dims[:, None, :]
    return centers[:, None, :] + torch.einsum("nkd,nda->nka", local, axes)


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((*points.shape[:-1], 1), dtype=points.dtype, device=points.device)
    homo = torch.cat([points, ones], dim=-1)
    return torch.einsum("nkj,nij->nki", homo, transform)[:, :, :3]


def project(points_cam: torch.Tensor, intrinsic: torch.Tensor, min_depth: float = 1.0e-5) -> torch.Tensor:
    z = points_cam[..., 2].clamp_min(float(min_depth))
    u = intrinsic[:, None, 0, 0] * points_cam[..., 0] / z + intrinsic[:, None, 0, 2]
    v = intrinsic[:, None, 1, 1] * points_cam[..., 1] / z + intrinsic[:, None, 1, 2]
    return torch.stack([u, v], dim=-1)


def projected_bbox(corners_px: torch.Tensor) -> torch.Tensor:
    mins = corners_px.min(dim=1).values
    maxs = corners_px.max(dim=1).values
    return torch.cat([mins, maxs], dim=1)


def vertical_edge_x(corners_px: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    edge_pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
    edge_x = torch.stack([corners_px[:, [a, b], 0].mean(dim=1) for a, b in edge_pairs], dim=1)
    left = edge_x.min(dim=1).values
    right = edge_x.max(dim=1).values
    return left, right
