from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .data import Observation
from .torch_geometry import camera_box_corners, project, projected_bbox, vertical_edge_x


@dataclass
class TrackResult:
    rows: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]
    summary: dict[str, Any]
    final_rows: list[dict[str, Any]]
    final_diagnostics: list[dict[str, Any]]
    final_summary: dict[str, Any]


def optimize_track(config: dict[str, Any], observations: list[Observation]) -> TrackResult:
    device = choose_device(str(config["solver"].get("device", "cuda")))
    dtype = torch.float32 if str(config["solver"].get("dtype", "float32")) == "float32" else torch.float64
    n = len(observations)
    centers0 = torch.tensor(np.stack([o.init_center_cam for o in observations]), dtype=dtype, device=device)
    size0_np = robust_track_size_init(observations)
    min_size_np = np.max(np.stack([o.min_size for o in observations]), axis=0)
    max_size_np = np.min(np.stack([o.max_size for o in observations]), axis=0)
    min_size_np = np.minimum(min_size_np, max_size_np * 0.99)
    size0_np = np.clip(size0_np, min_size_np, max_size_np)
    min_size = torch.tensor(min_size_np, dtype=dtype, device=device)
    max_size = torch.tensor(max_size_np, dtype=dtype, device=device)
    center_param = torch.nn.Parameter(centers0.clone())
    size_param = torch.nn.Parameter(size_to_unconstrained(torch.tensor(size0_np, dtype=dtype, device=device), min_size, max_size))
    opt = torch.optim.Adam([center_param, size_param], lr=float(config["solver"].get("learning_rate", 0.03)), weight_decay=float(config["solver"].get("weight_decay", 0.0)))
    scheduler = make_scheduler(config, opt)
    tensors = make_observation_tensors(config, observations, dtype, device)
    best_loss = float("inf")
    best_centers = centers0.detach().clone()
    best_size = torch.tensor(size0_np, dtype=dtype, device=device)
    max_iter_cfg = int(config["solver"].get("max_iterations", 300) or 0)
    convergence_cfg = config["solver"].get("convergence", {})
    convergence_enabled = bool(convergence_cfg.get("enabled", max_iter_cfg <= 0))
    max_iter = max_iter_cfg if max_iter_cfg > 0 else int(convergence_cfg.get("safety_max_iterations", 2000))
    min_iter = int(convergence_cfg.get("min_iterations", min(300, max_iter)))
    patience = int(convergence_cfg.get("patience", 120))
    check_every = max(int(convergence_cfg.get("check_every", 10)), 1)
    rel_tol = float(convergence_cfg.get("relative_improvement", 1.0e-5))
    abs_tol = float(convergence_cfg.get("absolute_improvement", 1.0e-7))
    no_improve_steps = 0
    iterations_used = 0
    stop_reason = "max_iterations"
    progress_interval = int(config["solver"].get("progress_interval", 0) or 0)
    progress_view = observations[0].view if observations else "unknown"
    progress_track = observations[0].track_id if observations else -1
    for step in range(max_iter):
        iterations_used = step + 1
        opt.zero_grad(set_to_none=True)
        size = unconstrained_to_size(size_param, min_size, max_size)
        losses = compute_losses(config, center_param, size, tensors, protect_size=True)
        total = losses["total"]
        if torch.isfinite(total):
            if step == 0 or iterations_used % check_every == 0 or iterations_used == max_iter:
                value = float(total.detach().cpu())
                threshold = max(abs_tol, rel_tol * max(abs(best_loss), 1.0)) if np.isfinite(best_loss) else 0.0
                if value < best_loss - threshold:
                    best_loss = value
                    best_centers = center_param.detach().clone()
                    best_size = size.detach().clone()
                    no_improve_steps = 0
                else:
                    no_improve_steps += check_every
                if convergence_enabled and iterations_used >= min_iter and no_improve_steps >= patience:
                    stop_reason = "converged"
                    break
            total.backward()
            clip = float(config["solver"].get("gradient_clip", 0.0) or 0.0)
            if clip > 0:
                torch.nn.utils.clip_grad_norm_([center_param, size_param], clip)
            opt.step()
            if scheduler is not None:
                scheduler.step()
                clamp_optimizer_learning_rate(config, opt)
            if progress_interval > 0 and (iterations_used == 1 or iterations_used % progress_interval == 0 or iterations_used == max_iter):
                current_value = float(total.detach().cpu())
                print(
                    f"TRACK_PROGRESS view={progress_view} track={progress_track} "
                    f"iter={iterations_used}/{max_iter} loss={current_value:.6g} best={best_loss:.6g} lr={current_learning_rate(opt):.6g}",
                    flush=True,
                )
        else:
            stop_reason = "non_finite_loss"
            break
    final_centers = center_param.detach().clone()
    final_size = unconstrained_to_size(size_param.detach(), min_size, max_size).detach().clone()
    best_losses = compute_losses(config, best_centers, best_size, tensors, reduce=False)
    final_losses = compute_losses(config, final_centers, final_size, tensors, reduce=False)
    best_result = build_result(config, observations, best_centers, best_size, best_losses, best_loss, device, iterations_used, stop_reason, "best")
    final_loss_value = float(final_losses["total_per_frame"].sum().detach().cpu())
    final_result = build_result(config, observations, final_centers, final_size, final_losses, final_loss_value, device, iterations_used, stop_reason, "final_iter")
    return TrackResult(
        rows=best_result.rows,
        diagnostics=best_result.diagnostics,
        summary=best_result.summary,
        final_rows=final_result.rows,
        final_diagnostics=final_result.diagnostics,
        final_summary=final_result.summary,
    )


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def robust_track_size_init(observations: list[Observation]) -> np.ndarray:
    sizes = np.stack([o.init_size for o in observations])
    return np.median(sizes, axis=0)


def size_to_unconstrained(size: torch.Tensor, min_size: torch.Tensor, max_size: torch.Tensor) -> torch.Tensor:
    t = ((size - min_size) / (max_size - min_size).clamp_min(1.0e-6)).clamp(1.0e-4, 1 - 1.0e-4)
    return torch.logit(t)


def unconstrained_to_size(param: torch.Tensor, min_size: torch.Tensor, max_size: torch.Tensor) -> torch.Tensor:
    return min_size + torch.sigmoid(param) * (max_size - min_size)


def make_scheduler(config: dict[str, Any], opt: torch.optim.Optimizer):
    sched_cfg = config["solver"].get("scheduler", {})
    if not sched_cfg.get("enabled", False):
        return None
    if sched_cfg.get("type", "exponential") == "exponential":
        gamma = float(sched_cfg.get("gamma", 0.98))
        step_size = int(sched_cfg.get("step_size", 25))
        return torch.optim.lr_scheduler.StepLR(opt, step_size=max(step_size, 1), gamma=gamma)
    return None


def clamp_optimizer_learning_rate(config: dict[str, Any], opt: torch.optim.Optimizer) -> None:
    sched_cfg = config["solver"].get("scheduler", {})
    min_lr = float(sched_cfg.get("minimum_learning_rate", 0.0) or 0.0)
    if min_lr <= 0:
        return
    for group in opt.param_groups:
        group["lr"] = max(float(group.get("lr", min_lr)), min_lr)


def current_learning_rate(opt: torch.optim.Optimizer) -> float:
    if not opt.param_groups:
        return float("nan")
    return float(opt.param_groups[0].get("lr", float("nan")))


def make_observation_tensors(config: dict[str, Any], observations: list[Observation], dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor | list[Any]]:
    mask_points, mask_points_valid = pack_mask_points(observations, dtype, device)
    return {
        "view_name": observations[0].view if observations else "",
        "box2d": torch.tensor(np.stack([o.box2d for o in observations]), dtype=dtype, device=device),
        "initial_center_cam": torch.tensor(np.stack([o.init_center_cam for o in observations]), dtype=dtype, device=device),
        "initial_camera_distance": torch.tensor([float(np.linalg.norm(o.init_center_cam)) for o in observations], dtype=dtype, device=device),
        "intrinsic": torch.tensor(np.stack([o.intrinsic for o in observations]), dtype=dtype, device=device),
        "image_size": torch.tensor(np.stack([observation_image_size(o) for o in observations]), dtype=dtype, device=device),
        "box_axes_cam": torch.tensor(np.stack([o.box_axes_cam for o in observations]), dtype=dtype, device=device),
        "yaw_fixed": torch.tensor([float(o.yaw_fixed) for o in observations], dtype=dtype, device=device),
        "camera_to_ego_rot": torch.tensor(np.stack([o.camera_to_ego[:3, :3] for o in observations]), dtype=dtype, device=device),
        "camera_to_ego_trans": torch.tensor(np.stack([o.camera_to_ego[:3, 3] for o in observations]), dtype=dtype, device=device),
        "camera_to_world_rot": torch.tensor(np.stack([o.camera_to_world[:3, :3] for o in observations]), dtype=dtype, device=device),
        "camera_to_world_trans": torch.tensor(np.stack([o.camera_to_world[:3, 3] for o in observations]), dtype=dtype, device=device),
        "reference_axes_world": torch.tensor(np.stack([o.axes_world for o in observations]), dtype=dtype, device=device),
        "frame_index": torch.tensor([o.frame for o in observations], dtype=dtype, device=device),
        "min_center_depth": torch.tensor(min_center_depths(config, observations), dtype=dtype, device=device),
        "mask_bbox": torch.tensor(np.stack([o.mask_bbox if o.mask_bbox is not None else o.box2d for o in observations]), dtype=dtype, device=device),
        "mask_area": torch.tensor([max(o.mask_area, 1.0) for o in observations], dtype=dtype, device=device),
        "has_mask": torch.tensor([o.mask_bbox is not None for o in observations], dtype=torch.bool, device=device),
        "camera_center_world": torch.tensor(np.stack([o.camera_center_world for o in observations]), dtype=dtype, device=device),
        "truncated_left": torch.tensor([o.truncated["left"] for o in observations], dtype=torch.bool, device=device),
        "truncated_right": torch.tensor([o.truncated["right"] for o in observations], dtype=torch.bool, device=device),
        "truncated_top": torch.tensor([o.truncated["top"] for o in observations], dtype=torch.bool, device=device),
        "truncated_bottom": torch.tensor([o.truncated["bottom"] for o in observations], dtype=torch.bool, device=device),
        "truncated_any": torch.tensor([any(o.truncated.values()) for o in observations], dtype=torch.bool, device=device),
        "is_rear": torch.tensor([o.view == "rear" for o in observations], dtype=torch.bool, device=device),
        "near_weight": torch.tensor(near_weights(observations), dtype=dtype, device=device),
        "size_ratio_target": torch.tensor(robust_track_size_init(observations), dtype=dtype, device=device),
        "class_size_target": torch.tensor(robust_track_size_init(observations), dtype=dtype, device=device),
        "mask_points": mask_points,
        "mask_points_valid": mask_points_valid,
        "mask_point_count": mask_points_valid.sum(dim=1),
    }


def observation_image_size(obs: Observation) -> np.ndarray:
    image_size = getattr(obs, "image_size", None)
    if image_size is not None:
        return np.asarray(image_size, dtype=np.float64)
    return np.asarray([float(obs.intrinsic[0, 2] * 2.0), float(obs.intrinsic[1, 2] * 2.0)], dtype=np.float64)


def pack_mask_points(observations: list[Observation], dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_points = max((len(o.mask_points) for o in observations if o.mask_points is not None), default=0)
    if max_points <= 0:
        return (
            torch.empty((len(observations), 0, 2), dtype=dtype, device=device),
            torch.zeros((len(observations), 0), dtype=torch.bool, device=device),
        )
    points_np = np.zeros((len(observations), max_points, 2), dtype=np.float32 if dtype == torch.float32 else np.float64)
    valid_np = np.zeros((len(observations), max_points), dtype=bool)
    for i, obs in enumerate(observations):
        if obs.mask_points is None or len(obs.mask_points) == 0:
            continue
        pts = np.asarray(obs.mask_points, dtype=points_np.dtype)
        count = min(len(pts), max_points)
        points_np[i, :count, :] = pts[:count]
        valid_np[i, :count] = True
    return torch.tensor(points_np, dtype=dtype, device=device), torch.tensor(valid_np, dtype=torch.bool, device=device)


def near_weights(observations: list[Observation]) -> np.ndarray:
    # The exact thresholds come from config in compute_losses; this keeps tensor shape handling simple.
    return np.ones(len(observations), dtype=np.float64)


def min_center_depths(config: dict[str, Any], observations: list[Observation]) -> np.ndarray:
    cfg = config.get("observations", {}).get("center_depth_safety", {})
    table = cfg.get("min_center_depth_m", {})
    default = float(cfg.get("default_min_center_depth_m", 0.0) or 0.0)
    out = []
    for obs in observations:
        value = default
        if isinstance(table, dict):
            value = float(table.get(obs.label, table.get(str(obs.label).lower(), value)) or value)
        out.append(value)
    return np.asarray(out, dtype=np.float64)


def compute_losses(
    config: dict[str, Any],
    centers: torch.Tensor,
    size: torch.Tensor,
    tensors: dict[str, Any],
    reduce: bool = True,
    protect_size: bool = False,
) -> dict[str, torch.Tensor]:
    yaw = tensors.get("yaw_fixed")
    if yaw is None:
        yaw = torch.zeros((centers.shape[0],), dtype=centers.dtype, device=centers.device)
    size_for_obs = observation_size_for_loss(config, size, tensors, protect_size)
    yaw_cfg = config.get("variables", {}).get("yaw", {})
    yaw_frame = str(yaw_cfg.get("fixed_yaw_frame", yaw_cfg.get("frame", "reference_axes")))
    axes_for_corners = None if yaw_frame == "camera_yaw" else tensors.get("box_axes_cam")
    corners_cam = camera_box_corners(centers, size_for_obs, yaw, axes_for_corners)
    min_projection_depth = float(config["observations"].get("projection", {}).get("min_depth_m", 1.0e-5))
    px = project(corners_cam, tensors["intrinsic"], min_projection_depth)
    center_px = project(centers[:, None, :], tensors["intrinsic"], min_projection_depth)[:, 0, :]
    px_clipped_loss = clip_projected_points_for_loss(config, px, tensors)
    use_clipped_geometry = tensors["truncated_any"][:, None, None]
    px_geometry_loss = torch.where(use_clipped_geometry, px_clipped_loss, px)
    pred_bbox = projected_bbox(px_geometry_loss)
    pred_left, pred_right = vertical_edge_x(px_geometry_loss)
    box = tensors["box2d"]
    box_w = (box[:, 2] - box[:, 0]).clamp_min(1.0)
    left_eq = (pred_left - box[:, 0]) / box_w
    right_eq = (pred_right - box[:, 2]) / box_w
    left_one_sided = torch.relu(pred_left - box[:, 0]) / box_w
    right_one_sided = torch.relu(box[:, 2] - pred_right) / box_w
    left = torch.where(tensors["truncated_left"], left_one_sided, left_eq)
    right = torch.where(tensors["truncated_right"], right_one_sided, right_eq)
    edge_loss = (left.square() + right.square()) * float(config["observations"]["supporting_edges"].get("weight", 1.0))
    top_bottom_edge_loss = compute_top_bottom_edge_loss(config, pred_bbox, box, centers, tensors)

    bbox_fit_loss = compute_bbox_fit_loss(config, pred_bbox, box, centers, tensors)
    no_mask_mode = str(config["observations"].get("no_mask_frames", {}).get("mode", "normal"))
    if no_mask_mode == "edge_only":
        bbox_fit_loss = bbox_fit_loss * tensors["has_mask"].to(dtype=bbox_fit_loss.dtype)

    contain_bbox = projected_bbox(px_clipped_loss)
    contain_loss = compute_mask_contain_loss(config, px_clipped_loss, contain_bbox, box, tensors)
    depth_safety_loss = compute_depth_safety_loss(config, corners_cam)
    center_depth_safety_loss = compute_center_depth_safety_loss(config, centers, tensors)
    ego_box_safety_loss, ego_box_safety_diag = compute_ego_box_safety_loss(config, centers, size_for_obs, tensors)
    initial_depth_prior_loss, initial_depth_prior_diag = compute_initial_depth_prior_loss(config, centers, tensors)
    size_ratio_prior_loss = compute_size_ratio_prior_loss(config, size, tensors, centers.shape[0])
    class_size_prior_loss, class_size_diag = compute_class_size_prior_loss(config, size, tensors, centers.shape[0])
    height_depth_prior_loss, height_depth_diag = compute_height_depth_prior_loss(config, centers, size, tensors)
    ground_loss, ground_diag = compute_ground_plane_loss(config, corners_cam, tensors)
    temporal_acc_loss, temporal_vertical_acc_loss, temporal_log_depth_loss = compute_temporal_smoothness_loss(config, centers, tensors)
    pred_bbox_area = (pred_bbox[:, 2] - pred_bbox[:, 0]).clamp_min(0.0) * (pred_bbox[:, 3] - pred_bbox[:, 1]).clamp_min(0.0)
    mask_cfg = config["observations"].get("mask", {})
    pred_clipped_area = projected_cuboid_visible_area(
        px_clipped_loss,
        tensors["image_size"],
    )
    pred_full_area = projected_cuboid_visible_area(px)
    # For truncated observations, projected area outside the image is unknown
    # foreground/background, so oversize should only use the visible in-image
    # part. For untruncated observations, a box projecting outside the image is
    # itself an oversize/location error, so keep the full projected area.
    pred_oversize_area = torch.where(tensors["truncated_any"], pred_clipped_area, pred_full_area)
    ratio = pred_oversize_area / tensors["mask_area"].clamp_min(1.0)
    oversize_max_area_ratio = float(config["observations"]["mask"].get("oversize_max_area_ratio", 2.5))
    oversize_weight = float(config["observations"]["mask"].get("oversize_weight", 0.05))
    oversize = torch.relu(ratio - oversize_max_area_ratio)
    oversize_unweighted_loss = oversize.square()
    oversize_weighted_loss = oversize_unweighted_loss * oversize_weight
    has_mask_float = tensors["has_mask"].to(dtype=oversize_weighted_loss.dtype)
    contain_loss = contain_loss * has_mask_float
    oversize_weighted_loss = oversize_weighted_loss * has_mask_float

    near_mult = near_size_multipliers(config, centers, tensors)
    oversize_final_loss = oversize_weighted_loss * near_mult
    total_per_frame = edge_loss + top_bottom_edge_loss + bbox_fit_loss + contain_loss + depth_safety_loss + center_depth_safety_loss + ego_box_safety_loss + initial_depth_prior_loss + size_ratio_prior_loss + class_size_prior_loss + height_depth_prior_loss + ground_loss + temporal_acc_loss + temporal_vertical_acc_loss + temporal_log_depth_loss + oversize_final_loss
    total = total_per_frame.sum()
    if reduce:
        return {
            "total": total,
            "edge": edge_loss.sum(),
            "top_bottom_edges": top_bottom_edge_loss.sum(),
            "bbox_fit": bbox_fit_loss.sum(),
            "mask_contain": contain_loss.sum(),
            "depth_safety": depth_safety_loss.sum(),
            "center_depth_safety": center_depth_safety_loss.sum(),
            "ego_box_safety": ego_box_safety_loss.sum(),
            "initial_depth_prior": initial_depth_prior_loss.sum(),
            "size_ratio_prior": size_ratio_prior_loss.sum(),
            "class_size_prior": class_size_prior_loss.sum(),
            "height_depth_prior": height_depth_prior_loss.sum(),
            "ground": ground_loss.sum(),
            "temporal_acceleration": temporal_acc_loss.sum(),
            "temporal_vertical_acceleration": temporal_vertical_acc_loss.sum(),
            "temporal_log_depth_acceleration": temporal_log_depth_loss.sum(),
            "mask_oversize": oversize_final_loss.sum(),
        }
    return {
        "total_per_frame": total_per_frame.detach(),
        "edge_per_frame": edge_loss.detach(),
        "top_bottom_edges_per_frame": top_bottom_edge_loss.detach(),
        "bbox_fit_per_frame": bbox_fit_loss.detach(),
        "mask_contain_per_frame": contain_loss.detach(),
        "depth_safety_per_frame": depth_safety_loss.detach(),
        "center_depth_safety_per_frame": center_depth_safety_loss.detach(),
        "ego_box_safety_per_frame": ego_box_safety_loss.detach(),
        **{key: value.detach() for key, value in ego_box_safety_diag.items()},
        "initial_depth_prior_per_frame": initial_depth_prior_loss.detach(),
        **{key: value.detach() for key, value in initial_depth_prior_diag.items()},
        "size_ratio_prior_per_frame": size_ratio_prior_loss.detach(),
        "class_size_prior_per_frame": class_size_prior_loss.detach(),
        **{key: value.detach() for key, value in class_size_diag.items()},
        "height_depth_prior_per_frame": height_depth_prior_loss.detach(),
        **{key: value.detach() for key, value in height_depth_diag.items()},
        "ground_per_frame": ground_loss.detach(),
        **{key: value.detach() for key, value in ground_diag.items()},
        "temporal_acceleration_per_frame": temporal_acc_loss.detach(),
        "temporal_vertical_acceleration_per_frame": temporal_vertical_acc_loss.detach(),
        "temporal_log_depth_acceleration_per_frame": temporal_log_depth_loss.detach(),
        "mask_oversize_per_frame": oversize_final_loss.detach(),
        "mask_oversize_unweighted_per_frame": oversize_unweighted_loss.detach() * has_mask_float.detach(),
        "mask_oversize_weighted_per_frame": oversize_weighted_loss.detach(),
        "mask_oversize_excess": oversize.detach() * has_mask_float.detach(),
        "mask_area_ratio": ratio.detach(),
        "mask_area": tensors["mask_area"].detach(),
        "pred_area": pred_oversize_area.detach(),
        "pred_clipped_area": pred_clipped_area.detach(),
        "pred_full_area": pred_full_area.detach(),
        "pred_bbox_area": pred_bbox_area.detach(),
        "mask_oversize_weight": torch.full_like(oversize_final_loss, oversize_weight).detach(),
        "mask_oversize_max_area_ratio": torch.full_like(oversize_final_loss, oversize_max_area_ratio).detach(),
        "mask_point_count": tensors["mask_point_count"].detach(),
        "pred_bbox": pred_bbox.detach(),
        "geometry_uses_clipped_projection": tensors["truncated_any"].to(dtype=pred_bbox.dtype).detach(),
        "pred_left": pred_left.detach(),
        "pred_right": pred_right.detach(),
        "pred_top": pred_bbox[:, 1].detach(),
        "pred_bottom": pred_bbox[:, 3].detach(),
        "corners_px": px.detach(),
        "loss_corners_px": px_geometry_loss.detach(),
        "clipped_corners_px": px_clipped_loss.detach(),
        "center_px": center_px.detach(),
        "size_weight": near_mult.detach(),
    }


def observation_size_for_loss(config: dict[str, Any], size: torch.Tensor, tensors: dict[str, Any], protect_size: bool) -> torch.Tensor:
    if not protect_size:
        return size
    cfg = config["observations"].get("size_update", {})
    if str(cfg.get("mode", "all_frames")) != "reliable_mask_untruncated_only":
        return size
    reliable = tensors["has_mask"]
    if bool(cfg.get("exclude_truncated", True)):
        reliable = reliable & (~tensors["truncated_any"])
    if not bool(reliable.any()) and bool(cfg.get("fallback_to_mask_frames_when_no_reliable", False)):
        fallback = tensors["has_mask"]
        fallback_scale = float(cfg.get("fallback_size_gradient_scale", 0.25))
        if bool(fallback.any()) and fallback_scale > 0:
            fallback_f = fallback.to(dtype=size.dtype)[:, None]
            scaled_size = size.detach()[None, :] + fallback_scale * (size[None, :] - size.detach()[None, :])
            return fallback_f * scaled_size + (1.0 - fallback_f) * size.detach()[None, :]
    reliable_f = reliable.to(dtype=size.dtype)[:, None]
    return reliable_f * size[None, :] + (1.0 - reliable_f) * size.detach()[None, :]


def compute_depth_safety_loss(config: dict[str, Any], corners_cam: torch.Tensor) -> torch.Tensor:
    cfg = config["observations"].get("depth_safety", {})
    if not cfg.get("enabled", False):
        return corners_cam.new_zeros((corners_cam.shape[0],))
    weight = float(cfg.get("weight", 50.0))
    mode = str(cfg.get("mode", "min_corner_z_depth"))
    if mode == "min_corner_distance_to_camera":
        min_distance = float(cfg.get("min_corner_distance_m", cfg.get("min_corner_depth_m", 0.2)))
        corner_distance = torch.linalg.norm(corners_cam, dim=-1)
        violation = torch.relu(min_distance - corner_distance)
    else:
        min_depth = float(cfg.get("min_corner_depth_m", 0.75))
        violation = torch.relu(min_depth - corners_cam[..., 2])
    return violation.square().mean(dim=1) * weight


def compute_center_depth_safety_loss(config: dict[str, Any], centers: torch.Tensor, tensors: dict[str, Any]) -> torch.Tensor:
    cfg = config["observations"].get("center_depth_safety", {})
    if not cfg.get("enabled", False):
        return centers.new_zeros((centers.shape[0],))
    weight = float(cfg.get("weight", 0.0))
    if weight <= 0:
        return centers.new_zeros((centers.shape[0],))
    min_depth = tensors["min_center_depth"].to(dtype=centers.dtype, device=centers.device).clamp_min(0.0)
    violation = torch.relu(min_depth - centers[:, 2])
    return violation.square() * weight


def compute_ego_box_safety_loss(
    config: dict[str, Any],
    centers: torch.Tensor,
    size: torch.Tensor,
    tensors: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = config["observations"].get("ego_box_safety", {})
    n = centers.shape[0]
    zeros = centers.new_zeros((n,))
    if not cfg.get("enabled", False):
        return zeros, {
            "ego_box_safety_min_clearance_m": zeros,
            "ego_box_safety_penetration_m": zeros,
            "ego_box_safety_active": zeros,
        }
    weight = float(cfg.get("weight", 500.0))
    if weight <= 0:
        return zeros, {
            "ego_box_safety_min_clearance_m": zeros,
            "ego_box_safety_penetration_m": zeros,
            "ego_box_safety_active": zeros,
        }
    ego_size = centers.new_tensor(cfg.get("ego_size", [4.8, 2.0, 1.6])).clamp_min(1.0e-4)
    ego_center = centers.new_tensor(cfg.get("ego_center", [0.0, 0.0, 0.8]))
    margin = float(cfg.get("margin_m", 0.2))
    # Transform optimized object center into current ego/car-center frame.
    center_ego = torch.einsum("nij,nj->ni", tensors["camera_to_ego_rot"], centers) + tensors["camera_to_ego_trans"]
    # The object box and ego box share the rear-reference yaw axes. Work in
    # those axes so length/width/height are never inferred from the longest edge.
    axes_ego = torch.einsum("nij,nkj->nki", tensors["camera_to_ego_rot"], tensors["box_axes_cam"])
    axes_ego = axes_ego / torch.linalg.norm(axes_ego, dim=-1, keepdim=True).clamp_min(1.0e-6)
    rel = center_ego - ego_center[None, :]
    axis_distance = torch.abs(torch.einsum("nj,nkj->nk", rel, axes_ego))
    half_extent = 0.5 * (size + ego_size[None, :]) + margin
    clearance = axis_distance - half_extent
    inside_inflated = (clearance.detach() < 0.0).all(dim=1)
    penetration = torch.relu(-clearance).amin(dim=1)
    loss = penetration.square() * weight * inside_inflated.to(dtype=centers.dtype)
    return loss, {
        "ego_box_safety_min_clearance_m": clearance.min(dim=1).values,
        "ego_box_safety_penetration_m": penetration * inside_inflated.to(dtype=centers.dtype),
        "ego_box_safety_active": inside_inflated.to(dtype=centers.dtype),
    }


def compute_initial_depth_prior_loss(
    config: dict[str, Any],
    centers: torch.Tensor,
    tensors: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = config["observations"].get("initial_depth_prior", {})
    n = centers.shape[0]
    zeros = centers.new_zeros((n,))
    if not cfg.get("enabled", False):
        return zeros, {
            "initial_depth_prior_target_z": zeros,
            "initial_depth_prior_log_deviation": zeros,
            "initial_depth_prior_active": zeros,
        }
    weight = float(cfg.get("weight", 0.0))
    if weight <= 0:
        return zeros, {
            "initial_depth_prior_target_z": zeros,
            "initial_depth_prior_log_deviation": zeros,
            "initial_depth_prior_active": zeros,
        }

    target_z = tensors["initial_center_cam"][:, 2].to(dtype=centers.dtype, device=centers.device).clamp_min(1.0e-4)
    z = centers[:, 2].clamp_min(1.0e-4)
    log_dev = torch.abs(torch.log(z / target_z))
    active_mode = str(cfg.get("active_on", "truncated_any"))
    if active_mode == "truncated_horizontal":
        active_bool = tensors["truncated_left"] | tensors["truncated_right"]
    elif active_mode == "truncated_vertical":
        active_bool = tensors["truncated_top"] | tensors["truncated_bottom"]
    elif active_mode == "all":
        active_bool = torch.ones((n,), dtype=torch.bool, device=centers.device)
    else:
        active_bool = tensors["truncated_any"]
    max_initial_depth = float(cfg.get("max_initial_depth_m", 0.0) or 0.0)
    if max_initial_depth > 0:
        active_bool = active_bool & (target_z <= max_initial_depth)
    min_initial_depth = float(cfg.get("min_initial_depth_m", 0.0) or 0.0)
    if min_initial_depth > 0:
        active_bool = active_bool & (target_z >= min_initial_depth)
    active = active_bool.to(dtype=centers.dtype)
    max_log_dev = float(cfg.get("max_log_depth_deviation", 0.15))
    if str(cfg.get("mode", "log_hinge")) == "log_l2":
        excess = log_dev
    else:
        excess = torch.relu(log_dev - max_log_dev)
    loss = excess.square() * weight * active
    return loss, {
        "initial_depth_prior_target_z": target_z,
        "initial_depth_prior_log_deviation": log_dev,
        "initial_depth_prior_active": active,
    }


def compute_size_ratio_prior_loss(config: dict[str, Any], size: torch.Tensor, tensors: dict[str, Any], frame_count: int) -> torch.Tensor:
    cfg = config["observations"].get("size_ratio_prior", {})
    if not cfg.get("enabled", False):
        return size.new_zeros((frame_count,))
    weight = float(cfg.get("weight", 0.0))
    if weight <= 0:
        return size.new_zeros((frame_count,))
    target = tensors["size_ratio_target"].to(dtype=size.dtype, device=size.device).clamp_min(1.0e-4)
    size_safe = size.clamp_min(1.0e-4)
    log_size = torch.log(size_safe)
    log_target = torch.log(target)
    if str(cfg.get("mode", "log_shape_ratio")) == "log_shape_ratio":
        log_size = log_size - log_size.mean()
        log_target = log_target - log_target.mean()
    loss = (log_size - log_target).square().mean() * weight
    return loss.expand(frame_count) / max(frame_count, 1)


def compute_class_size_prior_loss(
    config: dict[str, Any],
    size: torch.Tensor,
    tensors: dict[str, Any],
    frame_count: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = config["observations"].get("class_size_prior", {})
    zeros = size.new_zeros((frame_count,))
    diag_zeros = size.new_zeros((frame_count,))
    if not cfg.get("enabled", False):
        return zeros, {
            "class_size_prior_log_deviation_max": diag_zeros,
            "class_size_prior_excess_max": diag_zeros,
            "class_size_prior_target_length": diag_zeros,
            "class_size_prior_target_width": diag_zeros,
            "class_size_prior_target_height": diag_zeros,
        }
    weight = float(cfg.get("weight", 0.0))
    if weight <= 0:
        return zeros, {
            "class_size_prior_log_deviation_max": diag_zeros,
            "class_size_prior_excess_max": diag_zeros,
            "class_size_prior_target_length": diag_zeros,
            "class_size_prior_target_width": diag_zeros,
            "class_size_prior_target_height": diag_zeros,
        }
    target = tensors["class_size_target"].to(dtype=size.dtype, device=size.device).clamp_min(1.0e-4)
    size_safe = size.clamp_min(1.0e-4)
    log_deviation = torch.abs(torch.log(size_safe / target))
    max_dev_cfg = cfg.get("max_log_deviation", 0.35)
    if isinstance(max_dev_cfg, (list, tuple)):
        max_dev = torch.tensor(max_dev_cfg, dtype=size.dtype, device=size.device)
        if max_dev.numel() != 3:
            max_dev = torch.full((3,), float(max_dev.flatten()[0]), dtype=size.dtype, device=size.device)
    else:
        max_dev = torch.full((3,), float(max_dev_cfg), dtype=size.dtype, device=size.device)
    max_dev = max_dev.clamp_min(0.0)
    mode = str(cfg.get("mode", "log_hinge"))
    if mode == "log_l2":
        excess = log_deviation
    else:
        excess = torch.relu(log_deviation - max_dev)
    loss = excess.square().mean() * weight
    target_frame = target.expand(frame_count, 3)
    return loss.expand(frame_count) / max(frame_count, 1), {
        "class_size_prior_log_deviation_max": log_deviation.max().expand(frame_count),
        "class_size_prior_excess_max": excess.max().expand(frame_count),
        "class_size_prior_target_length": target_frame[:, 0],
        "class_size_prior_target_width": target_frame[:, 1],
        "class_size_prior_target_height": target_frame[:, 2],
    }


def compute_height_depth_prior_loss(
    config: dict[str, Any],
    centers: torch.Tensor,
    size: torch.Tensor,
    tensors: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = config["observations"].get("height_depth_prior", {})
    n = centers.shape[0]
    zeros = centers.new_zeros((n,))
    if not cfg.get("enabled", False):
        return zeros, {
            "height_depth_prior_z": zeros,
            "height_depth_prior_log_deviation": zeros,
            "height_depth_prior_excess": zeros,
            "height_depth_prior_height_m": zeros,
            "height_depth_prior_active": zeros,
        }
    weight = float(cfg.get("weight", 0.0))
    if weight <= 0:
        return zeros, {
            "height_depth_prior_z": zeros,
            "height_depth_prior_log_deviation": zeros,
            "height_depth_prior_excess": zeros,
            "height_depth_prior_height_m": zeros,
            "height_depth_prior_active": zeros,
        }

    box = tensors["box2d"]
    bbox_h = (box[:, 3] - box[:, 1]).clamp_min(1.0)
    fy = tensors["intrinsic"][:, 1, 1].clamp_min(1.0e-6)
    target_size = tensors["class_size_target"].to(dtype=centers.dtype, device=centers.device).clamp_min(1.0e-4)
    if str(cfg.get("height_source", "class_init")) == "optimized_size":
        height_m = size[2].clamp_min(1.0e-4).expand(n)
    else:
        height_m = target_size[2].expand(n)
    z_prior = (fy * height_m / bbox_h).clamp_min(1.0e-4)
    z = centers[:, 2].clamp_min(1.0e-4)
    log_dev = torch.abs(torch.log(z / z_prior))
    max_log_dev = float(cfg.get("max_log_depth_deviation", 0.25))
    if str(cfg.get("mode", "log_hinge")) == "log_l2":
        excess = log_dev
    else:
        excess = torch.relu(log_dev - max_log_dev)
    active = torch.ones((n,), dtype=centers.dtype, device=centers.device)
    if bool(cfg.get("exclude_truncated_vertical", True)):
        vertical_truncated = tensors["truncated_top"] | tensors["truncated_bottom"]
        active = active * (~vertical_truncated).to(dtype=centers.dtype)
    far_start = float(cfg.get("far_start_m", 0.0) or 0.0)
    if far_start > 0:
        active = active * (torch.linalg.norm(centers.detach(), dim=1) >= far_start).to(dtype=centers.dtype)
    loss = excess.square() * weight * active
    return loss, {
        "height_depth_prior_z": z_prior,
        "height_depth_prior_log_deviation": log_dev,
        "height_depth_prior_excess": excess * active,
        "height_depth_prior_height_m": height_m,
        "height_depth_prior_active": active,
    }


def gate_distance_for_loss(
    cfg: dict[str, Any],
    centers: torch.Tensor,
    tensors: dict[str, Any],
) -> torch.Tensor:
    gate_by = str(cfg.get("gate_by", "current_distance"))
    if gate_by in {"initial_distance", "init_distance", "observed_distance"}:
        return torch.linalg.norm(tensors["initial_center_cam"].detach(), dim=1)
    return torch.linalg.norm(centers.detach(), dim=1)


def compute_top_bottom_edge_loss(
    config: dict[str, Any],
    pred_bbox: torch.Tensor,
    box: torch.Tensor,
    centers: torch.Tensor,
    tensors: dict[str, Any],
) -> torch.Tensor:
    cfg = config["observations"].get("top_bottom_edges", {})
    if not cfg.get("enabled", False):
        return torch.zeros((box.shape[0],), dtype=box.dtype, device=box.device)
    distance = gate_distance_for_loss(cfg, centers, tensors)
    far_start = float(cfg.get("far_start_m", 10.0))
    gate = (distance > far_start).to(dtype=box.dtype)
    if bool(cfg.get("activate_untruncated", False)):
        # For fully visible objects, especially small motorcycles/e-bikes, the
        # projected cuboid should also fit the 2D box vertically at close range.
        # Keep the distance gate for vertically truncated frames, where one-sided
        # constraints are safer.
        vertical_untruncated = ~(tensors["truncated_top"] | tensors["truncated_bottom"])
        gate = torch.maximum(gate, vertical_untruncated.to(dtype=box.dtype))
    box_h = (box[:, 3] - box[:, 1]).clamp_min(1.0)
    top_eq = (pred_bbox[:, 1] - box[:, 1]) / box_h
    bottom_eq = (pred_bbox[:, 3] - box[:, 3]) / box_h
    if bool(cfg.get("one_sided_vertical_truncation", False)):
        # Image y points down.  If the top is truncated, the real cuboid top may
        # be above/outside the visible 2D bbox, so only penalize when projected
        # top is too low.  For bottom truncation, only penalize when projected
        # bottom is too high.
        top_one_sided = torch.relu(pred_bbox[:, 1] - box[:, 1]) / box_h
        bottom_one_sided = torch.relu(box[:, 3] - pred_bbox[:, 3]) / box_h
        top = torch.where(tensors["truncated_top"], top_one_sided, top_eq)
        bottom = torch.where(tensors["truncated_bottom"], bottom_one_sided, bottom_eq)
    else:
        top = top_eq
        bottom = bottom_eq
    return (top.square() + bottom.square()) * float(cfg.get("weight", 1.0)) * gate


def compute_ground_plane_loss(config: dict[str, Any], corners_cam: torch.Tensor, tensors: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = config["observations"].get("ground_plane", {})
    n = corners_cam.shape[0]
    zeros = corners_cam.new_zeros((n,))
    normal_zeros = corners_cam.new_zeros((n, 3))
    if not cfg.get("enabled", False):
        return zeros, {
            "ground_distance_mean": zeros,
            "ground_distance_abs_max": zeros,
            "ground_camera_height_m": zeros,
            "ground_loss_raw": zeros,
            "ground_distance_for_decay_m": zeros,
            "ground_distance_decay_multiplier": zeros,
            "ground_normal_cam": normal_zeros,
        }
    camera_height = ground_camera_height_for_view(cfg, str(tensors.get("view_name", "")))
    weight = float(cfg.get("weight", 5.0))
    ego_up = corners_cam.new_tensor([0.0, 0.0, 1.0])
    ground_normal_cam = torch.einsum("nij,j->ni", tensors["camera_to_ego_rot"].transpose(1, 2), ego_up)
    ground_normal_cam = ground_normal_cam / torch.linalg.norm(ground_normal_cam, dim=1, keepdim=True).clamp_min(1.0e-6)
    signed_height = torch.einsum("nkj,nj->nk", corners_cam, ground_normal_cam) + camera_height
    bottom_idx = torch.topk(signed_height.detach(), k=4, dim=1, largest=False).indices
    bottom_dist = torch.gather(signed_height, 1, bottom_idx)
    raw_loss = bottom_dist.square().mean(dim=1) * weight
    decay_distance, decay_multiplier = ground_distance_decay_multiplier(cfg, corners_cam, tensors)
    loss = raw_loss * decay_multiplier
    return loss, {
        "ground_distance_mean": bottom_dist.mean(dim=1),
        "ground_distance_abs_max": bottom_dist.abs().max(dim=1).values,
        "ground_camera_height_m": torch.full((n,), camera_height, dtype=corners_cam.dtype, device=corners_cam.device),
        "ground_loss_raw": raw_loss,
        "ground_distance_for_decay_m": decay_distance,
        "ground_distance_decay_multiplier": decay_multiplier,
        "ground_normal_cam": ground_normal_cam,
    }


def ground_camera_height_for_view(cfg: dict[str, Any], view: str) -> float:
    by_view = cfg.get("camera_height_m_by_view", {})
    if isinstance(by_view, dict) and view in by_view:
        return float(by_view[view])
    return float(cfg.get("camera_height_m", 0.5))


def ground_distance_decay_multiplier(
    cfg: dict[str, Any],
    corners_cam: torch.Tensor,
    tensors: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    n = corners_cam.shape[0]
    zeros = corners_cam.new_zeros((n,))
    ones = corners_cam.new_ones((n,))
    decay_cfg = cfg.get("distance_decay", {})
    if not bool(decay_cfg.get("enabled", False)):
        return zeros, ones
    source = str(decay_cfg.get("distance_source", "initial_camera_distance"))
    if source in {"current_camera_distance", "current_center_distance"}:
        center = corners_cam.mean(dim=1)
        distance = torch.linalg.norm(center.detach(), dim=1)
    elif source in {"initial_z", "initial_camera_z"}:
        distance = tensors["initial_center_cam"][:, 2].detach().abs()
    else:
        distance = tensors["initial_camera_distance"].detach()
    near_m = float(decay_cfg.get("near_m", 8.0))
    far_m = float(decay_cfg.get("far_m", 35.0))
    min_multiplier = float(decay_cfg.get("min_multiplier", 0.1))
    if far_m <= near_m:
        multiplier = torch.where(distance <= near_m, ones, corners_cam.new_full((n,), min_multiplier))
        return distance, multiplier
    t = ((distance - near_m) / (far_m - near_m)).clamp(0.0, 1.0)
    mode = str(decay_cfg.get("mode", "smoothstep"))
    if mode == "linear":
        smooth = t
    else:
        smooth = t * t * (3.0 - 2.0 * t)
    multiplier = 1.0 - smooth * (1.0 - min_multiplier)
    return distance, multiplier.clamp_min(0.0)


def compute_temporal_smoothness_loss(
    config: dict[str, Any],
    centers: torch.Tensor,
    tensors: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = config["observations"].get("temporal_smoothness", {})
    n = centers.shape[0]
    zeros = centers.new_zeros((n,))
    if not cfg.get("enabled", False) or n < 3:
        return zeros, zeros, zeros

    frames = tensors["frame_index"]
    gap = float(cfg.get("max_gap_frames", 3))
    prev_gap_ok = (frames[1:-1] - frames[:-2]) <= gap
    next_gap_ok = (frames[2:] - frames[1:-1]) <= gap
    valid = (prev_gap_ok & next_gap_ok).to(dtype=centers.dtype)
    if not bool(valid.detach().bool().any()):
        return zeros, zeros, zeros

    acc_cfg = cfg.get("acceleration", {})
    log_depth_cfg = cfg.get("log_depth_acceleration", {})
    acc_loss = zeros.clone()
    vertical_acc_loss = zeros.clone()
    log_depth_loss = zeros.clone()

    if bool(acc_cfg.get("enabled", True)):
        # Convert every optimized camera-frame center to world, then project to
        # the fixed first-frame rear/ego reference axes.  axes rows are:
        # [forward/length, lateral/width, vertical/up].  Planar and vertical
        # acceleration are reported separately so jump sources are easy to see.
        center_world = torch.einsum("nij,nj->ni", tensors["camera_to_world_rot"], centers) + tensors["camera_to_world_trans"]
        origin = center_world[0].detach()
        ref_axes = tensors["reference_axes_world"][0].detach()
        rel_world = center_world - origin[None, :]
        forward = torch.einsum("nj,j->n", rel_world, ref_axes[0])
        lateral = torch.einsum("nj,j->n", rel_world, ref_axes[1])
        vertical = torch.einsum("nj,j->n", rel_world, ref_axes[2])
        lateral_acc = lateral[2:] - 2.0 * lateral[1:-1] + lateral[:-2]
        forward_acc = forward[2:] - 2.0 * forward[1:-1] + forward[:-2]
        vertical_acc = vertical[2:] - 2.0 * vertical[1:-1] + vertical[:-2]
        delta_m = float(acc_cfg.get("robust_delta_m", cfg.get("robust_delta_m", 1.0)))
        lateral_weight = float(acc_cfg.get("lateral_weight", 1.0))
        forward_weight = float(acc_cfg.get("forward_weight", 0.5))
        vertical_weight = float(acc_cfg.get("vertical_weight", 0.0))
        middle_loss = (
            lateral_weight * charbonnier(lateral_acc, delta_m)
            + forward_weight * charbonnier(forward_acc, delta_m)
        ) * valid
        middle_vertical_loss = vertical_weight * charbonnier(vertical_acc, delta_m) * valid
        acc_loss = acc_loss.index_add(0, torch.arange(1, n - 1, device=centers.device), middle_loss)
        vertical_acc_loss = vertical_acc_loss.index_add(0, torch.arange(1, n - 1, device=centers.device), middle_vertical_loss)

    if bool(log_depth_cfg.get("enabled", True)):
        z = centers[:, 2].clamp_min(1.0e-4)
        log_z = torch.log(z)
        log_acc = log_z[2:] - 2.0 * log_z[1:-1] + log_z[:-2]
        delta_log = float(log_depth_cfg.get("robust_delta_log_depth", cfg.get("robust_delta_log_depth", 0.08)))
        weight = float(log_depth_cfg.get("weight", 1.0))
        middle_loss = weight * charbonnier(log_acc, delta_log) * valid
        log_depth_loss = log_depth_loss.index_add(0, torch.arange(1, n - 1, device=centers.device), middle_loss)

    return acc_loss, vertical_acc_loss, log_depth_loss


def charbonnier(value: torch.Tensor, delta: float) -> torch.Tensor:
    delta = max(float(delta), 1.0e-6)
    return (delta * delta) * (torch.sqrt((value / delta).square() + 1.0) - 1.0)


def compute_bbox_fit_loss(
    config: dict[str, Any],
    pred_bbox: torch.Tensor,
    box: torch.Tensor,
    centers: torch.Tensor,
    tensors: dict[str, Any],
) -> torch.Tensor:
    cfg = config["observations"].get("bbox_fit", {})
    if not cfg.get("enabled", False):
        return torch.zeros((box.shape[0],), dtype=box.dtype, device=box.device)
    box_w = (box[:, 2] - box[:, 0]).clamp_min(1.0)
    box_h = (box[:, 3] - box[:, 1]).clamp_min(1.0)
    left = torch.where(tensors["truncated_left"], torch.relu(pred_bbox[:, 0] - box[:, 0]) / box_w, (pred_bbox[:, 0] - box[:, 0]) / box_w)
    right = torch.where(tensors["truncated_right"], torch.relu(box[:, 2] - pred_bbox[:, 2]) / box_w, (pred_bbox[:, 2] - box[:, 2]) / box_w)
    if str(cfg.get("vertical_truncation_mode", "equality")) == "one_sided":
        top_eq = (pred_bbox[:, 1] - box[:, 1]) / box_h
        bottom_eq = (pred_bbox[:, 3] - box[:, 3]) / box_h
        top_one_sided = torch.relu(pred_bbox[:, 1] - box[:, 1]) / box_h
        bottom_one_sided = torch.relu(box[:, 3] - pred_bbox[:, 3]) / box_h
        top = torch.where(tensors["truncated_top"], top_one_sided, top_eq)
        bottom = torch.where(tensors["truncated_bottom"], bottom_one_sided, bottom_eq)
    else:
        top = (pred_bbox[:, 1] - box[:, 1]) / box_h
        bottom = (pred_bbox[:, 3] - box[:, 3]) / box_h
    base = left.square() + right.square() + top.square() + bottom.square()
    distance = gate_distance_for_loss(cfg, centers, tensors)
    far_start = float(cfg.get("far_start_m", 35.0))
    far_gate = (distance >= far_start).to(dtype=box.dtype)
    far_mult = far_gate * float(cfg.get("far_multiplier", 3.0))
    rear_mult = torch.where(
        tensors["is_rear"],
        torch.full_like(far_mult, float(cfg.get("rear_multiplier", 3.0))),
        torch.ones_like(far_mult),
    )
    return base * float(cfg.get("weight", 1.0)) * far_mult * rear_mult


def clip_projected_points_for_loss(
    config: dict[str, Any],
    corners_px: torch.Tensor,
    tensors: dict[str, Any],
) -> torch.Tensor:
    cfg = config["observations"].get("projection", {})
    if not bool(cfg.get("clip_loss_to_image", False)):
        return corners_px
    image_size = tensors["image_size"].to(dtype=corners_px.dtype, device=corners_px.device)
    width = image_size[:, 0].clamp_min(1.0)
    height = image_size[:, 1].clamp_min(1.0)
    x = torch.minimum(corners_px[..., 0].clamp(min=0.0), width[:, None])
    y = torch.minimum(corners_px[..., 1].clamp(min=0.0), height[:, None])
    return torch.stack([x, y], dim=-1)


def compute_mask_contain_loss(
    config: dict[str, Any],
    corners_px: torch.Tensor,
    pred_bbox: torch.Tensor,
    box: torch.Tensor,
    tensors: dict[str, Any],
) -> torch.Tensor:
    cfg = config["observations"].get("mask", {})
    weight = float(cfg.get("contain_weight", 0.25))
    diag = torch.sqrt((box[:, 2] - box[:, 0]).square() + (box[:, 3] - box[:, 1]).square()).clamp_min(1.0)
    use_points = bool(cfg.get("use_foreground_points", False))
    fallback_to_bbox = bool(cfg.get("fallback_to_mask_bbox", True))
    point_losses = pred_bbox.new_zeros((box.shape[0],))
    has_points = pred_bbox.new_zeros((box.shape[0],), dtype=torch.bool)
    points = tensors["mask_points"]
    valid = tensors["mask_points_valid"]
    if use_points and points.shape[1] > 0:
        px = points[:, :, 0]
        py = points[:, :, 1]
        outside = projected_cuboid_outside_distance(corners_px, points) / diag[:, None]
        outside_sq = outside.square() * valid.to(dtype=pred_bbox.dtype)
        counts = valid.sum(dim=1)
        has_points = counts > 0
        reduction = str(cfg.get("contain_reduction", "mean"))
        if reduction == "max":
            point_losses = outside_sq.masked_fill(~valid, 0.0).max(dim=1).values
        else:
            point_losses = outside_sq.sum(dim=1) / counts.clamp_min(1).to(dtype=pred_bbox.dtype)
    if fallback_to_bbox:
        mask_bbox = tensors["mask_bbox"]
        outside_bbox = (
            torch.relu(pred_bbox[:, 0] - mask_bbox[:, 0])
            + torch.relu(pred_bbox[:, 1] - mask_bbox[:, 1])
            + torch.relu(mask_bbox[:, 2] - pred_bbox[:, 2])
            + torch.relu(mask_bbox[:, 3] - pred_bbox[:, 3])
        ) / diag
        bbox_losses = outside_bbox.square()
    else:
        bbox_losses = pred_bbox.new_zeros((box.shape[0],))
    return torch.where(has_points, point_losses, bbox_losses) * weight


def projected_cuboid_outside_distance(corners_px: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Distance from 2D points to the projected 3D cuboid silhouette.

    The projected silhouette of a convex 3D box is the convex hull of its eight
    projected corners. Hull edge topology is selected from detached coordinates;
    signed distances are computed from live tensors so selected edges receive
    gradients.
    """
    hull_edge = projected_cuboid_hull_edges(corners_px)
    p_i = corners_px[:, :, None, :]  # [N, 8, 1, 2]
    p_j = corners_px[:, None, :, :]  # [N, 1, 8, 2]
    edge = p_j - p_i
    edge_len = torch.linalg.norm(edge, dim=-1).clamp_min(1.0)
    p = points[:, None, None, :, :]  # [N, 1, 1, M, 2]
    rel = p - p_i[:, :, :, None, :]
    signed_dist = (edge[:, :, :, None, 0] * rel[..., 1] - edge[:, :, :, None, 1] * rel[..., 0]) / edge_len[:, :, :, None]
    violation = torch.relu(-signed_dist) * hull_edge[:, :, :, None].to(dtype=corners_px.dtype)
    return violation.amax(dim=(1, 2))


def projected_cuboid_visible_area(
    corners_px: torch.Tensor,
    image_size: torch.Tensor | None = None,
) -> torch.Tensor:
    """Area of the projected cuboid silhouette.

    For a convex 3D box, the visible 2D silhouette is the convex hull of its
    eight projected corners. This vectorized half-plane edge test selects hull
    boundary edges on-device, then applies the shoelace formula over those
    directed edges. The binary edge topology is detached; selected corner
    coordinates still receive gradients.
    """
    if image_size is not None:
        width = image_size[:, 0].clamp_min(1.0)
        height = image_size[:, 1].clamp_min(1.0)
        clipped_x = torch.minimum(corners_px[..., 0].clamp(min=0.0), width[:, None])
        clipped_y = torch.minimum(corners_px[..., 1].clamp(min=0.0), height[:, None])
        clipped = torch.stack([clipped_x, clipped_y], dim=-1)
    else:
        clipped = corners_px
    center = clipped.mean(dim=1, keepdim=True)
    rel = clipped - center
    angles = torch.atan2(rel[..., 1], rel[..., 0])
    order = torch.argsort(angles.detach(), dim=1)
    poly = torch.gather(clipped, 1, order[..., None].expand(-1, -1, 2))
    x = poly[..., 0]
    y = poly[..., 1]
    area = 0.5 * torch.abs(torch.sum(x * torch.roll(y, shifts=-1, dims=1) - y * torch.roll(x, shifts=-1, dims=1), dim=1))
    if image_size is not None:
        bbox_area = (clipped[..., 0].max(dim=1).values - clipped[..., 0].min(dim=1).values).clamp_min(0.0) * (
            clipped[..., 1].max(dim=1).values - clipped[..., 1].min(dim=1).values
        ).clamp_min(0.0)
        area = torch.minimum(area, bbox_area)
    return area


def projected_cuboid_hull_edges(corners_px: torch.Tensor) -> torch.Tensor:
    p_i = corners_px[:, :, None, :]  # [N, 8, 1, 2]
    p_j = corners_px[:, None, :, :]  # [N, 1, 8, 2]
    p_k = corners_px[:, None, None, :, :]  # [N, 1, 1, 8, 2]
    edge = p_j[:, :, :, None, :] - p_i[:, :, :, None, :]
    rel = p_k - p_i[:, :, :, None, :]
    cross_to_points = edge[..., 0] * rel[..., 1] - edge[..., 1] * rel[..., 0]
    valid_pair = ~torch.eye(corners_px.shape[1], dtype=torch.bool, device=corners_px.device)[None, :, :]
    return (cross_to_points.detach() >= -1.0e-4).all(dim=-1) & valid_pair


def near_size_multipliers(config: dict[str, Any], centers: torch.Tensor, tensors: dict[str, Any]) -> torch.Tensor:
    cfg = config["weighting"]["near_observation_size_influence"]
    if not cfg.get("enabled", False):
        return torch.ones((centers.shape[0],), dtype=centers.dtype, device=centers.device)
    distance = torch.linalg.norm(centers.detach(), dim=1)
    near_threshold = float(cfg.get("near_threshold", 15.0))
    far_threshold = max(float(cfg.get("far_threshold", 60.0)), near_threshold + 1.0e-6)
    near_multiplier = float(cfg.get("near_multiplier", 2.0))
    minimum = float(cfg.get("minimum_weight", 1.0))
    alpha = ((far_threshold - distance) / (far_threshold - near_threshold)).clamp(0.0, 1.0)
    return minimum + alpha * (near_multiplier - minimum)


def build_result(
    config: dict[str, Any],
    observations: list[Observation],
    centers: torch.Tensor,
    size: torch.Tensor,
    losses: dict[str, torch.Tensor],
    best_loss: float,
    device: torch.device,
    iterations_used: int,
    stop_reason: str,
    solution_kind: str = "best",
) -> TrackResult:
    centers_np = centers.detach().cpu().numpy()
    size_np = size.detach().cpu().numpy()
    pred_bbox = losses["pred_bbox"].detach().cpu().numpy()
    geometry_uses_clipped_projection = losses["geometry_uses_clipped_projection"].detach().cpu().numpy()
    corners_px = losses["corners_px"].detach().cpu().numpy()
    center_px = losses["center_px"].detach().cpu().numpy()
    pred_left = losses["pred_left"].detach().cpu().numpy()
    pred_right = losses["pred_right"].detach().cpu().numpy()
    pred_top = losses["pred_top"].detach().cpu().numpy()
    pred_bottom = losses["pred_bottom"].detach().cpu().numpy()
    total_pf = losses["total_per_frame"].detach().cpu().numpy()
    edge_pf = losses["edge_per_frame"].detach().cpu().numpy()
    top_bottom_edges_pf = losses["top_bottom_edges_per_frame"].detach().cpu().numpy()
    bbox_fit_pf = losses["bbox_fit_per_frame"].detach().cpu().numpy()
    contain_pf = losses["mask_contain_per_frame"].detach().cpu().numpy()
    depth_safety_pf = losses["depth_safety_per_frame"].detach().cpu().numpy()
    center_depth_safety_pf = losses["center_depth_safety_per_frame"].detach().cpu().numpy()
    ego_box_safety_pf = losses["ego_box_safety_per_frame"].detach().cpu().numpy()
    ego_box_safety_min_clearance_m = losses["ego_box_safety_min_clearance_m"].detach().cpu().numpy()
    ego_box_safety_penetration_m = losses["ego_box_safety_penetration_m"].detach().cpu().numpy()
    ego_box_safety_active = losses["ego_box_safety_active"].detach().cpu().numpy()
    initial_depth_prior_pf = losses["initial_depth_prior_per_frame"].detach().cpu().numpy()
    initial_depth_prior_target_z = losses["initial_depth_prior_target_z"].detach().cpu().numpy()
    initial_depth_prior_log_deviation = losses["initial_depth_prior_log_deviation"].detach().cpu().numpy()
    initial_depth_prior_active = losses["initial_depth_prior_active"].detach().cpu().numpy()
    size_ratio_prior_pf = losses["size_ratio_prior_per_frame"].detach().cpu().numpy()
    class_size_prior_pf = losses["class_size_prior_per_frame"].detach().cpu().numpy()
    class_size_prior_log_deviation_max = losses["class_size_prior_log_deviation_max"].detach().cpu().numpy()
    class_size_prior_excess_max = losses["class_size_prior_excess_max"].detach().cpu().numpy()
    class_size_prior_target_length = losses["class_size_prior_target_length"].detach().cpu().numpy()
    class_size_prior_target_width = losses["class_size_prior_target_width"].detach().cpu().numpy()
    class_size_prior_target_height = losses["class_size_prior_target_height"].detach().cpu().numpy()
    height_depth_prior_pf = losses["height_depth_prior_per_frame"].detach().cpu().numpy()
    height_depth_prior_z = losses["height_depth_prior_z"].detach().cpu().numpy()
    height_depth_prior_log_deviation = losses["height_depth_prior_log_deviation"].detach().cpu().numpy()
    height_depth_prior_excess = losses["height_depth_prior_excess"].detach().cpu().numpy()
    height_depth_prior_height_m = losses["height_depth_prior_height_m"].detach().cpu().numpy()
    height_depth_prior_active = losses["height_depth_prior_active"].detach().cpu().numpy()
    ground_pf = losses["ground_per_frame"].detach().cpu().numpy()
    temporal_acc_pf = losses["temporal_acceleration_per_frame"].detach().cpu().numpy()
    temporal_vertical_acc_pf = losses["temporal_vertical_acceleration_per_frame"].detach().cpu().numpy()
    temporal_log_depth_pf = losses["temporal_log_depth_acceleration_per_frame"].detach().cpu().numpy()
    ground_distance_mean = losses["ground_distance_mean"].detach().cpu().numpy()
    ground_distance_abs_max = losses["ground_distance_abs_max"].detach().cpu().numpy()
    ground_camera_height_m = losses["ground_camera_height_m"].detach().cpu().numpy()
    ground_loss_raw = losses["ground_loss_raw"].detach().cpu().numpy()
    ground_distance_for_decay_m = losses["ground_distance_for_decay_m"].detach().cpu().numpy()
    ground_distance_decay_multiplier = losses["ground_distance_decay_multiplier"].detach().cpu().numpy()
    ground_normal_cam = losses["ground_normal_cam"].detach().cpu().numpy()
    oversize_pf = losses["mask_oversize_per_frame"].detach().cpu().numpy()
    oversize_unweighted_pf = losses["mask_oversize_unweighted_per_frame"].detach().cpu().numpy()
    oversize_weighted_pf = losses["mask_oversize_weighted_per_frame"].detach().cpu().numpy()
    oversize_excess = losses["mask_oversize_excess"].detach().cpu().numpy()
    mask_area_ratio = losses["mask_area_ratio"].detach().cpu().numpy()
    mask_area = losses["mask_area"].detach().cpu().numpy()
    pred_area = losses["pred_area"].detach().cpu().numpy()
    pred_clipped_area = losses["pred_clipped_area"].detach().cpu().numpy()
    pred_full_area = losses["pred_full_area"].detach().cpu().numpy()
    pred_bbox_area = losses["pred_bbox_area"].detach().cpu().numpy()
    oversize_weight = losses["mask_oversize_weight"].detach().cpu().numpy()
    oversize_max_area_ratio = losses["mask_oversize_max_area_ratio"].detach().cpu().numpy()
    mask_point_count = losses["mask_point_count"].detach().cpu().numpy()
    size_weight = losses["size_weight"].detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for i, obs in enumerate(observations):
        dominant = dominant_loss(edge_pf[i], top_bottom_edges_pf[i], bbox_fit_pf[i], contain_pf[i], depth_safety_pf[i], center_depth_safety_pf[i], ego_box_safety_pf[i], initial_depth_prior_pf[i], size_ratio_prior_pf[i], class_size_prior_pf[i], height_depth_prior_pf[i], ground_pf[i], temporal_acc_pf[i], temporal_vertical_acc_pf[i], temporal_log_depth_pf[i], oversize_pf[i])
        rows.append(
            {
                "view": obs.view,
                "frame": obs.frame,
                "timestamp": obs.timestamp,
                "track_id": obs.track_id,
                "class": obs.label,
                "cx": float(centers_np[i, 0]),
                "cy": float(centers_np[i, 1]),
                "cz": float(centers_np[i, 2]),
                "center_frame": "camera",
                "length": float(size_np[0]),
                "width": float(size_np[1]),
                "height": float(size_np[2]),
                "size_order": "length,width,height",
                "yaw": float(obs.yaw_fixed),
                "yaw_axis": obs.yaw_source,
                "yaw_optimized": False,
                "upright_box": bool(config.get("constraints", {}).get("upright_box", {}).get("enabled", True)),
                "ground_contact": bool(config.get("constraints", {}).get("ground_contact", {}).get("enabled", True)),
                "solver": "adam",
                "dtype": str(config["solver"].get("dtype", "float32")),
                "device": str(device),
                "iterations_used": iterations_used,
                "stop_reason": stop_reason,
                "optimized": True,
                "solution_kind": solution_kind,
            }
        )
        diagnostics.append(
            {
                "view": obs.view,
                "frame": obs.frame,
                "timestamp": obs.timestamp,
                "track_id": obs.track_id,
                "class": obs.label,
                "loss_total": float(total_pf[i]),
                "loss_edge": float(edge_pf[i]),
                "loss_top_bottom_edges": float(top_bottom_edges_pf[i]),
                "loss_bbox_fit": float(bbox_fit_pf[i]),
                "loss_mask_contain": float(contain_pf[i]),
                "loss_depth_safety": float(depth_safety_pf[i]),
                "loss_center_depth_safety": float(center_depth_safety_pf[i]),
                "loss_ego_box_safety": float(ego_box_safety_pf[i]),
                "ego_box_safety_min_clearance_m": float(ego_box_safety_min_clearance_m[i]),
                "ego_box_safety_penetration_m": float(ego_box_safety_penetration_m[i]),
                "ego_box_safety_active": float(ego_box_safety_active[i]),
                "loss_initial_depth_prior": float(initial_depth_prior_pf[i]),
                "initial_depth_prior_target_z": float(initial_depth_prior_target_z[i]),
                "initial_depth_prior_log_deviation": float(initial_depth_prior_log_deviation[i]),
                "initial_depth_prior_active": float(initial_depth_prior_active[i]),
                "loss_size_ratio_prior": float(size_ratio_prior_pf[i]),
                "loss_class_size_prior": float(class_size_prior_pf[i]),
                "loss_height_depth_prior": float(height_depth_prior_pf[i]),
                "loss_temporal_acceleration": float(temporal_acc_pf[i]),
                "loss_temporal_log_depth_acceleration": float(temporal_log_depth_pf[i]),
                "height_depth_prior_z": float(height_depth_prior_z[i]),
                "height_depth_prior_log_deviation": float(height_depth_prior_log_deviation[i]),
                "height_depth_prior_excess": float(height_depth_prior_excess[i]),
                "height_depth_prior_height_m": float(height_depth_prior_height_m[i]),
                "height_depth_prior_active": float(height_depth_prior_active[i]),
                "class_size_prior_log_deviation_max": float(class_size_prior_log_deviation_max[i]),
                "class_size_prior_excess_max": float(class_size_prior_excess_max[i]),
                "class_size_prior_target_length": float(class_size_prior_target_length[i]),
                "class_size_prior_target_width": float(class_size_prior_target_width[i]),
                "class_size_prior_target_height": float(class_size_prior_target_height[i]),
                "loss_ground": float(ground_pf[i]),
                "loss_ground_raw": float(ground_loss_raw[i]),
                "ground_distance_mean": float(ground_distance_mean[i]),
                "ground_distance_abs_max": float(ground_distance_abs_max[i]),
                "ground_camera_height_m": float(ground_camera_height_m[i]),
                "ground_distance_for_decay_m": float(ground_distance_for_decay_m[i]),
                "ground_distance_decay_multiplier": float(ground_distance_decay_multiplier[i]),
                "ground_normal_cam_x": float(ground_normal_cam[i, 0]),
                "ground_normal_cam_y": float(ground_normal_cam[i, 1]),
                "ground_normal_cam_z": float(ground_normal_cam[i, 2]),
                "loss_temporal_vertical_acceleration": float(temporal_vertical_acc_pf[i]),
                "loss_mask_oversize": float(oversize_pf[i]),
                "loss_mask_oversize_unweighted": float(oversize_unweighted_pf[i]),
                "loss_mask_oversize_weighted": float(oversize_weighted_pf[i]),
                "loss_mask_oversize_final": float(oversize_pf[i]),
                "mask_area": float(mask_area[i]),
                "pred_area": float(pred_area[i]),
                "pred_visible_area": float(pred_clipped_area[i]),
                "pred_clipped_area": float(pred_clipped_area[i]),
                "pred_full_area": float(pred_full_area[i]),
                "pred_bbox_area": float(pred_bbox_area[i]),
                "geometry_uses_clipped_projection": bool(geometry_uses_clipped_projection[i] > 0.5),
                "mask_area_ratio": float(mask_area_ratio[i]),
                "mask_oversize_excess": float(oversize_excess[i]),
                "mask_oversize_weight": float(oversize_weight[i]),
                "mask_oversize_max_area_ratio": float(oversize_max_area_ratio[i]),
                "mask_point_count": int(mask_point_count[i]),
                "size_weight": float(size_weight[i]),
                "dominant_loss": dominant,
                "iterations_used": iterations_used,
                "stop_reason": stop_reason,
                "solution_kind": solution_kind,
                "cx": float(centers_np[i, 0]),
                "cy": float(centers_np[i, 1]),
                "cz": float(centers_np[i, 2]),
                "length": float(size_np[0]),
                "width": float(size_np[1]),
                "height": float(size_np[2]),
                "image": obs.image,
                "mask_path": obs.mask_path,
                "obs_x1": float(obs.box2d[0]),
                "obs_y1": float(obs.box2d[1]),
                "obs_x2": float(obs.box2d[2]),
                "obs_y2": float(obs.box2d[3]),
                "pred_x1": float(pred_bbox[i, 0]),
                "pred_y1": float(pred_bbox[i, 1]),
                "pred_x2": float(pred_bbox[i, 2]),
                "pred_y2": float(pred_bbox[i, 3]),
                "support_left_x": float(pred_left[i]),
                "support_right_x": float(pred_right[i]),
                "support_top_y": float(pred_top[i]),
                "support_bottom_y": float(pred_bottom[i]),
                "center_u": float(center_px[i, 0]),
                "center_v": float(center_px[i, 1]),
                "has_mask": bool(obs.mask_bbox is not None),
                "truncated_left": obs.truncated["left"],
                "truncated_right": obs.truncated["right"],
                "truncated_top": obs.truncated["top"],
                "truncated_bottom": obs.truncated["bottom"],
                **bev_footprint_columns(centers_np[i], size_np, obs.box_axes_cam, obs.camera_to_ego, obs.reference_camera_to_ego),
                **corner_columns(corners_px[i]),
            }
        )
    summary = {
        "track_id": observations[0].track_id,
        "view": observations[0].view,
        "frames": len(observations),
        "best_loss": best_loss,
        "mean_frame_loss": float(np.mean(total_pf)) if len(total_pf) else float("nan"),
        "max_frame_loss": float(np.max(total_pf)) if len(total_pf) else float("nan"),
        "dominant_loss": dominant_loss(
            float(np.sum(edge_pf)),
            float(np.sum(top_bottom_edges_pf)),
            float(np.sum(bbox_fit_pf)),
            float(np.sum(contain_pf)),
            float(np.sum(depth_safety_pf)),
            float(np.sum(center_depth_safety_pf)),
            float(np.sum(ego_box_safety_pf)),
            float(np.sum(initial_depth_prior_pf)),
            float(np.sum(size_ratio_prior_pf)),
            float(np.sum(class_size_prior_pf)),
            float(np.sum(height_depth_prior_pf)),
            float(np.sum(ground_pf)),
            float(np.sum(temporal_acc_pf)),
            float(np.sum(temporal_vertical_acc_pf)),
            float(np.sum(temporal_log_depth_pf)),
            float(np.sum(oversize_pf)),
        ),
        "iterations_used": iterations_used,
        "stop_reason": stop_reason,
        "solution_kind": solution_kind,
    }
    return TrackResult(rows=rows, diagnostics=diagnostics, summary=summary, final_rows=[], final_diagnostics=[], final_summary={})


def dominant_loss(edge: float, top_bottom_edges: float, bbox_fit: float, contain: float, depth_safety: float, center_depth_safety: float, ego_box_safety: float, initial_depth_prior: float, size_ratio_prior: float, class_size_prior: float, height_depth_prior: float, ground: float, temporal_acceleration: float, temporal_vertical_acceleration: float, temporal_log_depth_acceleration: float, oversize: float) -> str:
    values = {
        "edge": edge,
        "top_bottom_edges": top_bottom_edges,
        "bbox_fit": bbox_fit,
        "mask_contain": contain,
        "depth_safety": depth_safety,
        "center_depth_safety": center_depth_safety,
        "ego_box_safety": ego_box_safety,
        "initial_depth_prior": initial_depth_prior,
        "size_ratio_prior": size_ratio_prior,
        "class_size_prior": class_size_prior,
        "height_depth_prior": height_depth_prior,
        "ground": ground,
        "temporal_acceleration": temporal_acceleration,
        "temporal_vertical_acceleration": temporal_vertical_acceleration,
        "temporal_log_depth_acceleration": temporal_log_depth_acceleration,
        "mask_oversize": oversize,
    }
    return max(values, key=values.get)


def corner_columns(corners: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for idx in range(corners.shape[0]):
        out[f"corner{idx}_x"] = float(corners[idx, 0])
        out[f"corner{idx}_y"] = float(corners[idx, 1])
    return out


def bev_footprint_columns(
    center: np.ndarray,
    size: np.ndarray,
    axes_cam: np.ndarray,
    camera_to_ego: np.ndarray,
    reference_camera_to_ego: np.ndarray,
) -> dict[str, float]:
    length_axis = np.asarray(axes_cam[0], dtype=np.float64)
    width_axis = np.asarray(axes_cam[1], dtype=np.float64)
    current_to_reference = np.linalg.inv(reference_camera_to_ego) @ camera_to_ego
    signs = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    out: dict[str, float] = {}
    center_ref = transform_point_np(center, current_to_reference)
    out["bev_center_x"] = float(center_ref[0])
    out["bev_center_z"] = float(center_ref[2])
    for idx, (sl, sw) in enumerate(signs):
        point = center + 0.5 * sl * size[0] * length_axis + 0.5 * sw * size[1] * width_axis
        point_ref = transform_point_np(point, current_to_reference)
        out[f"bev_corner{idx}_x"] = float(point_ref[0])
        out[f"bev_corner{idx}_z"] = float(point_ref[2])
    return out


def transform_point_np(point: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homo = np.concatenate([np.asarray(point, dtype=np.float64), np.ones(1, dtype=np.float64)])
    return (transform @ homo)[:3]
