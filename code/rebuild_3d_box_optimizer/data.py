from __future__ import annotations

import csv
import json
import math
from decimal import Decimal
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import resolve_path
from .geometry import PoseBundle, car_axes_from_orientation_camera, make_transform, quat_xyzw_to_matrix, transform_points


@dataclass
class Observation:
    view: str
    camera_name: str
    frame: int
    timestamp: int
    image: str
    track_id: int
    label: str
    box2d: np.ndarray
    init_center_world: np.ndarray
    init_center_cam: np.ndarray
    init_size: np.ndarray
    min_size: np.ndarray
    max_size: np.ndarray
    intrinsic: np.ndarray
    image_size: np.ndarray
    world_to_camera: np.ndarray
    camera_to_world: np.ndarray
    camera_center_world: np.ndarray
    axes_world: np.ndarray
    box_axes_cam: np.ndarray
    camera_to_ego: np.ndarray
    reference_camera_to_ego: np.ndarray
    yaw_fixed: float
    yaw_source: str
    mask_points: np.ndarray | None
    mask_bbox: np.ndarray | None
    mask_path: str
    mask_area: float
    truncated: dict[str, bool]


def load_view_observations(config: dict[str, Any], view: str) -> list[Observation]:
    view_cfg = config["inputs"]["views"][view]
    root = Path(config["_root_dir"])
    track_csv = resolve_path(config, view_cfg["track_csv"])
    mask_csv = resolve_path(config, view_cfg.get("mask_csv", ""))
    intrinsic_path = resolve_path(config, view_cfg["intrinsic"])
    intrinsic = load_intrinsic(intrinsic_path)
    image_size = load_image_size(intrinsic_path, intrinsic)
    camera_to_ego = load_extrinsic(resolve_path(config, view_cfg["camera_to_ego"]))
    reference_camera_to_ego = load_reference_camera_to_ego(config)
    axes_car = car_axes_from_orientation_camera(reference_camera_to_ego)
    box_axes_cam = reference_box_axes_in_current_camera(camera_to_ego, reference_camera_to_ego)
    annotations = load_annotation_poses(resolve_path(config, config["inputs"]["annotations_dir"]))
    masks = load_masks(config, mask_csv) if mask_csv.exists() else {}
    initial_center_overrides = load_initial_center_overrides(config)
    fixed_yaws = load_fixed_yaw_overrides(config)
    rows = read_csv_dicts(track_csv)
    out: list[Observation] = []
    image_w = float(image_size[0])
    image_h = float(image_size[1])
    for row in rows:
        try:
            frame = int(float(row["frame"]))
            track_id = int(float(row["track_id"]))
            box = np.asarray([float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])], dtype=np.float64)
        except (KeyError, ValueError):
            continue
        if track_id < 0:
            continue
        timestamp = parse_timestamp(row)
        pose = nearest_pose(timestamp, annotations, camera_to_ego)
        init_cam = initial_center_cam(row, box, intrinsic, config)
        override = initial_center_overrides.get((view, track_id, frame))
        if override is not None:
            init_cam = override.copy()
        init_world = transform_points(init_cam[None, :], pose.camera_to_world)[0]
        yaw_value, yaw_source = fixed_yaws.get((view, track_id, frame), (0.0, "zero_default"))
        raw_label = row.get("gt_label") or row.get("label") or row.get("prompt") or "default"
        label = canonical_label(config, raw_label)
        init_size, min_size, max_size = class_sizes(config, label)
        mask = match_mask(masks, frame, track_id, box, label)
        if mask and config["observations"]["mask"].get("use_foreground_points", False):
            mask_cfg = config["observations"]["mask"]
            points = load_mask_points(
                resolve_path(config, mask.get("path", "")),
                int(mask_cfg.get("max_foreground_points", 0)),
                str(mask_cfg.get("point_sample_mode", "foreground_pixels")),
            )
            if points is not None:
                mask["points"] = points
        supporting_edges_cfg = config["observations"]["supporting_edges"]
        truncation_margin = float(
            supporting_edges_cfg.get(
                "near_border_truncation_margin_px",
                supporting_edges_cfg.get("truncation_margin_px", 8.0),
            )
        )
        truncated = truncation_flags(box, image_w, image_h, truncation_margin)
        out.append(
            Observation(
                view=view,
                camera_name=str(view_cfg.get("camera_name", view)),
                frame=frame,
                timestamp=timestamp,
                image=str(row.get("image", "")),
                track_id=track_id,
                label=str(label),
                box2d=box,
                init_center_world=init_world,
                init_center_cam=init_cam,
                init_size=init_size,
                min_size=min_size,
                max_size=max_size,
                intrinsic=intrinsic,
                image_size=image_size,
                world_to_camera=pose.world_to_camera,
                camera_to_world=pose.camera_to_world,
                camera_center_world=pose.camera_center_world,
                axes_world=pose.ego_to_world[:3, :3] @ axes_car,
                box_axes_cam=box_axes_cam,
                camera_to_ego=camera_to_ego,
                reference_camera_to_ego=reference_camera_to_ego,
                yaw_fixed=float(yaw_value),
                yaw_source=str(yaw_source),
                mask_points=mask.get("points") if mask else None,
                mask_bbox=mask.get("bbox") if mask else None,
                mask_path=str(mask.get("path", "")) if mask else "",
                mask_area=float(mask.get("area", 0.0)) if mask else 0.0,
                truncated=truncated,
            )
        )
    refine_track_truncation_flags(config, out)
    return out


def group_by_track(observations: list[Observation]) -> dict[int, list[Observation]]:
    grouped: dict[int, list[Observation]] = {}
    for obs in observations:
        grouped.setdefault(obs.track_id, []).append(obs)
    for items in grouped.values():
        items.sort(key=lambda x: (x.frame, x.timestamp))
    return grouped


def refine_track_truncation_flags(config: dict[str, Any], observations: list[Observation]) -> None:
    cfg = config.get("observations", {}).get("supporting_edges", {}).get("track_level_truncation", {})
    if not bool(cfg.get("enabled", True)):
        return
    expand_margin = float(cfg.get("expand_margin_px", 12.0))
    stop_margin = float(cfg.get("stop_margin_px", 24.0))
    max_frame_gap = int(cfg.get("max_frame_gap", 3))
    grouped = group_by_track(observations)
    for items in grouped.values():
        mark_track_size_anomaly_truncation(items, cfg)
        for side in ("left", "right", "top", "bottom"):
            expand_track_side_truncation(items, side, expand_margin, stop_margin, max_frame_gap)


def mark_track_size_anomaly_truncation(items: list[Observation], cfg: dict[str, Any]) -> None:
    size_cfg = cfg.get("size_anomaly_near_border", {})
    if not bool(size_cfg.get("enabled", True)) or not items:
        return
    boxes = np.stack([obs.box2d for obs in items], axis=0)
    widths = np.maximum(boxes[:, 2] - boxes[:, 0], 1.0)
    heights = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
    areas = widths * heights
    reference_percentile = float(size_cfg.get("reference_percentile", 90.0))
    ref_width = float(np.percentile(widths, reference_percentile))
    ref_height = float(np.percentile(heights, reference_percentile))
    ref_area = float(np.percentile(areas, reference_percentile))
    min_reference_area = float(size_cfg.get("min_reference_area_px", 20000.0))
    if ref_area < min_reference_area:
        return
    near_margin = float(size_cfg.get("near_border_margin_px", 96.0))
    max_area_ratio = float(size_cfg.get("max_area_ratio_to_reference", 0.25))
    max_width_ratio = float(size_cfg.get("max_width_ratio_to_reference", 0.5))
    max_height_ratio = float(size_cfg.get("max_height_ratio_to_reference", 0.5))
    min_size_evidence = int(size_cfg.get("min_size_evidence_count", 1))
    for obs, width, height, area in zip(items, widths, heights, areas):
        area_small = area <= ref_area * max_area_ratio
        width_small = width <= ref_width * max_width_ratio
        height_small = height <= ref_height * max_height_ratio
        evidence = int(area_small) + int(width_small) + int(height_small)
        if evidence < min_size_evidence:
            continue
        for side in ("left", "right", "top", "bottom"):
            if boundary_gap(obs, side) <= near_margin:
                obs.truncated[side] = True


def expand_track_side_truncation(
    items: list[Observation],
    side: str,
    expand_margin: float,
    stop_margin: float,
    max_frame_gap: int,
) -> None:
    if not any(obs.truncated.get(side, False) for obs in items):
        return
    n = len(items)
    seed_indices = [idx for idx, obs in enumerate(items) if obs.truncated.get(side, False)]
    for seed in seed_indices:
        for direction in (-1, 1):
            prev = seed
            idx = seed + direction
            while 0 <= idx < n:
                if abs(items[idx].frame - items[prev].frame) > max_frame_gap:
                    break
                gap = boundary_gap(items[idx], side)
                if gap > stop_margin:
                    break
                if gap <= expand_margin:
                    items[idx].truncated[side] = True
                    prev = idx
                    idx += direction
                    continue
                break


def boundary_gap(obs: Observation, side: str) -> float:
    box = obs.box2d
    width = float(obs.image_size[0]) if hasattr(obs, "image_size") else float(obs.intrinsic[0, 2] * 2.0)
    height = float(obs.image_size[1]) if hasattr(obs, "image_size") else float(obs.intrinsic[1, 2] * 2.0)
    if side == "left":
        return float(box[0])
    if side == "right":
        return float(width - box[2])
    if side == "top":
        return float(box[1])
    if side == "bottom":
        return float(height - box[3])
    raise ValueError(f"Unknown boundary side: {side}")


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_initial_center_overrides(config: dict[str, Any]) -> dict[tuple[str, int, int], np.ndarray]:
    depth_cfg = config.get("observations", {}).get("depth", {})
    path_text = str(depth_cfg.get("initial_center_override_csv", "") or "")
    if not path_text:
        return {}
    path = resolve_path(config, path_text)
    if not path.exists():
        return {}
    out: dict[tuple[str, int, int], np.ndarray] = {}
    for row in read_csv_dicts(path):
        try:
            view = str(row["view"])
            track_id = int(float(row["track_id"]))
            frame = int(float(row["frame"]))
            center = np.asarray([float(row["cx"]), float(row["cy"]), float(row["cz"])], dtype=np.float64)
        except (KeyError, ValueError):
            continue
        if np.isfinite(center).all() and center[2] > 0:
            out[(view, track_id, frame)] = center
    return out


def load_fixed_yaw_overrides(config: dict[str, Any]) -> dict[tuple[str, int, int], tuple[float, str]]:
    yaw_cfg = config.get("variables", {}).get("yaw", {})
    path_value = yaw_cfg.get("fixed_csv") or yaw_cfg.get("preprocessed_csv") or ""
    if not path_value:
        return {}
    path = resolve_path(config, str(path_value))
    if not path.exists():
        raise FileNotFoundError(f"Fixed yaw CSV does not exist: {path}")
    yaw_column = str(yaw_cfg.get("column", "yaw"))
    source_column = str(yaw_cfg.get("source_column", "yaw_axis"))
    out: dict[tuple[str, int, int], tuple[float, str]] = {}
    for row in read_csv_dicts(path):
        try:
            view = str(row["view"])
            track_id = int(float(row["track_id"]))
            frame = int(float(row["frame"]))
            yaw = float(row[yaw_column])
        except (KeyError, ValueError):
            continue
        if not np.isfinite(yaw):
            continue
        source = str(row.get(source_column, "fixed_yaw_csv"))
        out[(view, track_id, frame)] = (yaw, source)
    return out


def load_intrinsic(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    for payload in data.values():
        param = payload.get("param", {})
        if "cam_K_new" in param:
            return np.asarray(param["cam_K_new"]["data"], dtype=np.float64)
        if "cam_K" in param:
            return np.asarray(param["cam_K"]["data"], dtype=np.float64)
    raise ValueError(f"No camera intrinsic found in {path}")


def load_image_size(path: Path, intrinsic: np.ndarray) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    for payload in data.values():
        param = payload.get("param", {})
        width = param.get("img_new_w") or param.get("img_dist_w")
        height = param.get("img_new_h") or param.get("img_dist_h")
        if width and height:
            return np.asarray([float(width), float(height)], dtype=np.float64)
    return np.asarray([float(intrinsic[0, 2] * 2.0), float(intrinsic[1, 2] * 2.0)], dtype=np.float64)


def load_extrinsic(path: Path) -> np.ndarray:
    data = json.loads(path.read_text(encoding="utf-8"))
    for payload in data.values():
        calib = payload.get("param", {}).get("sensor_calib")
        if calib and "data" in calib:
            return np.asarray(calib["data"], dtype=np.float64)
    raise ValueError(f"No extrinsic matrix found in {path}")


def load_reference_camera_to_ego(config: dict[str, Any]) -> np.ndarray:
    yaw_cfg = config.get("variables", {}).get("yaw", {})
    reference_view = str(yaw_cfg.get("reference_view", "rear"))
    views = config.get("inputs", {}).get("views", {})
    if reference_view not in views:
        raise ValueError(f"Yaw reference_view={reference_view!r} is not present in inputs.views")
    return load_extrinsic(resolve_path(config, views[reference_view]["camera_to_ego"]))


def reference_box_axes_in_current_camera(current_camera_to_ego: np.ndarray, reference_camera_to_ego: np.ndarray) -> np.ndarray:
    """Return [length,width,height] axes in the current camera frame.

    The length axis is defined by the reference camera (rear by default), then
    transformed into the current camera. This keeps side cameras as observers
    instead of making the 3D box face each side camera.
    """
    axes_ego = car_axes_from_orientation_camera(reference_camera_to_ego)
    ego_to_current_camera_rot = current_camera_to_ego[:3, :3].T
    axes_cam = np.stack([ego_to_current_camera_rot @ axis for axis in axes_ego], axis=0)
    axes_cam = axes_cam / np.maximum(np.linalg.norm(axes_cam, axis=1, keepdims=True), 1.0e-9)
    return axes_cam.astype(np.float64)


def load_annotation_poses(path: Path) -> list[tuple[int, np.ndarray]]:
    poses: list[tuple[int, np.ndarray]] = []
    for file in sorted(path.glob("*.json")):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            ts = int(data["timestamp"])
            rot = quat_xyzw_to_matrix(data["ego2global_rotation"])
            trans = np.asarray(data["ego2global_translation"], dtype=np.float64)
            poses.append((ts, make_transform(rot, trans)))
        except Exception:
            continue
    if not poses:
        raise ValueError(f"No annotation poses found in {path}")
    return poses


def nearest_pose(timestamp: int, poses: list[tuple[int, np.ndarray]], camera_to_ego: np.ndarray) -> PoseBundle:
    best_ts, best_pose = min(poses, key=lambda item: abs(item[0] - timestamp))
    _ = best_ts
    return PoseBundle(ego_to_world=best_pose, camera_to_ego=camera_to_ego)


def parse_timestamp(row: dict[str, str]) -> int:
    for key in ("timestamp", "annotation_timestamp"):
        value = row.get(key)
        if value not in (None, ""):
            return parse_int_value(value)
    image = Path(row.get("image", "")).stem
    return parse_int_value(image) if image else 0


def parse_int_value(value: str | int | float) -> int:
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return 0
    return int(Decimal(text))


def initial_center_cam(row: dict[str, str], box: np.ndarray, intrinsic: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    keys = ("x_cam", "y_cam", "z_cam")
    label = canonical_label(config, row.get("gt_label") or row.get("label") or row.get("prompt") or "default")
    init_size, _, _ = class_sizes(config, label)
    h_px = max(float(box[3] - box[1]), 1.0)
    height_prior_depth = float(intrinsic[1, 1] * init_size[2] / h_px)
    depth_cfg = config.get("observations", {}).get("depth", {})
    init_mode = str(depth_cfg.get("initialization_mode", "da3"))
    replace_beyond_m = float(depth_cfg.get("height_prior_replace_da3_beyond_m", 0.0) or 0.0)
    if init_mode == "height_prior":
        return center_from_depth(box, intrinsic, height_prior_depth)
    if all(row.get(k) not in (None, "") for k in keys):
        vals = np.asarray([float(row[k]) for k in keys], dtype=np.float64)
        if np.isfinite(vals).all() and vals[2] > 0:
            if init_mode == "height_prior_beyond_distance" and replace_beyond_m > 0 and vals[2] >= replace_beyond_m:
                return center_from_depth(box, intrinsic, height_prior_depth)
            return vals
    depth = float(row.get("depth_median") or row.get("depth_weighted_mean") or 0.0)
    if init_mode == "height_prior_beyond_distance" and replace_beyond_m > 0 and math.isfinite(depth) and depth >= replace_beyond_m:
        depth = height_prior_depth
    if not math.isfinite(depth) or depth <= 0:
        depth = height_prior_depth
    return center_from_depth(box, intrinsic, depth)


def center_from_depth(box: np.ndarray, intrinsic: np.ndarray, depth: float) -> np.ndarray:
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    x = (cx - intrinsic[0, 2]) * depth / intrinsic[0, 0]
    y = (cy - intrinsic[1, 2]) * depth / intrinsic[1, 1]
    return np.asarray([x, y, depth], dtype=np.float64)


def class_sizes(config: dict[str, Any], label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label = canonical_label(config, label)
    table = config.get("class_defaults", {})
    item = table.get(label) or table.get(str(label).upper()) or table.get("default", {})
    init = np.asarray(item.get("init_size", [4.5, 1.8, 1.6]), dtype=np.float64)
    mn = np.asarray(item.get("min_size", [0.2, 0.2, 0.2]), dtype=np.float64)
    mx = np.asarray(item.get("max_size", [20.0, 4.0, 5.0]), dtype=np.float64)
    return init, mn, mx


def canonical_label(config: dict[str, Any], label: str | None) -> str:
    text = str(label or "default").strip() or "default"
    aliases = config.get("label_aliases", {})
    if text in aliases:
        return str(aliases[text])
    upper = text.upper()
    if upper in aliases:
        return str(aliases[upper])
    lower = text.lower()
    if lower in aliases:
        return str(aliases[lower])
    return text


def axes_from_pose(camera_to_world: np.ndarray) -> np.ndarray:
    from .geometry import axes_from_camera_pose

    return axes_from_camera_pose(camera_to_world)


def load_masks(config: dict[str, Any], mask_csv: Path) -> dict[tuple[int, int], list[dict[str, Any]]]:
    masks: dict[tuple[int, int], list[dict[str, Any]]] = {}
    if not mask_csv.exists():
        return masks
    for row in read_csv_dicts(mask_csv):
        try:
            frame = int(float(row["frame"]))
            track_id = int(float(row.get("track_id", -1)))
            bbox = np.asarray([float(row["mask_x1"]), float(row["mask_y1"]), float(row["mask_x2"]), float(row["mask_y2"])], dtype=np.float64)
            box2d = np.asarray([float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])], dtype=np.float64)
        except (KeyError, ValueError):
            continue
        item: dict[str, Any] = {
            "bbox": bbox,
            "box2d": box2d,
            "label": row.get("label", ""),
            "area": float(row.get("mask_area") or max((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), 1.0)),
        }
        item["path"] = row.get("mask_path", "")
        masks.setdefault((frame, track_id), []).append(item)
        if track_id < 0:
            masks.setdefault((frame, -1), []).append(item)
    return masks


def load_mask_points(path: Path, max_points: int, mode: str = "foreground_pixels") -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        from PIL import Image
    except ImportError:
        return None
    arr = np.asarray(Image.open(path).convert("L"))
    mask = arr > 0
    if not np.any(mask):
        return None
    if mode in ("external_contour", "outer_contour", "contour"):
        pts = external_contour_points(mask)
    else:
        ys, xs = np.nonzero(mask)
        pts = np.stack([xs, ys], axis=1).astype(np.float64)
    if len(pts) == 0:
        return None
    return sample_points_uniform(pts, max_points)


def external_contour_points(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except ImportError:
        return boundary_points_without_cv2(mask)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return boundary_points_without_cv2(mask)
    pts = np.concatenate([c.reshape(-1, 2) for c in contours if len(c) > 0], axis=0)
    return pts.astype(np.float64)


def boundary_points_without_cv2(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    interior = (
        padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
        & padded[:-2, :-2]
        & padded[:-2, 2:]
        & padded[2:, :-2]
        & padded[2:, 2:]
    )
    ys, xs = np.nonzero(center & (~interior))
    return np.stack([xs, ys], axis=1).astype(np.float64) if len(xs) else np.empty((0, 2), dtype=np.float64)


def sample_points_uniform(pts: np.ndarray, max_points: int) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    if max_points > 0 and len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(np.int64)
        pts = pts[idx]
    return pts


def match_mask(masks: dict[tuple[int, int], list[dict[str, Any]]], frame: int, track_id: int, box: np.ndarray, label: str) -> dict[str, Any] | None:
    direct = masks.get((frame, track_id), [])
    if direct:
        return max(direct, key=lambda m: box_iou(box, m["box2d"]))
    return None


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / max(area_a + area_b - inter, 1.0e-9))


def truncation_flags(box: np.ndarray, image_w: float, image_h: float, margin: float) -> dict[str, bool]:
    return {
        "left": bool(box[0] <= margin),
        "right": bool(box[2] >= image_w - margin),
        "top": bool(box[1] <= margin),
        "bottom": bool(box[3] >= image_h - margin),
    }
