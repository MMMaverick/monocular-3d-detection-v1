from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .config import resolve_path
from .data import load_extrinsic
from .geometry import car_axes_from_orientation_camera


CUBOID_EDGES = [
    (0, 1),
    (2, 3),
    (1, 2),
    (3, 0),
    (4, 5),
    (6, 7),
    (5, 6),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def render_experiment_videos(config: dict[str, Any], diagnostics_csv: Path, out_dir: Path) -> list[Path]:
    video_cfg = config.get("output", {}).get("video", {})
    if not config.get("output", {}).get("videos", True) or not video_cfg.get("enabled", True):
        return []
    try:
        import cv2
    except ImportError:
        (out_dir / "video_render_skipped.txt").write_text("opencv-python is not available in this environment.\n", encoding="utf-8")
        return []

    rows = read_rows(diagnostics_csv)
    written: list[Path] = []
    for view, view_rows in group_rows(rows, "view").items():
        view_rows.sort(key=lambda r: (int(float(r["frame"])), int(float(r["track_id"]))))
        video_path = out_dir / f"{view}_overlay.mp4"
        if render_view_video(config, view_rows, video_path, video_cfg, cv2):
            written.append(video_path)
    return written


def render_track_video_from_rows(
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    video_path: Path,
) -> Path | None:
    video_cfg = config.get("output", {}).get("video", {})
    if not config.get("output", {}).get("videos", True) or not video_cfg.get("enabled", True):
        return None
    if not rows:
        return None
    try:
        import cv2
    except ImportError:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        (video_path.parent / "video_render_skipped.txt").write_text("opencv-python is not available in this environment.\n", encoding="utf-8")
        return None
    row_strings = [{key: "" if value is None else str(value) for key, value in row.items()} for row in rows]
    row_strings.sort(key=lambda r: (int(float(r["frame"])), int(float(r["track_id"]))))
    if render_view_video(config, row_strings, video_path, video_cfg, cv2):
        return video_path
    return None


def render_view_video(config: dict[str, Any], rows: list[dict[str, str]], video_path: Path, video_cfg: dict[str, Any], cv2) -> bool:
    writer = None
    fps = float(video_cfg.get("fps", 10.0))
    try:
        frame_groups = list(group_frame_rows(rows).values())
        frame_groups.sort(key=lambda g: (int(float(g[0]["frame"])), int(float(g[0].get("timestamp", 0)))))
        max_frames = int(video_cfg.get("max_frames_per_view", 0) or 0)
        if max_frames > 0:
            frame_groups = frame_groups[:max_frames]
        for frame_rows in frame_groups:
            row = frame_rows[0]
            image_path = resolve_data_path(config, row.get("image", ""))
            if not image_path.exists():
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            vis = image.copy()
            for item in frame_rows:
                color_seed = int(float(item.get("track_id", 0)))
                if bool(video_cfg.get("draw_mask_pixels", True)):
                    draw_mask_overlay(config, vis, item, float(video_cfg.get("mask_alpha", 0.35)), cv2)
                if bool(video_cfg.get("draw_2d_box", True)):
                    color = (255, 0, 0) if bool(video_cfg.get("debug_geometry_style", False)) else color_for_track(color_seed, "2d")
                    draw_box(
                        vis,
                        item,
                        ("obs_x1", "obs_y1", "obs_x2", "obs_y2"),
                        color,
                        f"2D {item.get('track_id')}",
                        cv2,
                        int(video_cfg.get("box_2d_thickness", 1)),
                    )
                if bool(video_cfg.get("draw_3d_box", True)):
                    color = (0, 255, 0) if bool(video_cfg.get("debug_geometry_style", False)) else color_for_track(color_seed, "3d")
                    draw_cuboid(vis, item, cv2, color, bool(video_cfg.get("draw_projected_bbox", True)), video_cfg)
                if bool(video_cfg.get("draw_corner_points", False)) or bool(video_cfg.get("draw_corner_labels", False)):
                    draw_corner_debug(vis, item, cv2, labels=bool(video_cfg.get("draw_corner_labels", False)))
                if bool(video_cfg.get("draw_center_projection", False)):
                    draw_center_projection(vis, item, cv2)
                if bool(video_cfg.get("draw_support_edges", True)):
                    draw_support_edges(vis, item, cv2)
            if bool(video_cfg.get("draw_loss_panel", True)):
                draw_loss_panel_multi(vis, frame_rows, cv2, video_cfg)
            if bool(video_cfg.get("draw_bev", True)):
                vis = append_bev_panel(config, vis, frame_rows, cv2, video_cfg)
            if writer is None:
                h, w = vis.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w, h))
            writer.write(vis)
    finally:
        if writer is not None:
            writer.release()
    return writer is not None


def draw_mask_overlay(config: dict[str, Any], image: np.ndarray, row: dict[str, str], alpha: float, cv2) -> None:
    mask_path_text = row.get("mask_path", "")
    if not mask_path_text:
        return
    mask_path = resolve_data_path(config, mask_path_text)
    if not mask_path.exists():
        return
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    color = np.zeros_like(image)
    color[:, :, 0] = 255
    active = mask > 0
    image[active] = (image[active] * (1.0 - alpha) + color[active] * alpha).astype(np.uint8)


def draw_box(image: np.ndarray, row: dict[str, str], keys: tuple[str, str, str, str], color: tuple[int, int, int], label: str, cv2, thickness: int = 2) -> None:
    x1, y1, x2, y2 = [safe_float(row.get(k, "nan")) for k in keys]
    if not np.isfinite([x1, y1, x2, y2]).all():
        return
    p1 = (int(round(x1)), int(round(y1)))
    p2 = (int(round(x2)), int(round(y2)))
    cv2.rectangle(image, p1, p2, color, max(1, int(thickness)), cv2.LINE_AA)
    cv2.putText(image, label, (p1[0], max(18, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, max(1, int(thickness)), cv2.LINE_AA)


def draw_cuboid(
    image: np.ndarray,
    row: dict[str, str],
    cv2,
    color: tuple[int, int, int] = (0, 0, 255),
    draw_projected_bbox: bool = True,
    video_cfg: dict[str, Any] | None = None,
) -> None:
    video_cfg = video_cfg or {}
    pts = []
    for idx in range(8):
        x = safe_float(row.get(f"corner{idx}_x", "nan"))
        y = safe_float(row.get(f"corner{idx}_y", "nan"))
        pts.append((x, y))
    if not all(np.isfinite([x, y]).all() for x, y in pts):
        return
    for a, b in CUBOID_EDGES:
        p1 = (int(round(pts[a][0])), int(round(pts[a][1])))
        p2 = (int(round(pts[b][0])), int(round(pts[b][1])))
        cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)
    if draw_projected_bbox:
        draw_box(image, row, ("pred_x1", "pred_y1", "pred_x2", "pred_y2"), (0, 255, 255), f"3D bbox {row.get('track_id')}", cv2)
    if bool(video_cfg.get("draw_box_dimensions", True)):
        draw_box_dimensions(image, row, pts, cv2, color, video_cfg)


def draw_box_dimensions(
    image: np.ndarray,
    row: dict[str, str],
    pts: list[tuple[float, float]],
    cv2,
    color: tuple[int, int, int],
    video_cfg: dict[str, Any],
) -> None:
    length = safe_float(row.get("length", "nan"))
    width = safe_float(row.get("width", "nan"))
    height = safe_float(row.get("height", "nan"))
    if not np.isfinite([length, width, height]).all():
        return
    pts_np = np.asarray(pts, dtype=np.float64)
    finite = np.isfinite(pts_np).all(axis=1)
    if not finite.any():
        return
    anchor = pts_np[finite].mean(axis=0)
    x = int(round(np.clip(anchor[0] + 8, 4, max(image.shape[1] - 520, 4))))
    y = int(round(np.clip(anchor[1] - 8, 22, max(image.shape[0] - 12, 22))))
    text = f"LWH={length:.2f},{width:.2f},{height:.2f}m"
    if bool(video_cfg.get("draw_truncation_label", True)):
        text = f"{text} trunc={truncation_label(row)}"
    scale = float(video_cfg.get("box_dimensions_font_scale", 0.62))
    thickness = int(video_cfg.get("box_dimensions_thickness", 2))
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    overlay = image.copy()
    cv2.rectangle(overlay, (x - 4, y - th - 6), (x + tw + 4, y + baseline + 4), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, image, 0.55, 0, image)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def truncation_label(row: dict[str, str]) -> str:
    sides = []
    for key, label in (
        ("truncated_left", "L"),
        ("truncated_right", "R"),
        ("truncated_top", "T"),
        ("truncated_bottom", "B"),
    ):
        if bool_text(row.get(key, "")):
            sides.append(label)
    return "".join(sides) if sides else "-"


def draw_corner_debug(image: np.ndarray, row: dict[str, str], cv2, labels: bool = True) -> None:
    for idx in range(8):
        x = safe_float(row.get(f"corner{idx}_x", "nan"))
        y = safe_float(row.get(f"corner{idx}_y", "nan"))
        if not np.isfinite([x, y]).all():
            continue
        p = (int(round(x)), int(round(y)))
        cv2.circle(image, p, 5, (0, 0, 255), -1, cv2.LINE_AA)
        if labels:
            cv2.putText(image, str(idx), (p[0] + 7, p[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2, cv2.LINE_AA)


def draw_center_projection(image: np.ndarray, row: dict[str, str], cv2) -> None:
    u = safe_float(row.get("center_u", "nan"))
    v = safe_float(row.get("center_v", "nan"))
    if not np.isfinite([u, v]).all():
        return
    p = (int(round(u)), int(round(v)))
    cv2.circle(image, p, 7, (255, 0, 255), -1, cv2.LINE_AA)
    cv2.putText(image, f"C {row.get('track_id')}", (p[0] + 10, p[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)


def draw_support_edges(image: np.ndarray, row: dict[str, str], cv2) -> None:
    obs_x1 = safe_float(row.get("obs_x1"))
    obs_x2 = safe_float(row.get("obs_x2"))
    obs_y1 = safe_float(row.get("obs_y1"))
    obs_y2 = safe_float(row.get("obs_y2"))
    if not np.isfinite([obs_x1, obs_x2, obs_y1, obs_y2]).all():
        return
    y1 = int(round(obs_y1))
    y2 = int(round(obs_y2))
    pts = cuboid_points(row)
    if pts is None:
        return
    edge_pairs = [(0, 4), (1, 5), (2, 6), (3, 7)]
    edge_x = np.asarray([(pts[a, 0] + pts[b, 0]) * 0.5 for a, b in edge_pairs], dtype=np.float64)
    left_i = int(np.argmin(edge_x))
    right_i = int(np.argmax(edge_x))
    left_edge = edge_pairs[left_i]
    right_edge = edge_pairs[right_i]
    draw_support_assignment(
        image,
        row,
        pts,
        left_edge,
        float(edge_x[left_i]),
        obs_x1,
        y1,
        y2,
        f"L {_vertical_edge_semantic_label(left_edge)} -> GT x1",
        (255, 0, 255),
        cv2,
        bool_text(row.get("truncated_left")),
    )
    draw_support_assignment(
        image,
        row,
        pts,
        right_edge,
        float(edge_x[right_i]),
        obs_x2,
        y1,
        y2,
        f"R {_vertical_edge_semantic_label(right_edge)} -> GT x2",
        (0, 255, 255),
        cv2,
        bool_text(row.get("truncated_right")),
    )
    draw_top_bottom_support_assignment(image, row, obs_x1, obs_x2, obs_y1, obs_y2, cv2)


def _vertical_edge_semantic_label(edge: tuple[int, int]) -> str:
    # Corner convention is fixed semantically:
    # 0/4=(-length,-width), 1/5=(+length,-width),
    # 2/6=(+length,+width), 3/7=(-length,+width).
    # The vehicle long side is always the length axis, never inferred from
    # projected 2D edge length.
    labels = {
        (0, 4): "rear-left vertical",
        (1, 5): "front-left vertical",
        (2, 6): "front-right vertical",
        (3, 7): "rear-right vertical",
    }
    return labels.get(edge, f"vertical {edge[0]}-{edge[1]}")


def draw_top_bottom_support_assignment(
    image: np.ndarray,
    row: dict[str, str],
    obs_x1: float,
    obs_x2: float,
    obs_y1: float,
    obs_y2: float,
    cv2,
) -> None:
    loss = safe_float(row.get("loss_top_bottom_edges", "nan"))
    if not np.isfinite(loss) or loss <= 0:
        return
    pred_top = safe_float(row.get("support_top_y", row.get("pred_y1", "nan")))
    pred_bottom = safe_float(row.get("support_bottom_y", row.get("pred_y2", "nan")))
    if not np.isfinite([pred_top, pred_bottom]).all():
        return
    x1, x2 = int(round(obs_x1)), int(round(obs_x2))
    gt_top, gt_bottom = int(round(obs_y1)), int(round(obs_y2))
    pt, pb = int(round(pred_top)), int(round(pred_bottom))
    color_top = (0, 128, 255)
    color_bottom = (128, 255, 0)
    cv2.line(image, (x1, gt_top), (x2, gt_top), color_top, 2, cv2.LINE_AA)
    cv2.line(image, (x1, pt), (x2, pt), color_top, 5, cv2.LINE_AA)
    cv2.line(image, (x1, gt_bottom), (x2, gt_bottom), color_bottom, 2, cv2.LINE_AA)
    cv2.line(image, (x1, pb), (x2, pb), color_bottom, 5, cv2.LINE_AA)
    text_x = int(round(min(max(x1, 4), max(image.shape[1] - 520, 4))))
    cv2.putText(image, f"T edge -> GT y1 d={pred_top - obs_y1:+.1f}px", (text_x, max(18, pt - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_top, 2, cv2.LINE_AA)
    cv2.putText(image, f"B edge -> GT y2 d={pred_bottom - obs_y2:+.1f}px", (text_x, min(image.shape[0] - 12, pb + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_bottom, 2, cv2.LINE_AA)


def cuboid_points(row: dict[str, str]) -> np.ndarray | None:
    pts = []
    for idx in range(8):
        x = safe_float(row.get(f"corner{idx}_x", "nan"))
        y = safe_float(row.get(f"corner{idx}_y", "nan"))
        pts.append((x, y))
    arr = np.asarray(pts, dtype=np.float64)
    if not np.isfinite(arr).all():
        return None
    return arr


def draw_support_assignment(
    image: np.ndarray,
    row: dict[str, str],
    pts: np.ndarray,
    edge: tuple[int, int],
    edge_x: float,
    gt_x: float,
    y1: int,
    y2: int,
    label: str,
    color: tuple[int, int, int],
    cv2,
    truncated: bool,
) -> None:
    a, b = edge
    p1 = (int(round(pts[a, 0])), int(round(pts[a, 1])))
    p2 = (int(round(pts[b, 0])), int(round(pts[b, 1])))
    gt_xi = int(round(gt_x))
    edge_xi = int(round(edge_x))
    cv2.line(image, p1, p2, color, 5, cv2.LINE_AA)
    cv2.circle(image, p1, 6, color, -1, cv2.LINE_AA)
    cv2.circle(image, p2, 6, color, -1, cv2.LINE_AA)
    cv2.line(image, (gt_xi, y1), (gt_xi, y2), color, 2, cv2.LINE_AA)
    mid_y = int(round((pts[a, 1] + pts[b, 1]) * 0.5))
    cv2.line(image, (edge_xi, mid_y), (gt_xi, mid_y), color, 2, cv2.LINE_AA)
    delta = edge_x - gt_x
    trunc_note = " one-sided" if truncated else ""
    text = f"{label}: {a}-{b} d={delta:+.1f}px{trunc_note}"
    text_x = int(round(min(max(min(edge_x, gt_x), 4), max(image.shape[1] - 520, 4))))
    text_y = int(round(min(max(mid_y - 8, 18), image.shape[0] - 12)))
    cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)


def bool_text(value: str | None) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def draw_loss_panel(image: np.ndarray, row: dict[str, str], cv2, video_cfg: dict[str, Any] | None = None) -> None:
    video_cfg = video_cfg or {}
    font_scale = float(video_cfg.get("loss_panel_font_scale", 0.78))
    thickness = int(video_cfg.get("loss_panel_thickness", 2))
    line_height = int(video_cfg.get("loss_panel_line_height", 30))
    lines = [
        f"view={row.get('view')} frame={row.get('frame')} track={row.get('track_id')}",
        f"class={row.get('class')} dominant={row.get('dominant_loss')}",
        f"total={fmt(row, 'loss_total')} edge={fmt(row, 'loss_edge')} top_bottom={fmt(row, 'loss_top_bottom_edges')}",
        f"bbox_fit={fmt(row, 'loss_bbox_fit')} mask_contain={fmt(row, 'loss_mask_contain')}",
        f"depth_safe={fmt(row, 'loss_depth_safety')} center_depth={fmt(row, 'loss_center_depth_safety')} ego_box={fmt(row, 'loss_ego_box_safety')} init_depth={fmt(row, 'loss_initial_depth_prior')}",
        f"ego_box: clear_min={fmt(row, 'ego_box_safety_min_clearance_m')} penetration={fmt(row, 'ego_box_safety_penetration_m')} active={fmt(row, 'ego_box_safety_active')}",
        f"height_depth={fmt(row, 'loss_height_depth_prior')} size_ratio={fmt(row, 'loss_size_ratio_prior')} class_size={fmt(row, 'loss_class_size_prior')} ground={fmt(row, 'loss_ground')}",
        f"temporal planar_acc={fmt(row, 'loss_temporal_acceleration')} vertical_acc={fmt(row, 'loss_temporal_vertical_acceleration')} log_depth_acc={fmt(row, 'loss_temporal_log_depth_acceleration')}",
        f"height_depth: z_prior={fmt(row, 'height_depth_prior_z')} h_prior={fmt(row, 'height_depth_prior_height_m')} logdev={fmt(row, 'height_depth_prior_log_deviation')} excess={fmt(row, 'height_depth_prior_excess')} active={fmt(row, 'height_depth_prior_active')}",
        f"class_size: target=[{fmt(row, 'class_size_prior_target_length')},{fmt(row, 'class_size_prior_target_width')},{fmt(row, 'class_size_prior_target_height')}] logdev_max={fmt(row, 'class_size_prior_log_deviation_max')} excess_max={fmt(row, 'class_size_prior_excess_max')}",
        f"ground: final={fmt(row, 'loss_ground')} raw={fmt(row, 'loss_ground_raw')} decay={fmt(row, 'ground_distance_decay_multiplier')} dist={fmt(row, 'ground_distance_for_decay_m')} mean_d={fmt(row, 'ground_distance_mean')} max_abs_d={fmt(row, 'ground_distance_abs_max')} h={fmt(row, 'ground_camera_height_m')} n=({fmt(row, 'ground_normal_cam_x')},{fmt(row, 'ground_normal_cam_y')},{fmt(row, 'ground_normal_cam_z')})",
        f"oversize: ratio={fmt(row, 'mask_area_ratio')} max={fmt(row, 'mask_oversize_max_area_ratio')} excess={fmt(row, 'mask_oversize_excess')}",
        f"oversize: raw=excess^2={fmt(row, 'loss_mask_oversize_unweighted')} *w={fmt(row, 'mask_oversize_weight')} => {fmt(row, 'loss_mask_oversize_weighted')} *size_w={fmt(row, 'size_weight')} => final={fmt(row, 'loss_mask_oversize_final')}",
        f"areas: used={fmt(row, 'pred_area')} clipped={fmt(row, 'pred_clipped_area')} full={fmt(row, 'pred_full_area')} mask={fmt(row, 'mask_area')} bbox_diag_area={fmt(row, 'pred_bbox_area')} mask_pts={fmt(row, 'mask_point_count')}",
        f"trunc L/R/T/B={row.get('truncated_left')}/{row.get('truncated_right')}/{row.get('truncated_top')}/{row.get('truncated_bottom')} geom_clip={row.get('geometry_uses_clipped_projection')} mask={row.get('has_mask')}",
    ]
    x, y = 12, 12
    width = min(image.shape[1] - 24, int(video_cfg.get("loss_panel_width", 920)))
    height = line_height * len(lines) + 16
    overlay = image.copy()
    cv2.rectangle(overlay, (x - 6, y - 4), (x + width, y + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
    for idx, text in enumerate(lines):
        cv2.putText(image, text, (x, y + line_height * (idx + 1)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_loss_panel_multi(image: np.ndarray, rows: list[dict[str, str]], cv2, video_cfg: dict[str, Any] | None = None) -> None:
    video_cfg = video_cfg or {}
    font_scale = float(video_cfg.get("loss_panel_font_scale", 0.72))
    thickness = int(video_cfg.get("loss_panel_thickness", 2))
    line_height = int(video_cfg.get("loss_panel_line_height", 26))
    total = sum(safe_float(r.get("loss_total")) for r in rows)
    edge = sum(safe_float(r.get("loss_edge")) for r in rows)
    top_bottom_edges = sum(safe_float(r.get("loss_top_bottom_edges")) for r in rows)
    bbox_fit = sum(safe_float(r.get("loss_bbox_fit")) for r in rows)
    contain = sum(safe_float(r.get("loss_mask_contain")) for r in rows)
    depth_safety = sum(safe_float(r.get("loss_depth_safety")) for r in rows)
    center_depth_safety = sum(safe_float(r.get("loss_center_depth_safety")) for r in rows)
    ego_box_safety = sum(safe_float(r.get("loss_ego_box_safety")) for r in rows)
    initial_depth_prior = sum(safe_float(r.get("loss_initial_depth_prior")) for r in rows)
    height_depth_prior = sum(safe_float(r.get("loss_height_depth_prior")) for r in rows)
    size_ratio_prior = sum(safe_float(r.get("loss_size_ratio_prior")) for r in rows)
    class_size_prior = sum(safe_float(r.get("loss_class_size_prior")) for r in rows)
    ground = sum(safe_float(r.get("loss_ground")) for r in rows)
    temporal_acc = sum(safe_float(r.get("loss_temporal_acceleration")) for r in rows)
    temporal_vertical_acc = sum(safe_float(r.get("loss_temporal_vertical_acceleration")) for r in rows)
    temporal_log_depth_acc = sum(safe_float(r.get("loss_temporal_log_depth_acceleration")) for r in rows)
    oversize = sum(safe_float(r.get("loss_mask_oversize")) for r in rows)
    oversize_unweighted = sum(safe_float(r.get("loss_mask_oversize_unweighted")) for r in rows)
    oversize_weighted = sum(safe_float(r.get("loss_mask_oversize_weighted")) for r in rows)
    oversize_final = sum(safe_float(r.get("loss_mask_oversize_final")) for r in rows)
    first = rows[0]
    if str(video_cfg.get("loss_panel_mode", "full")) == "dominant_only":
        by_loss: dict[str, float] = {}
        for row in rows:
            dom = str(row.get("dominant_loss") or "unknown")
            by_loss[dom] = by_loss.get(dom, 0.0) + safe_float(row.get("loss_total"))
        dominant = max(by_loss, key=by_loss.get) if by_loss else "unknown"
        lines = [
            f"view={first.get('view')} frame={first.get('frame')} targets={len(rows)}",
            f"dominant={dominant} total={total:.4g}",
        ]
        for row in rows[: max(1, int(video_cfg.get("loss_panel_max_targets", 3)))]:
            lines.append(
                f"id={row.get('track_id')} dom={row.get('dominant_loss')} "
                f"total={fmt(row,'loss_total')} contain={fmt(row,'loss_mask_contain')} "
                f"edge={fmt(row,'loss_edge')} ground={fmt(row,'loss_ground')}"
            )
        x, y = 12, 12
        width = min(image.shape[1] - 24, int(video_cfg.get("loss_panel_width", 900)))
        height = line_height * len(lines) + 16
        overlay = image.copy()
        cv2.rectangle(overlay, (x - 6, y - 4), (x + width, y + height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.58, image, 0.42, 0, image)
        for idx, text in enumerate(lines):
            cv2.putText(image, text, (x, y + line_height * (idx + 1)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return
    lines = [
        f"view={first.get('view')} frame={first.get('frame')} timestamp={first.get('timestamp')} targets={len(rows)}",
        f"SUM total={total:.4g} edge={edge:.4g} top_bottom={top_bottom_edges:.4g} bbox={bbox_fit:.4g} contain={contain:.4g} depth={depth_safety:.4g} center_depth={center_depth_safety:.4g} ego_box={ego_box_safety:.4g}",
        f"SUM priors height_depth={height_depth_prior:.4g} init_depth={initial_depth_prior:.4g} size_ratio={size_ratio_prior:.4g} class_size={class_size_prior:.4g} ground={ground:.4g}",
        f"SUM temporal planar_acc={temporal_acc:.4g} vertical_acc={temporal_vertical_acc:.4g} log_depth_acc={temporal_log_depth_acc:.4g} over={oversize:.4g}",
        f"SUM oversize: raw={oversize_unweighted:.4g} weighted={oversize_weighted:.4g} final={oversize_final:.4g}",
    ]
    max_lines = max(3, (image.shape[0] - 40) // line_height)
    shown_targets = min(len(rows), max(0, (max_lines - 5) // 9))
    for row in rows[:shown_targets]:
        lines.append(
            f"id={row.get('track_id')} cls={row.get('class')} total={fmt(row,'loss_total')} "
            f"edge={fmt(row,'loss_edge')} top_bottom={fmt(row,'loss_top_bottom_edges')} bbox={fmt(row,'loss_bbox_fit')} contain={fmt(row,'loss_mask_contain')} dom={row.get('dominant_loss')}"
        )
        lines.append(
            f"  safety/prior depth={fmt(row,'loss_depth_safety')} center_depth={fmt(row,'loss_center_depth_safety')} "
            f"ego_box={fmt(row,'loss_ego_box_safety')} init_depth={fmt(row,'loss_initial_depth_prior')} height_depth={fmt(row,'loss_height_depth_prior')} "
            f"size_ratio={fmt(row,'loss_size_ratio_prior')} class_size={fmt(row,'loss_class_size_prior')} ground={fmt(row,'loss_ground')}"
        )
        lines.append(
            f"  ego_box clear_min={fmt(row,'ego_box_safety_min_clearance_m')} penetration={fmt(row,'ego_box_safety_penetration_m')} active={fmt(row,'ego_box_safety_active')}"
        )
        lines.append(
            f"  temporal planar_acc={fmt(row,'loss_temporal_acceleration')} vertical_acc={fmt(row,'loss_temporal_vertical_acceleration')} "
            f"log_depth_acc={fmt(row,'loss_temporal_log_depth_acceleration')} over={fmt(row,'loss_mask_oversize_final')}"
        )
        lines.append(
            f"  height_depth z_prior={fmt(row,'height_depth_prior_z')} h_prior={fmt(row,'height_depth_prior_height_m')} "
            f"logdev={fmt(row,'height_depth_prior_log_deviation')} excess={fmt(row,'height_depth_prior_excess')} active={fmt(row,'height_depth_prior_active')}"
        )
        lines.append(
            f"  class_size target=[{fmt(row,'class_size_prior_target_length')},{fmt(row,'class_size_prior_target_width')},{fmt(row,'class_size_prior_target_height')}] "
            f"logdev_max={fmt(row,'class_size_prior_log_deviation_max')} excess_max={fmt(row,'class_size_prior_excess_max')}"
        )
        lines.append(
            f"  ground final={fmt(row,'loss_ground')} raw={fmt(row,'loss_ground_raw')} decay={fmt(row,'ground_distance_decay_multiplier')} dist={fmt(row,'ground_distance_for_decay_m')} mean_d={fmt(row,'ground_distance_mean')} max_abs_d={fmt(row,'ground_distance_abs_max')} h={fmt(row,'ground_camera_height_m')} "
            f"n=({fmt(row,'ground_normal_cam_x')},{fmt(row,'ground_normal_cam_y')},{fmt(row,'ground_normal_cam_z')})"
        )
        lines.append(
            f"  oversize ratio={fmt(row,'mask_area_ratio')} max={fmt(row,'mask_oversize_max_area_ratio')} excess={fmt(row,'mask_oversize_excess')} "
            f"raw={fmt(row,'loss_mask_oversize_unweighted')} *w={fmt(row,'mask_oversize_weight')} => weighted={fmt(row,'loss_mask_oversize_weighted')} "
            f"*size_w={fmt(row,'size_weight')} => final={fmt(row,'loss_mask_oversize_final')}"
        )
        lines.append(
            f"  areas used={fmt(row,'pred_area')} clipped={fmt(row,'pred_clipped_area')} full={fmt(row,'pred_full_area')} "
            f"mask={fmt(row,'mask_area')} bbox_diag_area={fmt(row,'pred_bbox_area')} mask_pts={fmt(row,'mask_point_count')}"
        )
        lines.append(
            f"  trunc L/R/T/B={row.get('truncated_left')}/{row.get('truncated_right')}/{row.get('truncated_top')}/{row.get('truncated_bottom')} "
            f"geom_clip={row.get('geometry_uses_clipped_projection')} mask={row.get('has_mask')}"
        )
    remaining = len(rows) - shown_targets
    if remaining > 0:
        lines.append(f"... {remaining} more targets; full per-target losses in frame_loss_diagnostics.csv")
    x, y = 12, 12
    width = min(image.shape[1] - 24, int(video_cfg.get("loss_panel_width", 1500)))
    height = line_height * len(lines) + 16
    overlay = image.copy()
    cv2.rectangle(overlay, (x - 6, y - 4), (x + width, y + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, image, 0.42, 0, image)
    for idx, text in enumerate(lines):
        cv2.putText(image, text, (x, y + line_height * (idx + 1)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def append_bev_panel(config: dict[str, Any], image: np.ndarray, rows: list[dict[str, str]], cv2, video_cfg: dict[str, Any]) -> np.ndarray:
    panel_w = int(video_cfg.get("bev_width_px", 560))
    panel_w = max(320, panel_w)
    h = image.shape[0]
    panel = np.full((h, panel_w, 3), 24, dtype=np.uint8)
    draw_bev_panel(config, panel, rows, cv2, video_cfg)
    return np.concatenate([image, panel], axis=1)


def draw_bev_panel(config: dict[str, Any], panel: np.ndarray, rows: list[dict[str, str]], cv2, video_cfg: dict[str, Any]) -> None:
    h, w = panel.shape[:2]
    margin = 34
    footprint_points = []
    centers = []
    for row in rows:
        cx = safe_float(row.get("bev_center_x"))
        cz = safe_float(row.get("bev_center_z"))
        if np.isfinite([cx, cz]).all():
            centers.append((cx, cz))
        pts = bev_points(row)
        if pts is not None:
            footprint_points.extend([(float(x), float(z)) for x, z in pts])
    ego_pts, ego_margin_pts = ego_box_bev_points(config, video_cfg)
    ego_point_list = []
    if ego_pts is not None:
        ego_point_list.extend([(float(x), float(z)) for x, z in ego_pts])
    if ego_margin_pts is not None:
        ego_point_list.extend([(float(x), float(z)) for x, z in ego_margin_pts])
    all_points = centers + footprint_points + ego_point_list
    if all_points:
        max_z_data = max([z for _, z in all_points] + [1.0])
        max_abs_x_data = max([abs(x) for x, _ in all_points] + [5.0])
    else:
        max_z_data = 60.0
        max_abs_x_data = 12.0
    min_z = float(video_cfg.get("bev_min_depth_m", -5.0))
    max_z = max(float(video_cfg.get("bev_max_depth_m", 0) or 0), max_z_data + 8.0, 40.0)
    depth_span = max(max_z - min_z, 1.0)
    max_abs_x = max(float(video_cfg.get("bev_half_width_m", 0) or 0), max_abs_x_data + 3.0, 8.0)

    def to_px(x: float, z: float) -> tuple[int, int]:
        u = margin + (x + max_abs_x) / (2.0 * max_abs_x) * max(w - 2 * margin, 1)
        v = h - margin - (z - min_z) / depth_span * max(h - 2 * margin, 1)
        return int(round(u)), int(round(v))

    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), (70, 70, 70), 1, cv2.LINE_AA)
    cv2.putText(panel, "BEV rear-reference x-z", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"x +/-{max_abs_x:.0f}m  z {min_z:.0f}..{max_z:.0f}m", (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1, cv2.LINE_AA)

    grid_start = np.ceil(min_z / 10.0) * 10.0
    for z in np.arange(grid_start, max_z + 1.0e-6, 10.0):
        p1 = to_px(-max_abs_x, z)
        p2 = to_px(max_abs_x, z)
        cv2.line(panel, p1, p2, (55, 55, 55), 1, cv2.LINE_AA)
        cv2.putText(panel, f"{int(z)}m", (p1[0] + 4, p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 130), 1, cv2.LINE_AA)
    cv2.line(panel, to_px(0.0, min_z), to_px(0.0, max_z), (85, 85, 85), 1, cv2.LINE_AA)
    cam = to_px(0.0, 0.0)
    cv2.circle(panel, cam, 7, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(panel, "rear ref", (cam[0] + 8, cam[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    if ego_margin_pts is not None:
        margin_poly = np.asarray([to_px(float(x), float(z)) for x, z in ego_margin_pts], dtype=np.int32)
        cv2.polylines(panel, [margin_poly], True, (60, 60, 210), 1, cv2.LINE_AA)
        cv2.putText(panel, "ego + margin", tuple(margin_poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (100, 100, 255), 1, cv2.LINE_AA)
    if ego_pts is not None:
        ego_poly = np.asarray([to_px(float(x), float(z)) for x, z in ego_pts], dtype=np.int32)
        cv2.polylines(panel, [ego_poly], True, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(panel, "ego box", tuple(ego_poly[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

    for row in rows:
        pts = bev_points(row)
        track_id = int(float(row.get("track_id", 0) or 0))
        color = color_for_track(track_id, "3d")
        if pts is not None:
            poly = np.asarray([to_px(float(x), float(z)) for x, z in pts], dtype=np.int32)
            cv2.polylines(panel, [poly], True, color, 2, cv2.LINE_AA)
            nose = ((pts[0] + pts[1]) * 0.5 + (pts[2] + pts[3]) * 0.5) * 0.5
            cv2.circle(panel, to_px(float(nose[0]), float(nose[1])), 3, color, -1, cv2.LINE_AA)
        cx = safe_float(row.get("bev_center_x"))
        cz = safe_float(row.get("bev_center_z"))
        if np.isfinite([cx, cz]).all():
            p = to_px(cx, cz)
            cv2.circle(panel, p, 5, (0, 255, 255), -1, cv2.LINE_AA)
            label = f"id={row.get('track_id')} z={cz:.1f}m"
            cv2.putText(panel, label, (p[0] + 7, p[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)


def bev_points(row: dict[str, str]) -> np.ndarray | None:
    pts = []
    for idx in range(4):
        x = safe_float(row.get(f"bev_corner{idx}_x", "nan"))
        z = safe_float(row.get(f"bev_corner{idx}_z", "nan"))
        pts.append((x, z))
    arr = np.asarray(pts, dtype=np.float64)
    if not np.isfinite(arr).all():
        return None
    return arr


def ego_box_bev_points(config: dict[str, Any], video_cfg: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    ego_cfg = dict(config.get("observations", {}).get("ego_box_safety", {}) or {})
    ego_cfg.update(video_cfg.get("ego_box", {}) or {})
    if not bool(video_cfg.get("draw_ego_box", True)):
        return None, None
    try:
        yaw_cfg = config.get("variables", {}).get("yaw", {})
        reference_view = str(yaw_cfg.get("reference_view", "rear"))
        views = config.get("inputs", {}).get("views", {})
        reference_camera_to_ego = load_extrinsic(resolve_path(config, views[reference_view]["camera_to_ego"]))
        ego_to_reference_camera = np.linalg.inv(reference_camera_to_ego)
        axes_ego = car_axes_from_orientation_camera(reference_camera_to_ego)
        ego_size = np.asarray(ego_cfg.get("ego_size", [4.8, 2.0, 1.6]), dtype=np.float64)
        ego_center = np.asarray(ego_cfg.get("ego_center", [0.0, 0.0, 0.8]), dtype=np.float64)
        margin = float(ego_cfg.get("margin_m", 0.2))
        base = ego_footprint_in_reference_camera(ego_center, ego_size, axes_ego, ego_to_reference_camera)
        margin_size = ego_size.copy()
        margin_size[:2] = margin_size[:2] + 2.0 * margin
        expanded = ego_footprint_in_reference_camera(ego_center, margin_size, axes_ego, ego_to_reference_camera)
        return base, expanded
    except Exception:
        return None, None


def ego_footprint_in_reference_camera(
    ego_center: np.ndarray,
    ego_size: np.ndarray,
    axes_ego: np.ndarray,
    ego_to_reference_camera: np.ndarray,
) -> np.ndarray:
    signs = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
    pts = []
    for sl, sw in signs:
        point_ego = ego_center + 0.5 * sl * ego_size[0] * axes_ego[0] + 0.5 * sw * ego_size[1] * axes_ego[1]
        point_ref = transform_point_np(point_ego, ego_to_reference_camera)
        pts.append((float(point_ref[0]), float(point_ref[2])))
    return np.asarray(pts, dtype=np.float64)


def transform_point_np(point: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homo = np.concatenate([np.asarray(point, dtype=np.float64), np.ones(1, dtype=np.float64)])
    return (transform @ homo)[:3]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def group_rows(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(row.get(key, ""), []).append(row)
    return out


def group_frame_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    out: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault((row.get("frame", ""), row.get("image", "")), []).append(row)
    for group in out.values():
        group.sort(key=lambda r: int(float(r.get("track_id", 0))))
    return out


def color_for_track(track_id: int, kind: str) -> tuple[int, int, int]:
    base = np.asarray(
        [
            80 + (track_id * 37) % 175,
            80 + (track_id * 67) % 175,
            80 + (track_id * 97) % 175,
        ],
        dtype=np.uint8,
    )
    if kind == "2d":
        return int(base[0]), int(base[1]), int(base[2])
    return int(base[2]), int(base[1]), int(base[0])


def resolve_data_path(config: dict[str, Any], value: str) -> Path:
    if not value:
        return Path("")
    return resolve_path(config, value.replace("\\", "/"))


def safe_float(value: str | None) -> float:
    try:
        return float(value) if value not in (None, "") else float("nan")
    except ValueError:
        return float("nan")


def fmt(row: dict[str, str], key: str) -> str:
    value = safe_float(row.get(key))
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4g}"
