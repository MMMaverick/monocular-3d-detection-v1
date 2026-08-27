#!/usr/bin/env python
"""Robustly smooth world-frame 3D tracks and visualize motion state.

This script is intentionally dependency-light: it only needs numpy and cv2.
It reads:
  1) a per-frame 3D box CSV with world_x/world_y/world_z columns;
  2) frame_loss_diagnostics.csv for image paths, projected 3D corners and 2D boxes.

The motion label is frame-wise:
  moving  if smoothed world-plane displacement over about 1s > threshold_m_per_s
  static  otherwise
  uncertain if the track is too short or still has large residual jumps
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


VIEW_TO_GT_CAMERA = {
    "center_front": "center_camera_fov120",
    "left_front": "left_front_camera",
    "right_front": "right_front_camera",
    "rear": "rear_camera",
    "left_rear": "left_rear_camera",
    "right_rear": "right_rear_camera",
    "camera_0_front": "center_camera_fov120",
    "camera_1_front_left": "left_front_camera",
    "camera_2_front_right": "right_front_camera",
}


@dataclass
class AnnotationFrame:
    ego_to_global_rotation: np.ndarray
    ego_to_global_translation: np.ndarray
    cams: dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser(description="Smooth world-frame 3D tracks and render speed/state videos.")
    parser.add_argument("--world-csv", required=True, help="CSV containing world_x/world_y/world_z per 3D box.")
    parser.add_argument("--diagnostics-csv", required=True, help="frame_loss_diagnostics.csv containing image/projection columns.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--annotations", default="", help="Optional annotation JSON dir. Used to convert cx/cy/cz camera centers to global/world coordinates.")
    parser.add_argument("--views", nargs="+", default=[], help="Optional view filter.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--speed-window-s", type=float, default=1.0)
    parser.add_argument("--moving-threshold-mps", type=float, default=2.0)
    parser.add_argument("--motion-policy", choices=["majority", "ratio", "median", "any"], default="ratio")
    parser.add_argument("--moving-ratio-threshold", type=float, default=0.30)
    parser.add_argument("--static-preference-ratio-threshold", type=float, default=0.10)
    parser.add_argument("--near-distance-m", type=float, default=20.0)
    parser.add_argument("--far-distance-m", type=float, default=50.0)
    parser.add_argument("--smooth-window", type=int, default=7)
    parser.add_argument("--hampel-window", type=int, default=5)
    parser.add_argument("--hampel-sigma", type=float, default=3.0)
    parser.add_argument("--max-residual-step-m", type=float, default=3.0)
    parser.add_argument("--min-track-frames", type=int, default=10)
    parser.add_argument("--min-track-duration-s", type=float, default=1.0)
    parser.add_argument("--repo-root", default="", help="Optional root used to resolve relative image paths.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(args.repo_root) if args.repo_root else Path.cwd()

    world_rows = read_csv(Path(args.world_csv))
    diag_rows = read_csv(Path(args.diagnostics_csv))
    annotations = load_annotations(Path(args.annotations)) if args.annotations else None
    views = set(args.views) if args.views else None

    motion_rows, track_summary = smooth_and_label(
        world_rows,
        annotations=annotations,
        views=views,
        fps=args.fps,
        speed_window_s=args.speed_window_s,
        moving_threshold_mps=args.moving_threshold_mps,
        motion_policy=args.motion_policy,
        moving_ratio_threshold=args.moving_ratio_threshold,
        static_preference_ratio_threshold=args.static_preference_ratio_threshold,
        near_distance_m=args.near_distance_m,
        far_distance_m=args.far_distance_m,
        smooth_window=args.smooth_window,
        hampel_window=args.hampel_window,
        hampel_sigma=args.hampel_sigma,
        max_residual_step_m=args.max_residual_step_m,
        min_track_frames=args.min_track_frames,
        min_track_duration_s=args.min_track_duration_s,
    )

    motion_csv = out_dir / "frame_world_motion_smoothed.csv"
    summary_csv = out_dir / "track_world_motion_summary_smoothed.csv"
    write_csv(motion_csv, motion_rows)
    write_csv(summary_csv, track_summary)

    motion_lookup = {
        (row["view"], int_float(row["track_id"]), int_float(row["frame"])): row
        for row in motion_rows
    }
    render_videos(
        diag_rows,
        motion_lookup,
        out_dir / "videos",
        fps=args.fps,
        repo_root=repo_root,
    )

    print(f"motion_csv={motion_csv}")
    print(f"summary_csv={summary_csv}")
    print(f"videos={out_dir / 'videos'}")
    print("states=", dict(Counter(row["motion_state"] for row in track_summary)))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def smooth_and_label(
    rows: list[dict[str, str]],
    *,
    annotations: list[AnnotationFrame] | None,
    views: set[str] | None,
    fps: float,
    speed_window_s: float,
    moving_threshold_mps: float,
    motion_policy: str,
    moving_ratio_threshold: float,
    static_preference_ratio_threshold: float,
    near_distance_m: float,
    far_distance_m: float,
    smooth_window: int,
    hampel_window: int,
    hampel_sigma: float,
    max_residual_step_m: float,
    min_track_frames: int,
    min_track_duration_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if views is not None and row.get("view") not in views:
            continue
        grouped[(row["view"], int_float(row["track_id"]))].append(row)

    frame_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for (view, track_id), items in sorted(grouped.items()):
        items.sort(key=lambda r: (int_float(r["frame"]), int_float(r["timestamp"])))
        ts = np.asarray([timestamp_seconds(r, fps) for r in items], dtype=np.float64)
        point_records = [extract_motion_point(r, annotations) for r in items]
        valid_points = [p for p in point_records if p is not None]
        if len(valid_points) != len(point_records):
            continue
        pts = np.asarray([p["motion_point"] for p in point_records], dtype=np.float64)
        distance_m = np.asarray([p["distance_m"] for p in point_records], dtype=np.float64)
        depth_m = np.asarray([p["depth_m"] for p in point_records], dtype=np.float64)
        coord_source = Counter(str(p["coordinate_source"]) for p in valid_points).most_common(1)[0][0]
        if len(items) == 0:
            continue

        filtered = pts.copy()
        outlier_mask = np.zeros(len(items), dtype=bool)
        for dim in range(3):
            filtered[:, dim], dim_outliers = hampel_filter_1d(
                filtered[:, dim], window=hampel_window, sigma=hampel_sigma
            )
            outlier_mask |= dim_outliers
        smoothed = triangular_smooth(filtered, window=smooth_window)

        speeds = one_second_speeds(ts, smoothed[:, :2], speed_window_s)
        steps = np.linalg.norm(np.diff(smoothed[:, :2], axis=0), axis=1) if len(items) > 1 else np.asarray([])
        duration = float(ts[-1] - ts[0]) if len(items) > 1 else 0.0
        displacement = float(np.linalg.norm(smoothed[-1, :2] - smoothed[0, :2])) if len(items) > 1 else 0.0
        path_len = float(np.sum(steps)) if len(steps) else 0.0
        max_step = float(np.max(steps)) if len(steps) else 0.0
        jitter = path_len / max(displacement, 1e-6) if len(items) > 2 else 1.0
        track_usable = len(items) >= min_track_frames and duration >= min_track_duration_s and max_step <= max_residual_step_m

        frame_states = []
        for speed in speeds:
            if not track_usable or not np.isfinite(speed):
                state = "uncertain"
            elif speed > moving_threshold_mps:
                state = "moving"
            else:
                state = "static"
            frame_states.append(state)

        counts = Counter(frame_states)
        valid_speeds = speeds[np.isfinite(speeds)]
        moving_ratio = float(np.mean(valid_speeds > moving_threshold_mps)) if len(valid_speeds) else float("nan")
        median_speed = float(np.nanmedian(speeds)) if len(speeds) else float("nan")
        p90_speed = float(np.nanpercentile(speeds, 90)) if len(speeds) else float("nan")
        max_speed = float(np.nanmax(speeds)) if np.isfinite(speeds).any() else float("nan")
        if not track_usable:
            track_state = "uncertain"
            state_reason = "too_short_or_large_residual_jump"
        elif motion_policy == "any":
            track_state = "moving" if max_speed > moving_threshold_mps else "static"
            state_reason = "any_speed_over_threshold"
        elif motion_policy == "median":
            track_state = "moving" if median_speed > moving_threshold_mps else "static"
            state_reason = "median_speed_over_threshold"
        elif motion_policy == "ratio":
            track_state = "moving" if moving_ratio >= moving_ratio_threshold else "static"
            state_reason = "moving_ratio_over_threshold"
        else:
            track_state = "moving" if counts["moving"] >= max(1, counts["static"]) else "static"
            state_reason = "moving_frames_at_least_static_frames"

        preference, confidence, moving_probability, static_probability = motion_preference_confidence(
            track_state=track_state,
            track_usable=track_usable,
            moving_ratio=moving_ratio,
            moving_ratio_threshold=moving_ratio_threshold,
            static_preference_ratio_threshold=static_preference_ratio_threshold,
            median_distance_m=float(np.nanmedian(distance_m)),
            near_distance_m=near_distance_m,
            far_distance_m=far_distance_m,
        )

        labels = [r.get("class") or r.get("label") or "" for r in items]
        majority_class = Counter(labels).most_common(1)[0][0] if labels else ""

        for idx, row in enumerate(items):
            out = dict(row)
            out.update(
                {
                    "world_x_raw": f"{pts[idx, 0]:.6f}",
                    "world_y_raw": f"{pts[idx, 1]:.6f}",
                    "world_z_raw": f"{pts[idx, 2]:.6f}",
                    "world_x_smooth": f"{smoothed[idx, 0]:.6f}",
                    "world_y_smooth": f"{smoothed[idx, 1]:.6f}",
                    "world_z_smooth": f"{smoothed[idx, 2]:.6f}",
                    "distance_to_ego_m": f"{distance_m[idx]:.6f}" if np.isfinite(distance_m[idx]) else "",
                    "depth_m": f"{depth_m[idx]:.6f}" if np.isfinite(depth_m[idx]) else "",
                    "speed_1s_mps": f"{speeds[idx]:.6f}" if np.isfinite(speeds[idx]) else "",
                    "motion_state": frame_states[idx],
                    "track_motion_state": track_state,
                    "track_motion_preference": preference,
                    "track_motion_confidence": fmt_float(confidence),
                    "track_moving_probability": fmt_float(moving_probability),
                    "track_static_probability": fmt_float(static_probability),
                    "track_usable_for_motion": str(bool(track_usable)),
                    "motion_outlier_replaced": str(bool(outlier_mask[idx])),
                }
            )
            frame_rows.append(out)

        summaries.append(
            {
                "view": view,
                "track_id": track_id,
                "class": majority_class,
                "num_frames": len(items),
                "first_frame": int_float(items[0].get("frame", -1)),
                "last_frame": int_float(items[-1].get("frame", -1)),
                "duration_s": f"{duration:.3f}",
                "coordinate_source": coord_source,
                "distance_to_ego_min_m": fmt_float(np.nanmin(distance_m)),
                "distance_to_ego_median_m": fmt_float(np.nanmedian(distance_m)),
                "distance_to_ego_max_m": fmt_float(np.nanmax(distance_m)),
                "distance_to_ego_first_m": fmt_float(distance_m[0]),
                "distance_to_ego_last_m": fmt_float(distance_m[-1]),
                "depth_min_m": fmt_float(np.nanmin(depth_m)),
                "depth_median_m": fmt_float(np.nanmedian(depth_m)),
                "depth_max_m": fmt_float(np.nanmax(depth_m)),
                "world_displacement_xy_m": f"{displacement:.6f}",
                "world_path_xy_m": f"{path_len:.6f}",
                "median_speed_1s_mps": f"{median_speed:.6f}" if np.isfinite(median_speed) else "",
                "p90_speed_1s_mps": f"{p90_speed:.6f}" if np.isfinite(p90_speed) else "",
                "max_speed_1s_mps": f"{max_speed:.6f}" if np.isfinite(max_speed) else "",
                "moving_ratio": f"{moving_ratio:.6f}" if np.isfinite(moving_ratio) else "",
                "max_smoothed_step_m": f"{max_step:.6f}",
                "jitter_ratio": f"{jitter:.6f}",
                "outlier_frames": int(outlier_mask.sum()),
                "usable_for_motion": str(bool(track_usable)),
                "motion_state": track_state,
                "motion_preference": preference,
                "motion_confidence": fmt_float(confidence),
                "moving_probability": fmt_float(moving_probability),
                "static_probability": fmt_float(static_probability),
                "motion_state_reason": state_reason,
                "moving_threshold_mps": moving_threshold_mps,
                "motion_policy": motion_policy,
                "moving_ratio_threshold": moving_ratio_threshold,
                "static_preference_ratio_threshold": static_preference_ratio_threshold,
                "near_distance_m": near_distance_m,
                "far_distance_m": far_distance_m,
                "speed_window_s": speed_window_s,
            }
        )

    frame_rows.sort(key=lambda r: (r["view"], int_float(r["frame"]), int_float(r["track_id"])))
    summaries.sort(key=lambda r: (r["view"], int_float(r["track_id"])))
    return frame_rows, summaries


def hampel_filter_1d(values: np.ndarray, *, window: int, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    if window < 3 or len(values) < 3:
        return values.copy(), np.zeros(len(values), dtype=bool)
    half = window // 2
    out = values.copy()
    mask = np.zeros(len(values), dtype=bool)
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        local = values[lo:hi]
        med = float(np.median(local))
        mad = float(np.median(np.abs(local - med)))
        scale = 1.4826 * mad
        if scale < 1e-6:
            continue
        if abs(values[i] - med) > sigma * scale:
            out[i] = med
            mask[i] = True
    return out, mask


def triangular_smooth(values: np.ndarray, *, window: int) -> np.ndarray:
    if window < 3 or len(values) < 3:
        return values.copy()
    window = int(window)
    if window % 2 == 0:
        window += 1
    half = window // 2
    weights = np.asarray([half + 1 - abs(i - half) for i in range(window)], dtype=np.float64)
    out = np.empty_like(values)
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        w_lo = half - (i - lo)
        w_hi = w_lo + (hi - lo)
        w = weights[w_lo:w_hi]
        w = w / np.sum(w)
        out[i] = np.sum(values[lo:hi] * w[:, None], axis=0)
    return out


def one_second_speeds(ts: np.ndarray, xy: np.ndarray, window_s: float) -> np.ndarray:
    speeds = np.full(len(ts), np.nan, dtype=np.float64)
    if len(ts) < 2:
        return speeds
    for i in range(len(ts)):
        target = ts[i] - window_s
        prev_candidates = np.where(ts <= target)[0]
        if len(prev_candidates):
            j = int(prev_candidates[-1])
        else:
            # Early frames: use the farthest available future point up to the same window.
            future = np.where(ts >= ts[i] + min(window_s, ts[-1] - ts[i]))[0]
            if len(future):
                j = int(future[0])
            else:
                j = 0 if i != 0 else min(1, len(ts) - 1)
        dt = abs(float(ts[i] - ts[j]))
        if dt <= 1e-6:
            continue
        speeds[i] = float(np.linalg.norm(xy[i] - xy[j]) / dt)
    return speeds


def motion_preference_confidence(
    *,
    track_state: str,
    track_usable: bool,
    moving_ratio: float,
    moving_ratio_threshold: float,
    static_preference_ratio_threshold: float,
    median_distance_m: float,
    near_distance_m: float,
    far_distance_m: float,
) -> tuple[str, float, float, float]:
    """Return practical preference, confidence, and normalized probabilities.

    `motion_state` remains the hard state.  For downstream use, uncertain
    tracks still receive a softer `motion_preference` when the moving-ratio
    evidence is clearly leaning to one side.

    The probability pair is deliberately simple and normalized:
      moving_probability = moving_ratio
      static_probability = 1 - moving_ratio
    It is an evidence score from speed samples, not a calibrated Bayesian
    probability of the object truly moving.
    """
    if not np.isfinite(moving_ratio):
        moving_probability = 0.5
        static_probability = 0.5
        preference = "unknown"
        confidence = 0.0
        return preference, confidence, moving_probability, static_probability

    moving_probability = float(np.clip(moving_ratio, 0.0, 1.0))
    static_probability = 1.0 - moving_probability

    if track_state in {"moving", "static"}:
        preference = track_state
    elif moving_ratio >= moving_ratio_threshold:
        preference = "moving"
    elif moving_ratio <= static_preference_ratio_threshold:
        preference = "static"
    else:
        preference = "unknown"

    reliability = 1.0 if track_usable else 0.5
    denom = max(float(moving_ratio_threshold), 1.0e-6)
    ratio_certainty = float(np.clip(abs(moving_ratio - moving_ratio_threshold) / denom, 0.0, 1.0))

    if not np.isfinite(median_distance_m):
        distance_reliability = 0.5
    elif median_distance_m < near_distance_m:
        distance_reliability = 1.0
    elif median_distance_m < far_distance_m:
        distance_reliability = 0.7
    else:
        distance_reliability = 0.4

    confidence = float(np.clip(reliability * ratio_certainty * distance_reliability, 0.0, 1.0))
    if preference == "unknown":
        confidence *= 0.5
    return preference, confidence, moving_probability, static_probability


def timestamp_seconds(row: dict[str, str], fps: float) -> float:
    value = row.get("timestamp", "")
    try:
        ts = float(value)
        if ts > 1e12:
            return ts / 1e9
        if ts > 1e6:
            return ts / 1e6
        return ts
    except Exception:
        return int_float(row.get("frame", 0)) / max(float(fps), 1e-6)


def extract_motion_point(row: dict[str, str], annotations: list[AnnotationFrame] | None) -> dict[str, Any] | None:
    if row.get("world_x", "") != "" and row.get("world_y", "") != "":
        point = np.asarray(
            [
                safe_float(row.get("world_x", "nan")),
                safe_float(row.get("world_y", "nan")),
                safe_float(row.get("world_z", "0")),
            ],
            dtype=np.float64,
        )
        cam = np.asarray(
            [
                safe_float(row.get("cx", "nan")),
                safe_float(row.get("cy", "nan")),
                safe_float(row.get("cz", "nan")),
            ],
            dtype=np.float64,
        )
        return {
            "motion_point": point,
            "distance_m": float(np.linalg.norm(cam)) if np.isfinite(cam).all() else float("nan"),
            "depth_m": float(cam[2]) if np.isfinite(cam).all() else float("nan"),
            "coordinate_source": "world_csv",
        }

    cam = np.asarray(
        [
            safe_float(row.get("cx", "nan")),
            safe_float(row.get("cy", "nan")),
            safe_float(row.get("cz", "nan")),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(cam).all():
        return None
    distance_m = float(np.linalg.norm(cam))
    depth_m = float(cam[2])

    if annotations is not None:
        frame = int_float(row.get("frame", -1))
        if 0 <= frame < len(annotations):
            ann = annotations[frame]
            gt_camera = VIEW_TO_GT_CAMERA.get(row.get("view", ""))
            cam_meta = ann.cams.get(gt_camera, {}) if gt_camera else {}
            extrinsic = cam_meta.get("extrinsic")
            if extrinsic is not None:
                ego_to_camera = np.asarray(extrinsic, dtype=np.float64).reshape(4, 4)
                point_ego = camera_to_ego(cam, ego_to_camera)
                point_world = ann.ego_to_global_rotation @ point_ego + ann.ego_to_global_translation
                return {
                    "motion_point": point_world,
                    "distance_m": distance_m,
                    "depth_m": depth_m,
                    "coordinate_source": "camera_to_global_from_annotations",
                }

    # Fallback: camera-local coordinates.  This is useful for visualization but
    # should not be treated as absolute motion if the ego vehicle is moving.
    point_camera_local = np.asarray([cam[0], cam[2], -cam[1]], dtype=np.float64)
    return {
        "motion_point": point_camera_local,
        "distance_m": distance_m,
        "depth_m": depth_m,
        "coordinate_source": "camera_local_fallback",
    }


def load_annotations(annotation_dir: Path) -> list[AnnotationFrame]:
    out: list[AnnotationFrame] = []
    for path in sorted(annotation_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        out.append(
            AnnotationFrame(
                ego_to_global_rotation=quat_to_rot(np.asarray(payload["ego2global_rotation"], dtype=np.float64)),
                ego_to_global_translation=np.asarray(payload["ego2global_translation"], dtype=np.float64),
                cams=payload.get("cams", {}) or {},
            )
        )
    if not out:
        raise FileNotFoundError(f"No annotation JSON found in {annotation_dir}")
    return out


def camera_to_ego(point_cam: np.ndarray, ego_to_camera: np.ndarray) -> np.ndarray:
    homog = np.concatenate([point_cam, np.ones(1, dtype=np.float64)])
    return (np.linalg.inv(ego_to_camera) @ homog)[:3]


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n <= 1e-12:
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


def fmt_float(value: float) -> str:
    return f"{float(value):.6f}" if np.isfinite(value) else ""


def render_videos(
    diag_rows: list[dict[str, str]],
    motion_lookup: dict[tuple[str, int, int], dict[str, Any]],
    out_dir: Path,
    *,
    fps: float,
    repo_root: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_view_frame: dict[str, dict[int, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in diag_rows:
        key = (row["view"], int_float(row["track_id"]), int_float(row["frame"]))
        if key in motion_lookup:
            by_view_frame[row["view"]][int_float(row["frame"])].append(row)

    for view, frames in sorted(by_view_frame.items()):
        writer = None
        video_path = out_dir / f"{view}_world_motion.mp4"
        try:
            for frame_idx in sorted(frames):
                image_path = resolve_path(frames[frame_idx][0].get("image", ""), repo_root)
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                vis = image.copy()
                for row in frames[frame_idx]:
                    motion = motion_lookup[(row["view"], int_float(row["track_id"]), int_float(row["frame"]))]
                    draw_box(vis, row, motion)
                if writer is None:
                    h, w = vis.shape[:2]
                    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                writer.write(vis)
        finally:
            if writer is not None:
                writer.release()
        print(f"video={video_path}")


def draw_box(image: np.ndarray, row: dict[str, str], motion: dict[str, Any]) -> None:
    tid = int_float(row["track_id"])
    state = str(motion.get("motion_state", "uncertain"))
    track_state = str(motion.get("track_motion_state", "uncertain"))
    speed_text = motion.get("speed_1s_mps", "")
    color = (0, 255, 0) if state == "moving" else (255, 180, 0) if state == "static" else (0, 200, 255)

    # 2D box: thin, just as a visual anchor.
    x1, y1, x2, y2 = [int(round(float(row.get(k, 0) or 0))) for k in ("obs_x1", "obs_y1", "obs_x2", "obs_y2")]
    cv2.rectangle(image, (x1, y1), (x2, y2), (160, 160, 255), 1, cv2.LINE_AA)

    corners = []
    for idx in range(8):
        x = row.get(f"corner{idx}_x", "")
        y = row.get(f"corner{idx}_y", "")
        if x == "" or y == "":
            return
        corners.append((float(x), float(y)))
    pts = np.asarray(corners, dtype=np.float32)
    for a, b in EDGES:
        pa = tuple(np.round(pts[a]).astype(int))
        pb = tuple(np.round(pts[b]).astype(int))
        cv2.line(image, pa, pb, color, 2, cv2.LINE_AA)

    valid = pts[np.isfinite(pts).all(axis=1)]
    if len(valid):
        anchor_x = int(np.clip(np.min(valid[:, 0]), 0, image.shape[1] - 1))
        anchor_y = int(np.clip(np.min(valid[:, 1]) - 8, 18, image.shape[0] - 1))
    else:
        anchor_x, anchor_y = x1, max(18, y1 - 8)
    speed = f"{float(speed_text):.1f}m/s" if str(speed_text) != "" else "NA"
    label = f"id={tid} {state} {speed}"
    if track_state != state:
        label += f" trk={track_state}"
    put_text(image, label, anchor_x, anchor_y, color)


def put_text(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(image, (x - 2, y - th - 6), (min(image.shape[1] - 1, x + tw + 4), y + 4), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [
        repo_root / path,
        Path(r"D:\vggt-omega") / path,
        Path(r"D:\vggt-omega") / "data" / path,
        Path(r"D:\vggt-omega\rebuild_data_transfer_package") / path,
        Path(r"D:\vggt-omega\rebuild_data_transfer_package\data") / path,
    ]
    for candidate in candidates:
        try:
            exists = candidate.exists()
        except OSError:
            # D:\mono-detect\data can be a stale junction in the local dev tree.
            # Skip inaccessible candidates and keep searching the real data roots.
            exists = False
        if exists:
            return candidate
    return repo_root / path


def int_float(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return -1


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


if __name__ == "__main__":
    main()
