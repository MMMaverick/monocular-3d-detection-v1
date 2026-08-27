#!/usr/bin/env python
"""Evaluate predicted 3D-track motion states against front-view GT boxes.

This script deliberately does NOT use optical flow.  It evaluates whether the
3D boxes are enough for motion/static judgment:

1. Match each predicted 2D observation to the visible GT 2D box by IoU.
2. Assign each predicted track to the majority matched GT track id.
3. Convert predicted camera-frame 3D centers and GT ego-frame 3D centers to
   global/world coordinates.
4. Classify each track by one-second displacement speed.
5. Report accuracy/confusion and write per-frame/per-track CSVs.
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

import numpy as np


VIEW_TO_GT_CAMERA = {
    "center_front": "center_camera_fov120",
    "left_front": "left_front_camera",
    "right_front": "right_front_camera",
    "camera_0_front": "center_camera_fov120",
    "camera_1_front_left": "left_front_camera",
    "camera_2_front_right": "right_front_camera",
}


@dataclass
class AnnotationFrame:
    frame: int
    timestamp: int
    ego_to_global_rotation: np.ndarray
    ego_to_global_translation: np.ndarray
    gt_boxes: list[list[float]]
    gt_names: list[str]
    gt_track_ids: list[int]
    valid_flags: list[bool]
    cams: dict[str, Any]


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate front-view 3D motion classification with GT boxes.")
    parser.add_argument("--diagnostics-csv", required=True, help="frame_loss_diagnostics.csv from 3D optimization.")
    parser.add_argument("--annotations", required=True, help="Annotation JSON directory with GT 3D boxes.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--views", nargs="+", default=list(VIEW_TO_GT_CAMERA), help="Prediction view names to evaluate.")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--horizon-sec", type=float, default=1.0, help="Time gap used to measure displacement speed.")
    parser.add_argument("--moving-threshold-mps", type=float, default=2.0, help="A speed sample is moving if it exceeds this.")
    parser.add_argument(
        "--pred-motion-policy",
        choices=("median", "ratio", "any"),
        default="median",
        help="How to classify predicted tracks from speed samples.",
    )
    parser.add_argument(
        "--gt-motion-policy",
        choices=("median", "ratio", "any"),
        default="median",
        help="How to classify GT tracks from speed samples. Use 'any' for strict GT: moved once means dynamic.",
    )
    parser.add_argument(
        "--pred-moving-ratio-threshold",
        type=float,
        default=0.30,
        help="For pred-motion-policy=ratio, classify as moving if this fraction of samples are above threshold.",
    )
    parser.add_argument(
        "--gt-moving-ratio-threshold",
        type=float,
        default=0.30,
        help="For gt-motion-policy=ratio, classify as moving if this fraction of samples are above threshold.",
    )
    parser.add_argument("--min-2d-iou", type=float, default=0.30, help="Pred/GT 2D IoU threshold for observation matching.")
    parser.add_argument("--min-matched-frames", type=int, default=3, help="Minimum matched frames to evaluate a pred track.")
    parser.add_argument("--min-motion-samples", type=int, default=2, help="Minimum one-second speed samples for a track state.")
    parser.add_argument("--class-aware-match", action="store_true", help="If set, only match pred/GT boxes in same coarse class group.")
    parser.add_argument("--smooth-window", type=int, default=5, help="Odd median smoothing window for world centers. 1 disables it.")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    annotations = load_annotations(Path(args.annotations))
    pred_rows_all = [r for r in read_csv(Path(args.diagnostics_csv)) if r.get("view") in set(args.views)]

    match_rows: list[dict[str, Any]] = []
    pred_points: dict[tuple[str, int], list[tuple[int, np.ndarray]]] = defaultdict(list)
    gt_points_by_camera: dict[tuple[str, int], list[tuple[int, np.ndarray]]] = defaultdict(list)

    for row in pred_rows_all:
        frame = int_float(row["frame"])
        if frame < 0 or frame >= len(annotations):
            continue
        ann = annotations[frame]
        view = row["view"]
        gt_camera = VIEW_TO_GT_CAMERA.get(view)
        if gt_camera is None:
            continue
        cam = ann.cams.get(gt_camera, {})
        gt2d = cam.get("boxes_2d", []) or []
        gt2d_indices = cam.get("boxes_2d_index", []) or []
        pred_box = read_box(row, ("obs_x1", "obs_y1", "obs_x2", "obs_y2"))

        best = match_gt_2d(
            pred_box,
            str(row.get("class", "")),
            gt2d,
            gt2d_indices,
            ann,
            min_iou=args.min_2d_iou,
            class_aware=args.class_aware_match,
        )
        if best is None:
            continue
        gt2d_idx, gt_index, gt_track_id, gt_name, iou = best

        pred_center_cam = np.asarray([float(row["cx"]), float(row["cy"]), float(row["cz"])], dtype=np.float64)
        pred_center_ego = camera_to_ego(pred_center_cam, np.asarray(cam["extrinsic"], dtype=np.float64).reshape(4, 4))
        pred_center_global = ego_to_global(pred_center_ego, ann)
        pred_key = (view, int_float(row["track_id"]))
        pred_points[pred_key].append((frame, pred_center_global))

        gt_box = ann.gt_boxes[gt_index]
        gt_center_global = ego_to_global(np.asarray(gt_box[:3], dtype=np.float64), ann)
        gt_key = (gt_camera, gt_track_id)
        gt_points_by_camera[gt_key].append((frame, gt_center_global))

        match_rows.append(
            {
                "view": view,
                "gt_camera": gt_camera,
                "frame": frame,
                "pred_track_id": int_float(row["track_id"]),
                "pred_class": row.get("class", ""),
                "gt2d_index": gt2d_idx,
                "gt_index": gt_index,
                "gt_track_id": gt_track_id,
                "gt_name": gt_name,
                "match_2d_iou": f"{iou:.6f}",
                "pred_global_x": f"{pred_center_global[0]:.6f}",
                "pred_global_y": f"{pred_center_global[1]:.6f}",
                "pred_global_z": f"{pred_center_global[2]:.6f}",
                "gt_global_x": f"{gt_center_global[0]:.6f}",
                "gt_global_y": f"{gt_center_global[1]:.6f}",
                "gt_global_z": f"{gt_center_global[2]:.6f}",
            }
        )

    pred_to_gt = majority_gt_for_pred(match_rows, min_frames=args.min_matched_frames)
    pred_states = {
        key: classify_points(
            points,
            fps=args.fps,
            horizon_sec=args.horizon_sec,
            threshold=args.moving_threshold_mps,
            min_samples=args.min_motion_samples,
            smooth_window=args.smooth_window,
            policy=args.pred_motion_policy,
            ratio_threshold=args.pred_moving_ratio_threshold,
        )
        for key, points in pred_points.items()
    }
    gt_states = {
        key: classify_points(
            points,
            fps=args.fps,
            horizon_sec=args.horizon_sec,
            threshold=args.moving_threshold_mps,
            min_samples=args.min_motion_samples,
            smooth_window=args.smooth_window,
            policy=args.gt_motion_policy,
            ratio_threshold=args.gt_moving_ratio_threshold,
        )
        for key, points in gt_points_by_camera.items()
    }

    track_rows: list[dict[str, Any]] = []
    for pred_key, gt_info in sorted(pred_to_gt.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        view, pred_track_id = pred_key
        gt_camera, gt_track_id = gt_info["gt_camera"], gt_info["gt_track_id"]
        pred_state = pred_states.get(pred_key, empty_state())
        gt_state = gt_states.get((gt_camera, gt_track_id), empty_state())
        ok = pred_state["state"] == gt_state["state"] and pred_state["state"] in {"moving", "static"}
        comparable = pred_state["state"] in {"moving", "static"} and gt_state["state"] in {"moving", "static"}
        track_rows.append(
            {
                "view": view,
                "pred_track_id": pred_track_id,
                "gt_camera": gt_camera,
                "gt_track_id": gt_track_id,
                "gt_name": gt_info["gt_name"],
                "matched_frames": gt_info["matched_frames"],
                "matched_iou_median": f"{gt_info['matched_iou_median']:.6f}",
                "pred_state": pred_state["state"],
                "gt_state": gt_state["state"],
                "is_correct": str(bool(ok)),
                "is_comparable": str(bool(comparable)),
                "pred_median_speed_mps": fmt(pred_state["median_speed"]),
                "gt_median_speed_mps": fmt(gt_state["median_speed"]),
                "pred_max_speed_mps": fmt(pred_state["max_speed"]),
                "gt_max_speed_mps": fmt(gt_state["max_speed"]),
                "pred_moving_ratio": fmt(pred_state["moving_ratio"]),
                "gt_moving_ratio": fmt(gt_state["moving_ratio"]),
                "pred_motion_samples": pred_state["num_samples"],
                "gt_motion_samples": gt_state["num_samples"],
            }
        )

    write_csv(out_dir / "frame_pred_gt_matches.csv", match_rows)
    write_csv(out_dir / "track_motion_eval.csv", track_rows)
    summary_text = build_summary(args, track_rows, match_rows, pred_rows_all)
    (out_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print(f"output_dir={out_dir}")
    return 0


def load_annotations(annotation_dir: Path) -> list[AnnotationFrame]:
    out: list[AnnotationFrame] = []
    for frame, path in enumerate(sorted(annotation_dir.glob("*.json"))):
        payload = json.loads(path.read_text(encoding="utf-8"))
        out.append(
            AnnotationFrame(
                frame=frame,
                timestamp=int(payload.get("timestamp", frame)),
                ego_to_global_rotation=quat_to_rot(np.asarray(payload["ego2global_rotation"], dtype=np.float64)),
                ego_to_global_translation=np.asarray(payload["ego2global_translation"], dtype=np.float64),
                gt_boxes=payload.get("gt_boxes", []) or [],
                gt_names=payload.get("gt_names", []) or [],
                gt_track_ids=[int(v) for v in (payload.get("track_id", []) or [])],
                valid_flags=[bool(v) for v in (payload.get("valid_flag", []) or [])],
                cams=payload.get("cams", {}) or {},
            )
        )
    if not out:
        raise FileNotFoundError(f"No annotation JSON found in {annotation_dir}")
    return out


def match_gt_2d(pred_box: np.ndarray, pred_class: str, gt2d: list[Any], gt2d_indices: list[Any], ann: AnnotationFrame, *, min_iou: float, class_aware: bool):
    best = None
    best_iou = -1.0
    pred_group = class_group(pred_class)
    for gt2d_idx, gt_box_raw in enumerate(gt2d):
        if gt_box_raw is None or len(gt_box_raw) != 4:
            continue
        if gt2d_idx >= len(gt2d_indices) or gt2d_indices[gt2d_idx] is None:
            continue
        gt_index = int(gt2d_indices[gt2d_idx])
        if gt_index < 0 or gt_index >= len(ann.gt_boxes) or gt_index >= len(ann.gt_track_ids):
            continue
        if gt_index < len(ann.valid_flags) and not ann.valid_flags[gt_index]:
            continue
        gt_name = ann.gt_names[gt_index] if gt_index < len(ann.gt_names) else "unknown"
        if class_aware and pred_group != "unknown" and class_group(gt_name) != pred_group:
            continue
        iou = box_iou(pred_box, np.asarray(gt_box_raw, dtype=np.float64))
        if iou > best_iou:
            best_iou = iou
            best = (gt2d_idx, gt_index, ann.gt_track_ids[gt_index], gt_name, iou)
    return best if best is not None and best_iou >= min_iou else None


def majority_gt_for_pred(match_rows: list[dict[str, Any]], *, min_frames: int) -> dict[tuple[str, int], dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in match_rows:
        grouped[(str(row["view"]), int(row["pred_track_id"]))].append(row)
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for key, rows in grouped.items():
        counts = Counter((str(r["gt_camera"]), int(r["gt_track_id"]), str(r["gt_name"])) for r in rows)
        (gt_camera, gt_track_id, gt_name), count = counts.most_common(1)[0]
        if count < min_frames:
            continue
        ious = [float(r["match_2d_iou"]) for r in rows if str(r["gt_camera"]) == gt_camera and int(r["gt_track_id"]) == gt_track_id]
        out[key] = {
            "gt_camera": gt_camera,
            "gt_track_id": gt_track_id,
            "gt_name": gt_name,
            "matched_frames": count,
            "matched_iou_median": float(np.median(ious)) if ious else float("nan"),
        }
    return out


def classify_points(
    points: list[tuple[int, np.ndarray]],
    *,
    fps: float,
    horizon_sec: float,
    threshold: float,
    min_samples: int,
    smooth_window: int,
    policy: str,
    ratio_threshold: float,
) -> dict[str, Any]:
    if len(points) < 2:
        return empty_state()
    by_frame: dict[int, np.ndarray] = {}
    for frame, point in points:
        by_frame[int(frame)] = np.asarray(point, dtype=np.float64)
    frames = sorted(by_frame)
    centers = np.stack([by_frame[f] for f in frames], axis=0)
    centers = median_smooth(centers, smooth_window)
    by_frame = {f: centers[i] for i, f in enumerate(frames)}
    horizon = max(1, int(round(horizon_sec * fps)))
    speeds: list[float] = []
    for f in frames:
        f2 = f + horizon
        if f2 not in by_frame:
            continue
        dt = (f2 - f) / max(fps, 1e-6)
        disp = float(np.linalg.norm(by_frame[f2][:2] - by_frame[f][:2]))
        speeds.append(disp / max(dt, 1e-6))
    if len(speeds) < min_samples:
        return {
            "state": "uncertain",
            "median_speed": float("nan"),
            "max_speed": float("nan"),
            "moving_ratio": float("nan"),
            "num_samples": len(speeds),
        }
    median_speed = float(np.median(speeds))
    max_speed = float(np.max(speeds))
    moving_ratio = float(np.mean(np.asarray(speeds, dtype=np.float64) > threshold))
    if policy == "any":
        is_moving = max_speed > threshold
    elif policy == "ratio":
        is_moving = moving_ratio >= ratio_threshold
    else:
        is_moving = median_speed > threshold
    return {
        "state": "moving" if is_moving else "static",
        "median_speed": median_speed,
        "max_speed": max_speed,
        "moving_ratio": moving_ratio,
        "num_samples": len(speeds),
    }


def median_smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 3:
        return values
    window = int(window)
    if window % 2 == 0:
        window += 1
    radius = window // 2
    out = values.copy()
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        out[i] = np.median(values[lo:hi], axis=0)
    return out


def build_summary(args: argparse.Namespace, track_rows: list[dict[str, Any]], match_rows: list[dict[str, Any]], pred_rows_all: list[dict[str, str]]) -> str:
    comparable = [r for r in track_rows if r["is_comparable"] == "True"]
    correct = [r for r in comparable if r["is_correct"] == "True"]
    lines = []
    lines.append("3D motion eval against front GT boxes")
    lines.append(f"views={','.join(args.views)}")
    lines.append(f"pred_observations={len(pred_rows_all)} matched_observations={len(match_rows)} match_rate={len(match_rows)/max(len(pred_rows_all),1):.3f}")
    lines.append(f"tracks_evaluated={len(track_rows)} comparable={len(comparable)} correct={len(correct)} accuracy={len(correct)/max(len(comparable),1):.3f}")
    lines.append(
        "params "
        f"fps={args.fps} horizon_sec={args.horizon_sec} moving_threshold_mps={args.moving_threshold_mps} "
        f"pred_motion_policy={args.pred_motion_policy} pred_moving_ratio_threshold={args.pred_moving_ratio_threshold} "
        f"gt_motion_policy={args.gt_motion_policy} gt_moving_ratio_threshold={args.gt_moving_ratio_threshold} "
        f"min_2d_iou={args.min_2d_iou} min_matched_frames={args.min_matched_frames} "
        f"min_motion_samples={args.min_motion_samples} class_aware_match={args.class_aware_match} "
        f"smooth_window={args.smooth_window}"
    )
    lines.append("")
    for view in args.views:
        rows = [r for r in track_rows if r["view"] == view]
        comp = [r for r in rows if r["is_comparable"] == "True"]
        ok = [r for r in comp if r["is_correct"] == "True"]
        lines.append(f"[{view}] tracks={len(rows)} comparable={len(comp)} correct={len(ok)} accuracy={len(ok)/max(len(comp),1):.3f}")
        lines.append(f"  pred_states={dict(Counter(r['pred_state'] for r in rows))}")
        lines.append(f"  gt_states={dict(Counter(r['gt_state'] for r in rows))}")
        lines.append(f"  confusion={dict(Counter((r['gt_state'], r['pred_state']) for r in comp))}")
    lines.append("")
    lines.append("Top mismatches:")
    bad = [r for r in comparable if r["is_correct"] != "True"]
    bad.sort(key=lambda r: abs(float_or_nan(r["pred_median_speed_mps"]) - float_or_nan(r["gt_median_speed_mps"])), reverse=True)
    for r in bad[:20]:
        lines.append(
            f"  {r['view']} pred={r['pred_track_id']} gt={r['gt_track_id']} {r['gt_name']} "
            f"pred={r['pred_state']}(median={r['pred_median_speed_mps']}m/s,max={r['pred_max_speed_mps']}m/s,ratio={r['pred_moving_ratio']}) "
            f"gt={r['gt_state']}(median={r['gt_median_speed_mps']}m/s,max={r['gt_max_speed_mps']}m/s,ratio={r['gt_moving_ratio']}) "
            f"matched={r['matched_frames']} iou={r['matched_iou_median']}"
        )
    return "\n".join(lines)


def empty_state() -> dict[str, Any]:
    return {
        "state": "uncertain",
        "median_speed": float("nan"),
        "max_speed": float("nan"),
        "moving_ratio": float("nan"),
        "num_samples": 0,
    }


def camera_to_ego(point_cam: np.ndarray, ego_to_camera: np.ndarray) -> np.ndarray:
    camera_to_ego = np.linalg.inv(ego_to_camera)
    homog = np.concatenate([point_cam, np.ones(1, dtype=np.float64)])
    return (camera_to_ego @ homog)[:3]


def ego_to_global(point_ego: np.ndarray, ann: AnnotationFrame) -> np.ndarray:
    return ann.ego_to_global_rotation @ point_ego + ann.ego_to_global_translation


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    # Annotation stores quaternion as [x, y, z, w].
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_box(row: dict[str, str], keys: tuple[str, str, str, str]) -> np.ndarray:
    return np.asarray([float(row[k]) for k in keys], dtype=np.float64)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(area_a + area_b - inter, 1e-9)


def class_group(name: str) -> str:
    text = str(name).lower()
    if any(k in text for k in ("vehicle", "car", "truck", "bus")):
        return "vehicle"
    if any(k in text for k in ("pedestrian", "person")):
        return "person"
    if any(k in text for k in ("cycle", "bike", "bicycle", "motorcycle")):
        return "cycle"
    return "unknown"


def int_float(value: Any) -> int:
    return int(float(value))


def float_or_nan(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def fmt(value: float) -> str:
    return f"{value:.6f}" if math.isfinite(value) else ""


if __name__ == "__main__":
    raise SystemExit(main())
