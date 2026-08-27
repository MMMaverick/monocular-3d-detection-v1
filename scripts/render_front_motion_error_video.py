#!/usr/bin/env python
"""Render a compact front-view motion-error video.

This script is for debugging the motion/static decision, not for pretty 3D
box evaluation.  It overlays selected predicted tracks, their one-second
world speed, the predicted/GT track state, and highlights false-moving cases
(`pred=moving, gt=static`) because that error is usually more expensive for
the downstream task.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Render selected front motion errors with speed and GT state.")
    parser.add_argument("--diagnostics-csv", required=True)
    parser.add_argument("--matches-csv", required=True, help="frame_pred_gt_matches.csv from evaluate_3d_motion_against_front_gt.py")
    parser.add_argument("--track-eval-csv", required=True, help="track_motion_eval.csv from evaluate_3d_motion_against_front_gt.py")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--tracks",
        nargs="+",
        default=[],
        help="Selected pred tracks, e.g. center_front:3000000. If omitted, render all comparable tracks in track-eval CSV.",
    )
    parser.add_argument("--views", nargs="+", default=["center_front", "left_front", "right_front"])
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--speed-window-sec", type=float, default=1.0)
    parser.add_argument("--moving-threshold-mps", type=float, default=2.0)
    parser.add_argument("--pred-moving-ratio-threshold", type=float, default=0.30)
    parser.add_argument("--repo-root", default=r"D:\mono-detect")
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--panel-height", type=int, default=360)
    args = parser.parse_args()

    selected = parse_tracks(args.tracks) if args.tracks else set()
    track_eval = {
        (row.get("view", ""), int_float(row.get("pred_track_id", -1))): row
        for row in read_csv(Path(args.track_eval_csv))
    }
    if not selected:
        selected = {
            key
            for key, row in track_eval.items()
            if row.get("is_comparable", "True") == "True"
        }
    diagnostics = [
        row
        for row in read_csv(Path(args.diagnostics_csv))
        if (row.get("view", ""), int_float(row.get("track_id", -1))) in selected
    ]
    matches = [
        row
        for row in read_csv(Path(args.matches_csv))
        if (row.get("view", ""), int_float(row.get("pred_track_id", -1))) in selected
    ]
    if not diagnostics:
        raise SystemExit("No selected diagnostic rows found.")

    speed_lookup = compute_speed_lookup(matches, args.speed_window_sec)
    diag_by_view_frame: dict[str, dict[int, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in diagnostics:
        diag_by_view_frame[row["view"]][int_float(row["frame"])].append(row)

    frames = sorted({int_float(row["frame"]) for row in diagnostics})
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for frame in frames:
            view_panels = []
            current_rows: list[dict[str, str]] = []
            for view in args.views:
                rows = diag_by_view_frame.get(view, {}).get(frame, [])
                current_rows.extend(rows)
                panel = render_view_panel(rows, view, args, speed_lookup, track_eval)
                view_panels.append(panel)
            bev = render_bev_panel(current_rows, args, speed_lookup, track_eval)
            canvas = np.vstack(view_panels)
            canvas = np.hstack([canvas, bev])
            if writer is None:
                h, w = canvas.shape[:2]
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            writer.write(canvas)
    finally:
        if writer is not None:
            writer.release()

    print(f"video={out_path}")
    return 0


def compute_speed_lookup(rows: list[dict[str, str]], window_sec: float) -> dict[tuple[str, int, int], float]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["view"], int_float(row["pred_track_id"]))].append(row)
    out: dict[tuple[str, int, int], float] = {}
    for key, items in grouped.items():
        items.sort(key=lambda r: int_float(r["frame"]))
        frames = np.asarray([int_float(r["frame"]) for r in items], dtype=np.int64)
        ts = np.asarray([int_float(r.get("frame", 0)) / 10.0 for r in items], dtype=np.float64)
        xy = np.asarray([[float(r["pred_global_x"]), float(r["pred_global_y"])] for r in items], dtype=np.float64)
        for i, frame in enumerate(frames):
            target = ts[i] - window_sec
            prev = np.where(ts <= target)[0]
            if len(prev):
                j = int(prev[-1])
            else:
                future = np.where(ts >= ts[i] + min(window_sec, ts[-1] - ts[i]))[0]
                j = int(future[0]) if len(future) else (0 if i != 0 else min(1, len(ts) - 1))
            dt = abs(float(ts[i] - ts[j]))
            if dt > 1e-6:
                out[(key[0], key[1], int(frame))] = float(np.linalg.norm(xy[i] - xy[j]) / dt)
    return out


def render_view_panel(
    rows: list[dict[str, str]],
    view: str,
    args: argparse.Namespace,
    speed_lookup: dict[tuple[str, int, int], float],
    track_eval: dict[tuple[str, int], dict[str, str]],
) -> np.ndarray:
    panel = np.zeros((args.panel_height, args.panel_width, 3), dtype=np.uint8)
    image = None
    if rows:
        image = cv2.imread(str(resolve_image(rows[0].get("image", ""), Path(args.repo_root))))
        if image is not None:
            panel = cv2.resize(image, (args.panel_width, args.panel_height), interpolation=cv2.INTER_AREA)
    put_text(panel, view, 10, 28, (255, 255, 255), scale=0.75)
    if not rows:
        return panel
    src_h, src_w = image.shape[:2] if image is not None else (1080, 1920)
    sx = args.panel_width / max(src_w, 1)
    sy = args.panel_height / max(src_h, 1)
    for row in rows:
        draw_track(panel, row, sx, sy, speed_lookup, track_eval, args)
    return panel


def draw_track(
    image: np.ndarray,
    row: dict[str, str],
    sx: float,
    sy: float,
    speed_lookup: dict[tuple[str, int, int], float],
    track_eval: dict[tuple[str, int], dict[str, str]],
    args: argparse.Namespace,
) -> None:
    view = row["view"]
    tid = int_float(row["track_id"])
    frame = int_float(row["frame"])
    eval_row = track_eval.get((view, tid), {})
    pred_state = eval_row.get("pred_state", "NA")
    gt_state = eval_row.get("gt_state", "NA")
    false_moving = pred_state == "moving" and gt_state == "static"
    false_static = pred_state == "static" and gt_state == "moving"
    color = (0, 0, 255) if false_moving else (0, 165, 255) if false_static else (0, 255, 0)

    x1 = safe_float(row.get("obs_x1", "nan")) * sx
    y1 = safe_float(row.get("obs_y1", "nan")) * sy
    x2 = safe_float(row.get("obs_x2", "nan")) * sx
    y2 = safe_float(row.get("obs_y2", "nan")) * sy
    if np.isfinite([x1, y1, x2, y2]).all():
        cv2.rectangle(image, (round_int(x1), round_int(y1)), (round_int(x2), round_int(y2)), (255, 160, 80), 1, cv2.LINE_AA)

    pts = []
    for idx in range(8):
        x = safe_float(row.get(f"corner{idx}_x", "nan")) * sx
        y = safe_float(row.get(f"corner{idx}_y", "nan")) * sy
        pts.append((x, y))
    if all(np.isfinite([x, y]).all() for x, y in pts):
        for a, b in EDGES:
            cv2.line(image, (round_int(pts[a][0]), round_int(pts[a][1])), (round_int(pts[b][0]), round_int(pts[b][1])), color, 2, cv2.LINE_AA)

    speed = speed_lookup.get((view, tid, frame), float("nan"))
    ratio = eval_row.get("pred_moving_ratio", "")
    err = "FP moving" if false_moving else "FN static" if false_static else "OK"
    label = f"{view}:{tid} v={fmt(speed)} pred={pred_state} gt={gt_state} {err}"
    if ratio:
        label += f" ratio={ratio}"
    ax = round_int(np.clip(x1 if np.isfinite(x1) else 8, 8, image.shape[1] - 40))
    ay = round_int(np.clip((y1 if np.isfinite(y1) else 40) - 8, 42, image.shape[0] - 8))
    put_text(image, label, ax, ay, color, scale=0.48)


def render_bev_panel(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    speed_lookup: dict[tuple[str, int, int], float],
    track_eval: dict[tuple[str, int], dict[str, str]],
) -> np.ndarray:
    h = args.panel_height * len(args.views)
    w = 460
    panel = np.full((h, w, 3), 24, dtype=np.uint8)
    put_text(panel, "Speed / Motion GT check", 16, 32, (255, 255, 255), scale=0.7)
    put_text(panel, "RED: pred moving but GT static (high penalty)", 16, 62, (0, 0, 255), scale=0.48)
    put_text(panel, "ORANGE: pred static but GT moving", 16, 86, (0, 165, 255), scale=0.48)

    origin = np.asarray([w * 0.50, h * 0.78], dtype=np.float64)
    scale = 6.0
    cv2.line(panel, (0, int(origin[1])), (w, int(origin[1])), (80, 80, 80), 1)
    cv2.line(panel, (int(origin[0]), 110), (int(origin[0]), h - 1), (80, 80, 80), 1)
    put_text(panel, "BEV global XY, local view", 16, 118, (180, 180, 180), scale=0.45)

    y = 150
    for row in rows:
        view = row["view"]
        tid = int_float(row["track_id"])
        frame = int_float(row["frame"])
        eval_row = track_eval.get((view, tid), {})
        pred_state = eval_row.get("pred_state", "NA")
        gt_state = eval_row.get("gt_state", "NA")
        false_moving = pred_state == "moving" and gt_state == "static"
        false_static = pred_state == "static" and gt_state == "moving"
        color = (0, 0, 255) if false_moving else (0, 165, 255) if false_static else (0, 255, 0)
        speed = speed_lookup.get((view, tid, frame), float("nan"))
        gx = safe_float(row.get("world_x", "nan"))
        gy = safe_float(row.get("world_y", "nan"))
        if np.isfinite([gx, gy]).all():
            # Draw relative to the first visible object to keep the panel readable.
            base_x = safe_float(rows[0].get("world_x", "nan"))
            base_y = safe_float(rows[0].get("world_y", "nan"))
            px = int(round(origin[0] + (gx - base_x) * scale))
            py = int(round(origin[1] - (gy - base_y) * scale))
            if 0 <= px < w and 110 <= py < h:
                cv2.circle(panel, (px, py), 5, color, -1, cv2.LINE_AA)
                put_text(panel, str(tid), px + 6, py - 4, color, scale=0.38)
        line = f"{view}:{tid} v={fmt(speed)} pred={pred_state} gt={gt_state}"
        put_text(panel, line, 16, y, color, scale=0.46)
        y += 26
        if y > h - 20:
            break
    return panel


def parse_tracks(values: list[str]) -> set[tuple[str, int]]:
    out = set()
    for value in values:
        if ":" not in value:
            raise SystemExit(f"Invalid track spec: {value}; expected view:track_id")
        view, tid = value.split(":", 1)
        out.add((view, int(tid)))
    return out


def resolve_image(value: str, repo_root: Path) -> Path:
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
            if candidate.exists():
                return candidate
        except OSError:
            pass
    return repo_root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def round_int(value: float) -> int:
    return int(round(float(value)))


def fmt(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.1f}m/s"


def put_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    *,
    scale: float = 0.55,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1 if scale < 0.55 else 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x2 = min(image.shape[1] - 1, x + tw + 5)
    y1 = max(0, y - th - 5)
    cv2.rectangle(image, (max(0, x - 3), y1), (x2, min(image.shape[0] - 1, y + baseline + 3)), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


if __name__ == "__main__":
    raise SystemExit(main())
