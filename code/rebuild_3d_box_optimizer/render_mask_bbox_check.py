from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, resolve_path, write_resolved_config
from .run import append_progress


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render raw 2D bbox + SAM mask check videos for rear/side cameras.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="outputs/mask_bbox_check_v1")
    parser.add_argument("--max-frames-per-view", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to render mask/bbox check videos.") from exc

    config = load_config(args.config)
    out_dir = resolve_path(config, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.log"
    progress_path.write_text("", encoding="utf-8")
    shutil.copy2(args.config, out_dir / "source_config.yaml")
    write_resolved_config(config, out_dir / "resolved_config.yaml")
    append_progress(progress_path, f"START output={out_dir}")

    views = list(config.get("scope", {}).get("cameras", ["rear", "left_rear", "right_rear"]))
    for view in views:
        render_view(config, view, out_dir, progress_path, cv2, args.max_frames_per_view)
    append_progress(progress_path, "DONE")
    return 0


def render_view(config: dict[str, Any], view: str, out_dir: Path, progress_path: Path, cv2, max_frames: int) -> None:
    view_cfg = config["inputs"]["views"][view]
    track_rows = read_csv(resolve_path(config, view_cfg["track_csv"]))
    mask_rows = read_csv(resolve_path(config, view_cfg["mask_csv"]))
    tracks_by_frame = group_by_int_field(track_rows, "frame")
    masks_by_frame = group_by_int_field(mask_rows, "frame")
    frames = sorted(set(tracks_by_frame) | set(masks_by_frame))
    if max_frames > 0:
        frames = frames[:max_frames]
    summary_rows = []
    writer = None
    video_path = out_dir / f"{view}_mask_bbox_check.mp4"
    append_progress(progress_path, f"VIEW_START view={view} frames={len(frames)} tracks={len(track_rows)} masks={len(mask_rows)}")
    try:
        for index, frame in enumerate(frames, start=1):
            rows_t = tracks_by_frame.get(frame, [])
            rows_m = masks_by_frame.get(frame, [])
            image_path = pick_image_path(config, rows_t, rows_m)
            if image_path is None or not image_path.exists():
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            vis = image.copy()
            draw_masks(config, vis, rows_m, cv2)
            draw_mask_bboxes(vis, rows_m, cv2)
            draw_track_bboxes(vis, rows_t, cv2)
            draw_header(vis, view, frame, rows_t, rows_m, cv2)
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
            writer.write(vis)
            summary_rows.append(frame_summary(view, frame, rows_t, rows_m))
            if index == 1 or index % 100 == 0 or index == len(frames):
                append_progress(progress_path, f"VIEW_PROGRESS view={view} frame_index={index}/{len(frames)} frame={frame}")
    finally:
        if writer is not None:
            writer.release()
    write_csv(out_dir / f"{view}_mask_bbox_summary.csv", summary_rows)
    append_progress(progress_path, f"VIEW_DONE view={view} video={video_path} summary_rows={len(summary_rows)}")


def draw_masks(config: dict[str, Any], image: np.ndarray, mask_rows: list[dict[str, str]], cv2) -> None:
    overlay = image.copy()
    any_mask = False
    for row in mask_rows:
        path_text = row.get("mask_path", "")
        if not path_text:
            continue
        mask_path = resolve_path(config, path_text)
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        color = mask_color(int_float(row.get("track_id", -1)), int_float(row.get("gt2d_index", 0)))
        active = mask > 0
        overlay[active] = color
        any_mask = True
    if any_mask:
        cv2.addWeighted(overlay, 0.42, image, 0.58, 0.0, dst=image)


def draw_mask_bboxes(image: np.ndarray, rows: list[dict[str, str]], cv2) -> None:
    for row in rows:
        box = parse_box(row, ("mask_x1", "mask_y1", "mask_x2", "mask_y2"))
        if box is None:
            continue
        color = mask_color(int_float(row.get("track_id", -1)), int_float(row.get("gt2d_index", 0)))
        draw_rect(image, box, color, 2, cv2)
        x1, y1, _, _ = box
        label = f"mask tid={row.get('track_id','?')} {row.get('label','')}"
        cv2.putText(image, label, (int(x1), max(18, int(y1) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def draw_track_bboxes(image: np.ndarray, rows: list[dict[str, str]], cv2) -> None:
    for row in rows:
        box = parse_box(row, ("x1", "y1", "x2", "y2"))
        if box is None:
            continue
        color = (0, 255, 0)
        draw_rect(image, box, color, 2, cv2)
        x1, _, _, y2 = box
        label = f"bbox tid={row.get('track_id','?')} {row.get('gt_label') or row.get('prompt') or row.get('label','')}"
        cv2.putText(image, label, (int(x1), min(image.shape[0] - 8, int(y2) + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def draw_header(image: np.ndarray, view: str, frame: int, track_rows: list[dict[str, str]], mask_rows: list[dict[str, str]], cv2) -> None:
    direct_ids = {int_float(r.get("track_id", -1)) for r in mask_rows if int_float(r.get("track_id", -1)) >= 0}
    fallback = sum(1 for r in mask_rows if int_float(r.get("track_id", -1)) < 0)
    text = f"{view} frame={frame} 2d_bbox={len(track_rows)} masks={len(mask_rows)} direct_masks={len(direct_ids)} track_-1_masks={fallback}"
    cv2.rectangle(image, (0, 0), (min(image.shape[1], 1500), 42), (0, 0, 0), -1)
    cv2.putText(image, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def frame_summary(view: str, frame: int, track_rows: list[dict[str, str]], mask_rows: list[dict[str, str]]) -> dict[str, object]:
    track_ids = sorted({int_float(r.get("track_id", -1)) for r in track_rows if int_float(r.get("track_id", -1)) >= 0})
    mask_track_ids = sorted({int_float(r.get("track_id", -1)) for r in mask_rows})
    direct = [tid for tid in mask_track_ids if tid >= 0]
    return {
        "view": view,
        "frame": frame,
        "num_2d_bbox": len(track_rows),
        "num_masks": len(mask_rows),
        "num_direct_mask_track_ids": len(direct),
        "num_track_minus1_masks": sum(1 for r in mask_rows if int_float(r.get("track_id", -1)) < 0),
        "bbox_track_ids": ";".join(str(x) for x in track_ids),
        "mask_track_ids": ";".join(str(x) for x in mask_track_ids),
    }


def pick_image_path(config: dict[str, Any], track_rows: list[dict[str, str]], mask_rows: list[dict[str, str]]) -> Path | None:
    for row in track_rows + mask_rows:
        image = row.get("image", "")
        if image:
            return resolve_path(config, image)
    return None


def parse_box(row: dict[str, str], keys: tuple[str, str, str, str]) -> tuple[float, float, float, float] | None:
    try:
        vals = tuple(float(row[k]) for k in keys)
    except (KeyError, ValueError):
        return None
    if not np.isfinite(vals).all():
        return None
    return vals


def draw_rect(image: np.ndarray, box: tuple[float, float, float, float], color: tuple[int, int, int], thickness: int, cv2) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), color, thickness, cv2.LINE_AA)


def mask_color(track_id: int, index: int) -> tuple[int, int, int]:
    if track_id < 0:
        return (255, 0, 255)
    base = (track_id * 37 + index * 17) % 255
    return (int((base + 60) % 255), int((base * 3 + 90) % 255), int((base * 7 + 130) % 255))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def group_by_int_field(rows: list[dict[str, str]], field: str) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        try:
            key = int(float(row[field]))
        except (KeyError, ValueError):
            continue
        grouped.setdefault(key, []).append(row)
    return grouped


def int_float(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return -1


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
