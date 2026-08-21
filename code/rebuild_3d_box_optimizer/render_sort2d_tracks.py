from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, resolve_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render 2D SORT track videos from track CSVs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="outputs/sort2d_track_vis_v1")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--thickness", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to render videos.") from exc

    config = load_config(args.config)
    out_dir = resolve_path(config, args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for view in config.get("scope", {}).get("cameras", ["rear", "left_rear", "right_rear"]):
        path = render_view(config, view, out_dir, cv2, float(args.fps), int(args.thickness), int(args.max_frames))
        if path is not None:
            written.append(path)
            print(f"VIDEO_DONE view={view} output={path}", flush=True)
    print(f"DONE videos={len(written)} output_dir={out_dir}", flush=True)
    return 0


def render_view(config: dict[str, Any], view: str, out_dir: Path, cv2, fps: float, thickness: int, max_frames: int) -> Path | None:
    view_cfg = config["inputs"]["views"][view]
    rows = read_csv(resolve_path(config, view_cfg["track_csv"]))
    groups = group_frame_rows(rows)
    if max_frames > 0:
        groups = dict(list(groups.items())[:max_frames])
    camera_name = str(view_cfg.get("camera_name", view))
    video_path = out_dir / f"{camera_name}_sort2d_tracks.mp4"
    writer = None
    try:
        for (_, image_text), frame_rows in groups.items():
            image_path = resolve_path(config, image_text)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            vis = image.copy()
            frame_index = frame_rows[0].get("frame", "")
            draw_header(vis, cv2, f"{view} frame={frame_index} targets={len(frame_rows)}")
            for row in frame_rows:
                draw_track_box(vis, row, cv2, thickness)
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(vis)
    finally:
        if writer is not None:
            writer.release()
    return video_path if writer is not None else None


def draw_header(image: np.ndarray, cv2, text: str) -> None:
    cv2.rectangle(image, (0, 0), (min(image.shape[1], 720), 34), (0, 0, 0), -1)
    cv2.putText(image, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)


def draw_track_box(image: np.ndarray, row: dict[str, str], cv2, thickness: int) -> None:
    box = parse_box(row)
    if box is None:
        return
    track_id = int_float(row.get("track_id", 0))
    color = color_for_track(track_id)
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, max(1, thickness), cv2.LINE_AA)
    label = row.get("track_majority_label") or row.get("gt_label") or row.get("label") or row.get("prompt") or ""
    source_id = row.get("source_track_id", "")
    text = f"id={track_id} {label}"
    if source_id not in ("", str(track_id)):
        text += f" src={source_id}"
    font_scale = 0.48
    text_thick = 1
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thick)
    tx, ty = x1, max(0, y1 - th - base - 3)
    cv2.rectangle(image, (tx, ty), (min(image.shape[1] - 1, tx + tw + 4), ty + th + base + 4), color, -1)
    cv2.putText(image, text, (tx + 2, ty + th + 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thick, cv2.LINE_AA)


def parse_box(row: dict[str, str]) -> tuple[float, float, float, float] | None:
    try:
        box = tuple(float(row[k]) for k in ("x1", "y1", "x2", "y2"))
    except (KeyError, ValueError):
        return None
    if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def group_frame_rows(rows: list[dict[str, str]]) -> dict[tuple[int, str], list[dict[str, str]]]:
    out: dict[tuple[int, str], list[dict[str, str]]] = {}
    for row in rows:
        frame = int_float(row.get("frame", -1))
        image = row.get("image", "")
        out.setdefault((frame, image), []).append(row)
    ordered: dict[tuple[int, str], list[dict[str, str]]] = {}
    for key in sorted(out):
        ordered[key] = sorted(out[key], key=lambda r: int_float(r.get("track_id", 0)))
    return ordered


def color_for_track(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id * 9973 + 17)
    vals = rng.integers(80, 255, size=3)
    return int(vals[0]), int(vals[1]), int(vals[2])


def int_float(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return -1


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    raise SystemExit(main())
