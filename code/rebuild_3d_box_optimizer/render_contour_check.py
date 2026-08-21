from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, resolve_path
from .data import external_contour_points
from .run import append_progress
from .visualization import CUBOID_EDGES, draw_box, draw_cuboid, draw_loss_panel_multi, read_rows, resolve_data_path, safe_float


def main() -> int:
    parser = argparse.ArgumentParser(description="Render mask external contour check videos for an experiment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-frames-per-view", type=int, default=0)
    parser.add_argument("--max-contour-points", type=int, default=512)
    args = parser.parse_args()
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required for contour visualization.") from exc

    config = load_config(args.config)
    rows = read_rows(Path(args.diagnostics))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.log"
    progress_path.write_text("", encoding="utf-8")
    append_progress(progress_path, f"START diagnostics={args.diagnostics} output={out_dir}")

    summary_rows: list[dict[str, object]] = []
    for view, view_rows in group_rows(rows, "view").items():
        video_path = out_dir / f"{view}_contour_check.mp4"
        summary_rows.extend(render_view(config, view, view_rows, video_path, progress_path, cv2, args.max_frames_per_view, args.max_contour_points))
    write_csv(out_dir / "contour_summary.csv", summary_rows)
    append_progress(progress_path, "DONE")
    return 0


def render_view(config: dict[str, Any], view: str, rows: list[dict[str, str]], video_path: Path, progress_path: Path, cv2, max_frames: int, max_points: int) -> list[dict[str, object]]:
    rows.sort(key=lambda r: (int(float(r["frame"])), int(float(r["track_id"]))))
    frame_groups = list(group_frame_rows(rows).values())
    frame_groups.sort(key=lambda g: (int(float(g[0]["frame"])), int(float(g[0].get("timestamp", 0)))))
    if max_frames > 0:
        frame_groups = frame_groups[:max_frames]
    append_progress(progress_path, f"VIEW_START view={view} frames={len(frame_groups)}")
    writer = None
    summary_rows: list[dict[str, object]] = []
    for idx, frame_rows in enumerate(frame_groups, start=1):
        image_path = resolve_data_path(config, frame_rows[0].get("image", ""))
        if not image_path.exists():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        vis = image.copy()
        for row in frame_rows:
            contour = load_contour(config, row, max_points, cv2)
            draw_box(vis, row, ("obs_x1", "obs_y1", "obs_x2", "obs_y2"), (255, 0, 0), f"2D {row.get('track_id')}", cv2)
            draw_cuboid(vis, row, cv2, (0, 255, 0), draw_projected_bbox=False)
            if contour is not None and len(contour) > 0:
                draw_contour(vis, contour, cv2)
                draw_contour_points(vis, contour, cv2)
            draw_header(vis, row, contour, cv2)
            summary_rows.append(make_summary_row(row, contour))
        draw_loss_panel_multi(vis, frame_rows, cv2, {"loss_panel_font_scale": 0.72, "loss_panel_thickness": 2, "loss_panel_line_height": 28, "loss_panel_width": 1700})
        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
        writer.write(vis)
        if idx in (1, len(frame_groups)) or int(float(frame_rows[0]["frame"])) in (597, 599):
            cv2.imwrite(str(video_path.parent / f"{view}_frame_{int(float(frame_rows[0]['frame']))}_contour.jpg"), vis)
        if idx == 1 or idx % 100 == 0 or idx == len(frame_groups):
            append_progress(progress_path, f"VIEW_PROGRESS view={view} frame_index={idx}/{len(frame_groups)}")
    if writer is not None:
        writer.release()
    append_progress(progress_path, f"VIEW_DONE view={view} video={video_path}")
    return summary_rows


def load_contour(config: dict[str, Any], row: dict[str, str], max_points: int, cv2) -> np.ndarray | None:
    path_text = row.get("mask_path", "")
    if not path_text:
        return None
    path = resolve_data_path(config, path_text)
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    points = external_contour_points(mask > 0)
    if max_points > 0 and len(points) > max_points:
        idx = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
        points = points[idx]
    return points


def draw_contour(image: np.ndarray, contour: np.ndarray, cv2) -> None:
    pts = contour.astype(np.int32).reshape(-1, 1, 2)
    if len(pts) >= 2:
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)


def draw_contour_points(image: np.ndarray, contour: np.ndarray, cv2) -> None:
    for x, y in contour[:: max(1, len(contour) // 160)]:
        cv2.circle(image, (int(round(x)), int(round(y))), 2, (0, 0, 255), -1, cv2.LINE_AA)


def draw_header(image: np.ndarray, row: dict[str, str], contour: np.ndarray | None, cv2) -> None:
    n = 0 if contour is None else len(contour)
    text = f"{row.get('view')} frame={row.get('frame')} track={row.get('track_id')} external_contour_points={n}"
    cv2.rectangle(image, (0, 0), (min(image.shape[1], 1500), 42), (0, 0, 0), -1)
    cv2.putText(image, text, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def make_summary_row(row: dict[str, str], contour: np.ndarray | None) -> dict[str, object]:
    if contour is None or len(contour) == 0:
        return {"view": row.get("view"), "frame": row.get("frame"), "track_id": row.get("track_id"), "contour_points": 0}
    return {
        "view": row.get("view"),
        "frame": row.get("frame"),
        "track_id": row.get("track_id"),
        "contour_points": len(contour),
        "contour_x1": float(contour[:, 0].min()),
        "contour_y1": float(contour[:, 1].min()),
        "contour_x2": float(contour[:, 0].max() + 1),
        "contour_y2": float(contour[:, 1].max() + 1),
        "obs_x1": row.get("obs_x1"),
        "obs_y1": row.get("obs_y1"),
        "obs_x2": row.get("obs_x2"),
        "obs_y2": row.get("obs_y2"),
    }


def group_rows(rows: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(row.get(key, ""), []).append(row)
    return out


def group_frame_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    out: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault((row.get("frame", ""), row.get("timestamp", "")), []).append(row)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

