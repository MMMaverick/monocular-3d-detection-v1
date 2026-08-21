from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, resolve_path, write_experiment_config_snapshot, write_resolved_config
from .data import Observation, group_by_track, load_view_observations
from .run import requested_tracks_for_view
from .visualization import render_experiment_videos


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render initial 3D tracks without running optimization.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="", help="Output directory. Defaults to <configured output>_initial_3d_tracks.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    configured_out = resolve_path(config, config["output"].get("dir", "outputs/rebuild_world_track_joint"))
    out_dir = Path(args.output) if args.output else configured_out.with_name(configured_out.name + "_initial_3d_tracks")
    if not out_dir.is_absolute():
        out_dir = resolve_path(config, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config.setdefault("output", {})["dir"] = str(out_dir)
    video_cfg = config.setdefault("output", {}).setdefault("video", {})
    config["output"]["videos"] = True
    video_cfg.update(
        {
            "enabled": True,
            "draw_3d_box": True,
            "draw_2d_box": False,
            "draw_support_edges": False,
            "draw_mask_pixels": False,
            "draw_corner_points": False,
            "draw_corner_labels": False,
            "draw_center_projection": False,
            "draw_loss_panel": False,
            "draw_bev": False,
            "draw_projected_bbox": False,
            "draw_box_dimensions": False,
        }
    )

    write_resolved_config(config, out_dir / "resolved_config.yaml")
    write_experiment_config_snapshot(config, out_dir / "experiment_config_snapshot.yaml")
    source = Path(args.config)
    if source.exists():
        shutil.copy2(source, out_dir / "source_config.yaml")

    progress_path = out_dir / "progress.log"
    progress_path.write_text("", encoding="utf-8")
    append_progress(progress_path, f"START config={args.config} output={out_dir}")

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    min_frames = int(config["solver"].get("min_track_frames", 3))
    max_tracks = int(config["solver"].get("max_tracks_per_view", 0) or 0)
    for view in config["scope"]["cameras"]:
        observations = load_view_observations(config, view)
        grouped = group_by_track(observations)
        eligible = [(track_id, items) for track_id, items in grouped.items() if len(items) >= min_frames]
        requested = requested_tracks_for_view(config, view)
        if requested is not None:
            eligible = [(track_id, items) for track_id, items in eligible if track_id in requested]
        if max_tracks > 0:
            eligible = eligible[:max_tracks]
        append_progress(progress_path, f"VIEW_START view={view} observations={len(observations)} tracks={len(eligible)}")
        for track_id, items in eligible:
            for obs in items:
                rows.append(initial_observation_row(obs, config))
            summaries.append({"view": view, "track_id": int(track_id), "frames": len(items), "source": "initialization"})
        append_progress(progress_path, f"VIEW_DONE view={view} rows={sum(1 for r in rows if r['view'] == view)}")

    csv_path = out_dir / "frame_initial_3d_tracks.csv"
    write_csv(csv_path, rows)
    write_csv(out_dir / "track_summary.csv", summaries)
    diagnostics_path = out_dir / "frame_loss_diagnostics.csv"
    write_csv(diagnostics_path, rows)
    append_progress(progress_path, f"CSV_DONE rows={len(rows)} tracks={len(summaries)}")

    append_progress(progress_path, "VIDEO_START")
    videos = render_experiment_videos(config, diagnostics_path, out_dir)
    append_progress(progress_path, "VIDEO_DONE " + " ".join(str(p) for p in videos))
    (out_dir / "summary.txt").write_text(
        "\n".join(
            [
                "initial_3d_tracks_visualization",
                f"rows={len(rows)}",
                f"tracks={len(summaries)}",
                f"csv={csv_path}",
                *[f"video={p}" for p in videos],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    append_progress(progress_path, "DONE")
    return 0


def initial_observation_row(obs: Observation, config: dict[str, Any]) -> dict[str, Any]:
    center = np.asarray(obs.init_center_cam, dtype=np.float64)
    size = np.asarray(obs.init_size, dtype=np.float64)
    corners_cam = box_corners(center, size, np.asarray(obs.box_axes_cam, dtype=np.float64))
    min_depth = float(config.get("observations", {}).get("projection", {}).get("min_depth_m", 0.2))
    corners_px = project(corners_cam, obs.intrinsic, min_depth)
    center_px = project(center[None, :], obs.intrinsic, min_depth)[0]
    row: dict[str, Any] = {
        "view": obs.view,
        "frame": obs.frame,
        "timestamp": obs.timestamp,
        "track_id": obs.track_id,
        "class": obs.label,
        "cx": float(center[0]),
        "cy": float(center[1]),
        "cz": float(center[2]),
        "center_frame": "camera_initialization",
        "length": float(size[0]),
        "width": float(size[1]),
        "height": float(size[2]),
        "size_order": "length,width,height",
        "yaw": 0.0,
        "yaw_axis": "rear_reference_axes_in_current_camera",
        "yaw_optimized": False,
        "optimized": False,
        "image": obs.image,
        "center_u": float(center_px[0]),
        "center_v": float(center_px[1]),
    }
    for idx, point in enumerate(corners_px):
        row[f"corner{idx}_x"] = float(point[0])
        row[f"corner{idx}_y"] = float(point[1])
    return row


def box_corners(center: np.ndarray, size: np.ndarray, axes_cam: np.ndarray) -> np.ndarray:
    coeffs = np.asarray(
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
        dtype=np.float64,
    )
    local = coeffs * size[None, :]
    return center[None, :] + local @ axes_cam


def project(points_cam: np.ndarray, intrinsic: np.ndarray, min_depth: float) -> np.ndarray:
    z = np.maximum(points_cam[:, 2], float(min_depth))
    u = intrinsic[0, 0] * points_cam[:, 0] / z + intrinsic[0, 2]
    v = intrinsic[1, 1] * points_cam[:, 1] / z + intrinsic[1, 2]
    return np.stack([u, v], axis=1)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def append_progress(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
