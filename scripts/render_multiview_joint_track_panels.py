from __future__ import annotations

"""跨视角联合结果的单目标检查视频渲染脚本。

用途：
    针对某几个 global_track_id，把同一辆车在多个相机里的画面拼到一个视频中，
    右侧同时显示 rear-reference BEV。这个脚本只做可视化，不参与优化。

典型输入：
    outputs/multiview_joint_loma_repair_from_singleview_v1/frame_loss_diagnostics.csv

典型输出：
    outputs/multiview_joint_loma_repair_from_singleview_v1/joint_track_videos/*.mp4
"""

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from rebuild_3d_box_optimizer.config import load_config, resolve_path
from rebuild_3d_box_optimizer.visualization import (
    color_for_track,
    draw_bev_panel,
    draw_box,
    draw_cuboid,
    draw_mask_overlay,
    safe_float,
)


VIEW_ORDER = ["rear", "left_rear", "right_rear"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render per-global-track joint multiview videos: camera panels on the left, "
            "BEV panel on the right."
        )
    )
    parser.add_argument("--config", required=True, help="Resolved experiment config yaml.")
    parser.add_argument("--diagnostics", required=True, help="Multiview frame_loss_diagnostics.csv.")
    parser.add_argument("--output-dir", required=True, help="Directory for rendered videos.")
    parser.add_argument(
        "--global-track-ids",
        default="",
        help="Comma-separated global ids. Empty means all ids in diagnostics.",
    )
    parser.add_argument("--panel-width", type=int, default=640)
    parser.add_argument("--panel-height", type=int, default=360)
    parser.add_argument("--bev-width", type=int, default=520)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--timestamp-tolerance-ms",
        type=float,
        default=-1.0,
        help="Rows whose timestamps differ by this amount are drawn as the same multiview time node. Negative reads config.",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="Debug limit, 0 means all.")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_row_paths(row: dict[str, str]) -> dict[str, str]:
    out = dict(row)
    for key in ("image", "mask_path"):
        if out.get(key):
            out[key] = out[key].replace("\\", "/")
    return out


def row_timestamp_ns(row: dict[str, str]) -> int:
    value = safe_float(row.get("timestamp"))
    if np.isfinite(value):
        return int(value)
    return int(float(row.get("frame", "0") or 0) * 1_000_000_000)


def group_by_time_node(rows: list[dict[str, str]], tolerance_ns: int) -> list[list[dict[str, str]]]:
    # 不同相机 timestamp 很少完全相等；这里按配置里的时间容忍度合成同一个展示节点。
    # 这只是可视化分组，不会改优化结果。
    rows = sorted(rows, key=lambda r: (row_timestamp_ns(r), str(r.get("view", "")), int(float(r.get("track_id", 0) or 0))))
    groups: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    current_center = 0
    for row in rows:
        ts = row_timestamp_ns(row)
        if not current:
            current = [row]
            current_center = ts
            continue
        if abs(ts - current_center) <= tolerance_ns:
            current.append(row)
            current_center = int(round(sum(row_timestamp_ns(r) for r in current) / len(current)))
        else:
            groups.append(current)
            current = [row]
            current_center = ts
    if current:
        groups.append(current)
    return groups


def ordered_views(rows: list[dict[str, str]]) -> list[str]:
    present = {str(r.get("view", "")) for r in rows}
    ordered = [v for v in VIEW_ORDER if v in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def best_row_per_view(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    # 同一个时间节点里理论上每个 view 只有一行。
    # 如果出现重复，取 loss 较小的一行用于展示，避免一个 panel 内重复画多个 box。
    out: dict[str, dict[str, str]] = {}
    for row in sorted(rows, key=lambda r: safe_float(r.get("loss_total"))):
        view = str(row.get("view", ""))
        out.setdefault(view, row)
    return out


def render_camera_panel(
    config: dict[str, Any],
    row: dict[str, str] | None,
    view: str,
    panel_size: tuple[int, int],
    cv2,
) -> np.ndarray:
    # camera panel 只画展示版信息：mask、2D box、3D box、尺寸/截断标签。
    # 不画 loss 面板和贴边 debug 线，避免干扰跨视角几何检查。
    panel_w, panel_h = panel_size
    if row is None:
        panel = np.full((panel_h, panel_w, 3), 18, dtype=np.uint8)
        cv2.putText(panel, f"{view}: missing", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 180, 180), 2, cv2.LINE_AA)
        return panel

    image_path = resolve_path(config, row.get("image", ""))
    image = cv2.imread(str(image_path))
    if image is None:
        image = np.full((panel_h, panel_w, 3), 18, dtype=np.uint8)
        cv2.putText(image, f"missing image: {view}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
        return image

    draw_mask_overlay(config, image, row, 0.35, cv2)
    track_id = int(float(row.get("track_id", 0) or 0))
    draw_box(image, row, ("obs_x1", "obs_y1", "obs_x2", "obs_y2"), color_for_track(track_id, "2d"), f"2D id={track_id}", cv2, thickness=1)
    video_cfg = {
        "draw_box_dimensions": True,
        "draw_truncation_label": True,
        "box_dimensions_font_scale": 0.55,
        "box_dimensions_thickness": 2,
    }
    draw_cuboid(image, row, cv2, color_for_track(track_id, "3d"), draw_projected_bbox=False, video_cfg=video_cfg)

    panel = cv2.resize(image, (panel_w, panel_h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (70, 70, 70), 1, cv2.LINE_AA)
    title = f"{view}  track={row.get('track_id')}  frame={row.get('frame')}"
    cv2.putText(panel, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def render_bev_panel(config: dict[str, Any], rows: list[dict[str, str]], size: tuple[int, int], cv2) -> np.ndarray:
    # BEV 使用 visualization.py 里统一的 rear-reference 坐标轴，便于和主展示视频一致。
    w, h = size
    panel = np.full((h, w, 3), 24, dtype=np.uint8)
    video_cfg = {
        "bev_min_depth_m": -5.0,
        "bev_width_px": w,
        "draw_ego_box": True,
        "ego_box": (config.get("observations", {}) or {}).get("ego_box_safety", {}) or {},
    }
    draw_bev_panel(config, panel, rows, cv2, video_cfg)
    return panel


def render_global_track(
    config: dict[str, Any],
    gid: str,
    rows: list[dict[str, str]],
    output_dir: Path,
    panel_size: tuple[int, int],
    bev_width: int,
    fps: float,
    tolerance_ns: int,
    max_frames: int,
) -> Path | None:
    import cv2

    rows = [normalize_row_paths(r) for r in rows]
    views = ordered_views(rows)
    groups = group_by_time_node(rows, tolerance_ns)
    if max_frames > 0:
        groups = groups[:max_frames]
    if not groups:
        return None

    panel_w, panel_h = panel_size
    frame_size = (panel_w * len(views) + bev_width, panel_h)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"global_{gid}_{'_'.join(views)}_bev.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    try:
        for idx, group in enumerate(groups):
            row_by_view = best_row_per_view(group)
            camera_panels = [render_camera_panel(config, row_by_view.get(view), view, panel_size, cv2) for view in views]
            bev = render_bev_panel(config, group, (bev_width, panel_h), cv2)
            frame = np.concatenate(camera_panels + [bev], axis=1)
            text = f"global={gid}  node={idx + 1}/{len(groups)}  views={','.join(views)}"
            cv2.putText(frame, text, (14, panel_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
    return video_path


def config_timestamp_tolerance_ms(config: dict[str, Any]) -> float:
    mv = config.get("multiview_joint", {}) or {}
    init = mv.get("initialization", {}) or {}
    value = init.get("timestamp_merge_tolerance_ms", 20.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 20.0


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    config["_root_dir"] = str(infer_project_root(config_path))
    diagnostics = Path(args.diagnostics)
    output_dir = Path(args.output_dir)
    rows = read_rows(diagnostics)
    selected = {x.strip() for x in args.global_track_ids.split(",") if x.strip()}
    by_gid: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        gid = str(row.get("global_track_id", "")).strip()
        if not gid:
            continue
        if selected and gid not in selected:
            continue
        by_gid.setdefault(gid, []).append(row)

    tolerance_ms = args.timestamp_tolerance_ms
    if tolerance_ms < 0:
        tolerance_ms = config_timestamp_tolerance_ms(config)
    tolerance_ns = max(0, int(round(tolerance_ms * 1_000_000.0)))

    rendered: list[Path] = []
    for gid in sorted(by_gid, key=lambda x: int(float(x)) if x.replace(".", "", 1).isdigit() else x):
        path = render_global_track(
            config=config,
            gid=gid,
            rows=by_gid[gid],
            output_dir=output_dir,
            panel_size=(args.panel_width, args.panel_height),
            bev_width=args.bev_width,
            fps=args.fps,
            tolerance_ns=tolerance_ns,
            max_frames=args.max_frames,
        )
        if path is not None:
            rendered.append(path)
            print(f"RENDERED {path}")
    print(f"done rendered={len(rendered)} output_dir={output_dir}")


def infer_project_root(config_path: Path) -> Path:
    """从普通 config 或输出目录里的 resolved_config 推断项目根目录。

    WSL 生成的 resolved_config 可能带 /mnt/d/...，Windows 下直接读取会错。
    这里向上找同时包含 code/ 和 data/ 的目录，让脚本在 Windows/WSL 两边都能用。
    """

    start = config_path.resolve()
    for parent in [start.parent, *start.parents]:
        if (parent / "code").exists() and (parent / "data").exists():
            return parent
    # Common layout for outputs/<experiment>/resolved_config.yaml.
    if len(start.parents) >= 3 and (start.parents[2] / "code").exists():
        return start.parents[2]
    return Path.cwd().resolve()


if __name__ == "__main__":
    main()
