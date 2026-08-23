#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from rebuild_3d_box_optimizer.config import load_config
from rebuild_3d_box_optimizer.visualization import render_experiment_videos


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    base = repo / "outputs" / "original_2dbox_full_gpu_v1" / "optimized_3d"
    out = base

    cfg = load_config(base / "resolved_config.yaml")
    cfg["_root_dir"] = str(repo)

    video_cfg = cfg.setdefault("output", {}).setdefault("video", {})
    # 展示版：只显示 mask、2D box、3D box、右侧 BEV。
    # 不显示专业版调试元素：loss、贴边、角点、center、尺寸、截断文字。
    video_cfg["debug_geometry_style"] = False
    video_cfg["draw_mask_pixels"] = True
    video_cfg["draw_2d_box"] = True
    video_cfg["box_2d_thickness"] = 1
    video_cfg["draw_3d_box"] = True
    video_cfg["draw_projected_bbox"] = False
    video_cfg["draw_bev"] = True
    video_cfg["draw_loss_panel"] = False
    video_cfg["loss_panel_mode"] = "none"
    video_cfg["draw_support_edges"] = False
    video_cfg["draw_corner_points"] = False
    video_cfg["draw_corner_labels"] = False
    video_cfg["draw_center_projection"] = False
    video_cfg["draw_box_dimensions"] = False
    video_cfg["draw_truncation_label"] = False

    out.mkdir(parents=True, exist_ok=True)
    print(f"render showcase best -> {out}")
    render_experiment_videos(cfg, base / "frame_loss_diagnostics.csv", out)

    final_out = out / "final_iter"
    final_out.mkdir(parents=True, exist_ok=True)
    print(f"render showcase final_iter -> {final_out}")
    render_experiment_videos(cfg, base / "frame_loss_diagnostics_final_iter.csv", final_out)

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
