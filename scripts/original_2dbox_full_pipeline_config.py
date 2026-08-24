from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


CAMERA_TO_VIEW = {
    "rear_camera": "rear",
    "left_rear_camera": "left_rear",
    "right_rear_camera": "right_rear",
    "center_camera_fov30": "front",
    "center_camera_fov120": "front_wide",
}


DEFAULTS: dict[str, Any] = {
    "pipeline": {
        "name": "original_2dbox_full_sam_da3_3dbox_v1",
        "output_root": "",
        "use_existing_2d_boxes": True,
        "run_detection": False,
    },
    "data": {
        "annotation_dir": "format_output/annotations/NV",
        "camera_root": "camera",
        "calib_root": "calib",
        "cameras": "rear_camera,left_rear_camera,right_rear_camera",
    },
    "runtime": {"device": "cpu", "fps": 10, "max_frames": -1},
    "sam": {
        "enabled": True,
        "implementation": "classic_sam",
        "checkpoint": "checkpoints/sam_vit_h_4b8939.pth",
        "model_type": "vit_h",
        "box_scale": 1.5,
        "positive_points": 5,
        "clip_mask_to_original_box": True,
        "mask_format": "png",
        "min_2d_score": 0.0,
    },
    "depth": {
        "enabled": True,
        "implementation": "da3_metric",
        "da3_root": "third_party/Depth-Anything-3",
        "model_dir": "third_party/Depth-Anything-3/checkpoints/da3metric-large",
        "process_res": 336,
        "chunk_size": 20,
        "stride": 1,
    },
    "tracking": {
        "enabled": True,
        "method": "robust_botsort",
        "render_video": True,
        "thickness": 1,
        "appearance": {
            "type": "dinov2",
            "model": "dinov2_vitl14",
            "device": "cpu",
            "batch_size": 24,
            "crop_size": 224,
            "box_padding": 0.08,
        },
        "botsort": {
            "frame_rate": 10,
            "track_high_thresh": 0.45,
            "track_low_thresh": 0.08,
            "new_track_thresh": 0.50,
            "track_buffer": 60,
            "match_thresh": 0.80,
            "proximity_thresh": 0.70,
            "appearance_thresh": 0.35,
            "second_match_thresh": 0.50,
            "unconfirmed_match_thresh": 0.70,
            "min_hits": 1,
            "max_obs": 100,
            "fuse_first_associate": True,
            "use_cmc": True,
            "cmc_method": "ecc",
        },
        # Kept as a light fallback/debug path; the reproduction path uses BoT-SORT.
        "sort": {},
    },
    "masks": {"ensure_for_every_track_box": True, "min_iou": 0.30, "allow_label_mismatch": False},
    "optimization_3d": {
        "enabled": True,
        "device": "cpu",
        "max_iterations": 3000,
        "workers": 4,
        "learning_rate": 0.01,
        "render_video": True,
        "output_videos": True,
        "initialization_mode": "da3",
        "losses": {},
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve original-2D-box full SAM/DA3/3D pipeline config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--write-generated", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    scene = Path(args.scene).resolve()
    cfg = resolve_config(load_yaml(Path(args.config)), repo, scene)
    if args.write_generated:
        write_generated_configs(cfg)
    else:
        print(json.dumps(cfg, ensure_ascii=False))
    return 0


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML must be a dict: {path}")
    return data


def resolve_config(user: dict[str, Any], repo: Path, scene: Path) -> dict[str, Any]:
    cfg = deep_merge(deepcopy(DEFAULTS), user)
    scene_name = scene.name
    out_text = str(cfg["pipeline"].get("output_root") or "")
    out_root = Path(out_text) if out_text else repo / "outputs" / f"{scene_name}_original_2dbox_full_v1"
    if not out_root.is_absolute():
        out_root = repo / out_root

    cameras = [x.strip() for x in str(cfg["data"]["cameras"]).split(",") if x.strip()]
    views = {camera: CAMERA_TO_VIEW.get(camera, camera.replace("_camera", "")) for camera in cameras}
    ann_dir = make_abs(scene, cfg["data"]["annotation_dir"])
    cam_root = make_abs(scene, cfg["data"]["camera_root"])
    calib_root = make_abs(scene, cfg["data"]["calib_root"])

    paths = {
        "repo": posix(repo),
        "scene": posix(scene),
        "output_root": posix(out_root),
        "annotation_dir": posix(ann_dir),
        "camera_root": posix(cam_root),
        "calib_root": posix(calib_root),
        "sam_output": posix(out_root / "masks" / "raw_sam"),
        "depth_output": posix(out_root / "depth"),
        "tracking_input": posix(out_root / "tracking_input"),
        "tracking": posix(out_root / "tracking"),
        "robust_tracks": posix(out_root / "tracking" / "robust_botsort_tracks"),
        "sort_tracks": posix(out_root / "tracking" / "sort2d_tracks"),
        "depth_tracks": posix(out_root / "tracks_with_depth"),
        "ensured_masks": posix(out_root / "masks" / "ensured"),
        "optimized_3d": posix(out_root / "optimized_3d"),
        "generated_configs": posix(out_root / "configs"),
    }
    cfg["paths"] = paths
    cfg["resolved_cameras"] = cameras
    cfg["resolved_views"] = views

    cfg["sam"]["checkpoint"] = posix(make_abs(repo, cfg["sam"]["checkpoint"]))
    cfg["depth"]["da3_root"] = posix(make_abs(repo, cfg["depth"]["da3_root"]))
    cfg["depth"]["model_dir"] = posix(make_abs(repo, cfg["depth"]["model_dir"]))
    return cfg


def make_abs(base: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else base / path


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def write_generated_configs(cfg: dict[str, Any]) -> None:
    config_dir = Path(cfg["paths"]["generated_configs"])
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "pipeline_resolved.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    write_robust_botsort_config(cfg, config_dir / "robust_botsort_2d.yaml")
    write_retrack_config(cfg, config_dir / "retrack_sort_2d.yaml")
    write_render_track_config(cfg, config_dir / "render_robust_botsort_tracks.yaml", base_path=cfg["paths"]["robust_tracks"], name="render_original_2dbox_robust_botsort_tracks")
    write_render_track_config(cfg, config_dir / "render_sort2d_tracks.yaml", base_path=cfg["paths"]["sort_tracks"], name="render_original_2dbox_sort_tracks")
    write_ensure_masks_config(cfg, config_dir / "ensure_masks.yaml")
    write_3d_optimizer_config(cfg, config_dir / "rebuild_3d.yaml")


def write_robust_botsort_config(cfg: dict[str, Any], path: Path) -> None:
    views = cfg["resolved_views"]
    lines = [
        "experiment:",
        "  name: original_2dbox_robust_botsort",
        "  mode: offline",
        "  description: Appearance-assisted BoT-SORT tracking from SAM/2D-box detections.",
        "inputs:",
        "  views:",
    ]
    detection_root = cfg["paths"]["sam_output"] if cfg["sam"].get("enabled", True) else cfg["paths"]["tracking_input"]
    for camera, view in views.items():
        lines += [
            f"    {view}:",
            f"      detections_csv: {detection_root}/{camera}/gt2d_sam_masks.csv" if cfg["sam"].get("enabled", True) else f"      detections_csv: {detection_root}/{camera}/tracks.csv",
            f"      image_dir: {cfg['paths']['camera_root']}/{camera}",
            f"      output_name: {camera}",
        ]
    lines += ["input_sequence:", f"  first_frame: 0"]
    append_common_labels(lines)
    lines += ["appearance:"]
    for key, value in cfg["tracking"].get("appearance", {}).items():
        lines.append(f"  {key}: {yaml_value(value)}")
    lines += ["tracker:"]
    for key, value in cfg["tracking"].get("botsort", {}).items():
        lines.append(f"  {key}: {yaml_value(value)}")
    lines += ["track_id_remaps: {}", "output:", f"  dir: {cfg['paths']['robust_tracks']}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_retrack_config(cfg: dict[str, Any], path: Path) -> None:
    views = cfg["resolved_views"]
    lines = [
        "schema_version: 1",
        "experiment:",
        "  name: original_2dbox_retrack_sort",
        "scope:",
        "  cameras: [" + ", ".join(views.values()) + "]",
        "inputs:",
        "  views:",
    ]
    for camera, view in views.items():
        lines += [
            f"    {view}:",
            f"      camera_name: {camera}",
            f"      image_dir: {cfg['paths']['camera_root']}/{camera}",
            f"      track_csv: {cfg['paths']['tracking_input']}/{camera}/tracks.csv",
        ]
    append_common_labels(lines)
    lines += ["retracking:", "  sort_2d:", f"    output_dir: {cfg['paths']['sort_tracks']}"]
    for key, value in cfg["tracking"]["sort"].items():
        lines.append(f"    {key}: {yaml_value(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_render_track_config(cfg: dict[str, Any], path: Path, base_path: str, name: str) -> None:
    views = cfg["resolved_views"]
    base = Path(base_path)
    lines = [
        "schema_version: 1",
        "experiment:",
        f"  name: {name}",
        "scope:",
        "  cameras: [" + ", ".join(views.values()) + "]",
        "inputs:",
        "  views:",
    ]
    for camera, view in views.items():
        track_csv = base / camera / "tracks.csv"
        lines += [
            f"    {view}:",
            f"      camera_name: {camera}",
            f"      track_csv: {posix(track_csv)}",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ensure_masks_config(cfg: dict[str, Any], path: Path) -> None:
    views = cfg["resolved_views"]
    lines = [
        "schema_version: 1",
        "experiment:",
        "  name: ensure_original_2dbox_masks",
        "scope:",
        "  cameras: [" + ", ".join(views.values()) + "]",
        "inputs:",
        "  views:",
    ]
    for camera, view in views.items():
        lines += view_input_lines(cfg, camera, view, f"{cfg['paths']['depth_tracks']}/{camera}/tracks.csv", f"{cfg['paths']['sam_output']}/{camera}/gt2d_sam_masks.csv")
    append_common_labels(lines)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_3d_optimizer_config(cfg: dict[str, Any], path: Path) -> None:
    views = cfg["resolved_views"]
    losses = cfg["optimization_3d"].get("losses", {})
    repo = cfg["paths"]["repo"]
    lines = [
        "base_configs:",
        f"  - {repo}/configs/common/rebuild_single_view_ensured_masks_common.yaml",
        "experiment:",
        f"  name: {cfg['pipeline']['name']}",
        "  mode: offline",
        "scope:",
        "  cameras: [" + ", ".join(views.values()) + "]",
        "output:",
        f"  dir: {cfg['paths']['optimized_3d']}",
        f"  videos: {yaml_value(cfg['optimization_3d'].get('output_videos', True))}",
        "  video:",
        "    # 展示版可视化：只保留 mask、2D box、3D box、BEV。",
        "    # 不输出专业版 loss 面板、贴边关系、角点编号、center 点、尺寸文字。",
        "    debug_geometry_style: false",
        "    draw_mask_pixels: true",
        "    draw_2d_box: true",
        "    box_2d_thickness: 1",
        "    draw_3d_box: true",
        "    draw_projected_bbox: false",
        "    draw_bev: true",
        "    draw_loss_panel: false",
        "    draw_support_edges: false",
        "    draw_corner_points: false",
        "    draw_corner_labels: false",
        "    draw_center_projection: false",
        "    draw_box_dimensions: false",
        "    draw_truncation_label: false",
        "inputs:",
        "  annotations_dir: " + cfg["paths"]["annotation_dir"],
        "  views:",
    ]
    for camera, view in views.items():
        lines += view_input_lines(
            cfg,
            camera,
            view,
            f"{cfg['paths']['depth_tracks']}/{camera}/tracks.csv",
            f"{cfg['paths']['ensured_masks']}/{camera}/gt2d_sam_masks_ensured_cropped.csv",
        )
    lines += [
        "solver:",
        f"  device: {cfg['optimization_3d']['device']}",
        f"  max_iterations: {int(cfg['optimization_3d']['max_iterations'])}",
        f"  learning_rate: {float(cfg['optimization_3d']['learning_rate'])}",
        "  track_parallel:",
        "    enabled: true",
        f"    workers: {int(cfg['optimization_3d']['workers'])}",
        "observations:",
        "  supporting_edges:",
        f"    weight: {float(losses.get('supporting_edges_weight', 400.0))}",
        f"    truncation_margin_px: {float(losses.get('truncation_margin_px', 8.0))}",
        "  mask:",
        "    enabled: true",
        "    use_foreground_points: true",
        "    point_sample_mode: foreground_pixels",
        "    max_foreground_points: 512",
        f"    contain_weight: {float(losses.get('mask_contain_weight', 1000.0))}",
        f"    oversize_weight: {float(losses.get('mask_oversize_weight', 0.002))}",
        "    oversize_max_area_ratio: 0.0",
        "    fallback_to_mask_bbox: false",
        "  depth:",
        "    enabled: true",
        "    role: initialization_only",
        f"    initialization_mode: {cfg['optimization_3d'].get('initialization_mode', 'da3')}",
        "  class_size_prior:",
        f"    enabled: {yaml_value(losses.get('class_size_prior_enabled', True))}",
        f"    weight: {float(losses.get('class_size_prior_weight', 100.0))}",
        "  size_ratio_prior:",
        f"    enabled: {yaml_value(losses.get('size_ratio_prior_enabled', True))}",
        f"    weight: {float(losses.get('size_ratio_prior_weight', 1.0))}",
        "  top_bottom_edges:",
        f"    enabled: {yaml_value(losses.get('top_bottom_edges_enabled', True))}",
        f"    weight: {float(losses.get('top_bottom_edges_weight', 100.0))}",
        f"    far_start_m: {float(losses.get('top_bottom_edges_far_start_m', 10.0))}",
        "  ground_plane:",
        f"    enabled: {yaml_value(losses.get('ground_plane_enabled', True))}",
        f"    weight: {float(losses.get('ground_plane_weight', 5.0))}",
        f"    camera_height_m: {float(losses.get('camera_height_m', 0.5))}",
        "  temporal_smoothness:",
        f"    enabled: {yaml_value(losses.get('temporal_smoothness_enabled', True))}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def view_input_lines(cfg: dict[str, Any], camera: str, view: str, track_csv: str, mask_csv: str) -> list[str]:
    calib_root = cfg["paths"]["calib_root"]
    return [
        f"    {view}:",
        f"      camera_name: {camera}",
        f"      image_dir: {cfg['paths']['camera_root']}/{camera}",
        f"      track_csv: {track_csv}",
        f"      mask_csv: {mask_csv}",
        f"      intrinsic: {calib_root}/{camera}/{camera}-intrinsic.json",
        f"      camera_to_ego: {calib_root}/{camera}/{camera}-to-car_center-extrinsic.json",
    ]


def append_common_labels(lines: list[str]) -> None:
    lines += [
        "label_aliases:",
        "  VEHICLE_CAR: car",
        "  VEHICLE_SUV: car",
        "  VEHICLE_TRUCK: truck",
        "  VEHICLE_TRUCK_SMALL: truck",
        "  VEHICLE_TRAILER: truck",
        "  VEHICLE_BUS: bus",
        "  PEDESTRIAN_NORMAL: person",
        "  MOTOR: motorcycle",
        "  CYCLIST_MOTOR: motorcycle",
        "  BICYCLE: bicycle",
        "  CYCLIST_BICYCLE: bicycle",
        "  VEHICLE_TRIKE: motorcycle",
    ]


def yaml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def posix(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


if __name__ == "__main__":
    raise SystemExit(main())
