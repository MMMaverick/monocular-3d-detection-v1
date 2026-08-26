from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "experiment": {"name": "rebuild_world_track_joint", "mode": "offline"},
    "scope": {
        "cameras": ["rear", "left_rear", "right_rear"],
        "classes": {"source": "all_2d_gt_box_classes"},
        "single_view_first": True,
        "multiview_enabled": False,
    },
    "inputs": {
        "annotations_dir": "data/format_output/annotations/NV",
        "views": {
            "rear": {
                "camera_name": "rear_camera",
                "image_dir": "data/camera/rear_camera",
                "track_csv": "preprocessed/tracks/other/rear/tracking_masks/rear_gt2d_da3metric_tracking/tracks.csv",
                "mask_csv": "preprocessed/masks/rear_camera/gt2d_sam_masks.csv",
                "intrinsic": "data/calib/rear_camera/rear_camera-intrinsic.json",
                "camera_to_ego": "data/calib/rear_camera/rear_camera-to-car_center-extrinsic.json",
            },
            "left_rear": {
                "camera_name": "left_rear_camera",
                "image_dir": "data/camera/left_rear_camera",
                "track_csv": "preprocessed/tracks/other/left_rear/tracking_masks/left_rear_sam3_da3metric_tracking_robust/tracks.csv",
                "mask_csv": "preprocessed/masks/left_rear_camera/gt2d_sam_masks.csv",
                "intrinsic": "data/calib/left_rear_camera/left_rear_camera-intrinsic.json",
                "camera_to_ego": "data/calib/left_rear_camera/left_rear_camera-to-car_center-extrinsic.json",
            },
            "right_rear": {
                "camera_name": "right_rear_camera",
                "image_dir": "data/camera/right_rear_camera",
                "track_csv": "preprocessed/tracks/other/right_rear/tracking_masks/right_rear_sam3_da3metric_tracking_robust/tracks.csv",
                "mask_csv": "preprocessed/masks/right_rear_camera/gt2d_sam_masks.csv",
                "intrinsic": "data/calib/right_rear_camera/right_rear_camera-intrinsic.json",
                "camera_to_ego": "data/calib/right_rear_camera/right_rear_camera-to-car_center-extrinsic.json",
            },
        },
    },
    "coordinates": {
        "optimization_frame": "world",
        "size_order": ["length", "width", "height"],
        "length_unit": "meter",
        "quaternion_order": "xyzw",
    },
    "variables": {
        "center_xyz": {"scope": "per_frame", "solve_jointly_per_track": True},
        "size": {"scope": "shared_per_track", "optimize": True, "parameterization": "log"},
        "yaw": {
            "optimize": False,
            "default": "parallel_to_rear_camera_and_ego_motion_direction",
            "reference_view": "rear",
        },
        "prior_losses": {"center": False, "depth": False, "size": False, "yaw": False},
    },
    "constraints": {
        "upright_box": {"enabled": True, "up_axis": "world_z", "pitch": 0.0, "roll": 0.0},
        "ground_contact": {
            "enabled": False,
            "mode": "fixed_track_bottom_z_from_initialization",
            "optimize_bottom_z": False,
        },
    },
    "label_aliases": {
        "VEHICLE_CAR": "car",
        "VEHICLE_SUV": "car",
        "VEHICLE_TRUCK": "truck",
        "VEHICLE_TRUCK_SMALL": "truck",
        "VEHICLE_TRAILER": "truck",
        "VEHICLE_BUS": "bus",
        "PEDESTRIAN_NORMAL": "person",
        "MOTOR": "motorcycle",
        "CYCLIST_MOTOR": "motorcycle",
        "BICYCLE": "bicycle",
        "CYCLIST_BICYCLE": "bicycle",
        "VEHICLE_TRIKE": "motorcycle",
    },
    "class_defaults": {
        "default": {"init_size": [4.5, 1.8, 1.6], "min_size": [4.2, 1.7, 1.4], "max_size": [6.5, 2.5, 2.4]},
        "car": {"init_size": [4.5, 1.8, 1.6], "min_size": [4.2, 1.7, 1.4], "max_size": [6.5, 2.5, 2.4]},
        "truck": {"init_size": [15.0, 2.5, 3.2], "min_size": [12.0, 2.0, 2.4], "max_size": [25.0, 5.0, 5.0]},
        "bus": {"init_size": [11.0, 2.6, 3.3], "min_size": [8.0, 2.0, 2.5], "max_size": [15.0, 3.5, 4.2]},
        "person": {"init_size": [0.8, 0.8, 1.7], "min_size": [0.6, 0.6, 1.25], "max_size": [1.2, 1.2, 2.2]},
        "motorcycle": {"init_size": [2.0, 0.8, 1.5], "min_size": [1.5, 0.6, 1.1], "max_size": [3.0, 1.2, 2.0]},
        "bicycle": {"init_size": [1.8, 0.6, 1.5], "min_size": [1.35, 0.45, 1.1], "max_size": [2.5, 1.0, 2.0]},
    },
    "observations": {
        "supporting_edges": {
            "enabled": True,
            "weight": 400.0,
            "normalization": "box_width",
            "edge_selection": "hard_leftmost_rightmost_vertical_edge",
            "truncation_margin_px": 2.0,
        },
        "bbox_fit": {
            "enabled": False,
            "weight": 1.0,
            "rear_multiplier": 3.0,
            "far_multiplier": 3.0,
            "far_start_m": 35.0,
            "normalization": "box_width_height",
        },
        "mask": {
            "enabled": True,
            "use_foreground_points": False,
            "point_sample_mode": "foreground_pixels",
            "contain_weight": 400.0,
            "contain_reduction": "mean",
            "oversize_weight": 0.002,
            "oversize_max_area_ratio": 0.0,
            "max_foreground_points": 512,
            "fallback_to_mask_bbox": True,
        },
        "depth": {
            "enabled": True,
            "role": "initialization_only",
            "initialization_mode": "da3",
            "height_prior_replace_da3_beyond_m": 0.0,
            "initial_center_override_csv": "",
        },
        "depth_safety": {
            "enabled": False,
            "mode": "min_corner_distance_to_camera",
            "min_corner_distance_m": 0.2,
            "min_corner_depth_m": 0.75,
            "weight": 50.0,
        },
        "ego_box_safety": {
            "enabled": False,
            "weight": 500.0,
            "ego_size": [4.8, 2.0, 1.6],
            "ego_center": [0.0, 0.0, 0.8],
            "margin_m": 0.2,
        },
        "top_bottom_edges": {"enabled": False, "weight": 100.0, "far_start_m": 10.0},
        "size_ratio_prior": {"enabled": False, "weight": 1.0, "mode": "log_shape_ratio"},
        "class_size_prior": {
            "enabled": False,
            "weight": 10.0,
            "mode": "log_hinge",
            "max_log_deviation": [0.45, 0.35, 0.35],
        },
        "height_depth_prior": {
            "enabled": False,
            "weight": 50.0,
            "mode": "log_hinge",
            "height_source": "class_init",
            "max_log_depth_deviation": 0.25,
            "exclude_truncated_vertical": True,
            "far_start_m": 0.0,
        },
        "ground_plane": {
            "enabled": False,
            "camera_height_m": 0.4,
            "camera_height_m_by_view": {},
            "distance_decay": {
                "enabled": False,
                "distance_source": "initial_camera_distance",
                "near_m": 8.0,
                "far_m": 35.0,
                "min_multiplier": 0.1,
                "mode": "smoothstep",
            },
            "weight": 5.0,
            "mode": "bottom_corners_to_ground_plane",
        },
        "projection": {"min_depth_m": 0.2},
        "no_mask_frames": {"mode": "normal"},
        "size_update": {
            "mode": "all_frames",
            "exclude_truncated": True,
            "fallback_to_mask_frames_when_no_reliable": False,
            "fallback_size_gradient_scale": 0.25,
        },
        "temporal": {"enabled": False},
        "ground": {"enabled": False},
        "point_cloud": {"enabled": False},
        "multiview": {"enabled": False},
    },
    "weighting": {
        "near_observation_size_influence": {
            "enabled": True,
            "distance_measure": "initialized_3d_euclidean_distance_to_camera",
            "near_threshold": 15.0,
            "far_threshold": 60.0,
            "near_multiplier": 2.0,
            "minimum_weight": 1.0,
        }
    },
    "solver": {
        "device": "cuda",
        "framework": "pytorch",
        "algorithm": "adam",
        "dtype": "float32",
        "progress_interval": 100,
        "learning_rate": 0.01,
        "max_iterations": 300,
        "gradient_clip": 10.0,
        "scheduler": {"enabled": True, "type": "exponential", "gamma": 0.98, "step_size": 25, "minimum_learning_rate": 1.0e-4},
        "weight_decay": 0.0,
        "min_track_frames": 3,
        "max_tracks_per_view": 0,
        "track_parallel": {"enabled": False, "workers": 1},
    },
    "fallback": {
        "accept_partial_track": True,
        "revert_failed_frames": False,
        "revert_failed_track": False,
        "failure_diagnostics": {"per_frame_loss_values": True, "per_loss_breakdown": True, "identify_failed_loss_term": True},
    },
    "output": {
        "dir": "outputs/rebuild_world_track_joint_draft_v1",
        "resolved_config": True,
        "copy_source_config": True,
        "per_residual_diagnostics": True,
        "per_frame_loss_diagnostics": True,
        "videos": True,
        "track_videos": False,
        "video": {
            "enabled": True,
            "fps": 10.0,
            "draw_3d_box": True,
            "draw_2d_box": True,
            "box_2d_thickness": 1,
            "draw_support_edges": True,
            "draw_mask_pixels": True,
            "draw_corner_points": False,
            "draw_corner_labels": False,
            "draw_center_projection": False,
            "draw_box_dimensions": True,
            "box_dimensions_font_scale": 0.62,
            "box_dimensions_thickness": 2,
            "draw_loss_panel": True,
            "mask_alpha": 0.35,
            "max_frames_per_view": 0,
        },
    },
}


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    cfg = deepcopy(DEFAULT_CONFIG)
    user = load_config_with_bases(config_path)
    deep_merge(cfg, user)
    cfg["_config_path"] = str(config_path)
    cfg["_root_dir"] = str(config_path.resolve().parents[1])
    return cfg


def load_config_with_bases(config_path: Path) -> dict[str, Any]:
    user = load_yaml_file(config_path)
    cfg: dict[str, Any] = {}
    for base_path in user.pop("base_configs", []) or []:
        base_cfg = load_config_with_bases(resolve_base_config_path(config_path, base_path))
        deep_merge(cfg, base_cfg)
    deep_merge(cfg, user)
    return cfg


def load_yaml_file(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:  # pragma: no cover
        yaml = None
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        data = yaml.safe_load(text) or {}
    else:
        data = parse_simple_yaml(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def resolve_base_config_path(config_path: Path, base_path: str | Path) -> Path:
    path = Path(base_path)
    if path.is_absolute():
        return path
    candidate = config_path.parent / path
    if candidate.exists():
        return candidate
    return config_path.resolve().parents[1] / path


def write_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    payload = deepcopy(config)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except ImportError:  # pragma: no cover
        Path(path).write_text(to_simple_yaml(payload), encoding="utf-8")
    else:
        with Path(path).open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def write_experiment_config_snapshot(config: dict[str, Any], path: str | Path) -> None:
    """Write the fully merged runtime config as an experiment-local snapshot.

    This snapshot is intentionally self-contained: it removes ``base_configs``
    and runtime-only metadata, so future changes to common config files cannot
    change what this experiment meant.
    """
    payload = deepcopy(config)
    payload.pop("_config_path", None)
    payload.pop("_root_dir", None)
    payload.pop("base_configs", None)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except ImportError:  # pragma: no cover
        Path(path).write_text(to_simple_yaml(payload), encoding="utf-8")
    else:
        with Path(path).open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_root_dir"]) / path


def parse_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the small YAML subset used by this project's config.

    This fallback supports indentation-based mappings, list items, inline
    scalar lists such as ``[length, width, height]``, booleans, nulls and
    numbers. It intentionally does not try to be a full YAML parser.
    """
    prepared: list[tuple[int, str]] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        prepared.append((len(raw) - len(raw.lstrip(" ")), raw.strip()))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(prepared):
            return {}, index
        is_list = prepared[index][1].startswith("- ")
        if is_list:
            items: list[Any] = []
            while index < len(prepared):
                cur_indent, stripped = prepared[index]
                if cur_indent < indent or not stripped.startswith("- "):
                    break
                if cur_indent > indent:
                    break
                item_text = stripped[2:].strip()
                index += 1
                if item_text == "":
                    child, index = parse_block(index, indent + 2)
                    items.append(child)
                elif ":" in item_text and not item_text.startswith(("'", '"')):
                    key, value = split_key_value(item_text)
                    item: dict[str, Any] = {}
                    if value == "":
                        child, index = parse_block(index, indent + 2)
                        item[key] = child
                    else:
                        item[key] = parse_scalar(value)
                    items.append(item)
                else:
                    items.append(parse_scalar(item_text))
            return items, index

        mapping: dict[str, Any] = {}
        while index < len(prepared):
            cur_indent, stripped = prepared[index]
            if cur_indent < indent or stripped.startswith("- "):
                break
            if cur_indent > indent:
                break
            key, value = split_key_value(stripped)
            index += 1
            if value == "":
                if index < len(prepared) and prepared[index][0] > cur_indent:
                    child, index = parse_block(index, prepared[index][0])
                else:
                    child = {}
                mapping[key] = child
            else:
                mapping[key] = parse_scalar(value)
        return mapping, index

    parsed, final_index = parse_block(0, prepared[0][0] if prepared else 0)
    if final_index != len(prepared):
        raise ValueError(f"Could not parse config near: {prepared[final_index]}")
    if not isinstance(parsed, dict):
        raise ValueError("Top-level YAML config must be a mapping.")
    return parsed


def split_key_value(text: str) -> tuple[str, str]:
    key, sep, value = text.partition(":")
    if not sep:
        raise ValueError(f"Expected key: value line, got {text!r}")
    return key.strip(), value.strip()


def parse_scalar(value: str) -> Any:
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    if value in ("null", "Null", "None", "~"):
        return None
    if value.startswith("[") and value.endswith("]"):
        inside = value[1:-1].strip()
        if not inside:
            return []
        return [parse_scalar(part.strip()) for part in inside.split(",")]
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value


def to_simple_yaml(value: Any, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(to_simple_yaml(item, indent + 2))
            else:
                lines.append(f"{pad}{key}: {format_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.append(to_simple_yaml(item, indent + 2))
            else:
                lines.append(f"{pad}- {format_scalar(item)}")
        return "\n".join(lines)
    return f"{pad}{format_scalar(value)}"


def format_scalar(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, list):
        return "[" + ", ".join(format_scalar(v) for v in value) + "]"
    return str(value)
