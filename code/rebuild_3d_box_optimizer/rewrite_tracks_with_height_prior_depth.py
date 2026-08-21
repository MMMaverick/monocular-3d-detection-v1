from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, resolve_path
from .data import canonical_label, class_sizes, load_intrinsic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rewrite track CSV depth fields using class-height pinhole prior.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", default="preprocessed/tracks/height_prior")
    parser.add_argument(
        "--fallback-da3-when-top-truncated",
        action="store_true",
        help="Keep DA3metric depth for boxes whose top edge touches the image boundary.",
    )
    parser.add_argument("--top-truncation-margin-px", type=float, default=2.0)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    output_root = resolve_path(config, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    for view in config["scope"]["cameras"]:
        view_cfg = config["inputs"]["views"][view]
        src = resolve_path(config, view_cfg["track_csv"])
        intrinsic = load_intrinsic(resolve_path(config, view_cfg["intrinsic"]))
        dst_dir = output_root / view_cfg.get("camera_name", view)
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "tracks_height_prior_depth.csv"
        count, changed, labels, fallback = rewrite_one_csv(
            config,
            src,
            dst,
            intrinsic,
            fallback_da3_when_top_truncated=bool(args.fallback_da3_when_top_truncated),
            top_truncation_margin_px=float(args.top_truncation_margin_px),
        )
        summary_rows.append(
            {
                "view": view,
                "camera_name": view_cfg.get("camera_name", view),
                "input": str(src),
                "output": str(dst),
                "rows": count,
                "rows_changed": changed,
                "rows_da3metric_top_truncated_fallback": fallback,
                "canonical_labels": ",".join(sorted(labels)),
            }
        )
        print(f"VIEW_DONE view={view} rows={count} changed={changed} da3_top_fallback={fallback} output={dst}", flush=True)

    write_csv(output_root / "summary.csv", summary_rows)
    print(f"DONE output_root={output_root}", flush=True)
    return 0


def rewrite_one_csv(
    config: dict[str, Any],
    src: Path,
    dst: Path,
    intrinsic,
    fallback_da3_when_top_truncated: bool = False,
    top_truncation_margin_px: float = 2.0,
) -> tuple[int, int, set[str], int]:
    rows: list[dict[str, str]] = []
    labels: set[str] = set()
    changed = 0
    fallback = 0
    with src.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header: {src}")
        fieldnames = list(reader.fieldnames)
        for key in ["depth_source", "depth_original_median", "depth_original_weighted_mean", "x_cam_original", "y_cam_original", "z_cam_original", "canonical_label", "height_prior_m"]:
            if key not in fieldnames:
                fieldnames.append(key)
        for row in reader:
            label_raw = row.get("gt_label") or row.get("label") or row.get("prompt") or "default"
            label = canonical_label(config, label_raw)
            init_size, _, _ = class_sizes(config, label)
            height_prior = float(init_size[2])
            x1, y1, x2, y2 = [float(row[k]) for k in ("x1", "y1", "x2", "y2")]
            bbox_h = max(y2 - y1, 1.0)
            depth = float(intrinsic[1, 1]) * height_prior / bbox_h
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            x_cam = (cx - float(intrinsic[0, 2])) * depth / float(intrinsic[0, 0])
            y_cam = (cy - float(intrinsic[1, 2])) * depth / float(intrinsic[1, 1])
            original_depth_median = row.get("depth_median", "")
            original_depth_weighted_mean = row.get("depth_weighted_mean", "")
            original_x_cam = row.get("x_cam", "")
            original_y_cam = row.get("y_cam", "")
            original_z_cam = row.get("z_cam", "")
            top_truncated = y1 <= float(top_truncation_margin_px)
            use_da3 = bool(fallback_da3_when_top_truncated and top_truncated and is_positive_finite(original_z_cam))

            row["depth_original_median"] = original_depth_median
            row["depth_original_weighted_mean"] = original_depth_weighted_mean
            row["x_cam_original"] = original_x_cam
            row["y_cam_original"] = original_y_cam
            row["z_cam_original"] = original_z_cam
            row["canonical_label"] = label
            row["height_prior_m"] = f"{height_prior:.9g}"
            if use_da3:
                row["depth_source"] = "da3metric_top_truncated_fallback"
                row["depth_median"] = original_depth_median
                row["depth_weighted_mean"] = original_depth_weighted_mean or original_depth_median
                row["x_cam"] = original_x_cam
                row["y_cam"] = original_y_cam
                row["z_cam"] = original_z_cam
                fallback += 1
            else:
                row["depth_source"] = "height_prior"
                row["depth_median"] = f"{depth:.9g}"
                row["depth_weighted_mean"] = f"{depth:.9g}"
                row["x_cam"] = f"{x_cam:.9g}"
                row["y_cam"] = f"{y_cam:.9g}"
                row["z_cam"] = f"{depth:.9g}"
            labels.add(label)
            changed += 1
            rows.append(row)
    with dst.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), changed, labels, fallback


def is_positive_finite(value: object) -> bool:
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        return False
    return np.isfinite(out) and out > 0


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
