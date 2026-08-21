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
    parser = argparse.ArgumentParser(description="Bind GT2D SAM masks back to tracking ids and crop masks to the matched 2D bbox.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="preprocessed/masks_bound_to_tracks")
    parser.add_argument("--min-iou", type=float, default=0.50)
    parser.add_argument("--allow-label-mismatch", action="store_true", default=False)
    args = parser.parse_args(argv)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to crop mask PNGs.") from exc

    config = load_config(args.config)
    out_root = resolve_path(config, args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    progress_path = out_root / "progress.log"
    progress_path.write_text("", encoding="utf-8")
    shutil.copy2(args.config, out_root / "source_config.yaml")
    write_resolved_config(config, out_root / "resolved_config.yaml")
    append_progress(progress_path, f"START output={out_root} min_iou={args.min_iou}")

    summary_rows: list[dict[str, object]] = []
    for view in config.get("scope", {}).get("cameras", ["rear", "left_rear", "right_rear"]):
        summary_rows.append(process_view(config, view, out_root, progress_path, cv2, float(args.min_iou), bool(args.allow_label_mismatch)))
    write_csv(out_root / "summary.csv", summary_rows)
    append_progress(progress_path, "DONE")
    return 0


def process_view(config: dict[str, Any], view: str, out_root: Path, progress_path: Path, cv2, min_iou: float, allow_label_mismatch: bool) -> dict[str, object]:
    view_cfg = config["inputs"]["views"][view]
    track_rows = read_csv(resolve_path(config, view_cfg["track_csv"]))
    mask_rows = read_csv(resolve_path(config, view_cfg["mask_csv"]))
    tracks_by_frame = group_by_int_field(track_rows, "frame")
    out_dir = out_root / str(view_cfg.get("camera_name", view))
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "gt2d_sam_masks_bound_cropped.csv"
    append_progress(progress_path, f"VIEW_START view={view} tracks={len(track_rows)} masks={len(mask_rows)}")

    output_rows: list[dict[str, object]] = []
    matched = 0
    cropped_nonempty = 0
    rejected = 0
    direct_kept = 0
    for idx, mask_row in enumerate(mask_rows):
        frame = int_float(mask_row.get("frame", -1))
        candidates = tracks_by_frame.get(frame, [])
        match = choose_track(mask_row, candidates, min_iou, allow_label_mismatch)
        if match is None:
            rejected += 1
            continue
        track_row, iou = match
        if int_float(mask_row.get("track_id", -1)) == int_float(track_row.get("track_id", -1)):
            direct_kept += 1
        matched += 1
        cropped = crop_mask_to_track_box(config, mask_row, track_row, cv2)
        if cropped is None:
            rejected += 1
            continue
        cropped_mask, bbox, area = cropped
        if area <= 0:
            rejected += 1
            continue
        cropped_nonempty += 1
        out_path = mask_dir / f"frame_{frame:06d}_track_{int_float(track_row.get('track_id'))}_src_{idx:05d}.png"
        cv2.imwrite(str(out_path), cropped_mask.astype(np.uint8) * 255)
        row = dict(mask_row)
        row.update(
            {
                "track_id": int_float(track_row.get("track_id")),
                "matched_track_id": int_float(track_row.get("track_id")),
                "source_track_id": int_float(mask_row.get("track_id", -1)),
                "match_iou": float(iou),
                "match_method": "frame_bbox_iou_then_crop_to_track_bbox",
                "x1": float(track_row["x1"]),
                "y1": float(track_row["y1"]),
                "x2": float(track_row["x2"]),
                "y2": float(track_row["y2"]),
                "track_prompt": track_row.get("prompt", ""),
                "track_label": track_row.get("gt_label", ""),
                "mask_area": int(area),
                "mask_x1": float(bbox[0]),
                "mask_y1": float(bbox[1]),
                "mask_x2": float(bbox[2]),
                "mask_y2": float(bbox[3]),
                "mask_path": str(out_path.relative_to(Path(config["_root_dir"]))).replace("\\", "/"),
            }
        )
        output_rows.append(row)
        if matched == 1 or matched % 500 == 0:
            append_progress(progress_path, f"VIEW_PROGRESS view={view} matched={matched} output_masks={cropped_nonempty}")

    write_csv(out_csv, output_rows)
    summary = {
        "view": view,
        "input_masks": len(mask_rows),
        "input_tracks": len(track_rows),
        "matched_masks": matched,
        "direct_track_id_kept": direct_kept,
        "cropped_nonempty_masks": cropped_nonempty,
        "rejected_masks": rejected,
        "output_csv": str(out_csv),
    }
    append_progress(progress_path, f"VIEW_DONE view={view} matched={matched} output={cropped_nonempty} rejected={rejected} csv={out_csv}")
    return summary


def choose_track(mask_row: dict[str, str], candidates: list[dict[str, str]], min_iou: float, allow_label_mismatch: bool) -> tuple[dict[str, str], float] | None:
    mask_box = parse_box(mask_row, ("x1", "y1", "x2", "y2"))
    if mask_box is None:
        mask_box = parse_box(mask_row, ("mask_x1", "mask_y1", "mask_x2", "mask_y2"))
    if mask_box is None:
        return None
    mask_label = normalize_label(mask_row.get("label", ""))
    best = None
    best_iou = 0.0
    for row in candidates:
        track_box = parse_box(row, ("x1", "y1", "x2", "y2"))
        if track_box is None:
            continue
        if not allow_label_mismatch:
            track_label = normalize_label(row.get("gt_label") or row.get("prompt") or row.get("label", ""))
            if mask_label and track_label and mask_label != track_label and not compatible_vehicle(mask_label, track_label):
                continue
        iou = box_iou(np.asarray(mask_box), np.asarray(track_box))
        if iou > best_iou:
            best = row
            best_iou = iou
    if best is None or best_iou < min_iou:
        return None
    return best, best_iou


def crop_mask_to_track_box(config: dict[str, Any], mask_row: dict[str, str], track_row: dict[str, str], cv2) -> tuple[np.ndarray, tuple[int, int, int, int], int] | None:
    mask_path_text = mask_row.get("mask_path", "")
    if not mask_path_text:
        return None
    mask_path = resolve_path(config, mask_path_text)
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    box = parse_box(track_row, ("x1", "y1", "x2", "y2"))
    if box is None:
        return None
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = clip_int_box(box, w, h)
    cropped = np.zeros(mask.shape, dtype=np.bool_)
    if x2 <= x1 or y2 <= y1:
        return None
    active = mask[y1:y2, x1:x2] > 0
    cropped[y1:y2, x1:x2] = active
    ys, xs = np.nonzero(cropped)
    if len(xs) == 0:
        return cropped, (x1, y1, x1, y1), 0
    return cropped, (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)), int(len(xs))


def normalize_label(label: str) -> str:
    text = str(label).strip().lower()
    for prefix in ("vehicle_",):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if "truck" in text:
        return "truck"
    if "bus" in text:
        return "bus"
    if "suv" in text:
        return "suv"
    if "car" in text:
        return "car"
    if "pedestrian" in text or "person" in text:
        return "person"
    if "motor" in text:
        return "motor"
    if "cycl" in text or "bicycle" in text:
        return "bicycle"
    return text


def compatible_vehicle(a: str, b: str) -> bool:
    vehicles = {"car", "suv", "truck", "bus"}
    return a in vehicles and b in vehicles


def parse_box(row: dict[str, str], keys: tuple[str, str, str, str]) -> tuple[float, float, float, float] | None:
    try:
        vals = tuple(float(row[k]) for k in keys)
    except (KeyError, ValueError):
        return None
    return vals if np.isfinite(vals).all() else None


def clip_int_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1 = int(np.ceil(np.clip(box[0], 0, width)))
    y1 = int(np.ceil(np.clip(box[1], 0, height)))
    x2 = int(np.floor(np.clip(box[2], 0, width)))
    y2 = int(np.floor(np.clip(box[3], 0, height)))
    return x1, y1, x2, y2


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(inter / max(area_a + area_b - inter, 1.0e-9))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def group_by_int_field(rows: list[dict[str, str]], field: str) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        key = int_float(row.get(field, -1))
        grouped.setdefault(key, []).append(row)
    return grouped


def int_float(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return -1


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
