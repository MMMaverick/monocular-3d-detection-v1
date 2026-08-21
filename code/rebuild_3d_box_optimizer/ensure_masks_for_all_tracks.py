from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, resolve_path, write_resolved_config
from .run import append_progress


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create one cropped mask for every tracking bbox.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="preprocessed/masks_ensured_for_tracks")
    parser.add_argument("--min-iou", type=float, default=0.30)
    parser.add_argument("--allow-label-mismatch", action="store_true", default=False)
    args = parser.parse_args(argv)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to write ensured mask PNGs.") from exc

    config = load_config(args.config)
    out_root = resolve_path(config, args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    progress_path = out_root / "progress.log"
    progress_path.write_text("", encoding="utf-8")
    shutil.copy2(args.config, out_root / "source_config.yaml")
    write_resolved_config(config, out_root / "resolved_config.yaml")
    append_progress(progress_path, f"START output={out_root} min_iou={args.min_iou}")

    summaries = []
    for view in config.get("scope", {}).get("cameras", ["rear", "left_rear", "right_rear"]):
        summaries.append(process_view(config, view, out_root, progress_path, cv2, float(args.min_iou), bool(args.allow_label_mismatch)))
    write_csv(out_root / "summary.csv", summaries)
    append_progress(progress_path, "DONE")
    return 0


def process_view(config: dict[str, Any], view: str, out_root: Path, progress_path: Path, cv2, min_iou: float, allow_label_mismatch: bool) -> dict[str, object]:
    view_cfg = config["inputs"]["views"][view]
    track_rows = read_csv(resolve_path(config, view_cfg["track_csv"]))
    raw_mask_rows = read_csv(resolve_path(config, view_cfg["mask_csv"]))
    raw_masks_by_frame = group_by_int_field(raw_mask_rows, "frame")
    out_dir = out_root / str(view_cfg.get("camera_name", view))
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    append_progress(progress_path, f"VIEW_START view={view} track_boxes={len(track_rows)} raw_masks={len(raw_mask_rows)}")

    output_rows: list[dict[str, object]] = []
    matched_sam = 0
    bbox_fallback = 0
    no_raw_mask_same_frame = 0
    low_iou = 0
    empty_after_crop = 0
    missing_image = 0
    for idx, track_row in enumerate(track_rows):
        frame = int_float(track_row.get("frame", -1))
        track_id = int_float(track_row.get("track_id", -1))
        if track_id < 0:
            continue
        image_shape = read_image_shape(config, track_row, cv2)
        if image_shape is None:
            missing_image += 1
            continue
        candidates = raw_masks_by_frame.get(frame, [])
        if not candidates:
            no_raw_mask_same_frame += 1
        chosen, best_iou = choose_mask_for_track(track_row, candidates, allow_label_mismatch)
        mask_source = "matched_sam_crop"
        mask = None
        reason = ""
        if chosen is not None and best_iou >= min_iou:
            mask = load_and_crop_sam_mask(config, chosen, track_row, image_shape, cv2)
            if mask is None or int(mask.sum()) <= 0:
                empty_after_crop += 1
                reason = "matched_sam_empty_after_crop"
                mask = None
            else:
                matched_sam += 1
        else:
            if candidates:
                low_iou += 1
                reason = f"best_iou_below_threshold:{best_iou:.4f}"
            else:
                reason = "no_raw_mask_same_frame"

        if mask is None:
            mask_source = "bbox_filled_fallback"
            bbox_fallback += 1
            mask = bbox_filled_mask(track_row, image_shape)
        bbox = mask_bbox(mask)
        area = int(mask.sum())
        out_path = mask_dir / f"frame_{frame:06d}_track_{track_id}_ensured.png"
        cv2.imwrite(str(out_path), mask.astype(np.uint8) * 255)
        row = make_output_row(config, view, track_row, chosen, out_path, bbox, area, best_iou, mask_source, reason)
        output_rows.append(row)
        if idx == 0 or (idx + 1) % 500 == 0 or idx + 1 == len(track_rows):
            append_progress(progress_path, f"VIEW_PROGRESS view={view} processed={idx + 1}/{len(track_rows)} matched_sam={matched_sam} bbox_fallback={bbox_fallback}")

    out_csv = out_dir / "gt2d_sam_masks_ensured_cropped.csv"
    write_csv(out_csv, output_rows)
    summary = {
        "view": view,
        "track_boxes": len(track_rows),
        "output_masks": len(output_rows),
        "matched_sam_crop": matched_sam,
        "bbox_filled_fallback": bbox_fallback,
        "no_raw_mask_same_frame": no_raw_mask_same_frame,
        "best_iou_below_threshold": low_iou,
        "empty_after_crop": empty_after_crop,
        "missing_image": missing_image,
        "output_csv": str(out_csv),
    }
    append_progress(progress_path, f"VIEW_DONE view={view} output_masks={len(output_rows)} matched_sam={matched_sam} bbox_fallback={bbox_fallback} csv={out_csv}")
    return summary


def choose_mask_for_track(track_row: dict[str, str], candidates: list[dict[str, str]], allow_label_mismatch: bool) -> tuple[dict[str, str] | None, float]:
    track_box = parse_box(track_row, ("x1", "y1", "x2", "y2"))
    if track_box is None:
        return None, 0.0
    track_label = normalize_label(track_row.get("gt_label") or track_row.get("prompt") or track_row.get("label", ""))
    best = None
    best_iou = 0.0
    for row in candidates:
        mask_box = parse_box(row, ("x1", "y1", "x2", "y2")) or parse_box(row, ("mask_x1", "mask_y1", "mask_x2", "mask_y2"))
        if mask_box is None:
            continue
        if not allow_label_mismatch:
            mask_label = normalize_label(row.get("label", ""))
            if mask_label and track_label and mask_label != track_label and not compatible_vehicle(mask_label, track_label):
                continue
        iou = box_iou(np.asarray(track_box), np.asarray(mask_box))
        if iou > best_iou:
            best = row
            best_iou = iou
    return best, best_iou


def load_and_crop_sam_mask(config: dict[str, Any], mask_row: dict[str, str], track_row: dict[str, str], image_shape: tuple[int, int], cv2) -> np.ndarray | None:
    path_text = mask_row.get("mask_path", "")
    if not path_text:
        return None
    path = resolve_path(config, path_text)
    if not path.exists():
        return None
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        return None
    h, w = image_shape
    if raw.shape[:2] != (h, w):
        raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
    box = parse_box(track_row, ("x1", "y1", "x2", "y2"))
    if box is None:
        return None
    x1, y1, x2, y2 = clip_int_box(box, w, h)
    mask = np.zeros((h, w), dtype=np.bool_)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = raw[y1:y2, x1:x2] > 0
    return mask


def bbox_filled_mask(track_row: dict[str, str], image_shape: tuple[int, int]) -> np.ndarray:
    h, w = image_shape
    mask = np.zeros((h, w), dtype=np.bool_)
    box = parse_box(track_row, ("x1", "y1", "x2", "y2"))
    if box is None:
        return mask
    x1, y1, x2, y2 = clip_int_box(box, w, h)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


def make_output_row(
    config: dict[str, Any],
    view: str,
    track_row: dict[str, str],
    source_mask_row: dict[str, str] | None,
    out_path: Path,
    bbox: tuple[int, int, int, int],
    area: int,
    best_iou: float,
    mask_source: str,
    fallback_reason: str,
) -> dict[str, object]:
    root = Path(config["_root_dir"])
    x1, y1, x2, y2 = parse_box(track_row, ("x1", "y1", "x2", "y2")) or (0.0, 0.0, 0.0, 0.0)
    return {
        "frame": int_float(track_row.get("frame", -1)),
        "timestamp": track_row.get("annotation_timestamp") or Path(track_row.get("image", "")).stem,
        "image": track_row.get("image", ""),
        "gt2d_index": source_mask_row.get("gt2d_index", "") if source_mask_row else "",
        "gt_index": track_row.get("gt_index", source_mask_row.get("gt_index", "") if source_mask_row else ""),
        "track_id": int_float(track_row.get("track_id", -1)),
        "label": track_row.get("gt_label") or track_row.get("prompt") or source_mask_row.get("label", "") if source_mask_row else track_row.get("gt_label") or track_row.get("prompt") or "",
        "gt2d_score": source_mask_row.get("gt2d_score", "") if source_mask_row else "",
        "x1": float(x1),
        "y1": float(y1),
        "x2": float(x2),
        "y2": float(y2),
        "sam_score": source_mask_row.get("sam_score", "") if source_mask_row else "",
        "mask_area": int(area),
        "mask_x1": float(bbox[0]),
        "mask_y1": float(bbox[1]),
        "mask_x2": float(bbox[2]),
        "mask_y2": float(bbox[3]),
        "mask_path": str(out_path.relative_to(root)).replace("\\", "/"),
        "mask_source": mask_source,
        "source_mask_track_id": int_float(source_mask_row.get("track_id", -1)) if source_mask_row else "",
        "source_mask_path": source_mask_row.get("mask_path", "") if source_mask_row else "",
        "match_iou": float(best_iou),
        "fallback_reason": fallback_reason,
        "view": view,
    }


def read_image_shape(config: dict[str, Any], track_row: dict[str, str], cv2) -> tuple[int, int] | None:
    image = track_row.get("image", "")
    if not image:
        return None
    path = resolve_path(config, image)
    frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        return None
    return frame.shape[:2]


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))


def normalize_label(label: str) -> str:
    text = str(label).strip().lower()
    if text.startswith("vehicle_"):
        text = text[len("vehicle_") :]
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
    return a in {"car", "suv", "truck", "bus"} and b in {"car", "suv", "truck", "bus"}


def parse_box(row: dict[str, str], keys: tuple[str, str, str, str]) -> tuple[float, float, float, float] | None:
    try:
        vals = tuple(float(row[k]) for k in keys)
    except (KeyError, ValueError):
        return None
    return vals if np.isfinite(vals).all() else None


def clip_int_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    return (
        int(np.ceil(np.clip(box[0], 0, width))),
        int(np.ceil(np.clip(box[1], 0, height))),
        int(np.floor(np.clip(box[2], 0, width))),
        int(np.floor(np.clip(box[3], 0, height))),
    )


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
        grouped.setdefault(int_float(row.get(field, -1)), []).append(row)
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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
