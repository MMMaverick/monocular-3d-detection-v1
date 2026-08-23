from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export existing 2D boxes from original annotation JSONs to per-camera CSVs.")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--camera-root", required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-frames", type=int, default=-1)
    args = parser.parse_args()

    ann_paths = sorted(Path(args.annotations).glob("*.json"))
    if int(args.max_frames) >= 0:
        ann_paths = ann_paths[: int(args.max_frames)]
    cameras = [x.strip() for x in str(args.cameras).split(",") if x.strip()]
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary = []
    for camera in cameras:
        rows = export_camera(ann_paths, Path(args.camera_root), camera)
        out_dir = out_root / camera
        out_csv = out_dir / "tracks.csv"
        write_csv(out_csv, rows)
        summary.append({"camera": camera, "rows": len(rows), "csv": str(out_csv)})
        print(f"EXPORT_2DBOX camera={camera} rows={len(rows)} csv={out_csv}", flush=True)
    write_csv(out_root / "summary.csv", summary)
    return 0


def export_camera(ann_paths: list[Path], camera_root: Path, camera: str) -> list[dict[str, object]]:
    image_index = index_images(camera_root / camera)
    rows: list[dict[str, object]] = []
    for frame, ann_path in enumerate(ann_paths):
        payload = json.loads(ann_path.read_text(encoding="utf-8"))
        cam = (payload.get("cams") or payload.get("cameras") or {}).get(camera)
        if not cam:
            continue
        timestamp = int(cam.get("timestamp", -1))
        image_path = image_index.get(timestamp)
        if image_path is None:
            image_path = nearest_image(image_index, timestamp)
        boxes = cam.get("boxes_2d", []) or []
        labels = cam.get("boxes_2d_label", []) or cam.get("sub_type_name", []) or []
        scores = cam.get("boxes_2d_score", []) or []
        indices = cam.get("boxes_2d_index", []) or []
        gt_track_ids = payload.get("track_id", []) or []
        for i, box in enumerate(boxes):
            if len(box) < 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            if x2 <= x1 or y2 <= y1:
                continue
            gt_index = int(indices[i]) if i < len(indices) and indices[i] is not None else -1
            source_track_id = -1
            if 0 <= gt_index < len(gt_track_ids):
                try:
                    source_track_id = int(gt_track_ids[gt_index])
                except Exception:
                    source_track_id = gt_index
            label = str(labels[i]) if i < len(labels) else "default"
            score = float(scores[i]) if i < len(scores) and scores[i] is not None else 1.0
            rows.append(
                {
                    "frame": frame,
                    "timestamp": timestamp,
                    "image": str(image_path or ""),
                    "source_track_id": source_track_id,
                    "track_id": source_track_id,
                    "gt2d_index": i,
                    "gt_index": gt_index,
                    "gt_label": label,
                    "label": label,
                    "prompt": label,
                    "score": score,
                    "gt2d_score": score,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": 0.5 * (x1 + x2),
                    "cy": 0.5 * (y1 + y2),
                }
            )
    return rows


def index_images(image_dir: Path) -> dict[int, Path]:
    out = {}
    for path in sorted(image_dir.glob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        try:
            out[int(path.stem)] = path
        except ValueError:
            continue
    return out


def nearest_image(index: dict[int, Path], timestamp: int) -> Path | None:
    if not index:
        return None
    return index[min(index, key=lambda key: abs(key - timestamp))]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
