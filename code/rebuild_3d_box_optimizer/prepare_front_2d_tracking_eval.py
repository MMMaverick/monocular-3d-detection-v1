from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMERAS = ("center_camera_fov120", "left_front_camera", "right_front_camera")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split front-view GT 2D detections from held-out track-ID evaluation labels.")
    parser.add_argument("--annotations", default="data/format_output/annotations/NV")
    parser.add_argument("--image-root", default="data/camera")
    parser.add_argument("--output", default="preprocessed/front_2d_tracking_eval")
    args = parser.parse_args(argv)
    annotations = sorted(resolve(args.annotations).glob("*.json"))
    output = resolve(args.output)
    for camera in CAMERAS:
        image_map = {int(path.stem): path for path in (resolve(args.image_root) / camera).glob("*.jpg")}
        detections: list[dict[str, Any]] = []
        truth: list[dict[str, Any]] = []
        for frame, path in enumerate(annotations):
            payload = json.loads(path.read_text(encoding="utf-8"))
            cam = payload.get("cams", {}).get(camera, {})
            timestamp = int(cam.get("timestamp", -1))
            image = image_map.get(timestamp)
            if image is None:
                continue
            boxes = cam.get("boxes_2d", []) or []
            labels = cam.get("boxes_2d_label", []) or []
            scores = cam.get("boxes_2d_score", []) or []
            indices = cam.get("boxes_2d_index", []) or []
            gt_ids = payload.get("track_id", []) or []
            for index, box in enumerate(boxes):
                if len(box) != 4 or float(box[2]) <= float(box[0]) or float(box[3]) <= float(box[1]):
                    continue
                row = {
                    "frame": frame,
                    "timestamp": timestamp,
                    "image": str(image.relative_to(PROJECT_ROOT)),
                    "gt2d_index": index,
                    "label": str(labels[index]) if index < len(labels) else "unknown",
                    "gt2d_score": float(scores[index]) if index < len(scores) and scores[index] is not None else 1.0,
                    "x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3]),
                    "mask_path": "", "sam_score": "",
                }
                detections.append(row)
                gt_index = int(indices[index]) if index < len(indices) and indices[index] is not None else -1
                gt_id = int(gt_ids[gt_index]) if 0 <= gt_index < len(gt_ids) else -1
                truth.append({"view": camera, "frame": frame, "timestamp": timestamp, "gt2d_index": index, "gt_index": gt_index, "gt_track_id": gt_id, "label": row["label"], "x1": row["x1"], "y1": row["y1"], "x2": row["x2"], "y2": row["y2"]})
        camera_dir = output / camera
        camera_dir.mkdir(parents=True, exist_ok=True)
        write_csv(camera_dir / "detections.csv", detections)
        write_csv(camera_dir / "evaluation_gt.csv", truth)
        print(f"{camera}: frames={len(annotations)} detections={len(detections)} gt_labels={sum(int(r['gt_track_id']) >= 0 for r in truth)}")
    return 0


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not rows:
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
