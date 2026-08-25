from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry


DEFAULT_CAMERAS = "left_rear_camera,right_rear_camera,rear_camera"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment every GT 2D box with classic SAM box prompts.")
    parser.add_argument("--annotations", default="data/format_output/annotations/NV")
    parser.add_argument("--camera-root", default="data/camera")
    parser.add_argument("--cameras", default=DEFAULT_CAMERAS)
    parser.add_argument("--output", default="outputs/rear_3view_gt2d_sam_box15_masks")
    parser.add_argument("--checkpoint", default="checkpoints/sam_vit_h_4b8939.pth")
    parser.add_argument("--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b", "default"])
    parser.add_argument("--box-scale", type=float, default=1.5)
    parser.add_argument("--positive-points", type=int, default=5, help="Sample positive point prompts inside the original GT 2D box.")
    parser.add_argument("--clip-mask-to-original-box", action="store_true", default=True, help="Force final mask to stay inside the original GT 2D box.")
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--mask-format", choices=["png", "npz"], default="png")
    parser.add_argument("--min-gt2d-score", type=float, default=0.0)
    parser.add_argument("--multimask-output", action="store_true", help="Ask SAM for 3 masks and keep the highest-score one.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {checkpoint}\n"
            "Download one of Meta SAM checkpoints and place it there, for example sam_vit_h_4b8939.pth."
        )
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print(f"Loading SAM {args.model_type}: {checkpoint}")
    sam = sam_model_registry[str(args.model_type)](checkpoint=str(checkpoint))
    sam.to(device=device)
    sam.eval()
    predictor = SamPredictor(sam)

    ann_paths = sorted(Path(args.annotations).glob("*.json"))
    if int(args.max_frames) > 0:
        ann_paths = ann_paths[: int(args.max_frames)]
    cameras = [x.strip() for x in str(args.cameras).split(",") if x.strip()]
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for camera in cameras:
        summaries.append(_process_camera(camera, ann_paths, predictor, args, output_root / camera))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    text = "\n".join(f"{s['camera']}: frames={s['frames']} gt2d={s['gt2d']} masks={s['masks']} video={s['video']}" for s in summaries)
    (output_root / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


def _process_camera(camera: str, ann_paths: list[Path], predictor: SamPredictor, args: argparse.Namespace, out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = out_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    image_map = _image_path_map(Path(args.camera_root) / camera)
    rows: list[dict[str, object]] = []
    video_path = out_dir / "gt2d_sam_box15_masks.mp4"
    csv_path = out_dir / "gt2d_sam_masks.csv"
    writer = None
    rendered = 0
    total_gt2d = 0
    total_masks = 0
    last_image_path: Path | None = None

    for frame_idx, ann_path in enumerate(ann_paths):
        ann = json.loads(ann_path.read_text(encoding="utf-8"))
        cam = ann.get("cams", {}).get(camera)
        if not cam:
            continue
        image_path = image_map.get(int(cam.get("timestamp", -1)))
        if image_path is None:
            continue
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            continue
        height, width = bgr.shape[:2]
        if writer is None:
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), int(args.fps), (width, height))
        if last_image_path != image_path:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            predictor.set_image(rgb)
            last_image_path = image_path
        overlay = bgr.copy()
        frame_rows: list[dict[str, object]] = []
        for item in _camera_gt2d_items(ann, cam, frame_idx, image_path, width, height, args):
            total_gt2d += 1
            point_coords = _prompt_points(item["box"], int(args.positive_points))
            masks, scores, _logits = predictor.predict(
                point_coords=point_coords,
                point_labels=np.ones(len(point_coords), dtype=np.int32) if point_coords is not None else None,
                box=np.asarray(item["expanded_box"], dtype=np.float32),
                multimask_output=bool(args.multimask_output),
            )
            if masks is None or len(masks) == 0:
                item.update({"sam_score": float("nan"), "mask_area": 0, "mask_path": "", "mask_x1": float("nan"), "mask_y1": float("nan"), "mask_x2": float("nan"), "mask_y2": float("nan")})
                frame_rows.append(item)
                _draw_item(overlay, item, None)
                continue
            best_idx = int(np.argmax(scores))
            mask = masks[best_idx].astype(bool)
            if bool(args.clip_mask_to_original_box):
                mask = _clip_mask_to_box(mask, item["box"])
            total_masks += 1
            mask_path = _save_mask(mask_dir, item, mask, str(args.mask_format))
            mx1, my1, mx2, my2 = _mask_bbox(mask)
            item.update({"sam_score": float(scores[best_idx]), "mask_area": int(mask.sum()), "mask_path": str(mask_path), "mask_x1": mx1, "mask_y1": my1, "mask_x2": mx2, "mask_y2": my2})
            frame_rows.append(item)
            _draw_item(overlay, item, mask)
        rows.extend(frame_rows)
        blended = cv2.addWeighted(overlay, 0.62, bgr, 0.38, 0)
        writer.write(blended)
        rendered += 1
        if rendered % 25 == 0 or rendered == len(ann_paths):
            print(f"{camera}: frames={rendered}/{len(ann_paths)} gt2d={total_gt2d} masks={total_masks}")
    if writer is not None:
        writer.release()
    _write_csv(csv_path, rows)
    summary = {"camera": camera, "frames": rendered, "gt2d": total_gt2d, "masks": total_masks, "video": str(video_path), "csv": str(csv_path)}
    (out_dir / "summary.txt").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _camera_gt2d_items(ann: dict, cam: dict, frame_idx: int, image_path: Path, width: int, height: int, args: argparse.Namespace) -> list[dict[str, object]]:
    boxes = cam.get("boxes_2d", []) or []
    labels = cam.get("boxes_2d_label", []) or []
    scores = cam.get("boxes_2d_score", []) or []
    indices = cam.get("boxes_2d_index", []) or []
    gt_track_ids = ann.get("track_id", []) or []
    out = []
    for i, box in enumerate(boxes):
        score = float(scores[i]) if i < len(scores) and scores[i] is not None else 1.0
        if score < float(args.min_gt2d_score):
            continue
        gt_index = indices[i] if i < len(indices) else None
        gt_index_i = int(gt_index) if gt_index is not None else -1
        track_id = int(gt_track_ids[gt_index_i]) if 0 <= gt_index_i < len(gt_track_ids) else -1
        box_np = _clip_box(np.asarray(box, dtype=np.float64), width, height)
        if box_np[2] <= box_np[0] or box_np[3] <= box_np[1]:
            continue
        expanded = _expand_box(box_np, float(args.box_scale), width, height)
        out.append(
            {
                "frame": int(frame_idx),
                "timestamp": int(cam.get("timestamp", -1)),
                "image": str(image_path),
                "gt2d_index": int(i),
                "gt_index": gt_index_i,
                "track_id": track_id,
                "label": str(labels[i]) if i < len(labels) else "",
                "gt2d_score": score,
                "x1": float(box_np[0]),
                "y1": float(box_np[1]),
                "x2": float(box_np[2]),
                "y2": float(box_np[3]),
                "ex1": float(expanded[0]),
                "ey1": float(expanded[1]),
                "ex2": float(expanded[2]),
                "ey2": float(expanded[3]),
                "expanded_box": expanded,
                "box": box_np,
            }
        )
    return out


def _save_mask(mask_dir: Path, item: dict[str, object], mask: np.ndarray, mask_format: str) -> Path:
    label = _safe_name(str(item["label"]))
    stem = f"frame_{int(item['frame']):06d}_gt2d_{int(item['gt2d_index']):03d}_track_{int(item['track_id'])}_{label}"
    if mask_format == "npz":
        path = mask_dir / f"{stem}.npz"
        np.savez_compressed(path, mask=mask.astype(np.bool_))
    else:
        path = mask_dir / f"{stem}.png"
        cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
    return path


def _prompt_points(box: np.ndarray, num_points: int) -> np.ndarray | None:
    # Points are inside the original GT 2D box, not the enlarged SAM box prompt.
    if int(num_points) <= 0:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    if x2 <= x1 or y2 <= y1:
        return None
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    dx, dy = 0.22 * (x2 - x1), 0.22 * (y2 - y1)
    base = [[cx, cy], [cx - dx, cy], [cx + dx, cy], [cx, cy - dy], [cx, cy + dy]]
    if num_points <= len(base):
        return np.asarray(base[:num_points], dtype=np.float32)
    grid = []
    for yy in np.linspace(y1 + 0.25 * (y2 - y1), y2 - 0.25 * (y2 - y1), 3):
        for xx in np.linspace(x1 + 0.25 * (x2 - x1), x2 - 0.25 * (x2 - x1), 3):
            grid.append([float(xx), float(yy)])
    points = base + grid
    return np.asarray(points[:num_points], dtype=np.float32)


def _clip_mask_to_box(mask: np.ndarray, box: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in box]
    x1, x2 = int(np.clip(x1, 0, w)), int(np.clip(x2, 0, w))
    y1, y2 = int(np.clip(y1, 0, h)), int(np.clip(y2, 0, h))
    if x2 > x1 and y2 > y1:
        out[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return out


def _draw_item(frame: np.ndarray, item: dict[str, object], mask: np.ndarray | None) -> None:
    color = _color_for_label(str(item["label"]), int(item["gt2d_index"]))
    if mask is not None:
        frame[mask] = (0.42 * frame[mask] + 0.58 * np.asarray(color, dtype=np.float32)).astype(np.uint8)
    x1, y1, x2, y2 = [int(round(float(item[k]))) for k in ("x1", "y1", "x2", "y2")]
    ex1, ey1, ex2, ey2 = [int(round(float(item[k]))) for k in ("ex1", "ey1", "ex2", "ey2")]
    cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (170, 170, 170), 1, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    _put_label(frame, f"{item['label']} id={item['track_id']} sam={float(item.get('sam_score', float('nan'))):.2f}", x1, max(0, y1 - 20), color)


def _image_path_map(image_dir: Path) -> dict[int, Path]:
    out = {}
    for path in _list_images(image_dir):
        try:
            out[int(path.stem)] = path
        except ValueError:
            pass
    return out


def _list_images(image_dir: Path) -> list[Path]:
    """List camera frames locally; avoids depending on old visualization helpers."""

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions)


def _expand_box(box: np.ndarray, scale: float, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    return _clip_box(np.asarray([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dtype=np.float64), width, height)


def _clip_box(box: np.ndarray, width: int, height: int) -> np.ndarray:
    out = np.asarray(box, dtype=np.float64).copy()
    out[[0, 2]] = np.clip(out[[0, 2]], 0.0, max(width - 1, 0))
    out[[1, 3]] = np.clip(out[[1, 3]], 0.0, max(height - 1, 0))
    return out


def _mask_bbox(mask: np.ndarray) -> tuple[float, float, float, float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "frame",
        "timestamp",
        "image",
        "gt2d_index",
        "gt_index",
        "track_id",
        "label",
        "gt2d_score",
        "x1",
        "y1",
        "x2",
        "y2",
        "ex1",
        "ey1",
        "ex2",
        "ey2",
        "sam_score",
        "mask_area",
        "mask_x1",
        "mask_y1",
        "mask_x2",
        "mask_y2",
        "mask_path",
    ]
    cleaned = [{k: v for k, v in row.items() if k != "expanded_box"} for row in rows]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cleaned)


def _color_for_label(label: str, idx: int) -> tuple[int, int, int]:
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(label)) + idx * 9973
    rng = np.random.default_rng(seed)
    vals = rng.integers(48, 255, size=3)
    return int(vals[0]), int(vals[1]), int(vals[2])


def _put_label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = int(np.clip(x, 0, max(image.shape[1] - tw - 6, 0)))
    y = int(np.clip(y, th + 4, max(image.shape[0] - baseline - 4, th + 4)))
    cv2.rectangle(image, (x, y - th - 4), (x + tw + 6, y + baseline + 2), (0, 0, 0), -1)
    cv2.putText(image, text, (x + 3, y), font, scale, color, thickness, cv2.LINE_AA)


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)


if __name__ == "__main__":
    main()
