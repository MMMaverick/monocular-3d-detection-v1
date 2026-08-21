from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_DIR = PROJECT_ROOT / ".deps"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))


CANONICAL_LABELS = {
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
}


@dataclass
class Detection:
    row: dict[str, str]
    box: np.ndarray
    score: float
    label: str
    class_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run independent appearance-assisted BoT-SORT on rear/left-rear/right-rear 2D detections."
    )
    parser.add_argument("--config", default="configs/robust_botsort_2d_rear_views.yaml")
    parser.add_argument("--views", nargs="*", default=None, help="Optional subset of configured views.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional smoke-test frame limit per view.")
    parser.add_argument("--device", default=None, help="Override appearance device, e.g. cuda or cpu.")
    parser.add_argument("--force-embeddings", action="store_true", help="Recompute cached crop embeddings.")
    return parser.parse_args()


def main(argv: list[str] | None = None) -> int:
    args = parse_args() if argv is None else parse_args_from(argv)
    config_path = resolve_project_path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = resolve_project_path(config["output"]["dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    configured_views = config["inputs"]["views"]
    views = args.views or list(configured_views)
    summaries: list[dict[str, Any]] = []
    for view in views:
        if view not in configured_views:
            raise KeyError(f"Unknown view {view!r}; available={list(configured_views)}")
        summaries.append(
            process_view(
                view=view,
                view_cfg=configured_views[view],
                config=config,
                output_root=output_root,
                max_frames=args.max_frames,
                device_override=args.device,
                force_embeddings=args.force_embeddings,
            )
        )
    write_csv(output_root / "summary.csv", summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0


def parse_args_from(argv: list[str]) -> argparse.Namespace:
    original = sys.argv
    try:
        sys.argv = [original[0], *argv]
        return parse_args()
    finally:
        sys.argv = original


def process_view(
    view: str,
    view_cfg: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
    max_frames: int | None,
    device_override: str | None,
    force_embeddings: bool,
) -> dict[str, Any]:
    source_csv = resolve_project_path(view_cfg["detections_csv"])
    rows = read_csv(source_csv)
    detections, class_names = build_detections(rows, config)
    by_frame: dict[int, list[Detection]] = defaultdict(list)
    for det in detections:
        by_frame[int(float(det.row["frame"]))].append(det)

    all_frames = list(range(int(config["input_sequence"].get("first_frame", 0)), max(by_frame, default=-1) + 1))
    if max_frames is not None:
        all_frames = all_frames[:max_frames]
    image_paths = list_frame_images(resolve_project_path(view_cfg["image_dir"]))
    if len(image_paths) <= max(all_frames, default=-1):
        raise ValueError(
            f"{view}: image_dir has {len(image_paths)} frames but tracking needs frame {max(all_frames, default=-1)}"
        )

    view_dir = output_root / str(view_cfg.get("output_name", view))
    view_dir.mkdir(parents=True, exist_ok=True)
    appearance_cfg = config["appearance"]
    embedder = make_embedder(appearance_cfg, device_override)
    cache_path = view_dir / "detection_embeddings.npz"
    embeddings_by_key = load_or_create_embeddings(
        by_frame=by_frame,
        frames=all_frames,
        image_paths=image_paths,
        embedder=embedder,
        appearance_cfg=appearance_cfg,
        cache_path=cache_path,
        force=force_embeddings,
    )

    trackers: dict[int, Any] = {}
    output_rows: list[dict[str, Any]] = []
    tracked_detection_keys: set[tuple[int, int]] = set()
    for index, frame in enumerate(all_frames):
        frame_dets = by_frame.get(frame, [])
        image = load_frame_image(frame_dets, image_paths[frame])
        class_ids = sorted(set(trackers) | {det.class_id for det in frame_dets})
        for class_id in class_ids:
            if class_id not in trackers:
                trackers[class_id] = make_tracker(config["tracker"])
            tracker = trackers[class_id]
            det_indices = [i for i, det in enumerate(frame_dets) if det.class_id == class_id]
            det_array = np.asarray(
                [[*frame_dets[i].box.tolist(), frame_dets[i].score, class_id] for i in det_indices],
                dtype=np.float32,
            ).reshape(-1, 6)
            emb_array = np.asarray(
                [embeddings_by_key[(frame, i)] for i in det_indices], dtype=np.float32
            )
            if not det_indices:
                emb_array = np.empty((0, embedder.dimension), dtype=np.float32)

            tracks = np.asarray(tracker.update(det_array, image, embs=emb_array), dtype=np.float32).reshape(-1, 8)
            for track in tracks:
                local_index = int(round(float(track[7])))
                if local_index < 0 or local_index >= len(det_indices):
                    continue
                det_index = det_indices[local_index]
                det = frame_dets[det_index]
                key = (frame, det_index)
                if key in tracked_detection_keys:
                    raise RuntimeError(f"Detection assigned twice: view={view} frame={frame} index={det_index}")
                tracked_detection_keys.add(key)
                track_id = class_id * 1_000_000 + int(round(float(track[4])))
                output_rows.append(make_output_row(det, view, track_id, track[:4], "matched"))
            # Preserve the first assignment of a newly created track.
            for pending in tracker.active_tracks:
                if pending.is_activated or int(pending.frame_id) != int(tracker.frame_count):
                    continue
                local_index = int(round(float(pending.det_ind)))
                if local_index < 0 or local_index >= len(det_indices):
                    continue
                det_index = det_indices[local_index]
                key = (frame, det_index)
                if key in tracked_detection_keys:
                    continue
                det = frame_dets[det_index]
                tracked_detection_keys.add(key)
                track_id = class_id * 1_000_000 + int(pending.id)
                output_rows.append(make_output_row(det, view, track_id, pending.xyxy, "tentative"))

        if (index + 1) % 100 == 0 or index + 1 == len(all_frames):
            print(f"[{view}] frames={index + 1}/{len(all_frames)} output_rows={len(output_rows)}")

    # Keep every source detection visible in the result. Very weak detections that
    # BoT-SORT deliberately rejects retain track_id=-1 and an explicit status.
    for frame in all_frames:
        for det_index, det in enumerate(by_frame.get(frame, [])):
            if (frame, det_index) not in tracked_detection_keys:
                output_rows.append(make_output_row(det, view, -1, det.box, "unassigned"))

    remapped_track_ids = apply_track_id_remaps(output_rows, config.get("track_id_remaps", {}).get(view, {}))
    apply_majority_track_labels(output_rows, remapped_track_ids)
    output_rows.sort(key=lambda row: (int(row["frame"]), int(row["track_id"]), int(row["source_detection_index"])))
    tracks_csv = view_dir / "tracks.csv"
    write_csv(tracks_csv, output_rows)
    track_summary = summarize_tracks(output_rows)
    write_csv(view_dir / "track_summary.csv", track_summary)
    assigned = sum(int(row["track_id"]) >= 0 for row in output_rows)
    return {
        "view": view,
        "source_csv": str(source_csv.relative_to(PROJECT_ROOT)),
        "frames_processed": len(all_frames),
        "source_detections": sum(len(by_frame.get(frame, [])) for frame in all_frames),
        "assigned_detections": assigned,
        "unassigned_detections": len(output_rows) - assigned,
        "tracks": len(track_summary),
        "tracks_csv": str(tracks_csv.relative_to(PROJECT_ROOT)),
        "embedding_cache": str(cache_path.relative_to(PROJECT_ROOT)),
        "appearance": embedder.name,
    }


def build_detections(
    rows: list[dict[str, str]], config: dict[str, Any]
) -> tuple[list[Detection], dict[int, str]]:
    aliases = dict(CANONICAL_LABELS)
    aliases.update(config.get("label_aliases", {}))
    canonical = sorted({aliases.get(str(row.get("label", "")), str(row.get("label", "")).lower()) for row in rows})
    class_to_id = {name: index for index, name in enumerate(canonical)}
    out: list[Detection] = []
    for row in rows:
        try:
            box = np.asarray([float(row[k]) for k in ("x1", "y1", "x2", "y2")], dtype=np.float32)
            frame = int(float(row["frame"]))
        except (KeyError, TypeError, ValueError):
            continue
        if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1] or frame < 0:
            continue
        raw_label = str(row.get("label", "unknown"))
        label = aliases.get(raw_label, raw_label.lower())
        score = float(row.get("gt2d_score") or row.get("score") or 1.0)
        out.append(Detection(row=row, box=box, score=score, label=label, class_id=class_to_id[label]))
    return out, {value: key for key, value in class_to_id.items()}


def make_embedder(cfg: dict[str, Any], device_override: str | None):
    kind = str(cfg.get("type", "dinov2")).lower()
    if kind == "dinov2":
        return DinoV2Embedder(
            model_name=str(cfg.get("model", "dinov2_vitl14")),
            device=device_override or str(cfg.get("device", "cuda")),
            batch_size=int(cfg.get("batch_size", 24)),
            crop_size=int(cfg.get("crop_size", 224)),
            padding=float(cfg.get("box_padding", 0.08)),
        )
    if kind == "hsv_histogram":
        return HSVHistogramEmbedder(padding=float(cfg.get("box_padding", 0.08)))
    raise ValueError(f"Unsupported appearance.type={kind!r}")


class DinoV2Embedder:
    def __init__(self, model_name: str, device: str, batch_size: int, crop_size: int, padding: float):
        self.name = model_name
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.padding = padding
        hub_dir = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
        if not hub_dir.exists():
            raise FileNotFoundError(
                f"Cached DINOv2 source not found at {hub_dir}. Use appearance.type=hsv_histogram or cache DINOv2."
            )
        self.model = torch.hub.load(str(hub_dir), model_name, source="local", pretrained=True)
        self.model.eval().to(self.device)
        if self.device.type == "cuda":
            self.model.half()
        self.dimension = int(getattr(self.model, "embed_dim", 1024))

    @torch.inference_mode()
    def encode(self, image: np.ndarray, boxes: list[np.ndarray]) -> np.ndarray:
        if not boxes:
            return np.empty((0, self.dimension), dtype=np.float32)
        crops = [prepare_crop(image, box, self.crop_size, self.padding) for box in boxes]
        outputs = []
        for start in range(0, len(crops), self.batch_size):
            batch = torch.from_numpy(np.stack(crops[start : start + self.batch_size])).to(self.device)
            batch = batch.half() if self.device.type == "cuda" else batch.float()
            features = self.model(batch)
            features = torch.nn.functional.normalize(features.float(), dim=1)
            outputs.append(features.cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)


class HSVHistogramEmbedder:
    def __init__(self, padding: float):
        self.name = "hsv_histogram"
        self.padding = padding
        self.dimension = 16 * 8 + 16

    def encode(self, image: np.ndarray, boxes: list[np.ndarray]) -> np.ndarray:
        features = []
        for box in boxes:
            crop = crop_box(image, box, self.padding)
            if crop.size == 0:
                features.append(np.zeros(self.dimension, dtype=np.float32))
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hs = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).reshape(-1)
            v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).reshape(-1)
            feat = np.concatenate([hs, v]).astype(np.float32)
            feat /= max(float(np.linalg.norm(feat)), 1.0e-12)
            features.append(feat)
        return np.stack(features).astype(np.float32) if features else np.empty((0, self.dimension), np.float32)


def load_or_create_embeddings(
    by_frame: dict[int, list[Detection]],
    frames: list[int],
    image_paths: list[Path],
    embedder,
    appearance_cfg: dict[str, Any],
    cache_path: Path,
    force: bool,
) -> dict[tuple[int, int], np.ndarray]:
    expected_keys = [(frame, index) for frame in frames for index in range(len(by_frame.get(frame, [])))]
    if cache_path.exists() and not force:
        payload = np.load(cache_path)
        keys = payload["keys"]
        embeddings = payload["embeddings"]
        cached = {(int(key[0]), int(key[1])): emb for key, emb in zip(keys, embeddings, strict=True)}
        if all(key in cached for key in expected_keys):
            return cached

    keys: list[tuple[int, int]] = []
    features: list[np.ndarray] = []
    for position, frame in enumerate(frames):
        frame_dets = by_frame.get(frame, [])
        if not frame_dets:
            continue
        image = load_frame_image(frame_dets, image_paths[frame])
        encoded = embedder.encode(image, [det.box for det in frame_dets])
        for index, feature in enumerate(encoded):
            keys.append((frame, index))
            features.append(feature)
        if (position + 1) % 50 == 0 or position + 1 == len(frames):
            print(f"[embeddings] frames={position + 1}/{len(frames)} detections={len(features)}")
    array = np.stack(features).astype(np.float32) if features else np.empty((0, embedder.dimension), np.float32)
    np.savez_compressed(cache_path, keys=np.asarray(keys, dtype=np.int32), embeddings=array)
    return {key: emb for key, emb in zip(keys, array, strict=True)}


def make_tracker(cfg: dict[str, Any]):
    try:
        from boxmot.trackers.bbox.botsort import BotSort
    except ImportError as exc:
        raise RuntimeError(
            "BoxMOT is required. Install it in the environment or into PROJECT_ROOT/.deps."
        ) from exc
    return BotSort(
        reid_model=None,
        with_reid=True,
        use_cmc=bool(cfg.get("use_cmc", True)),
        cmc_method=str(cfg.get("cmc_method", "ecc")),
        frame_rate=int(cfg.get("frame_rate", 10)),
        track_high_thresh=float(cfg.get("track_high_thresh", 0.45)),
        track_low_thresh=float(cfg.get("track_low_thresh", 0.08)),
        new_track_thresh=float(cfg.get("new_track_thresh", 0.50)),
        track_buffer=int(cfg.get("track_buffer", 50)),
        match_thresh=float(cfg.get("match_thresh", 0.80)),
        proximity_thresh=float(cfg.get("proximity_thresh", 0.70)),
        appearance_thresh=float(cfg.get("appearance_thresh", 0.35)),
        second_match_thresh=float(cfg.get("second_match_thresh", 0.50)),
        unconfirmed_match_thresh=float(cfg.get("unconfirmed_match_thresh", 0.70)),
        fuse_first_associate=bool(cfg.get("fuse_first_associate", True)),
        per_class=False,
        min_hits=int(cfg.get("min_hits", 1)),
        max_obs=int(cfg.get("max_obs", 100)),
    )


def load_frame_image(frame_dets: list[Detection], fallback_path: Path) -> np.ndarray:
    path = resolve_project_path(frame_dets[0].row["image"]) if frame_dets else fallback_path
    if not path.exists():
        path = fallback_path
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read frame image: {path}")
    return image


def list_frame_images(image_dir: Path) -> list[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in extensions)


def prepare_crop(image: np.ndarray, box: np.ndarray, crop_size: int, padding: float) -> np.ndarray:
    crop = crop_box(image, box, padding)
    if crop.size == 0:
        crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
    h, w = crop.shape[:2]
    scale = crop_size / max(h, w, 1)
    resized = cv2.resize(crop, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))))
    canvas = np.full((crop_size, crop_size, 3), 114, dtype=np.uint8)
    y = (crop_size - resized.shape[0]) // 2
    x = (crop_size - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.asarray([0.485, 0.456, 0.406], np.float32)) / np.asarray(
        [0.229, 0.224, 0.225], np.float32
    )
    return np.transpose(rgb, (2, 0, 1)).astype(np.float32)


def crop_box(image: np.ndarray, box: np.ndarray, padding: float) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in box]
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    h, w = image.shape[:2]
    ix1 = max(0, int(np.floor(x1 - pad_x)))
    iy1 = max(0, int(np.floor(y1 - pad_y)))
    ix2 = min(w, int(np.ceil(x2 + pad_x)))
    iy2 = min(h, int(np.ceil(y2 + pad_y)))
    return image[iy1:iy2, ix1:ix2]


def make_output_row(
    det: Detection, view: str, track_id: int, filtered_box: np.ndarray, status: str
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "view": view,
        "frame": int(float(det.row["frame"])),
        "timestamp": det.row.get("timestamp", ""),
        "image": det.row.get("image", ""),
        "track_id": track_id,
        "class_id": det.class_id,
        "class_name": det.label,
        "score": det.score,
        "x1": float(det.box[0]),
        "y1": float(det.box[1]),
        "x2": float(det.box[2]),
        "y2": float(det.box[3]),
        "filtered_x1": float(filtered_box[0]),
        "filtered_y1": float(filtered_box[1]),
        "filtered_x2": float(filtered_box[2]),
        "filtered_y2": float(filtered_box[3]),
        "association_status": status,
        "source_detection_index": int(float(det.row.get("gt2d_index") or -1)),
        "raw_label": det.row.get("label", ""),
        "mask_path": det.row.get("mask_path", ""),
        "mask_score": det.row.get("sam_score", ""),
    }
    return row


def summarize_tracks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row["track_id"]) >= 0:
            grouped[int(row["track_id"])].append(row)
    out = []
    for track_id, items in sorted(grouped.items()):
        items.sort(key=lambda row: int(row["frame"]))
        frames = [int(item["frame"]) for item in items]
        labels = defaultdict(int)
        for item in items:
            labels[str(item["class_name"])] += 1
        out.append(
            {
                "track_id": track_id,
                "class_name": max(labels, key=labels.get),
                "confirmed": any(item["association_status"] == "matched" for item in items),
                "detections": len(items),
                "first_frame": min(frames),
                "last_frame": max(frames),
                "span_frames": max(frames) - min(frames) + 1,
                "observed_ratio": len(set(frames)) / max(max(frames) - min(frames) + 1, 1),
                "max_internal_gap": maximum_internal_gap(frames),
            }
        )
    return out


def apply_track_id_remaps(rows: list[dict[str, Any]], remaps: dict[int | str, int | str]) -> set[int]:
    normalized = {int(source): int(target) for source, target in remaps.items()}
    for row in rows:
        track_id = int(row["track_id"])
        if track_id in normalized:
            row["track_id"] = normalized[track_id]
    return set(normalized.values())


def apply_majority_track_labels(rows: list[dict[str, Any]], track_ids: set[int] | None = None) -> None:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        track_id = int(row["track_id"])
        if track_id >= 0 and (track_ids is None or track_id in track_ids):
            grouped[track_id].append(row)
    for items in grouped.values():
        votes: dict[str, int] = defaultdict(int)
        first_seen: dict[str, int] = {}
        class_ids: dict[str, int] = {}
        for index, item in enumerate(items):
            label = str(item["class_name"])
            votes[label] += 1
            first_seen.setdefault(label, index)
            class_ids.setdefault(label, int(item["class_id"]))
        majority = min(votes, key=lambda label: (-votes[label], first_seen[label]))
        for item in items:
            item["class_name"] = majority
            item["class_id"] = class_ids[majority]


def maximum_internal_gap(frames: list[int]) -> int:
    unique = sorted(set(frames))
    return max((b - a - 1 for a, b in zip(unique, unique[1:])), default=0)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
