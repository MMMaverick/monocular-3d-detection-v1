from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, resolve_path, write_resolved_config
from .data import canonical_label
from .run import append_progress


@dataclass
class SortDetection:
    row: dict[str, str]
    frame: int
    box: np.ndarray
    score: float
    raw_label: str
    canonical: str


@dataclass
class SortTrack:
    track_id: int
    canonical: str
    state: np.ndarray
    covariance: np.ndarray
    last_box: np.ndarray
    last_frame: int
    age: int = 1
    hits: int = 1
    missed: int = 0
    rows: list[SortDetection] = field(default_factory=list)

    def predict(self, process_noise: float) -> np.ndarray:
        f = np.eye(8, dtype=np.float64)
        f[0, 4] = 1.0
        f[1, 5] = 1.0
        f[2, 6] = 1.0
        f[3, 7] = 1.0
        q = np.eye(8, dtype=np.float64) * float(process_noise)
        self.state = f @ self.state
        self.covariance = f @ self.covariance @ f.T + q
        self.state[2:4] = np.maximum(self.state[2:4], 1.0)
        self.last_box = xywh_to_xyxy(self.state[:4])
        self.age += 1
        self.missed += 1
        return self.last_box

    def update(self, detection: SortDetection, measurement_noise: float) -> None:
        z = xyxy_to_xywh(detection.box)
        h = np.zeros((4, 8), dtype=np.float64)
        h[0, 0] = h[1, 1] = h[2, 2] = h[3, 3] = 1.0
        r = np.eye(4, dtype=np.float64) * float(measurement_noise)
        innovation = z - h @ self.state
        s = h @ self.covariance @ h.T + r
        k = self.covariance @ h.T @ np.linalg.inv(s)
        self.state = self.state + k @ innovation
        self.covariance = (np.eye(8, dtype=np.float64) - k @ h) @ self.covariance
        self.state[2:4] = np.maximum(self.state[2:4], 1.0)
        self.last_box = detection.box.astype(np.float64)
        self.last_frame = detection.frame
        self.hits += 1
        self.missed = 0
        self.rows.append(detection)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run category-agnostic CSV-level SORT on existing 2D track/detection CSVs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    retrack_cfg = config.get("retracking", {}).get("sort_2d", {})
    out_root = resolve_path(config, args.output_dir or retrack_cfg.get("output_dir", "preprocessed/tracks/sort2d_majority_label_v1"))
    out_root.mkdir(parents=True, exist_ok=True)
    progress_path = out_root / "progress.log"
    progress_path.write_text("", encoding="utf-8")
    if Path(args.config).exists():
        shutil.copy2(args.config, out_root / "source_config.yaml")
    write_resolved_config(config, out_root / "resolved_config.yaml")
    append_progress(progress_path, f"START config={args.config} output={out_root}")

    summary_rows: list[dict[str, object]] = []
    for view in config.get("scope", {}).get("cameras", ["rear", "left_rear", "right_rear"]):
        summary_rows.append(process_view(config, view, retrack_cfg, out_root, progress_path))
    write_csv(out_root / "summary.csv", summary_rows)
    append_progress(progress_path, "DONE")
    return 0


def process_view(config: dict[str, Any], view: str, cfg: dict[str, Any], out_root: Path, progress_path: Path) -> dict[str, object]:
    view_cfg = config["inputs"]["views"][view]
    camera_name = str(view_cfg.get("camera_name", view))
    source_csv = resolve_path(config, view_cfg["track_csv"])
    rows = read_csv(source_csv)
    detections = build_detections(config, rows)
    by_frame = group_by_frame(detections)
    frames = sorted(by_frame)
    append_progress(progress_path, f"VIEW_START view={view} source={source_csv} detections={len(detections)} frames={len(frames)}")

    tracker = Sort2DTracker(
        config=cfg,
        next_id=int(cfg.get("start_track_id", 1)),
    )
    for index, frame in enumerate(frames):
        tracker.step(by_frame[frame], frame)
        if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(frames):
            append_progress(
                progress_path,
                f"VIEW_PROGRESS view={view} frames={index + 1}/{len(frames)} active={len(tracker.active)} finished={len(tracker.finished)}",
            )
    tracker.finish_all()

    min_hits = int(cfg.get("min_hits", 1))
    kept_tracks = [t for t in tracker.finished if len(t.rows) >= min_hits]
    output_rows, track_summary = flatten_tracks(config, kept_tracks)
    out_dir = out_root / camera_name
    out_csv = out_dir / "tracks.csv"
    write_csv(out_csv, output_rows)
    write_csv(out_dir / "track_summary.csv", track_summary)
    append_progress(
        progress_path,
        f"VIEW_DONE view={view} source_rows={len(rows)} output_rows={len(output_rows)} tracks={len(track_summary)} csv={out_csv}",
    )
    return {
        "view": view,
        "camera_name": camera_name,
        "source_csv": str(source_csv),
        "source_rows": len(rows),
        "output_rows": len(output_rows),
        "tracks": len(track_summary),
        "output_csv": str(out_csv),
    }


class Sort2DTracker:
    def __init__(self, config: dict[str, Any], next_id: int = 1) -> None:
        self.config = config
        self.next_id = next_id
        self.active: list[SortTrack] = []
        self.finished: list[SortTrack] = []

    def step(self, detections: list[SortDetection], frame: int) -> None:
        process_noise = float(self.config.get("process_noise", 10.0))
        measurement_noise = float(self.config.get("measurement_noise", 25.0))
        max_age = int(self.config.get("max_age", 6))
        for track in self.active:
            track.predict(process_noise)

        matches, unmatched_tracks, unmatched_detections = associate(self.active, detections, self.config)
        for track_idx, det_idx in matches:
            self.active[track_idx].update(detections[det_idx], measurement_noise)

        for det_idx in unmatched_detections:
            self.active.append(make_track(self.next_id, detections[det_idx], measurement_noise))
            self.next_id += 1

        still_active: list[SortTrack] = []
        for idx, track in enumerate(self.active):
            if idx in unmatched_tracks and track.missed > max_age:
                self.finished.append(track)
            else:
                still_active.append(track)
        self.active = still_active

    def finish_all(self) -> None:
        self.finished.extend(self.active)
        self.active = []


def associate(
    tracks: list[SortTrack],
    detections: list[SortDetection],
    cfg: dict[str, Any],
) -> tuple[list[tuple[int, int]], set[int], set[int]]:
    if not tracks:
        return [], set(), set(range(len(detections)))
    if not detections:
        return [], set(range(len(tracks))), set()
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scipy is required for SORT Hungarian assignment.") from exc

    iou_weight = float(cfg.get("iou_weight", 1.0))
    center_weight = float(cfg.get("center_weight", 0.25))
    size_weight = float(cfg.get("size_weight", 0.10))
    min_iou = float(cfg.get("min_iou", 0.05))
    center_gate = float(cfg.get("center_gate", 1.75))
    max_cost = float(cfg.get("max_cost", 2.0))

    cost = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    valid = np.zeros_like(cost, dtype=bool)
    class_aware = bool(cfg.get("class_aware", False))
    for ti, track in enumerate(tracks):
        for di, det in enumerate(detections):
            if class_aware and track.canonical != det.canonical:
                cost[ti, di] = 1.0e6
                valid[ti, di] = False
                continue
            iou = box_iou(track.last_box, det.box)
            center_dist = normalized_center_distance(track.last_box, det.box)
            size_dist = normalized_size_change(track.last_box, det.box)
            pair_cost = iou_weight * (1.0 - iou) + center_weight * center_dist + size_weight * size_dist
            allowed = (iou >= min_iou or center_dist <= center_gate) and pair_cost <= max_cost
            cost[ti, di] = pair_cost if allowed else 1.0e6
            valid[ti, di] = allowed

    row_ind, col_ind = linear_sum_assignment(cost)
    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_detections: set[int] = set()
    for ti, di in zip(row_ind.tolist(), col_ind.tolist()):
        if valid[ti, di]:
            matches.append((ti, di))
            matched_tracks.add(ti)
            matched_detections.add(di)
    return matches, set(range(len(tracks))) - matched_tracks, set(range(len(detections))) - matched_detections


def make_track(track_id: int, detection: SortDetection, measurement_noise: float) -> SortTrack:
    state = np.zeros(8, dtype=np.float64)
    state[:4] = xyxy_to_xywh(detection.box)
    covariance = np.eye(8, dtype=np.float64) * float(measurement_noise)
    covariance[4:, 4:] *= 10.0
    return SortTrack(
        track_id=track_id,
        canonical=detection.canonical,
        state=state,
        covariance=covariance,
        last_box=detection.box.astype(np.float64),
        last_frame=detection.frame,
        rows=[detection],
    )


def build_detections(config: dict[str, Any], rows: list[dict[str, str]]) -> list[SortDetection]:
    detections: list[SortDetection] = []
    for row in rows:
        try:
            frame = int(float(row["frame"]))
            box = np.asarray([float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])], dtype=np.float64)
        except (KeyError, ValueError):
            continue
        if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
            continue
        raw_label = row.get("gt_label") or row.get("label") or row.get("prompt") or "default"
        detections.append(
            SortDetection(
                row=dict(row),
                frame=frame,
                box=box,
                score=safe_float(row.get("score"), 1.0),
                raw_label=str(raw_label),
                canonical=canonical_label(config, str(raw_label)),
            )
        )
    detections.sort(key=lambda d: (d.frame, -d.score, -box_area(d.box)))
    return detections


def flatten_tracks(config: dict[str, Any], tracks: list[SortTrack]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for track in sorted(tracks, key=lambda t: (min(d.frame for d in t.rows), t.track_id)):
        labels = [d.canonical for d in track.rows]
        majority = majority_label(labels)
        raw_majority = majority_label([d.raw_label for d in track.rows])
        source_ids = sorted({str(d.row.get("track_id", "")) for d in track.rows if str(d.row.get("track_id", "")) != ""})
        frames = [d.frame for d in track.rows]
        for det in sorted(track.rows, key=lambda d: d.frame):
            out = dict(det.row)
            out["source_track_id"] = det.row.get("track_id", "")
            out["track_id"] = int(track.track_id)
            out["raw_label"] = det.raw_label
            out["raw_canonical_label"] = det.canonical
            out["track_majority_label"] = majority
            out["track_majority_raw_label"] = raw_majority
            out["gt_label"] = majority
            out["label"] = majority
            out["prompt"] = majority
            rows.append(out)
        summaries.append(
            {
                "track_id": int(track.track_id),
                "frames": len(track.rows),
                "start_frame": min(frames),
                "end_frame": max(frames),
                "track_majority_label": majority,
                "track_majority_raw_label": raw_majority,
                "label_counts": ";".join(f"{k}:{v}" for k, v in Counter(labels).most_common()),
                "source_track_ids": ";".join(source_ids),
            }
        )
    rows.sort(key=lambda r: (int(float(r.get("frame", 0))), int(float(r.get("track_id", 0)))))
    return rows, summaries


def group_by_frame(detections: list[SortDetection]) -> dict[int, list[SortDetection]]:
    grouped: dict[int, list[SortDetection]] = {}
    for det in detections:
        grouped.setdefault(det.frame, []).append(det)
    return grouped


def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            0.5 * (box[0] + box[2]),
            0.5 * (box[1] + box[3]),
            max(box[2] - box[0], 1.0),
            max(box[3] - box[1], 1.0),
        ],
        dtype=np.float64,
    )


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    cx, cy, w, h = [float(v) for v in xywh]
    w, h = max(w, 1.0), max(h, 1.0)
    return np.asarray([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dtype=np.float64)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1, iy1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    ix2, iy2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return float(inter / max(box_area(a) + box_area(b) - inter, 1.0e-9))


def box_area(box: np.ndarray) -> float:
    return float(max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1])))


def normalized_center_distance(a: np.ndarray, b: np.ndarray) -> float:
    ac = np.asarray([0.5 * (a[0] + a[2]), 0.5 * (a[1] + a[3])], dtype=np.float64)
    bc = np.asarray([0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])], dtype=np.float64)
    diag_a = np.hypot(max(a[2] - a[0], 1.0), max(a[3] - a[1], 1.0))
    diag_b = np.hypot(max(b[2] - b[0], 1.0), max(b[3] - b[1], 1.0))
    return float(np.linalg.norm(ac - bc) / max(0.5 * (diag_a + diag_b), 1.0))


def normalized_size_change(a: np.ndarray, b: np.ndarray) -> float:
    aw, ah = max(a[2] - a[0], 1.0), max(a[3] - a[1], 1.0)
    bw, bh = max(b[2] - b[0], 1.0), max(b[3] - b[1], 1.0)
    return float(abs(np.log(aw / bw)) + abs(np.log(ah / bh)))


def majority_label(labels: list[str]) -> str:
    if not labels:
        return "default"
    return Counter(labels).most_common(1)[0][0]


def safe_float(value: object, default: float) -> float:
    try:
        out = float(str(value))
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
