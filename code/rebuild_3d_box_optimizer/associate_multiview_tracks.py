from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, resolve_path, write_experiment_config_snapshot, write_resolved_config
from .data import Observation, group_by_track, load_view_observations


@dataclass
class Tracklet:
    view: str
    track_id: int
    label: str
    frames: int
    start_ts: int
    end_ts: int
    start_frame: int
    end_frame: int
    timestamps: np.ndarray
    centers_world: np.ndarray
    centers_cam: np.ndarray
    box_areas: np.ndarray


@dataclass
class MatchEdge:
    a: tuple[str, int]
    b: tuple[str, int]
    cost: float
    score: float
    mean_dist: float
    median_dist: float
    overlap_pairs: int
    label_compatible: bool
    time_overlap_sec: float


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Associate single-view tracks across rear/left_rear/right_rear cameras.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    out_dir = resolve_path(config, config.get("association", {}).get("output_dir", "outputs/multiview_track_association_v1"))
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.log"
    progress_path.write_text("", encoding="utf-8")
    log(progress_path, f"START config={args.config} output={out_dir}")
    shutil.copy2(args.config, out_dir / "source_config.yaml")
    write_resolved_config(config, out_dir / "resolved_config.yaml")
    write_experiment_config_snapshot(config, out_dir / "experiment_config_snapshot.yaml")

    assoc_cfg = config.get("association", {})
    views = list(assoc_cfg.get("views", config.get("scope", {}).get("cameras", ["rear", "left_rear", "right_rear"])))
    min_frames = int(assoc_cfg.get("min_track_frames", 3))

    all_tracklets: dict[tuple[str, int], Tracklet] = {}
    for view in views:
        log(progress_path, f"LOAD_VIEW_START view={view}")
        observations = load_view_observations(config, view)
        grouped = group_by_track(observations)
        kept = 0
        for track_id, items in grouped.items():
            if len(items) < min_frames:
                continue
            tr = make_tracklet(view, track_id, items)
            all_tracklets[(view, track_id)] = tr
            kept += 1
        log(progress_path, f"LOAD_VIEW_DONE view={view} observations={len(observations)} tracklets={kept}")

    edges: list[MatchEdge] = []
    view_pairs = association_view_pairs(views, assoc_cfg)
    for a_view, b_view in view_pairs:
        a_items = [t for t in all_tracklets.values() if t.view == a_view]
        b_items = [t for t in all_tracklets.values() if t.view == b_view]
        log(progress_path, f"MATCH_PAIR_START {a_view}<->{b_view} candidates={len(a_items)}x{len(b_items)}")
        pair_edges = match_pair(a_items, b_items, assoc_cfg)
        edges.extend(pair_edges)
        log(progress_path, f"MATCH_PAIR_DONE {a_view}<->{b_view} accepted={len(pair_edges)}")

    components = connected_components(all_tracklets.keys(), edges)
    rows = build_assignment_rows(components, all_tracklets)
    component_rows = build_component_rows(components, all_tracklets)
    candidate_rows = build_candidate_diagnostic_rows(view_pairs, all_tracklets, assoc_cfg)
    edge_rows = [edge_to_row(e) for e in sorted(edges, key=lambda x: (x.a[0], x.a[1], x.b[0], x.b[1]))]
    track_rows = [tracklet_to_row(t) for t in sorted(all_tracklets.values(), key=lambda x: (x.view, x.track_id))]
    write_csv(out_dir / "global_track_assignments.csv", rows)
    write_csv(out_dir / "global_track_components.csv", component_rows)
    write_csv(out_dir / "candidate_match_diagnostics.csv", candidate_rows)
    write_csv(out_dir / "match_edges.csv", edge_rows)
    write_csv(out_dir / "tracklet_summary.csv", track_rows)
    write_summary(out_dir / "summary.txt", all_tracklets, edges, components)
    log(progress_path, f"DONE tracklets={len(all_tracklets)} edges={len(edges)} global_tracks={len(components)}")
    return 0


def make_tracklet(view: str, track_id: int, observations: list[Observation]) -> Tracklet:
    label = majority_label([o.label for o in observations])
    timestamps = np.asarray([o.timestamp for o in observations], dtype=np.int64)
    centers_world = np.stack([o.init_center_world for o in observations]).astype(np.float64)
    centers_cam = np.stack([o.init_center_cam for o in observations]).astype(np.float64)
    boxes = np.stack([o.box2d for o in observations]).astype(np.float64)
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return Tracklet(
        view=view,
        track_id=track_id,
        label=label,
        frames=len(observations),
        start_ts=int(timestamps.min()),
        end_ts=int(timestamps.max()),
        start_frame=int(min(o.frame for o in observations)),
        end_frame=int(max(o.frame for o in observations)),
        timestamps=timestamps,
        centers_world=centers_world,
        centers_cam=centers_cam,
        box_areas=areas,
    )


def majority_label(labels: list[str]) -> str:
    counts: dict[str, int] = {}
    for label in labels:
        counts[str(label)] = counts.get(str(label), 0) + 1
    return max(counts, key=counts.get) if counts else "default"


def match_pair(a_items: list[Tracklet], b_items: list[Tracklet], cfg: dict[str, Any]) -> list[MatchEdge]:
    candidates: list[MatchEdge] = []
    for a in a_items:
        for b in b_items:
            edge = score_pair(a, b, cfg)
            if edge is not None:
                candidates.append(edge)
    candidates.sort(key=lambda e: e.cost)
    used_a: set[tuple[str, int]] = set()
    used_b: set[tuple[str, int]] = set()
    selected: list[MatchEdge] = []
    for edge in candidates:
        if edge.a in used_a or edge.b in used_b:
            continue
        selected.append(edge)
        used_a.add(edge.a)
        used_b.add(edge.b)
    return selected


def association_view_pairs(views: list[str], cfg: dict[str, Any]) -> list[tuple[str, str]]:
    allowed = cfg.get("allowed_view_pairs")
    if allowed:
        pairs: list[tuple[str, str]] = []
        available = set(views)
        for item in allowed:
            if isinstance(item, str):
                if "-" in item:
                    a, b = item.split("-", 1)
                elif "," in item:
                    a, b = item.split(",", 1)
                else:
                    continue
            else:
                seq = list(item)
                if len(seq) != 2:
                    continue
                a, b = str(seq[0]), str(seq[1])
            a, b = a.strip(), b.strip()
            if a in available and b in available:
                pairs.append((a, b))
        return pairs
    return [(a, b) for i, a in enumerate(views) for b in views[i + 1 :]]


def score_pair(a: Tracklet, b: Tracklet, cfg: dict[str, Any]) -> MatchEdge | None:
    max_time_diff_ns = int(float(cfg.get("max_time_diff_ms", 120.0)) * 1.0e6)
    max_mean_dist = float(cfg.get("max_mean_3d_distance_m", 6.0))
    max_median_dist = float(cfg.get("max_median_3d_distance_m", 5.0))
    min_pairs = int(cfg.get("min_overlap_pairs", 3))
    label_compatible = compatible_labels(a.label, b.label, cfg)
    if not label_compatible and bool(cfg.get("reject_label_mismatch", False)):
        return None
    pairs = nearest_time_pairs(a.timestamps, b.timestamps, max_time_diff_ns)
    if len(pairs) < min_pairs:
        return None
    da = a.centers_world[[i for i, _ in pairs]]
    db = b.centers_world[[j for _, j in pairs]]
    dists = np.linalg.norm(da - db, axis=1)
    mean_dist = float(np.mean(dists))
    median_dist = float(np.median(dists))
    if mean_dist > max_mean_dist or median_dist > max_median_dist:
        return None
    overlap_sec = overlap_seconds(a, b)
    label_penalty = 0.0 if label_compatible else float(cfg.get("label_mismatch_penalty", 2.0))
    time_bonus = min(float(cfg.get("time_overlap_bonus", 0.5)), overlap_sec * float(cfg.get("time_overlap_bonus_per_sec", 0.05)))
    cost = mean_dist + 0.5 * median_dist + label_penalty - time_bonus
    score = 1.0 / (1.0 + max(cost, 0.0))
    return MatchEdge(
        a=(a.view, a.track_id),
        b=(b.view, b.track_id),
        cost=float(cost),
        score=float(score),
        mean_dist=mean_dist,
        median_dist=median_dist,
        overlap_pairs=len(pairs),
        label_compatible=label_compatible,
        time_overlap_sec=overlap_sec,
    )


def diagnose_pair(a: Tracklet, b: Tracklet, cfg: dict[str, Any]) -> dict[str, object]:
    max_time_diff_ns = int(float(cfg.get("max_time_diff_ms", 120.0)) * 1.0e6)
    max_mean_dist = float(cfg.get("max_mean_3d_distance_m", 6.0))
    max_median_dist = float(cfg.get("max_median_3d_distance_m", 5.0))
    min_pairs = int(cfg.get("min_overlap_pairs", 3))
    label_compatible = compatible_labels(a.label, b.label, cfg)
    reject_label_mismatch = bool(cfg.get("reject_label_mismatch", False))
    pairs = nearest_time_pairs(a.timestamps, b.timestamps, max_time_diff_ns)
    reason = "accepted_by_thresholds"
    mean_dist = math.inf
    median_dist = math.inf
    cost = math.inf
    score = 0.0
    if len(pairs) < min_pairs:
        reason = "too_few_time_overlap_pairs"
    else:
        da = a.centers_world[[i for i, _ in pairs]]
        db = b.centers_world[[j for _, j in pairs]]
        dists = np.linalg.norm(da - db, axis=1)
        mean_dist = float(np.mean(dists))
        median_dist = float(np.median(dists))
        overlap_sec = overlap_seconds(a, b)
        label_penalty = 0.0 if label_compatible else float(cfg.get("label_mismatch_penalty", 2.0))
        time_bonus = min(float(cfg.get("time_overlap_bonus", 0.5)), overlap_sec * float(cfg.get("time_overlap_bonus_per_sec", 0.05)))
        cost = mean_dist + 0.5 * median_dist + label_penalty - time_bonus
        score = 1.0 / (1.0 + max(cost, 0.0))
        if not label_compatible and reject_label_mismatch:
            reason = "label_mismatch"
        elif mean_dist > max_mean_dist:
            reason = "mean_3d_distance_too_large"
        elif median_dist > max_median_dist:
            reason = "median_3d_distance_too_large"
    return {
        "view_a": a.view,
        "track_id_a": a.track_id,
        "class_a": a.label,
        "frames_a": a.frames,
        "view_b": b.view,
        "track_id_b": b.track_id,
        "class_b": b.label,
        "frames_b": b.frames,
        "reason": reason,
        "passes_thresholds": reason == "accepted_by_thresholds",
        "cost": cost,
        "score": score,
        "mean_3d_distance_m": mean_dist,
        "median_3d_distance_m": median_dist,
        "overlap_pairs": len(pairs),
        "time_overlap_sec": overlap_seconds(a, b),
        "label_compatible": label_compatible,
    }


def build_candidate_diagnostic_rows(
    view_pairs: list[tuple[str, str]],
    tracklets: dict[tuple[str, int], Tracklet],
    cfg: dict[str, Any],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    keep_topk = int(cfg.get("candidate_diagnostics_topk_per_track", 8))
    for a_view, b_view in view_pairs:
        a_items = [t for t in tracklets.values() if t.view == a_view]
        b_items = [t for t in tracklets.values() if t.view == b_view]
        for a in a_items:
            local = [diagnose_pair(a, b, cfg) for b in b_items]
            local.sort(key=lambda row: (float(row["cost"]) if math.isfinite(float(row["cost"])) else 1.0e9, -int(row["overlap_pairs"])))
            rows.extend(local[:keep_topk])
    return rows


def nearest_time_pairs(a_ts: np.ndarray, b_ts: np.ndarray, max_diff_ns: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    j = 0
    for i, ts in enumerate(a_ts):
        while j + 1 < len(b_ts) and abs(int(b_ts[j + 1]) - int(ts)) <= abs(int(b_ts[j]) - int(ts)):
            j += 1
        if j < len(b_ts) and abs(int(b_ts[j]) - int(ts)) <= max_diff_ns:
            pairs.append((i, j))
    dedup: dict[int, tuple[int, int]] = {}
    for i, j in pairs:
        prev = dedup.get(j)
        if prev is None or abs(int(a_ts[i]) - int(b_ts[j])) < abs(int(a_ts[prev[0]]) - int(b_ts[j])):
            dedup[j] = (i, j)
    return list(dedup.values())


def compatible_labels(a: str, b: str, cfg: dict[str, Any]) -> bool:
    if norm_label(a) == norm_label(b):
        return True
    groups = cfg.get("compatible_label_groups", [["car", "truck", "bus", "VEHICLE_CAR", "VEHICLE_TRUCK", "VEHICLE_BUS", "VEHICLE_SUV", "VEHICLE_TRUCK_SMALL"]])
    na, nb = norm_label(a), norm_label(b)
    for group in groups:
        normalized = {norm_label(str(x)) for x in group}
        if na in normalized and nb in normalized:
            return True
    return False


def norm_label(label: str) -> str:
    text = str(label).strip().lower()
    for prefix in ("vehicle_",):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    return text


def overlap_seconds(a: Tracklet, b: Tracklet) -> float:
    lo = max(a.start_ts, b.start_ts)
    hi = min(a.end_ts, b.end_ts)
    return max(0.0, (hi - lo) * 1.0e-9)


def connected_components(nodes: Any, edges: list[MatchEdge]) -> list[list[tuple[str, int]]]:
    parent = {node: node for node in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for edge in edges:
        union(edge.a, edge.b)
    groups: dict[tuple[str, int], list[tuple[str, int]]] = {}
    for node in parent:
        groups.setdefault(find(node), []).append(node)
    return [sorted(g) for g in groups.values()]


def build_assignment_rows(components: list[list[tuple[str, int]]], tracklets: dict[tuple[str, int], Tracklet]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    components = sorted(components, key=lambda comp: min(tracklets[n].start_ts for n in comp))
    for gid, comp in enumerate(components, start=1):
        views = [view for view, _ in comp]
        view_conflict = len(set(views)) != len(views)
        component_views = ",".join(sorted(set(views)))
        for view, track_id in comp:
            t = tracklets[(view, track_id)]
            rows.append(
                {
                    "global_track_id": gid,
                    "component_size": len(comp),
                    "component_views": component_views,
                    "view_conflict": view_conflict,
                    "view": view,
                    "track_id": track_id,
                    "class": t.label,
                    "frames": t.frames,
                    "start_frame": t.start_frame,
                    "end_frame": t.end_frame,
                    "start_ts": t.start_ts,
                    "end_ts": t.end_ts,
                }
            )
    return rows


def build_component_rows(components: list[list[tuple[str, int]]], tracklets: dict[tuple[str, int], Tracklet]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    components = sorted(components, key=lambda comp: min(tracklets[n].start_ts for n in comp))
    for gid, comp in enumerate(components, start=1):
        views = [view for view, _ in comp]
        view_conflict = len(set(views)) != len(views)
        labels = [tracklets[n].label for n in comp]
        starts = [tracklets[n].start_ts for n in comp]
        ends = [tracklets[n].end_ts for n in comp]
        rows.append(
            {
                "global_track_id": gid,
                "component_size": len(comp),
                "component_views": ",".join(sorted(set(views))),
                "view_conflict": view_conflict,
                "tracks": ";".join(f"{view}:{track_id}" for view, track_id in comp),
                "classes": ",".join(sorted(set(labels))),
                "start_ts": min(starts),
                "end_ts": max(ends),
            }
        )
    return rows


def edge_to_row(edge: MatchEdge) -> dict[str, object]:
    return {
        "view_a": edge.a[0],
        "track_id_a": edge.a[1],
        "view_b": edge.b[0],
        "track_id_b": edge.b[1],
        "cost": edge.cost,
        "score": edge.score,
        "mean_3d_distance_m": edge.mean_dist,
        "median_3d_distance_m": edge.median_dist,
        "overlap_pairs": edge.overlap_pairs,
        "label_compatible": edge.label_compatible,
        "time_overlap_sec": edge.time_overlap_sec,
    }


def tracklet_to_row(t: Tracklet) -> dict[str, object]:
    return {
        "view": t.view,
        "track_id": t.track_id,
        "class": t.label,
        "frames": t.frames,
        "start_frame": t.start_frame,
        "end_frame": t.end_frame,
        "start_ts": t.start_ts,
        "end_ts": t.end_ts,
        "median_center_x": float(np.median(t.centers_world[:, 0])),
        "median_center_y": float(np.median(t.centers_world[:, 1])),
        "median_center_z": float(np.median(t.centers_world[:, 2])),
        "median_depth_cam": float(np.median(t.centers_cam[:, 2])),
        "median_box_area": float(np.median(t.box_areas)),
    }


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


def write_summary(path: Path, tracklets: dict[tuple[str, int], Tracklet], edges: list[MatchEdge], components: list[list[tuple[str, int]]]) -> None:
    multi = [c for c in components if len(c) > 1]
    conflicts = [c for c in components if len({view for view, _ in c}) != len(c)]
    lines = [
        "multiview track association",
        f"tracklets={len(tracklets)}",
        f"accepted_edges={len(edges)}",
        f"global_tracks={len(components)}",
        f"multi_view_global_tracks={len(multi)}",
        f"view_conflict_global_tracks={len(conflicts)}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def log(path: Path, message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
    print(line, flush=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
