from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIEW_SPECS = {
    "center_front": ("center_camera_fov120", "outputs/robust_botsort_2d_front_views_eval/center_camera_fov120/tracks.csv"),
    "left_front": ("left_front_camera", "outputs/robust_botsort_2d_front_views_eval/left_front_camera/tracks.csv"),
    "right_front": ("right_front_camera", "outputs/robust_botsort_2d_front_views_eval/right_front_camera/tracks.csv"),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate LoMa cross-view edges using held-out GT identities.")
    parser.add_argument("--predictions", default="outputs/loma_global_2d_front_eval_v1")
    parser.add_argument("--truth-root", default="preprocessed/front_2d_tracking_eval")
    parser.add_argument("--min-track-purity", type=float, default=0.8)
    parser.add_argument("--min-overlap-pairs", type=int, default=3)
    parser.add_argument("--max-time-diff-ms", type=float, default=20.0)
    args = parser.parse_args(argv)
    prediction_dir, truth_root = resolve(args.predictions), resolve(args.truth_root)
    mapping, track_times, mapping_rows = build_track_gt_mapping(truth_root, float(args.min_track_purity))
    diagnostics = read_csv(prediction_dir / "candidate_diagnostics.csv")
    accepted = read_csv(prediction_dir / "accepted_edges.csv")
    positive_pairs = gt_positive_pairs(mapping, track_times, int(args.min_overlap_pairs), int(args.max_time_diff_ms * 1e6))
    candidate_keys = {edge_key(row) for row in diagnostics}
    accepted_keys = {edge_key(row) for row in accepted}
    edge_rows = evaluate_edges(accepted, mapping)
    evaluable = [row for row in edge_rows if row["evaluable"]]
    tp = sum(row["is_true_match"] for row in evaluable)
    fp = len(evaluable) - tp
    metrics = {
        "accepted_edges": len(accepted),
        "evaluable_accepted_edges": len(evaluable),
        "true_positive_edges": tp,
        "false_positive_edges": fp,
        "pair_precision": tp / len(evaluable) if evaluable else None,
        "gt_positive_track_pairs": len(positive_pairs),
        "gt_positive_pairs_reaching_candidate_stage": len(positive_pairs & candidate_keys),
        "candidate_recall": len(positive_pairs & candidate_keys) / len(positive_pairs) if positive_pairs else None,
        "accepted_recall": len(positive_pairs & accepted_keys) / len(positive_pairs) if positive_pairs else None,
        "mapped_tracks": len(mapping),
    }
    write_csv(prediction_dir / "evaluation_track_gt_mapping.csv", mapping_rows)
    write_csv(prediction_dir / "evaluation_accepted_edges.csv", edge_rows)
    write_csv(prediction_dir / "evaluation_candidates.csv", evaluate_edges(diagnostics, mapping))
    write_csv(prediction_dir / "evaluation_threshold_sweep.csv", threshold_sweep(diagnostics, mapping, positive_pairs))
    (prediction_dir / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


def build_track_gt_mapping(truth_root: Path, min_purity: float):
    mapping: dict[tuple[str, int], int] = {}
    times: dict[tuple[str, int], list[int]] = defaultdict(list)
    output = []
    for view, (camera, tracks_path) in VIEW_SPECS.items():
        truth = read_csv(truth_root / camera / "evaluation_gt.csv")
        truth_by_box = {(int(r["frame"]), box_key(r)): int(r["gt_track_id"]) for r in truth}
        votes: dict[int, list[int]] = defaultdict(list)
        for row in read_csv(resolve(tracks_path)):
            track_id = int(row["track_id"])
            if track_id < 0:
                continue
            gt_id = truth_by_box.get((int(row["frame"]), box_key(row)), -1)
            if gt_id >= 0:
                votes[track_id].append(gt_id)
            times[(view, track_id)].append(int(row["timestamp"]))
        for track_id, gt_ids in votes.items():
            gt_id, count = Counter(gt_ids).most_common(1)[0]
            purity = count / len(gt_ids)
            valid = purity >= min_purity
            if valid:
                mapping[(view, track_id)] = gt_id
            output.append({"view": view, "track_id": track_id, "dominant_gt_track_id": gt_id, "gt_labeled_observations": len(gt_ids), "dominant_observations": count, "purity": purity, "valid_mapping": valid})
    return mapping, times, output


def gt_positive_pairs(mapping, times, minimum: int, tolerance: int):
    out = set()
    for a_view, b_view in (("center_front", "left_front"), ("center_front", "right_front")):
        aa = [key for key in mapping if key[0] == a_view]
        bb = [key for key in mapping if key[0] == b_view]
        for a in aa:
            for b in bb:
                if mapping[a] == mapping[b] and overlap_count(times[a], times[b], tolerance) >= minimum:
                    out.add((a, b))
    return out


def overlap_count(a: list[int], b: list[int], tolerance: int) -> int:
    a, b = sorted(a), sorted(b)
    count = j = 0
    for value in a:
        while j + 1 < len(b) and abs(b[j + 1] - value) <= abs(b[j] - value):
            j += 1
        count += bool(b and abs(b[j] - value) <= tolerance)
    return count


def evaluate_edges(rows, mapping):
    out = []
    for row in rows:
        a, b = edge_key(row)
        ga, gb = mapping.get(a), mapping.get(b)
        item = dict(row)
        item.update({"gt_track_id_a": ga if ga is not None else -1, "gt_track_id_b": gb if gb is not None else -1, "evaluable": ga is not None and gb is not None, "is_true_match": ga is not None and ga == gb})
        out.append(item)
    return out


def threshold_sweep(diagnostics, mapping, positives):
    out = []
    for threshold in [x / 100 for x in range(0, 101, 2)]:
        predicted = []
        for row in diagnostics:
            if float(row["score"]) >= threshold and int(row["verified_frames"]) >= 3:
                predicted.append(edge_key(row))
        evaluable = [key for key in predicted if key[0] in mapping and key[1] in mapping]
        tp = sum(key in positives for key in evaluable)
        out.append({"threshold": threshold, "predicted_evaluable_edges": len(evaluable), "true_positive_edges": tp, "precision": tp / len(evaluable) if evaluable else "", "recall": tp / len(positives) if positives else ""})
    return out


def edge_key(row):
    return ((row["view_a"], int(row["track_id_a"])), (row["view_b"], int(row["track_id_b"])))


def box_key(row):
    return tuple(round(float(row[k]), 3) for k in ("x1", "y1", "x2", "y2"))


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows: list[dict[str, Any]]):
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
