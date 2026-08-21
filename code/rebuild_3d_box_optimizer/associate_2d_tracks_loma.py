from __future__ import annotations

"""使用 LoMa/外观/极线几何做跨视角 2D track 关联。

当前主线定位：
    这个脚本负责生成 global_track_assignments.csv。
    后续跨视角 3D 联合优化会固定使用该文件，而不是重新根据 3D 距离猜同车关系。

匹配大致流程：
    1. 读取每个视角的 2D track、mask/image 路径、DINO/检测特征缓存。
    2. 按 timestamp 找候选帧对，并用极线几何先做粗筛。
    3. 对候选帧对运行 LoMa 特征匹配，统计 inlier、重投影误差、三角化质量等。
    4. 结合 track 级外观特征打分。
    5. 用二分图选择每个 view pair 中较可信的匹配边。

注意：
    这是“同一辆车跨相机 ID”的来源。如果这里错配，后续联合优化会把两辆车硬绑在一起。
"""

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
LOMA_SRC = PROJECT_ROOT / "external" / "LoMa" / "src"
if str(LOMA_SRC) not in sys.path:
    sys.path.insert(0, str(LOMA_SRC))


@dataclass
class Obs:
    view: str
    track_id: int
    frame: int
    timestamp: int
    image: Path
    mask: Path | None
    box: np.ndarray
    label: str
    row: dict[str, str]


@dataclass
class FrameMatch:
    a: Obs
    b: Obs
    points_a: np.ndarray
    points_b: np.ndarray
    errors: np.ndarray
    matches: int
    inliers: int
    inlier_ratio: float
    score: float
    triangulated_points: int
    positive_depth_ratio: float
    median_reprojection_error: float
    median_ray_angle_deg: float
    triangulated_center: np.ndarray | None


@dataclass
class Edge:
    a: tuple[str, int]
    b: tuple[str, int]
    score: float
    loma_score: float
    appearance_score: float
    verified_frames: int
    candidate_frames: int
    sampled_frames: int
    median_matches: float
    median_inliers: float
    median_error: float
    triangulation_verified_frames: int
    median_triangulated_points: float
    median_positive_depth_ratio: float
    median_reprojection_error: float
    median_ray_angle_deg: float
    frames: list[FrameMatch]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Associate 2D tracks across cameras using only LoMa image matches.")
    parser.add_argument("--config", default="configs/loma_global_2d_tracks_v1.yaml")
    parser.add_argument("--max-candidates", type=int, default=None)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(resolve(args.config).read_text(encoding="utf-8"))
    out_dir = resolve(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    observations, rows_by_view = load_observations(cfg)
    # track_features 是 track 级外观特征，主要用于避免只靠几何导致的错配。
    track_features = load_track_features(cfg, observations)
    calibration = load_calibration(cfg)
    matcher = make_matcher(cfg["loma"])
    assoc = cfg["association"]
    accepted: list[Edge] = []
    scored_edges: list[Edge] = []
    diagnostics: list[dict[str, Any]] = []
    processed = 0
    for view_a, view_b in assoc["view_pairs"]:
        tracks_a = group_tracks(observations[view_a], int(assoc["min_track_frames"]))
        tracks_b = group_tracks(observations[view_b], int(assoc["min_track_frames"]))
        fundamental = fundamental_matrix(calibration[view_a], calibration[view_b])
        candidates = build_candidates(tracks_a, tracks_b, fundamental, assoc)
        print(f"[{view_a}<->{view_b}] candidates={len(candidates)}")
        edges: list[Edge] = []
        for candidate_index, (key_a, key_b, pairs, total_pairs) in enumerate(candidates):
            if args.max_candidates is not None and processed >= args.max_candidates:
                break
            edge = score_candidate(key_a, key_b, pairs, total_pairs, fundamental, calibration[view_a], calibration[view_b], matcher, track_features, cfg)
            scored_edges.append(edge)
            processed += 1
            diagnostics.append(edge_row(edge, accepted=False))
            if passes(edge, assoc):
                edges.append(edge)
            if (candidate_index + 1) % 20 == 0:
                print(f"[{view_a}<->{view_b}] scored={candidate_index + 1}/{len(candidates)} accepted_raw={len(edges)}")
        selected = hungarian_edges(edges)
        accepted.extend(selected)
        selected_keys = {(edge.a, edge.b) for edge in selected}
        for row in diagnostics:
            if ((row["view_a"], int(row["track_id_a"])), (row["view_b"], int(row["track_id_b"]))) in selected_keys:
                row["selected"] = True

    assignments = global_assignments(observations, accepted)
    write_outputs(out_dir, rows_by_view, assignments, accepted, diagnostics)
    render_edges(out_dir / "visualizations", accepted, cfg["visualization"])
    render_edges(out_dir / "candidate_visualizations", scored_edges, cfg["visualization"])
    print(json.dumps({"candidates_scored": processed, "accepted_edges": len(accepted), "global_tracks": len(set(assignments.values()))}, indent=2))
    return 0


def load_observations(cfg: dict[str, Any]) -> tuple[dict[str, list[Obs]], dict[str, list[dict[str, str]]]]:
    by_view: dict[str, list[Obs]] = {}
    raw: dict[str, list[dict[str, str]]] = {}
    for view, spec in cfg["inputs"]["views"].items():
        rows = read_csv(resolve(spec["tracks_csv"]))
        raw[view] = rows
        items = []
        for row in rows:
            track_id = int(row["track_id"])
            if track_id < 0:
                continue
            mask_value = row.get("mask_path", "")
            items.append(Obs(view, track_id, int(row["frame"]), int(row["timestamp"]), resolve(row["image"]), resolve(mask_value) if mask_value else None, np.asarray([float(row[k]) for k in ("x1", "y1", "x2", "y2")]), row.get("class_name", ""), row))
        by_view[view] = items
    return by_view, raw


def group_tracks(items: list[Obs], min_frames: int) -> dict[tuple[str, int], list[Obs]]:
    grouped: dict[tuple[str, int], list[Obs]] = defaultdict(list)
    for item in items:
        grouped[(item.view, item.track_id)].append(item)
    return {key: sorted(value, key=lambda o: o.timestamp) for key, value in grouped.items() if len(value) >= min_frames}


def load_track_features(cfg: dict[str, Any], observations: dict[str, list[Obs]]) -> dict[tuple[str, int], np.ndarray]:
    """读取每帧检测特征，并聚合成 track 级外观特征。

    当前优先用 source_detection_index 找特征；这是为了避免 bbox 被修正/裁剪后，
    再按 bbox 坐标精确匹配导致找不到对应检测特征。
    """
    output: dict[tuple[str, int], np.ndarray] = {}
    for view, spec in cfg["inputs"]["views"].items():
        cache = np.load(resolve(spec["embedding_cache"]))
        cached = {(int(key[0]), int(key[1])): feature.astype(np.float32) for key, feature in zip(cache["keys"], cache["embeddings"], strict=True)}
        source_by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in read_csv(resolve(spec["detections_csv"])):
            source_by_frame[int(row["frame"])].append(row)
        feature_by_detection = {}
        feature_by_index = {}
        for frame, rows in source_by_frame.items():
            for index, row in enumerate(rows):
                feature = cached[(frame, index)]
                feature_by_index[(frame, index)] = feature
                feature_by_detection[(frame, tuple(round(float(row[k]), 3) for k in ("x1", "y1", "x2", "y2")))] = feature
        grouped: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
        for obs in observations[view]:
            source_index = obs.row.get("source_detection_index", "")
            feature = None
            if source_index not in ("", None):
                try:
                    feature = feature_by_index.get((obs.frame, int(float(source_index))))
                except ValueError:
                    feature = None
            if feature is None:
                key = (obs.frame, tuple(round(float(value), 3) for value in obs.box))
                feature = feature_by_detection.get(key)
            if feature is None:
                raise KeyError(f"Missing DINOv2 feature for {view} frame={obs.frame} source_detection_index={source_index} box={obs.box.tolist()}")
            grouped[(view, obs.track_id)].append(feature)
        for track_key, features in grouped.items():
            feature = np.mean(np.stack(features), axis=0)
            output[track_key] = feature / max(float(np.linalg.norm(feature)), 1e-12)
    return output


def nearest_pairs(a: list[Obs], b: list[Obs], tolerance_ns: int) -> list[tuple[Obs, Obs]]:
    result = []
    j = 0
    for oa in a:
        while j + 1 < len(b) and abs(b[j + 1].timestamp - oa.timestamp) <= abs(b[j].timestamp - oa.timestamp):
            j += 1
        if b and abs(b[j].timestamp - oa.timestamp) <= tolerance_ns:
            result.append((oa, b[j]))
    return result


def build_candidates(a_tracks, b_tracks, fundamental: np.ndarray, cfg: dict[str, Any]):
    # 先用时间同步和极线几何做粗筛，减少 LoMa 需要跑的候选对数量。
    tolerance = int(float(cfg["max_time_diff_ms"]) * 1e6)
    minimum = int(cfg["min_candidate_frame_pairs"])
    margin = float(cfg["box_epipolar_margin_px"])
    sample_grid = int(cfg.get("epipolar_box_sample_grid", 3))
    maximum = int(cfg["max_frame_pairs_per_candidate"])
    out = []
    for key_a, a in a_tracks.items():
        for key_b, b in b_tracks.items():
            pairs = [(oa, ob) for oa, ob in nearest_pairs(a, b, tolerance) if boxes_epipolar_compatible(oa.box, ob.box, fundamental, margin, sample_grid)]
            if len(pairs) >= minimum:
                indices = np.linspace(0, len(pairs) - 1, min(maximum, len(pairs))).round().astype(int)
                out.append((key_a, key_b, [pairs[i] for i in sorted(set(indices))], len(pairs)))
    out.sort(key=lambda item: (-item[3], item[0], item[1]))
    return out


def boxes_epipolar_compatible(box_a: np.ndarray, box_b: np.ndarray, f: np.ndarray, margin: float, sample_grid: int = 3) -> bool:
    points_a = sample_box_points(box_a, sample_grid)
    points_b = sample_box_points(box_b, sample_grid)
    a_reaches_b = np.any(line_box_distances((f @ points_a.T).T, box_b) <= margin)
    b_reaches_a = np.any(line_box_distances((f.T @ points_b.T).T, box_a) <= margin)
    return a_reaches_b and b_reaches_a


def sample_box_points(box: np.ndarray, grid_size: int) -> np.ndarray:
    if grid_size < 1:
        raise ValueError("epipolar_box_sample_grid must be at least 1")
    fractions = np.linspace(0.0, 1.0, grid_size) if grid_size > 1 else np.asarray([0.5])
    xs = box[0] + fractions * (box[2] - box[0])
    ys = box[1] + fractions * (box[3] - box[1])
    return np.asarray([[x, y, 1.0] for y in ys for x in xs], dtype=np.float64)


def line_box_distances(lines: np.ndarray, box: np.ndarray) -> np.ndarray:
    corners = np.asarray([
        [box[0], box[1], 1.0], [box[2], box[1], 1.0],
        [box[0], box[3], 1.0], [box[2], box[3], 1.0],
    ])
    signed = lines @ corners.T
    intersects = (np.min(signed, axis=1) <= 0.0) & (np.max(signed, axis=1) >= 0.0)
    distances = np.min(np.abs(signed), axis=1) / np.maximum(np.linalg.norm(lines[:, :2], axis=1), 1e-12)
    distances[intersects] = 0.0
    return distances


def point_line_box_distance(line: np.ndarray, box: np.ndarray) -> float:
    x = np.clip(-(line[1] * ((box[1] + box[3]) / 2) + line[2]) / (line[0] + 1e-12), box[0], box[2])
    y = np.clip(-(line[0] * ((box[0] + box[2]) / 2) + line[2]) / (line[1] + 1e-12), box[1], box[3])
    points = np.asarray([[x, box[1]], [x, box[3]], [box[0], y], [box[2], y]])
    return float(np.min(np.abs(points @ line[:2] + line[2]) / max(np.linalg.norm(line[:2]), 1e-12)))


def make_matcher(cfg: dict[str, Any]):
    from loma import LoMa, LoMaB
    if str(cfg.get("model", "loma_b")).lower() != "loma_b":
        raise ValueError("The first implementation supports loma_b only")
    return LoMa(LoMaB(num_keypoints=int(cfg["num_keypoints"])))


def score_candidate(key_a, key_b, pairs, total_pairs, fundamental, calibration_a, calibration_b, matcher, track_features, cfg) -> Edge:
    results = [match_frame(a, b, fundamental, calibration_a, calibration_b, matcher, cfg) for a, b in pairs]
    valid = [r for r in results if r.matches >= int(cfg["association"]["min_matches_per_frame"])]
    scores = [r.score for r in valid]
    errors = [float(np.median(r.errors[r.errors <= float(cfg["association"]["max_epipolar_error_px"])])) for r in valid if r.inliers]
    loma_score = float(np.median(scores)) if scores else 0.0
    appearance_score = float(np.dot(track_features[key_a], track_features[key_b]))
    assoc = cfg["association"]
    score = float(assoc["fused_loma_weight"]) * loma_score + float(assoc["fused_appearance_weight"]) * appearance_score
    tri_cfg = assoc["triangulation"]
    tri_valid = [r for r in results if triangulation_frame_passes(r, tri_cfg)]
    return Edge(key_a, key_b, score, loma_score, appearance_score, sum(r.inlier_ratio >= float(assoc["min_epipolar_inlier_ratio"]) for r in valid), total_pairs, len(results), float(np.median([r.matches for r in results])), float(np.median([r.inliers for r in results])), float(np.median(errors)) if errors else float("inf"), len(tri_valid), float(np.median([r.triangulated_points for r in results])), float(np.median([r.positive_depth_ratio for r in results])), float(np.median([r.median_reprojection_error for r in results])), float(np.median([r.median_ray_angle_deg for r in results])), results)


def match_frame(a: Obs, b: Obs, fundamental: np.ndarray, calibration_a, calibration_b, matcher, cfg) -> FrameMatch:
    tensor_a, transform_a = crop_tensor(a, cfg["loma"])
    tensor_b, transform_b = crop_tensor(b, cfg["loma"])
    pa, pb = matcher.match(tensor_a, tensor_b, filter_threshold=float(cfg["loma"]["filter_threshold"]), num_keypoints=int(cfg["loma"]["num_keypoints"]))
    pa = crop_to_image(pa, transform_a)
    pb = crop_to_image(pb, transform_b)
    errors = symmetric_epipolar_errors(pa, pb, fundamental)
    threshold = float(cfg["association"]["max_epipolar_error_px"])
    inliers = int(np.sum(errors <= threshold))
    ratio = inliers / max(len(errors), 1)
    support = min(inliers / 20.0, 1.0)
    score = float(ratio * support)
    tri = triangulate_matches(pa[errors <= threshold], pb[errors <= threshold], calibration_a, calibration_b)
    return FrameMatch(a, b, pa, pb, errors, len(errors), inliers, ratio, score, *tri)


def triangulate_matches(points_a: np.ndarray, points_b: np.ndarray, calibration_a, calibration_b):
    if len(points_a) == 0:
        return 0, 0.0, float("inf"), 0.0, None
    ka, camera_a_to_ego = calibration_a
    kb, camera_b_to_ego = calibration_b
    ego_to_a, ego_to_b = np.linalg.inv(camera_a_to_ego), np.linalg.inv(camera_b_to_ego)
    pa, pb = ka @ ego_to_a[:3], kb @ ego_to_b[:3]
    homogeneous = cv2.triangulatePoints(pa, pb, points_a.T.astype(np.float64), points_b.T.astype(np.float64))
    valid_w = np.abs(homogeneous[3]) > 1e-9
    xyz = (homogeneous[:3, valid_w] / homogeneous[3:4, valid_w]).T
    if not len(xyz):
        return 0, 0.0, float("inf"), 0.0, None
    depth_a = ((ego_to_a[:3, :3] @ xyz.T) + ego_to_a[:3, 3:4])[2]
    depth_b = ((ego_to_b[:3, :3] @ xyz.T) + ego_to_b[:3, 3:4])[2]
    positive = (depth_a > 0) & (depth_b > 0)
    reproj_a = project_points(xyz, pa)
    reproj_b = project_points(xyz, pb)
    source_a, source_b = points_a[valid_w], points_b[valid_w]
    reprojection = 0.5 * (np.linalg.norm(reproj_a - source_a, axis=1) + np.linalg.norm(reproj_b - source_b, axis=1))
    center_a, center_b = camera_a_to_ego[:3, 3], camera_b_to_ego[:3, 3]
    ray_a, ray_b = xyz - center_a, xyz - center_b
    cosine = np.sum(ray_a * ray_b, axis=1) / np.maximum(np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1), 1e-12)
    angles = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    usable = positive & np.isfinite(reprojection) & np.isfinite(angles)
    center = np.median(xyz[usable], axis=0) if np.any(usable) else None
    return int(np.sum(usable)), float(np.mean(positive)), float(np.median(reprojection[usable])) if np.any(usable) else float("inf"), float(np.median(angles[usable])) if np.any(usable) else 0.0, center


def project_points(points: np.ndarray, projection: np.ndarray) -> np.ndarray:
    projected = (projection @ np.c_[points, np.ones(len(points))].T).T
    return projected[:, :2] / np.maximum(projected[:, 2:3], 1e-12)


def triangulation_frame_passes(frame: FrameMatch, cfg: dict[str, Any]) -> bool:
    return frame.triangulated_points >= int(cfg["min_points_per_frame"]) and frame.positive_depth_ratio >= float(cfg["min_positive_depth_ratio"]) and frame.median_reprojection_error <= float(cfg["max_median_reprojection_error_px"]) and frame.median_ray_angle_deg >= float(cfg["min_median_ray_angle_deg"])


def crop_tensor(obs: Obs, cfg: dict[str, Any]):
    image = cv2.imread(str(obs.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(obs.image)
    h, w = image.shape[:2]
    x1, y1, x2, y2 = obs.box
    px, py = (x2 - x1) * float(cfg["crop_padding"]), (y2 - y1) * float(cfg["crop_padding"])
    ix1, iy1, ix2, iy2 = max(0, int(x1 - px)), max(0, int(y1 - py)), min(w, int(x2 + px + 1)), min(h, int(y2 + py + 1))
    crop = image[iy1:iy2, ix1:ix2]
    if cfg.get("mask_background") and obs.mask and obs.mask.exists():
        mask = cv2.imread(str(obs.mask), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            local = mask[iy1:iy2, ix1:ix2] > 0
            crop = np.where(local[..., None], crop, 114).astype(np.uint8)
    size = int(cfg["input_size"])
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return tensor, (ix1, iy1, max(ix2 - ix1, 1), max(iy2 - iy1, 1), size)


def crop_to_image(points: np.ndarray, transform) -> np.ndarray:
    x, y, w, h, size = transform
    out = points.astype(np.float64).copy()
    out[:, 0] = x + out[:, 0] * w / size
    out[:, 1] = y + out[:, 1] * h / size
    return out


def symmetric_epipolar_errors(a: np.ndarray, b: np.ndarray, f: np.ndarray) -> np.ndarray:
    if not len(a):
        return np.empty(0)
    ah = np.c_[a, np.ones(len(a))]
    bh = np.c_[b, np.ones(len(b))]
    lb, la = (f @ ah.T).T, (f.T @ bh.T).T
    numerator = np.abs(np.sum(bh * lb, axis=1))
    return 0.5 * (numerator / np.maximum(np.linalg.norm(lb[:, :2], axis=1), 1e-12) + numerator / np.maximum(np.linalg.norm(la[:, :2], axis=1), 1e-12))


def load_calibration(cfg):
    from .data import load_extrinsic, load_intrinsic
    return {view: (load_intrinsic(resolve(spec["intrinsic"])), load_extrinsic(resolve(spec["camera_to_ego"]))) for view, spec in cfg["inputs"]["views"].items()}


def fundamental_matrix(a, b):
    ka, ta = a
    kb, tb = b
    a_to_b = np.linalg.inv(tb) @ ta
    r, t = a_to_b[:3, :3], a_to_b[:3, 3]
    skew = np.asarray([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    return np.linalg.inv(kb).T @ skew @ r @ np.linalg.inv(ka)


def passes(edge: Edge, cfg) -> bool:
    standard_loma = edge.loma_score >= float(cfg["accept_score"]) and edge.verified_frames >= int(cfg["min_verified_frames"])
    short_track = edge.candidate_frames <= int(cfg["short_track_max_pairs"]) and edge.verified_frames >= int(cfg["short_track_min_verified_frames"]) and edge.loma_score >= float(cfg["short_track_min_loma_score"])
    fused = edge.score >= float(cfg["fused_accept_score"]) and edge.appearance_score >= float(cfg["appearance_min_score"]) and edge.verified_frames >= int(cfg["fused_min_verified_frames"])
    tri_cfg = cfg["triangulation"]
    triangulated = edge.triangulation_verified_frames >= int(tri_cfg["min_verified_frames"]) and edge.median_triangulated_points >= float(tri_cfg["min_median_points"]) and edge.loma_score >= float(tri_cfg["min_track_loma_score"]) and edge.appearance_score >= float(cfg["appearance_min_score"])
    strong_triangulation = edge.triangulation_verified_frames >= int(tri_cfg["strong_min_verified_frames"]) and edge.median_triangulated_points >= float(tri_cfg["strong_min_median_points"]) and edge.loma_score >= float(tri_cfg["min_track_loma_score"])
    return standard_loma or short_track or fused or triangulated or strong_triangulation


def hungarian_edges(edges: list[Edge]) -> list[Edge]:
    if not edges:
        return []
    from scipy.optimize import linear_sum_assignment
    aa, bb = sorted({e.a for e in edges}), sorted({e.b for e in edges})
    matrix = np.full((len(aa), len(bb)), 1e6)
    lookup = {(e.a, e.b): e for e in edges}
    for e in edges:
        matrix[aa.index(e.a), bb.index(e.b)] = 1.0 - e.score
    rows, cols = linear_sum_assignment(matrix)
    return [lookup[(aa[i], bb[j])] for i, j in zip(rows, cols) if (aa[i], bb[j]) in lookup]


def global_assignments(observations, edges):
    nodes = {(o.view, o.track_id) for items in observations.values() for o in items}
    parent = {n: n for n in nodes}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for edge in sorted(edges, key=lambda e: -e.score):
        ra, rb = find(edge.a), find(edge.b)
        if ra != rb:
            parent[rb] = ra
    roots = sorted({find(n) for n in nodes})
    root_id = {root: i for i, root in enumerate(roots)}
    return {n: root_id[find(n)] for n in nodes}


def edge_row(edge, accepted):
    return {"view_a": edge.a[0], "track_id_a": edge.a[1], "view_b": edge.b[0], "track_id_b": edge.b[1], "score": edge.score, "loma_score": edge.loma_score, "appearance_score": edge.appearance_score, "verified_frames": edge.verified_frames, "candidate_frames": edge.candidate_frames, "sampled_frames": edge.sampled_frames, "median_matches": edge.median_matches, "median_inliers": edge.median_inliers, "median_epipolar_error_px": edge.median_error, "triangulation_verified_frames": edge.triangulation_verified_frames, "median_triangulated_points": edge.median_triangulated_points, "median_positive_depth_ratio": edge.median_positive_depth_ratio, "median_reprojection_error_px": edge.median_reprojection_error, "median_ray_angle_deg": edge.median_ray_angle_deg, "selected": accepted}


def write_outputs(out_dir, rows_by_view, assignments, accepted, diagnostics):
    write_csv(out_dir / "candidate_diagnostics.csv", diagnostics)
    write_csv(out_dir / "accepted_edges.csv", [edge_row(e, True) for e in accepted])
    assignment_rows = [{"view": view, "track_id": track, "global_track_id": gid} for (view, track), gid in sorted(assignments.items())]
    write_csv(out_dir / "global_track_assignments.csv", assignment_rows)
    combined = []
    for view, rows in rows_by_view.items():
        for row in rows:
            item = dict(row)
            tid = int(row["track_id"])
            item["global_track_id"] = assignments.get((view, tid), -1)
            combined.append(item)
    write_csv(out_dir / "global_2d_observations.csv", combined)


def render_edges(out_dir: Path, edges: list[Edge], cfg):
    out_dir.mkdir(parents=True, exist_ok=True)
    for rank, edge in enumerate(sorted(edges, key=lambda e: -e.score)[: int(cfg["max_edges"])]):
        for fi, match in enumerate(sorted(edge.frames, key=lambda f: -f.score)[: int(cfg["max_frames_per_edge"])]):
            canvas = draw_match(match)
            name = f"{rank:03d}_{edge.a[0]}_{edge.a[1]}__{edge.b[0]}_{edge.b[1]}_f{fi}_s{edge.score:.3f}.jpg"
            cv2.imwrite(str(out_dir / name), canvas)


def draw_match(match: FrameMatch):
    ia, ib = cv2.imread(str(match.a.image)), cv2.imread(str(match.b.image))
    ca, ta = visual_crop(ia, match.a.box); cb, tb = visual_crop(ib, match.b.box)
    height = max(ca.shape[0], cb.shape[0]); ca = pad_height(ca, height); cb = pad_height(cb, height)
    canvas = np.concatenate([ca, cb], axis=1)
    for i, (pa, pb, error) in enumerate(zip(match.points_a, match.points_b, match.errors)):
        if error > 3.0: continue
        color = tuple(int(x) for x in np.random.default_rng(i).integers(80, 255, 3))
        p1 = (int(pa[0] - ta[0]), int(pa[1] - ta[1])); p2 = (int(pb[0] - tb[0] + ca.shape[1]), int(pb[1] - tb[1]))
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA); cv2.circle(canvas, p1, 3, color, -1); cv2.circle(canvas, p2, 3, color, -1)
    cv2.putText(canvas, f"{match.a.view}:{match.a.track_id}  {match.b.view}:{match.b.track_id} matches={match.matches} inliers={match.inliers} ratio={match.inlier_ratio:.2f}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return canvas


def visual_crop(image, box):
    h, w = image.shape[:2]; x1, y1, x2, y2 = box.astype(int); p = 30
    x1, y1, x2, y2 = max(0, x1-p), max(0, y1-p), min(w, x2+p), min(h, y2+p)
    return image[y1:y2, x1:x2].copy(), (x1, y1)


def pad_height(image, height):
    return cv2.copyMakeBorder(image, 0, height-image.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(25,25,25))


def resolve(value):
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
