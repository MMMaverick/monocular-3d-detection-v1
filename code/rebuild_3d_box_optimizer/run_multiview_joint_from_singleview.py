from __future__ import annotations

"""从单视角 3D 优化结果出发，做跨视角联合后优化。

当前主线用法：
    1. 先运行 run.py 得到每个视角各自的 3D track CSV。
    2. 再运行 associate_2d_tracks_loma.py 得到跨视角 global id。
    3. 本脚本读取这两类结果，把同一个 global id 下的多视角观测合并到一个优化问题里。

注意：
    这里默认使用 LoMa/特征匹配生成的 fixed_global_assignments。
    旧的“根据单视角 3D 位置重新跨视角关联”逻辑仍保留，但当前主线暂不推荐使用，
    因为单视角 3D 初值本身可能被 bbox/mask 退化拉偏。
"""

import argparse
import csv
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .associate_multiview_tracks import (
    MatchEdge,
    Tracklet,
    build_assignment_rows,
    build_candidate_diagnostic_rows,
    build_component_rows,
    connected_components,
    edge_to_row,
    association_view_pairs,
    match_pair,
)
from .config import load_config, resolve_path, write_experiment_config_snapshot, write_resolved_config
from .data import Observation, group_by_track, load_view_observations
from .optimizer import (
    TrackResult,
    build_result,
    choose_device,
    compute_losses,
    make_observation_tensors,
    make_scheduler,
    robust_track_size_init,
    size_to_unconstrained,
    unconstrained_to_size,
)
from .run import append_progress, write_csv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Joint multiview refinement initialized from already optimized single-view results.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    joint_cfg = config.get("joint_from_singleview", {})
    out_dir = resolve_path(config, config["output"].get("dir", "outputs/multiview_joint_from_singleview_v1"))
    source_dir = resolve_path(config, joint_cfg.get("singleview_output_dir", "outputs/rebuild_three_views_no_fallback_mask_size_reliable_v1"))
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.log"
    progress_path.write_text("", encoding="utf-8")
    append_progress(progress_path, f"START config={args.config} output={out_dir} singleview_source={source_dir}")
    if config["output"].get("copy_source_config", True):
        shutil.copy2(args.config, out_dir / "source_config.yaml")
    if config["output"].get("resolved_config", True):
        write_resolved_config(config, out_dir / "resolved_config.yaml")
        write_experiment_config_snapshot(config, out_dir / "experiment_config_snapshot.yaml")

    run_start = time.time()
    # 单视角 3D 优化 CSV 是二次优化的初始几何来源。
    source_rows = read_singleview_boxes(source_dir / str(joint_cfg.get("singleview_boxes_csv", "frame_3d_boxes_world_track_joint.csv")))
    observations_by_view, optimized_by_key = load_observations_with_singleview_init(config, source_rows, progress_path)
    tracklets = build_optimized_tracklets(config, observations_by_view, optimized_by_key, progress_path)
    # 当前主线优先走 fixed_assignment_components：直接读取 LoMa 输出的 global_track_assignments.csv。
    components, edges, view_pairs, component_ids = associate_optimized_tracklets(config, tracklets, progress_path)
    write_association_outputs(config, out_dir, tracklets, components, edges, view_pairs, component_ids)

    jobs = build_component_jobs(config, components, observations_by_view, component_ids)
    max_global_tracks = int(joint_cfg.get("max_global_tracks", 0) or 0)
    if max_global_tracks > 0:
        jobs = jobs[:max_global_tracks]
    append_progress(progress_path, f"OPTIMIZE_START global_tracks={len(jobs)}")

    all_rows: list[dict[str, object]] = []
    all_diag: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for index, (global_track_id, component, observations) in enumerate(jobs, start=1):
        label = ",".join(f"{view}:{track_id}" for view, track_id in component)
        track_start = time.time()
        append_progress(progress_path, f"GLOBAL_TRACK_START gid={global_track_id} tracks={label} observations={len(observations)} index={index}/{len(jobs)}")
        try:
            result = optimize_global_track_from_singleview(config, global_track_id, component, observations)
        except Exception as exc:
            append_progress(progress_path, f"GLOBAL_TRACK_FAILED gid={global_track_id} error={type(exc).__name__}:{exc}")
            continue
        all_rows.extend(result.rows)
        all_diag.extend(result.diagnostics)
        summaries.append(result.summary)
        append_progress(
            progress_path,
            "GLOBAL_TRACK_DONE "
            f"gid={global_track_id} observations={len(observations)} index={index}/{len(jobs)} "
            f"iters={result.summary.get('iterations_used')} stop={result.summary.get('stop_reason')} "
            f"best_loss={float(result.summary.get('best_loss', float('nan'))):.6g} "
            f"mean_frame_loss={float(result.summary.get('mean_frame_loss', float('nan'))):.6g} "
            f"max_frame_loss={float(result.summary.get('max_frame_loss', float('nan'))):.6g} "
            f"dominant={result.summary.get('dominant_loss')} elapsed_sec={time.time() - track_start:.1f}"
        )

    boxes_path = out_dir / "frame_3d_boxes_multiview_joint_from_singleview.csv"
    diagnostics_path = out_dir / "frame_loss_diagnostics.csv"
    write_csv(boxes_path, all_rows)
    write_csv(diagnostics_path, all_diag)
    write_csv(out_dir / "global_track_optimization_summary.csv", summaries)
    write_summary(out_dir / "summary.txt", source_dir, all_rows, all_diag, summaries, components, edges)
    append_progress(progress_path, f"CSV_DONE frames={len(all_rows)} global_tracks={len(summaries)} diagnostics={len(all_diag)}")

    if config.get("output", {}).get("videos", True):
        from .visualization import render_experiment_videos

        append_progress(progress_path, "VIDEO_START")
        render_experiment_videos(config, diagnostics_path, out_dir)
        append_progress(progress_path, "VIDEO_DONE")
    append_progress(progress_path, f"DONE total_elapsed_sec={time.time() - run_start:.1f}")
    return 0


def read_singleview_boxes(path: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[tuple[str, int, int], dict[str, str]] = {}
    for row in rows:
        try:
            key = (str(row["view"]), int(float(row["track_id"])), int(row["timestamp"]))
        except (KeyError, ValueError):
            continue
        out[key] = row
    return out


def load_observations_with_singleview_init(
    config: dict[str, Any],
    source_rows: dict[tuple[str, int, int], dict[str, str]],
    progress_path: Path,
) -> tuple[dict[str, dict[int, list[Observation]]], dict[tuple[str, int, int], dict[str, Any]]]:
    """读取原始观测，并把单视角优化出的 center/size 写回 Observation 初值。

    这里会丢弃没有单视角优化结果的帧，保证二次优化只作用于当前确认可用的结果。
    """
    views = list(config.get("association", {}).get("views", config.get("scope", {}).get("cameras", ["rear", "left_rear", "right_rear"])))
    observations_by_view: dict[str, dict[int, list[Observation]]] = {}
    optimized_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    for view in views:
        append_progress(progress_path, f"LOAD_VIEW_START view={view}")
        observations = load_view_observations(config, view)
        patched: list[Observation] = []
        matched = 0
        for obs in observations:
            key = (obs.view, obs.track_id, obs.timestamp)
            source = source_rows.get(key)
            if source is None:
                continue
            center_cam = np.asarray([float(source["cx"]), float(source["cy"]), float(source["cz"])], dtype=np.float64)
            size = np.asarray([float(source["length"]), float(source["width"]), float(source["height"])], dtype=np.float64)
            center_world = transform_point(center_cam, obs.camera_to_world)
            patched.append(replace(obs, init_center_cam=center_cam, init_center_world=center_world, init_size=size))
            optimized_by_key[key] = {"center_cam": center_cam, "center_world": center_world, "size": size}
            matched += 1
        observations_by_view[view] = group_by_track(patched)
        append_progress(progress_path, f"LOAD_VIEW_DONE view={view} usable_singleview_frames={matched} tracks={len(observations_by_view[view])}")
    return observations_by_view, optimized_by_key


def build_optimized_tracklets(
    config: dict[str, Any],
    observations_by_view: dict[str, dict[int, list[Observation]]],
    optimized_by_key: dict[tuple[str, int, int], dict[str, Any]],
    progress_path: Path,
) -> dict[tuple[str, int], Tracklet]:
    min_frames = int(config.get("association", {}).get("min_track_frames", config["solver"].get("min_track_frames", 3)))
    tracklets: dict[tuple[str, int], Tracklet] = {}
    for view, grouped in observations_by_view.items():
        kept = 0
        for track_id, observations in grouped.items():
            if len(observations) < min_frames:
                continue
            tracklets[(view, track_id)] = make_optimized_tracklet(view, track_id, observations, optimized_by_key)
            kept += 1
        append_progress(progress_path, f"TRACKLETS_BUILT view={view} kept={kept}")
    return tracklets


def make_optimized_tracklet(
    view: str,
    track_id: int,
    observations: list[Observation],
    optimized_by_key: dict[tuple[str, int, int], dict[str, Any]],
) -> Tracklet:
    timestamps = np.asarray([o.timestamp for o in observations], dtype=np.int64)
    centers_world = np.stack([optimized_by_key[(o.view, o.track_id, o.timestamp)]["center_world"] for o in observations]).astype(np.float64)
    centers_cam = np.stack([optimized_by_key[(o.view, o.track_id, o.timestamp)]["center_cam"] for o in observations]).astype(np.float64)
    boxes = np.stack([o.box2d for o in observations]).astype(np.float64)
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    labels = [o.label for o in observations]
    label = max(set(labels), key=labels.count) if labels else "default"
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


def associate_optimized_tracklets(config: dict[str, Any], tracklets: dict[tuple[str, int], Tracklet], progress_path: Path):
    assoc_cfg = config.get("association", {})
    mv_cfg = config.get("multiview_joint", {})
    if bool(mv_cfg.get("enabled", False)) and str(assoc_cfg.get("mode", "")).lower() in {"fixed_global_assignments", "fixed_global_track_assignments", "loma_global_assignments"}:
        return fixed_assignment_components(config, tracklets, progress_path)
    # 暂未作为主线使用：根据单视角 3D 轨迹相似度重新匹配跨视角 track。
    # 保留是为了后续消融/兜底，但当前应优先相信 LoMa 输出的 global id 文件。
    views = list(assoc_cfg.get("views", config.get("scope", {}).get("cameras", ["rear", "left_rear", "right_rear"])))
    edges: list[MatchEdge] = []
    view_pairs = association_view_pairs(views, assoc_cfg)
    for a_view, b_view in view_pairs:
        a_items = [t for t in tracklets.values() if t.view == a_view]
        b_items = [t for t in tracklets.values() if t.view == b_view]
        append_progress(progress_path, f"MATCH_PAIR_START {a_view}<->{b_view} candidates={len(a_items)}x{len(b_items)}")
        pair_edges = match_pair(a_items, b_items, assoc_cfg)
        edges.extend(pair_edges)
        append_progress(progress_path, f"MATCH_PAIR_DONE {a_view}<->{b_view} accepted={len(pair_edges)}")
    return connected_components(tracklets.keys(), edges), edges, view_pairs, None


def fixed_assignment_components(config: dict[str, Any], tracklets: dict[tuple[str, int], Tracklet], progress_path: Path):
    assoc_cfg = config.get("association", {})
    assignment_path = resolve_path(config, assoc_cfg["global_track_assignments_csv"])
    append_progress(progress_path, f"FIXED_ASSIGNMENTS_START path={assignment_path}")
    rows = read_csv_rows(assignment_path)
    by_gid: dict[int, list[tuple[str, int]]] = {}
    missing = 0
    for row in rows:
        try:
            gid = int(float(row["global_track_id"]))
            key = (str(row["view"]), int(float(row["track_id"])))
        except (KeyError, ValueError):
            continue
        if key not in tracklets:
            missing += 1
            continue
        by_gid.setdefault(gid, []).append(key)
    multi_only = bool(config.get("multiview_joint", {}).get("multi_view_only", assoc_cfg.get("multi_view_only", True)))
    components: list[list[tuple[str, int]]] = []
    component_ids: list[int] = []
    for gid in sorted(by_gid):
        comp = sorted(set(by_gid[gid]))
        if multi_only and len({view for view, _ in comp}) < 2:
            continue
        components.append(comp)
        component_ids.append(gid)
    edges = fixed_edges_from_csv(config)
    view_pairs = [tuple(pair) for pair in assoc_cfg.get("view_pairs", assoc_cfg.get("allowed_view_pairs", []))]
    append_progress(progress_path, f"FIXED_ASSIGNMENTS_DONE groups={len(components)} missing_tracks={missing} edges={len(edges)}")
    return components, edges, view_pairs, component_ids


def fixed_edges_from_csv(config: dict[str, Any]) -> list[MatchEdge]:
    assoc_cfg = config.get("association", {})
    path_value = assoc_cfg.get("accepted_edges_csv", "")
    if not path_value:
        return []
    path = resolve_path(config, path_value)
    if not path.exists():
        return []
    edges: list[MatchEdge] = []
    for row in read_csv_rows(path):
        try:
            a = (str(row["view_a"]), int(float(row["track_id_a"])))
            b = (str(row["view_b"]), int(float(row["track_id_b"])))
            score = float(row.get("score", row.get("loma_score", 0.0)) or 0.0)
            verified = int(float(row.get("verified_frames", row.get("candidate_frames", 0)) or 0))
        except (KeyError, ValueError):
            continue
        edges.append(MatchEdge(a=a, b=b, cost=1.0 - score, score=score, mean_dist=float("nan"), median_dist=float("nan"), overlap_pairs=verified, label_compatible=True, time_overlap_sec=float("nan")))
    return edges


def write_association_outputs(config: dict[str, Any], out_dir: Path, tracklets: dict[tuple[str, int], Tracklet], components, edges, view_pairs, component_ids=None) -> None:
    assoc_cfg = config.get("association", {})
    if component_ids is None:
        write_csv(out_dir / "global_track_assignments.csv", build_assignment_rows(components, tracklets))
        write_csv(out_dir / "global_track_components.csv", build_component_rows(components, tracklets))
    else:
        write_csv(out_dir / "global_track_assignments.csv", fixed_assignment_rows(components, component_ids, tracklets))
        write_csv(out_dir / "global_track_components.csv", fixed_component_rows(components, component_ids, tracklets))
    write_csv(out_dir / "match_edges.csv", [edge_to_row(e) for e in sorted(edges, key=lambda x: (x.a[0], x.a[1], x.b[0], x.b[1]))])
    write_csv(out_dir / "candidate_match_diagnostics.csv", build_candidate_diagnostic_rows(view_pairs, tracklets, assoc_cfg))


def fixed_assignment_rows(components: list[list[tuple[str, int]]], component_ids: list[int], tracklets: dict[tuple[str, int], Tracklet]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for gid, comp in zip(component_ids, components, strict=True):
        views = [view for view, _ in comp]
        component_views = ",".join(sorted(set(views)))
        view_conflict = len(set(views)) != len(views)
        for view, track_id in comp:
            t = tracklets[(view, track_id)]
            rows.append({"global_track_id": gid, "component_size": len(comp), "component_views": component_views, "view_conflict": view_conflict, "view": view, "track_id": track_id, "class": t.label, "frames": t.frames, "start_frame": t.start_frame, "end_frame": t.end_frame, "start_ts": t.start_ts, "end_ts": t.end_ts})
    return rows


def fixed_component_rows(components: list[list[tuple[str, int]]], component_ids: list[int], tracklets: dict[tuple[str, int], Tracklet]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for gid, comp in zip(component_ids, components, strict=True):
        views = [view for view, _ in comp]
        labels = [tracklets[n].label for n in comp]
        starts = [tracklets[n].start_ts for n in comp]
        ends = [tracklets[n].end_ts for n in comp]
        rows.append({"global_track_id": gid, "component_size": len(comp), "component_views": ",".join(sorted(set(views))), "view_conflict": len(set(views)) != len(views), "tracks": ";".join(f"{view}:{track_id}" for view, track_id in comp), "classes": ",".join(sorted(set(labels))), "start_ts": min(starts), "end_ts": max(ends)})
    return rows


def build_component_jobs(config: dict[str, Any], components: list[list[tuple[str, int]]], observations_by_view: dict[str, dict[int, list[Observation]]], component_ids: list[int] | None = None):
    min_obs = int(config.get("joint_from_singleview", {}).get("min_observations", config["solver"].get("min_track_frames", 3)))
    include_singletons = bool(config.get("joint_from_singleview", {}).get("optimize_single_view_components", True))
    if component_ids is None:
        pairs = [(gid, comp) for gid, comp in enumerate(sorted(components, key=lambda comp: min(observations_by_view[v][tid][0].timestamp for v, tid in comp)), start=1)]
    else:
        pairs = list(zip(component_ids, components, strict=True))
    pairs = filter_requested_components(config, pairs)
    jobs = []
    for gid, comp in pairs:
        if len(comp) <= 1 and not include_singletons:
            continue
        observations: list[Observation] = []
        for view, track_id in comp:
            observations.extend(observations_by_view[view][track_id])
        observations.sort(key=lambda o: (o.timestamp, o.view, o.frame, o.track_id))
        if len(observations) >= min_obs:
            jobs.append((gid, comp, observations))
    return jobs


def filter_requested_components(config: dict[str, Any], pairs: list[tuple[int, list[tuple[str, int]]]]) -> list[tuple[int, list[tuple[str, int]]]]:
    mv_cfg = config.get("multiview_joint", {})
    gids = {int(x) for x in mv_cfg.get("only_global_track_ids", [])}
    source_specs = set(str(x) for x in mv_cfg.get("only_source_tracks", []))
    if not gids and not source_specs:
        return pairs
    out = []
    for gid, comp in pairs:
        sources = {f"{view}:{track_id}" for view, track_id in comp}
        if (gids and gid in gids) or (source_specs and sources.intersection(source_specs)):
            out.append((gid, comp))
    return out


def optimize_global_track_from_singleview(
    config: dict[str, Any],
    global_track_id: int,
    component: list[tuple[str, int]],
    observations: list[Observation],
) -> TrackResult:
    device = choose_device(str(config["solver"].get("device", "cuda")))
    dtype = torch.float32 if str(config["solver"].get("dtype", "float32")) == "float32" else torch.float64
    timestamps, obs_to_node = unique_timestamp_nodes(config, observations)
    centers0_world = initial_world_centers(observations, timestamps, obs_to_node, config)
    centers_param = torch.nn.Parameter(torch.tensor(centers0_world, dtype=dtype, device=device))

    size0_np = robust_track_size_init(observations)
    min_size_np = np.max(np.stack([o.min_size for o in observations]), axis=0)
    max_size_np = np.min(np.stack([o.max_size for o in observations]), axis=0)
    min_size_np = np.minimum(min_size_np, max_size_np * 0.99)
    size0_np = np.clip(size0_np, min_size_np, max_size_np)
    min_size = torch.tensor(min_size_np, dtype=dtype, device=device)
    max_size = torch.tensor(max_size_np, dtype=dtype, device=device)
    size_param = torch.nn.Parameter(size_to_unconstrained(torch.tensor(size0_np, dtype=dtype, device=device), min_size, max_size))

    tensors = make_observation_tensors(config, observations, dtype, device)
    tensors["world_to_camera"] = torch.tensor(np.stack([o.world_to_camera for o in observations]), dtype=dtype, device=device)
    tensors["obs_to_node"] = torch.tensor(obs_to_node, dtype=torch.long, device=device)
    opt = torch.optim.Adam([centers_param, size_param], lr=float(config["solver"].get("learning_rate", 0.01)), weight_decay=float(config["solver"].get("weight_decay", 0.0)))
    scheduler = make_scheduler(config, opt)

    max_iter_cfg = int(config["solver"].get("max_iterations", 300) or 0)
    convergence_cfg = config["solver"].get("convergence", {})
    convergence_enabled = bool(convergence_cfg.get("enabled", max_iter_cfg <= 0))
    max_iter = max_iter_cfg if max_iter_cfg > 0 else int(convergence_cfg.get("safety_max_iterations", 2500))
    min_iter = int(convergence_cfg.get("min_iterations", min(400, max_iter)))
    patience = int(convergence_cfg.get("patience", 180))
    check_every = max(int(convergence_cfg.get("check_every", 10)), 1)
    rel_tol = float(convergence_cfg.get("relative_improvement", 1.0e-5))
    abs_tol = float(convergence_cfg.get("absolute_improvement", 1.0e-7))
    best_loss = float("inf")
    best_centers_world = centers_param.detach().clone()
    best_size = torch.tensor(size0_np, dtype=dtype, device=device)
    no_improve_steps = 0
    iterations_used = 0
    stop_reason = "max_iterations"
    progress_interval = int(config["solver"].get("progress_interval", 0) or 0)
    progress_label = ";".join(f"{view}:{track_id}" for view, track_id in component)
    for step in range(max_iter):
        iterations_used = step + 1
        opt.zero_grad(set_to_none=True)
        size = unconstrained_to_size(size_param, min_size, max_size)
        centers_cam = centers_world_to_obs_camera(centers_param, tensors)
        losses = compute_losses(config, centers_cam, size, tensors, protect_size=True)
        total = losses["total"] + temporal_smoothness_loss(config, centers_param, timestamps)
        if not torch.isfinite(total):
            stop_reason = "non_finite_loss"
            break
        total.backward()
        clip = float(config["solver"].get("gradient_clip", 0.0) or 0.0)
        if clip > 0:
            torch.nn.utils.clip_grad_norm_([centers_param, size_param], clip)
        opt.step()
        if scheduler is not None:
            scheduler.step()
        if step == 0 or iterations_used % check_every == 0 or iterations_used == max_iter:
            value = float(total.detach().cpu())
            threshold = max(abs_tol, rel_tol * max(abs(best_loss), 1.0)) if np.isfinite(best_loss) else 0.0
            if value < best_loss - threshold:
                best_loss = value
                best_centers_world = centers_param.detach().clone()
                best_size = size.detach().clone()
                no_improve_steps = 0
            else:
                no_improve_steps += check_every
            if convergence_enabled and iterations_used >= min_iter and no_improve_steps >= patience:
                stop_reason = "converged"
                break
        if progress_interval > 0 and (iterations_used == 1 or iterations_used % progress_interval == 0 or iterations_used == max_iter):
            print(
                f"GLOBAL_TRACK_PROGRESS gid={global_track_id} tracks={progress_label} "
                f"iter={iterations_used}/{max_iter} loss={float(total.detach().cpu()):.6g} "
                f"best={best_loss:.6g} lr={current_learning_rate(opt):.6g}",
                flush=True,
            )

    final_centers_cam = centers_world_to_obs_camera(best_centers_world, tensors)
    final_losses = compute_losses(config, final_centers_cam, best_size, tensors, reduce=False)
    result = build_result(config, observations, final_centers_cam, best_size, final_losses, best_loss, device, iterations_used, stop_reason)
    source_tracks = ";".join(f"{view}:{track_id}" for view, track_id in component)
    for row, obs, node_index in zip(result.rows, observations, obs_to_node, strict=True):
        center_world = best_centers_world[node_index].detach().cpu().numpy()
        row["global_track_id"] = global_track_id
        row["source_track_id"] = obs.track_id
        row["source_tracks"] = source_tracks
        row["world_cx"] = float(center_world[0])
        row["world_cy"] = float(center_world[1])
        row["world_cz"] = float(center_world[2])
        row["initialized_from"] = "singleview_optimized_result"
    for row, obs, node_index in zip(result.diagnostics, observations, obs_to_node, strict=True):
        center_world = best_centers_world[node_index].detach().cpu().numpy()
        row["global_track_id"] = global_track_id
        row["source_track_id"] = obs.track_id
        row["source_tracks"] = source_tracks
        row["world_cx"] = float(center_world[0])
        row["world_cy"] = float(center_world[1])
        row["world_cz"] = float(center_world[2])
        row["initialized_from"] = "singleview_optimized_result"
    result.summary["global_track_id"] = global_track_id
    result.summary["source_tracks"] = source_tracks
    result.summary["views"] = ",".join(sorted({v for v, _ in component}))
    result.summary["single_view_track_count"] = len(component)
    result.summary["world_nodes"] = len(timestamps)
    result.summary["initialized_from"] = "singleview_optimized_result"
    return result


def unique_timestamp_nodes(config: dict[str, Any], observations: list[Observation]) -> tuple[np.ndarray, np.ndarray]:
    init_cfg = config.get("multiview_joint", {}).get("initialization", {})
    tolerance_ns = int(float(init_cfg.get("timestamp_merge_tolerance_ms", config.get("association", {}).get("max_time_diff_ms", 0.0))) * 1.0e6)
    sorted_obs = sorted(enumerate(observations), key=lambda item: item[1].timestamp)
    nodes: list[list[int]] = []
    node_ts: list[int] = []
    for obs_index, obs in sorted_obs:
        if not nodes or (tolerance_ns > 0 and abs(obs.timestamp - node_ts[-1]) > tolerance_ns) or tolerance_ns <= 0 and obs.timestamp != node_ts[-1]:
            nodes.append([obs_index])
            node_ts.append(obs.timestamp)
        else:
            nodes[-1].append(obs_index)
            node_ts[-1] = int(round(np.mean([observations[i].timestamp for i in nodes[-1]])))
    obs_to_node = np.zeros(len(observations), dtype=np.int64)
    for node_index, indices in enumerate(nodes):
        for obs_index in indices:
            obs_to_node[obs_index] = node_index
    return np.asarray(node_ts, dtype=np.int64), obs_to_node


def initial_world_centers(observations: list[Observation], timestamps: np.ndarray, obs_to_node: np.ndarray, config: dict[str, Any] | None = None) -> np.ndarray:
    config = config or {}
    init_cfg = config.get("multiview_joint", {}).get("initialization", {})
    centers = np.zeros((len(timestamps), 3), dtype=np.float64)
    for i in range(len(timestamps)):
        node_obs = [obs for obs, node in zip(observations, obs_to_node, strict=True) if node == i]
        pts = np.stack([obs.init_center_world for obs in node_obs])
        weights = np.asarray([initial_observation_weight(obs, init_cfg) for obs in node_obs], dtype=np.float64)
        centers[i] = robust_fused_center(pts, weights, init_cfg)
    return smooth_initial_centers(centers, timestamps, init_cfg)


def initial_observation_weight(obs: Observation, init_cfg: dict[str, Any]) -> float:
    view_weights = init_cfg.get("view_weights", {})
    weight = float(view_weights.get(obs.view, 1.0))
    trunc_cfg = init_cfg.get("truncation_weight", {})
    truncated_count = sum(1 for value in obs.truncated.values() if value)
    if truncated_count >= 2:
        weight *= float(trunc_cfg.get("multi_side", trunc_cfg.get("truncated", 0.25)))
    elif truncated_count == 1:
        weight *= float(trunc_cfg.get("single_side", trunc_cfg.get("truncated", 0.5)))
    else:
        weight *= float(trunc_cfg.get("none", 1.0))
    return max(weight, float(init_cfg.get("min_observation_weight", 1.0e-3)))


def robust_fused_center(points: np.ndarray, weights: np.ndarray, init_cfg: dict[str, Any]) -> np.ndarray:
    if len(points) == 1:
        return points[0]
    weights = weights / max(float(weights.sum()), 1.0e-12)
    x0 = np.sum(points * weights[:, None], axis=0)
    distances = np.linalg.norm(points - x0[None, :], axis=1)
    hard = float(init_cfg.get("hard_outlier_m", 8.0))
    if np.max(distances) > hard:
        return points[int(np.argmax(weights))]
    soft = max(float(init_cfg.get("soft_outlier_m", 3.0)), 1.0e-6)
    robust = np.minimum(1.0, soft / np.maximum(distances, 1.0e-6))
    weights = weights * robust
    weights = weights / max(float(weights.sum()), 1.0e-12)
    return np.sum(points * weights[:, None], axis=0)


def smooth_initial_centers(centers: np.ndarray, timestamps: np.ndarray, init_cfg: dict[str, Any]) -> np.ndarray:
    smooth_cfg = init_cfg.get("temporal_smoothing", {})
    if not smooth_cfg.get("enabled", True) or len(centers) < 3:
        return centers
    max_gap_ns = int(float(smooth_cfg.get("max_gap_ms", 300.0)) * 1.0e6)
    prev_w = float(smooth_cfg.get("prev_weight", 0.25))
    cur_w = float(smooth_cfg.get("current_weight", 0.50))
    next_w = float(smooth_cfg.get("next_weight", 0.25))
    out = centers.copy()
    for i in range(1, len(centers) - 1):
        if timestamps[i] - timestamps[i - 1] > max_gap_ns or timestamps[i + 1] - timestamps[i] > max_gap_ns:
            continue
        denom = prev_w + cur_w + next_w
        out[i] = (prev_w * centers[i - 1] + cur_w * centers[i] + next_w * centers[i + 1]) / max(denom, 1.0e-12)
    return out


def centers_world_to_obs_camera(centers_world: torch.Tensor, tensors: dict[str, Any]) -> torch.Tensor:
    obs_centers = centers_world[tensors["obs_to_node"]]
    transform = tensors["world_to_camera"]
    return torch.einsum("nij,nj->ni", transform[:, :3, :3], obs_centers) + transform[:, :3, 3]


def temporal_smoothness_loss(config: dict[str, Any], centers_world: torch.Tensor, timestamps: np.ndarray) -> torch.Tensor:
    cfg = config.get("joint_from_singleview", {}).get("temporal_smoothness", {})
    if not cfg.get("enabled", True) or centers_world.shape[0] < 3:
        return centers_world.new_tensor(0.0)
    weight = float(cfg.get("weight", 0.02))
    accel = centers_world[2:] - 2.0 * centers_world[1:-1] + centers_world[:-2]
    robust = float(cfg.get("robust_m", 3.0))
    sq = accel.square().sum(dim=1)
    return torch.where(sq < robust * robust, sq, 2.0 * robust * torch.sqrt(sq.clamp_min(1.0e-12)) - robust * robust).mean() * weight


def transform_point(point: np.ndarray, transform: np.ndarray) -> np.ndarray:
    homo = np.concatenate([point.astype(np.float64), np.ones(1, dtype=np.float64)])
    return (transform @ homo)[:3]


def current_learning_rate(opt: torch.optim.Optimizer) -> float:
    return float(opt.param_groups[0].get("lr", 0.0))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_summary(path: Path, source_dir: Path, rows, diag, summaries, components, edges) -> None:
    multi = [c for c in components if len(c) > 1]
    lines = [
        "multiview joint optimization initialized from single-view optimized results",
        f"singleview_source={source_dir}",
        f"frames={len(rows)}",
        f"optimized_global_tracks={len(summaries)}",
        f"diagnostic_rows={len(diag)}",
        f"association_edges={len(edges)}",
        f"multi_view_global_tracks={len(multi)}",
        f"outputs={path.parent}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
