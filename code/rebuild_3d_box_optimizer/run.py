from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_config, resolve_path, write_experiment_config_snapshot, write_resolved_config
from .data import Observation
from .data import group_by_track, load_view_observations


@dataclass
class TrackJob:
    view: str
    track_id: int
    items: list[Observation]
    index: int
    total: int


@dataclass
class TrackJobResult:
    job: TrackJob
    ok: bool
    rows: list[dict[str, Any]]
    diagnostics: list[dict[str, Any]]
    summary: dict[str, Any]
    final_rows: list[dict[str, Any]]
    final_diagnostics: list[dict[str, Any]]
    final_summary: dict[str, Any]
    elapsed_sec: float
    error: str = ""


def run_track_job(config: dict[str, Any], job: TrackJob) -> TrackJobResult:
    start = time.time()
    try:
        from .optimizer import optimize_track

        result = optimize_track(config, job.items)
        return TrackJobResult(
            job=job,
            ok=True,
            rows=result.rows,
            diagnostics=result.diagnostics,
            summary=result.summary,
            final_rows=result.final_rows,
            final_diagnostics=result.final_diagnostics,
            final_summary=result.final_summary,
            elapsed_sec=time.time() - start,
        )
    except Exception:
        return TrackJobResult(
            job=job,
            ok=False,
            rows=[],
            diagnostics=[],
            summary={},
            final_rows=[],
            final_diagnostics=[],
            final_summary={},
            elapsed_sec=time.time() - start,
            error=traceback.format_exc(),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run track-level world-coordinate 3D box optimization.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value using dot paths, e.g. --set solver.learning_rate=0.01",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    for override in args.set:
        apply_config_override(config, override)
    out_dir = resolve_path(config, config["output"].get("dir", "outputs/rebuild_world_track_joint"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if config["output"].get("resolved_config", True):
        write_resolved_config(config, out_dir / "resolved_config.yaml")
        write_experiment_config_snapshot(config, out_dir / "experiment_config_snapshot.yaml")
    if config["output"].get("copy_source_config", True):
        source = Path(args.config)
        if source.exists():
            shutil.copy2(source, out_dir / "source_config.yaml")
    progress_path = out_dir / "progress.log"
    progress_path.write_text("", encoding="utf-8")
    append_progress(progress_path, f"START config={args.config} output={out_dir}")
    incremental_csv_paths = get_global_csv_paths(out_dir)
    reset_incremental_csvs(incremental_csv_paths)

    all_rows: list[dict[str, object]] = []
    all_diag: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    all_final_rows: list[dict[str, object]] = []
    all_final_diag: list[dict[str, object]] = []
    final_summaries: list[dict[str, object]] = []
    min_frames = int(config["solver"].get("min_track_frames", 3))
    max_tracks = int(config["solver"].get("max_tracks_per_view", 0) or 0)
    parallel_cfg = config["solver"].get("track_parallel", {})
    parallel_enabled = bool(parallel_cfg.get("enabled", False))
    workers = int(parallel_cfg.get("workers", 1) or 1)
    run_start = time.time()
    for view in config["scope"]["cameras"]:
        observations = load_view_observations(config, view)
        grouped = group_by_track(observations)
        eligible = [(track_id, items) for track_id, items in grouped.items() if len(items) >= min_frames]
        requested_track_ids = requested_tracks_for_view(config, view)
        if requested_track_ids is not None:
            eligible = [(track_id, items) for track_id, items in eligible if track_id in requested_track_ids]
        if max_tracks > 0:
            eligible = eligible[:max_tracks]
        append_progress(progress_path, f"VIEW_START view={view} observations={len(observations)} eligible_tracks={len(eligible)}")
        jobs = [TrackJob(view=view, track_id=int(track_id), items=items, index=i + 1, total=len(eligible)) for i, (track_id, items) in enumerate(eligible)]
        processed = run_track_jobs(config, jobs, progress_path, out_dir, all_rows, all_diag, summaries, all_final_rows, all_final_diag, final_summaries, parallel_enabled, workers)
        append_progress(progress_path, f"VIEW_DONE view={view} processed_tracks={processed}")

    write_csv(out_dir / "frame_3d_boxes_world_track_joint.csv", all_rows)
    diagnostics_path = out_dir / "frame_loss_diagnostics.csv"
    write_csv(diagnostics_path, all_diag)
    write_csv(out_dir / "track_summary.csv", summaries)
    final_diagnostics_path = out_dir / "frame_loss_diagnostics_final_iter.csv"
    write_csv(out_dir / "frame_3d_boxes_world_track_joint_final_iter.csv", all_final_rows)
    write_csv(final_diagnostics_path, all_final_diag)
    write_csv(out_dir / "track_summary_final_iter.csv", final_summaries)
    write_summary(out_dir / "summary.txt", all_rows, all_diag, summaries)
    append_progress(progress_path, f"CSV_DONE frames={len(all_rows)} tracks={len(summaries)} diagnostics={len(all_diag)} final_frames={len(all_final_rows)} final_diagnostics={len(all_final_diag)}")
    if config.get("output", {}).get("videos", True):
        from .visualization import render_experiment_videos

        append_progress(progress_path, "VIDEO_START solution=best")
        render_experiment_videos(config, diagnostics_path, out_dir)
        append_progress(progress_path, "VIDEO_DONE solution=best")
        append_progress(progress_path, "VIDEO_START solution=final_iter")
        render_experiment_videos(config, final_diagnostics_path, out_dir / "final_iter")
        append_progress(progress_path, "VIDEO_DONE solution=final_iter")
    append_progress(progress_path, f"DONE total_elapsed_sec={time.time() - run_start:.1f}")
    return 0


def run_track_jobs(
    config: dict[str, Any],
    jobs: list[TrackJob],
    progress_path: Path,
    out_dir: Path,
    all_rows: list[dict[str, object]],
    all_diag: list[dict[str, object]],
    summaries: list[dict[str, object]],
    all_final_rows: list[dict[str, object]],
    all_final_diag: list[dict[str, object]],
    final_summaries: list[dict[str, object]],
    parallel_enabled: bool,
    workers: int,
) -> int:
    if not jobs:
        return 0
    if not parallel_enabled or workers <= 1 or len(jobs) <= 1:
        processed = 0
        for job in jobs:
            append_progress(progress_path, f"TRACK_START view={job.view} track={job.track_id} frames={len(job.items)} index={job.index}/{job.total}")
            result = run_track_job(config, job)
            handle_track_result(result, config, progress_path, out_dir, all_rows, all_diag, summaries, all_final_rows, all_final_diag, final_summaries)
            processed += 1
        return processed

    actual_workers = min(max(workers, 1), len(jobs))
    append_progress(progress_path, f"TRACK_PARALLEL_START workers={actual_workers} jobs={len(jobs)}")
    for job in jobs:
        append_progress(progress_path, f"TRACK_QUEUED view={job.view} track={job.track_id} frames={len(job.items)} index={job.index}/{job.total}")
    processed = 0
    results: list[TrackJobResult] = []
    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        future_to_job = {executor.submit(run_track_job, config, job): job for job in jobs}
        for future in as_completed(future_to_job):
            result = future.result()
            results.append(result)
            processed += 1
            handle_track_result(result, config, progress_path, out_dir, all_rows, all_diag, summaries, all_final_rows, all_final_diag, final_summaries, completed_index=processed, total=len(jobs))
    results.sort(key=lambda item: item.job.index)
    return processed


def handle_track_result(
    result: TrackJobResult,
    config: dict[str, Any],
    progress_path: Path,
    out_dir: Path,
    all_rows: list[dict[str, object]],
    all_diag: list[dict[str, object]],
    summaries: list[dict[str, object]],
    all_final_rows: list[dict[str, object]],
    all_final_diag: list[dict[str, object]],
    final_summaries: list[dict[str, object]],
    completed_index: int | None = None,
    total: int | None = None,
) -> None:
    job = result.job
    progress = f"completed={completed_index}/{total} " if completed_index is not None and total is not None else ""
    if not result.ok:
        append_progress(
            progress_path,
            "TRACK_FAILED "
            f"{progress}view={job.view} track={job.track_id} frames={len(job.items)} index={job.index}/{job.total} "
            f"elapsed_sec={result.elapsed_sec:.1f}",
        )
        raise RuntimeError(f"Track optimization failed for {job.view}:{job.track_id}\n{result.error}")
    all_rows.extend(result.rows)
    all_diag.extend(result.diagnostics)
    summaries.append(result.summary)
    all_final_rows.extend(result.final_rows)
    all_final_diag.extend(result.final_diagnostics)
    final_summaries.append(result.final_summary)
    write_single_track_outputs(config, out_dir, job, result, progress_path)
    append_global_track_outputs(out_dir, result)
    summary = result.summary
    append_progress(
        progress_path,
        "TRACK_DONE "
        f"{progress}view={job.view} track={job.track_id} frames={len(job.items)} index={job.index}/{job.total} "
        f"iters={summary.get('iterations_used')} stop={summary.get('stop_reason')} "
        f"best_loss={float(summary.get('best_loss', float('nan'))):.6g} "
        f"mean_frame_loss={float(summary.get('mean_frame_loss', float('nan'))):.6g} "
        f"max_frame_loss={float(summary.get('max_frame_loss', float('nan'))):.6g} "
        f"dominant={summary.get('dominant_loss')} elapsed_sec={result.elapsed_sec:.1f}",
    )


def write_single_track_outputs(
    config: dict[str, Any],
    out_dir: Path,
    job: TrackJob,
    result: TrackJobResult,
    progress_path: Path,
) -> None:
    track_root = out_dir / "tracks" / f"{job.view}_{job.track_id}"
    best_dir = track_root / "best"
    final_dir = track_root / "final_iter"
    write_csv(best_dir / "frame_3d_boxes_world_track_joint.csv", result.rows)
    write_csv(best_dir / "frame_loss_diagnostics.csv", result.diagnostics)
    write_csv(best_dir / "track_summary.csv", [result.summary])
    write_csv(final_dir / "frame_3d_boxes_world_track_joint.csv", result.final_rows)
    write_csv(final_dir / "frame_loss_diagnostics.csv", result.final_diagnostics)
    write_csv(final_dir / "track_summary.csv", [result.final_summary])
    # Per-track videos are diagnostic-only and can dominate runtime on large runs.
    # Keep the final per-view experiment overlays controlled by output.videos.
    if config.get("output", {}).get("track_videos", False):
        try:
            from .visualization import render_track_video_from_rows

            append_progress(progress_path, f"TRACK_VIDEO_START view={job.view} track={job.track_id} solution=best")
            best_video = render_track_video_from_rows(config, result.diagnostics, best_dir / f"{job.view}_{job.track_id}_best_overlay.mp4")
            append_progress(progress_path, f"TRACK_VIDEO_DONE view={job.view} track={job.track_id} solution=best path={best_video}")
            append_progress(progress_path, f"TRACK_VIDEO_START view={job.view} track={job.track_id} solution=final_iter")
            final_video = render_track_video_from_rows(config, result.final_diagnostics, final_dir / f"{job.view}_{job.track_id}_final_iter_overlay.mp4")
            append_progress(progress_path, f"TRACK_VIDEO_DONE view={job.view} track={job.track_id} solution=final_iter path={final_video}")
        except Exception:
            append_progress(progress_path, f"TRACK_VIDEO_FAILED view={job.view} track={job.track_id}\n{traceback.format_exc()}")


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


def get_global_csv_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "rows": out_dir / "frame_3d_boxes_world_track_joint.csv",
        "diagnostics": out_dir / "frame_loss_diagnostics.csv",
        "summary": out_dir / "track_summary.csv",
        "final_rows": out_dir / "frame_3d_boxes_world_track_joint_final_iter.csv",
        "final_diagnostics": out_dir / "frame_loss_diagnostics_final_iter.csv",
        "final_summary": out_dir / "track_summary_final_iter.csv",
    }


def reset_incremental_csvs(paths: dict[str, Path]) -> None:
    for path in paths.values():
        if path.exists():
            path.unlink()


def append_global_track_outputs(out_dir: Path, result: TrackJobResult) -> None:
    paths = get_global_csv_paths(out_dir)
    append_csv(paths["rows"], result.rows)
    append_csv(paths["diagnostics"], result.diagnostics)
    append_csv(paths["summary"], [result.summary])
    append_csv(paths["final_rows"], result.final_rows)
    append_csv(paths["final_diagnostics"], result.final_diagnostics)
    append_csv(paths["final_summary"], [result.final_summary])


def append_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)


def requested_tracks_for_view(config: dict[str, object], view: str) -> set[int] | None:
    track_ids_cfg = config.get("scope", {}).get("track_ids", {})  # type: ignore[union-attr]
    if not isinstance(track_ids_cfg, dict) or view not in track_ids_cfg:
        return None
    raw = track_ids_cfg.get(view)
    if raw in (None, ""):
        return None
    if isinstance(raw, (int, float, str)):
        raw_items = [raw]
    else:
        raw_items = list(raw)
    return {int(float(item)) for item in raw_items}


def apply_config_override(config: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"--set override must use KEY=VALUE form, got: {expression!r}")
    key, raw_value = expression.split("=", 1)
    if not key:
        raise ValueError(f"--set override key is empty: {expression!r}")
    try:
        import yaml

        value = yaml.safe_load(raw_value)
    except Exception:
        value = raw_value
    cursor: dict[str, Any] = config
    parts = key.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def write_summary(path: Path, rows: list[dict[str, object]], diag: list[dict[str, object]], summaries: list[dict[str, object]]) -> None:
    text = [
        "rebuild_world_track_joint first runnable version",
        f"frames={len(rows)}",
        f"tracks={len(summaries)}",
        f"diagnostic_rows={len(diag)}",
        f"outputs={path.parent}",
    ]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def append_progress(path: Path, message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {message}"
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
    print(line, flush=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
