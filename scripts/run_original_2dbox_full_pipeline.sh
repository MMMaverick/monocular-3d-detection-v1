#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_original_2dbox_full_pipeline.sh /path/to/original/scene [config.yaml]

Input is the original dataset with existing 2D boxes in annotation JSONs.
This pipeline does NOT run object detection.

All parameters live in config yaml.
Default config:
  configs/original_2dbox_full_pipeline_default.yaml
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

if [[ $# -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  [[ $# -eq 0 ]] && exit 1 || exit 0
fi

if [[ $# -gt 2 ]]; then
  fail "Too many arguments. Put parameters in yaml config, not bash flags."
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCENE_DIR="$(cd "$1" && pwd)"
CONFIG_PATH="${2:-${REPO_ROOT}/configs/original_2dbox_full_pipeline_default.yaml}"
CONFIG_PATH="$(cd "$(dirname "${CONFIG_PATH}")" && pwd)/$(basename "${CONFIG_PATH}")"

[[ -f "${CONFIG_PATH}" ]] || fail "Missing config: ${CONFIG_PATH}"

PYTHON_BIN="${PYTHON:-python}"
export PYTHONPATH="${REPO_ROOT}/code:${REPO_ROOT}/code/examples:${REPO_ROOT}/third_party/segment_anything:${REPO_ROOT}/third_party/Depth-Anything-3/src:${REPO_ROOT}/third_party/Depth-Anything-3/.local_deps:${PYTHONPATH:-}"

CONFIG_JSON="$("${PYTHON_BIN}" "${REPO_ROOT}/scripts/original_2dbox_full_pipeline_config.py" --config "${CONFIG_PATH}" --scene "${SCENE_DIR}" --repo "${REPO_ROOT}")"

cfg() {
  CFG_KEY="$1" CONFIG_JSON="${CONFIG_JSON}" "${PYTHON_BIN}" - <<'PY'
import json, os
cur = json.loads(os.environ["CONFIG_JSON"])
for part in os.environ["CFG_KEY"].split("."):
    cur = cur[part]
if isinstance(cur, bool):
    print("true" if cur else "false")
else:
    print(cur)
PY
}

OUT_ROOT="$(cfg paths.output_root)"
mkdir -p "${OUT_ROOT}/logs"
LOG_FILE="${OUT_ROOT}/logs/pipeline.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
TIMING_FILE="${OUT_ROOT}/logs/stage_timings.csv"
PIPELINE_START_SECONDS="$(date +%s)"
printf 'stage,start_utc,end_utc,elapsed_seconds\n' > "${TIMING_FILE}"

run_stage() {
  local stage="$1"
  shift
  local start_seconds end_seconds start_utc end_utc
  start_seconds="$(date +%s)"
  start_utc="$(date -u '+%FT%TZ')"
  log "STAGE_START stage=${stage}"
  "$@"
  end_seconds="$(date +%s)"
  end_utc="$(date -u '+%FT%TZ')"
  printf '%s,%s,%s,%s\n' "${stage}" "${start_utc}" "${end_utc}" "$((end_seconds - start_seconds))" >> "${TIMING_FILE}"
  log "STAGE_DONE stage=${stage} elapsed_sec=$((end_seconds - start_seconds))"
}

log "repo=${REPO_ROOT}"
log "scene=${SCENE_DIR}"
log "config=${CONFIG_PATH}"
log "out=${OUT_ROOT}"
log "pipeline=no_detection_existing_2dbox"

cd "${REPO_ROOT}"

export YOLO_CONFIG_DIR="${OUT_ROOT}/ultralytics_config"
export MPLCONFIGDIR="${OUT_ROOT}/matplotlib_cache"
if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

run_stage "generate_configs" "${PYTHON_BIN}" "${REPO_ROOT}/scripts/original_2dbox_full_pipeline_config.py" \
  --config "${CONFIG_PATH}" \
  --scene "${SCENE_DIR}" \
  --repo "${REPO_ROOT}" \
  --write-generated

PIPELINE_JSON="$(cfg paths.generated_configs)/pipeline_resolved.json"
ROBUST_TRACK_CONFIG="$(cfg paths.generated_configs)/robust_botsort_2d.yaml"
RETRACK_CONFIG="$(cfg paths.generated_configs)/retrack_sort_2d.yaml"
TRACK_RENDER_CONFIG="$(cfg paths.generated_configs)/render_robust_botsort_tracks.yaml"
ENSURE_MASKS_CONFIG="$(cfg paths.generated_configs)/ensure_masks.yaml"
REBUILD_CONFIG="$(cfg paths.generated_configs)/rebuild_3d.yaml"

run_stage "export_2d_boxes" "${PYTHON_BIN}" "${REPO_ROOT}/scripts/export_original_2dbox_csv.py" \
  --annotations "$(cfg paths.annotation_dir)" \
  --camera-root "$(cfg paths.camera_root)" \
  --cameras "$(cfg data.cameras)" \
  --output-root "$(cfg paths.tracking_input)" \
  --max-frames "$(cfg runtime.max_frames)"

if [[ "$(cfg sam.enabled)" == "true" ]]; then
  [[ -f "$(cfg sam.checkpoint)" ]] || fail "Missing SAM checkpoint: $(cfg sam.checkpoint)"
  run_stage "sam_masks" "${PYTHON_BIN}" "${REPO_ROOT}/code/examples/sam_segment_gt2d_boxes.py" \
    --annotations "$(cfg paths.annotation_dir)" \
    --camera-root "$(cfg paths.camera_root)" \
    --cameras "$(cfg data.cameras)" \
    --output "$(cfg paths.sam_output)" \
    --checkpoint "$(cfg sam.checkpoint)" \
    --model-type "$(cfg sam.model_type)" \
    --box-scale "$(cfg sam.box_scale)" \
    --positive-points "$(cfg sam.positive_points)" \
    --max-frames "$(cfg runtime.max_frames)" \
    --device "$(cfg runtime.device)" \
    --fps "$(cfg runtime.fps)" \
    --mask-format "$(cfg sam.mask_format)" \
    --min-gt2d-score "$(cfg sam.min_2d_score)"
fi

if [[ "$(cfg depth.enabled)" == "true" ]]; then
  [[ -d "$(cfg depth.da3_root)" ]] || fail "Missing DA3 root: $(cfg depth.da3_root)"
  [[ -d "$(cfg depth.model_dir)" ]] || fail "Missing DA3 model dir: $(cfg depth.model_dir)"
  log "run DA3 metric depth per camera"
  IFS=',' read -r -a CAMERAS <<< "$(cfg data.cameras)"
  for camera in "${CAMERAS[@]}"; do
    camera="$(echo "$camera" | xargs)"
    [[ -n "${camera}" ]] || continue
    run_stage "da3_${camera}" "${PYTHON_BIN}" "${REPO_ROOT}/code/examples/da3_metric_rear_depth_export.py" \
      --da3-root "$(cfg depth.da3_root)" \
      --model-dir "$(cfg depth.model_dir)" \
      --input "$(cfg paths.camera_root)/${camera}" \
      --output "$(cfg paths.depth_output)/${camera}" \
      --intrinsic "$(cfg paths.calib_root)/${camera}/${camera}-intrinsic.json" \
      --max-frames "$(cfg runtime.max_frames)" \
      --stride "$(cfg depth.stride)" \
      --chunk-size "$(cfg depth.chunk_size)" \
      --process-res "$(cfg depth.process_res)" \
      --device "$(cfg runtime.device)"
  done
fi

TRACK_ROOT="$(cfg paths.robust_tracks)"
if [[ "$(cfg tracking.method)" == "robust_botsort" ]]; then
  run_stage "botsort_2d_tracking" "${PYTHON_BIN}" -m rebuild_3d_box_optimizer.robust_track_2d --config "${ROBUST_TRACK_CONFIG}"
else
  run_stage "sort_2d_tracking" "${PYTHON_BIN}" -m rebuild_3d_box_optimizer.retrack_sort_2d --config "${RETRACK_CONFIG}"
  TRACK_ROOT="$(cfg paths.sort_tracks)"
  TRACK_RENDER_CONFIG="$(cfg paths.generated_configs)/render_sort2d_tracks.yaml"
fi

if [[ "$(cfg tracking.render_video)" == "true" ]]; then
  run_stage "render_2d_tracking" "${PYTHON_BIN}" -m rebuild_3d_box_optimizer.render_sort2d_tracks \
    --config "${TRACK_RENDER_CONFIG}" \
    --output-dir "$(cfg paths.tracking)/sort2d_track_videos" \
    --fps "$(cfg runtime.fps)" \
    --thickness "$(cfg tracking.thickness)"
fi

if [[ "$(cfg depth.enabled)" == "true" ]]; then
  run_stage "attach_depth_to_tracks" "${PYTHON_BIN}" "${REPO_ROOT}/scripts/attach_da3_depth_to_tracks.py" \
    --track-root "${TRACK_ROOT}" \
    --depth-root "$(cfg paths.depth_output)" \
    --calib-root "$(cfg paths.calib_root)" \
    --cameras "$(cfg data.cameras)" \
    --image-root "$(cfg paths.scene)" \
    --output-root "$(cfg paths.depth_tracks)"
else
  log "depth disabled; copy tracked boxes as depth-track input"
  mkdir -p "$(cfg paths.depth_tracks)"
  cp -a "${TRACK_ROOT}/." "$(cfg paths.depth_tracks)/"
fi

if [[ "$(cfg masks.ensure_for_every_track_box)" == "true" ]]; then
  run_stage "ensure_track_masks" "${PYTHON_BIN}" -m rebuild_3d_box_optimizer.ensure_masks_for_all_tracks \
    --config "${ENSURE_MASKS_CONFIG}" \
    --output-dir "$(cfg paths.ensured_masks)" \
    --min-iou "$(cfg masks.min_iou)"
fi

if [[ "$(cfg optimization_3d.enabled)" == "true" ]]; then
  run_stage "optimize_3d_boxes" "${PYTHON_BIN}" -m rebuild_3d_box_optimizer.run --config "${REBUILD_CONFIG}"
fi

PIPELINE_END_SECONDS="$(date +%s)"
printf 'total,%s,%s,%s\n' "$(date -u -d "@${PIPELINE_START_SECONDS}" '+%FT%TZ')" "$(date -u '+%FT%TZ')" "$((PIPELINE_END_SECONDS - PIPELINE_START_SECONDS))" >> "${TIMING_FILE}"
log "DONE"
log "stage timings: ${TIMING_FILE}"
log "resolved config: ${PIPELINE_JSON}"
log "outputs: ${OUT_ROOT}"
log "pipeline log: ${LOG_FILE}"
