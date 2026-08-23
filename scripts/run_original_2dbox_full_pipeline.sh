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

log "write generated configs"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/original_2dbox_full_pipeline_config.py" \
  --config "${CONFIG_PATH}" \
  --scene "${SCENE_DIR}" \
  --repo "${REPO_ROOT}" \
  --write-generated

PIPELINE_JSON="$(cfg paths.generated_configs)/pipeline_resolved.json"
RETRACK_CONFIG="$(cfg paths.generated_configs)/retrack_sort_2d.yaml"
TRACK_RENDER_CONFIG="$(cfg paths.generated_configs)/render_sort2d_tracks.yaml"
ENSURE_MASKS_CONFIG="$(cfg paths.generated_configs)/ensure_masks.yaml"
REBUILD_CONFIG="$(cfg paths.generated_configs)/rebuild_3d.yaml"

log "export existing 2D boxes from original annotations"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/export_original_2dbox_csv.py" \
  --annotations "$(cfg paths.annotation_dir)" \
  --camera-root "$(cfg paths.camera_root)" \
  --cameras "$(cfg data.cameras)" \
  --output-root "$(cfg paths.tracking_input)" \
  --max-frames "$(cfg runtime.max_frames)"

if [[ "$(cfg sam.enabled)" == "true" ]]; then
  [[ -f "$(cfg sam.checkpoint)" ]] || fail "Missing SAM checkpoint: $(cfg sam.checkpoint)"
  log "run SAM masks from existing 2D boxes"
  "${PYTHON_BIN}" "${REPO_ROOT}/code/examples/sam_segment_gt2d_boxes.py" \
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
    log "DA3 camera=${camera}"
    "${PYTHON_BIN}" "${REPO_ROOT}/code/examples/da3_metric_rear_depth_export.py" \
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

log "run 2D SORT tracking on existing 2D boxes"
"${PYTHON_BIN}" -m rebuild_3d_box_optimizer.retrack_sort_2d --config "${RETRACK_CONFIG}"

if [[ "$(cfg tracking.render_video)" == "true" ]]; then
  log "render 2D tracking videos"
  "${PYTHON_BIN}" -m rebuild_3d_box_optimizer.render_sort2d_tracks \
    --config "${TRACK_RENDER_CONFIG}" \
    --output-dir "$(cfg paths.tracking)/sort2d_track_videos" \
    --fps "$(cfg runtime.fps)" \
    --thickness "$(cfg tracking.thickness)"
fi

if [[ "$(cfg depth.enabled)" == "true" ]]; then
  log "attach DA3 depth to tracked 2D boxes"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/attach_da3_depth_to_tracks.py" \
    --track-root "$(cfg paths.tracking)/sort2d_tracks" \
    --depth-root "$(cfg paths.depth_output)" \
    --calib-root "$(cfg paths.calib_root)" \
    --cameras "$(cfg data.cameras)" \
    --output-root "$(cfg paths.depth_tracks)"
else
  log "depth disabled; copy tracked boxes as depth-track input"
  mkdir -p "$(cfg paths.depth_tracks)"
  cp -a "$(cfg paths.tracking)/sort2d_tracks/." "$(cfg paths.depth_tracks)/"
fi

if [[ "$(cfg masks.ensure_for_every_track_box)" == "true" ]]; then
  log "ensure one cropped mask for every tracked 2D box"
  "${PYTHON_BIN}" -m rebuild_3d_box_optimizer.ensure_masks_for_all_tracks \
    --config "${ENSURE_MASKS_CONFIG}" \
    --output-dir "$(cfg paths.ensured_masks)" \
    --min-iou "$(cfg masks.min_iou)"
fi

if [[ "$(cfg optimization_3d.enabled)" == "true" ]]; then
  log "run 3D box optimization"
  "${PYTHON_BIN}" -m rebuild_3d_box_optimizer.run --config "${REBUILD_CONFIG}"
fi

log "DONE"
log "resolved config: ${PIPELINE_JSON}"
log "outputs: ${OUT_ROOT}"
log "pipeline log: ${LOG_FILE}"
