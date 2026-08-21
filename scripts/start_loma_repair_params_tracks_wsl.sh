#!/usr/bin/env bash
set -Eeuo pipefail

# 当前主线第 2 步：用 LoMa/外观/极线几何做跨视角 2D track 关联。
# 输出 global_track_assignments.csv，供第 3 步跨视角 3D 联合优化固定使用。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_NAME="${MONO_DETECT_ENV:-mono-detect-original-2dbox-full-gpu}"
CONDA_SH="${CONDA_SH:-/opt/miniforge3/etc/profile.d/conda.sh}"
CONFIG_PATH="configs/loma_global_2d_repair_params_tracks_v1.yaml"
OUT_DIR="${REPO_ROOT}/outputs/loma_global_2d_repair_params_tracks_v1"
LOG_DIR="${OUT_DIR}/logs"
RUN_LOG="${LOG_DIR}/run.log"
LAUNCHER_LOG="${LOG_DIR}/launcher.log"
PID_FILE="${LOG_DIR}/run.pid"

mkdir -p "${LOG_DIR}"

if [[ -f "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}" || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "ALREADY_RUNNING pid=${old_pid} log=${RUN_LOG}"
    exit 0
  fi
fi

cat >> "${LAUNCHER_LOG}" <<EOF
[launcher] request_start $(date '+%F %T')
[launcher] repo=${REPO_ROOT}
[launcher] env=${ENV_NAME}
[launcher] config=${CONFIG_PATH}
[launcher] output=${OUT_DIR}
[launcher] log=${RUN_LOG}
EOF

(
  source "${CONDA_SH}"
  conda activate "${ENV_NAME}"
  cd "${REPO_ROOT}"

  export PYTHONPATH="${REPO_ROOT}/code:${REPO_ROOT}/external/LoMa/src:${PYTHONPATH:-}"
  if [[ -n "${CONDA_PREFIX:-}" && -d "${CONDA_PREFIX}/lib" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  fi

  echo "[loma] started $(date '+%F %T')"
  echo "[loma] config=${CONFIG_PATH}"
  echo "[loma] output=${OUT_DIR}"
  echo "[loma] python=$(which python)"
  python - <<'PY'
import torch
print(f"[loma] torch={torch.__version__} cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[loma] gpu={torch.cuda.get_device_name(0)}")
PY

  python -m rebuild_3d_box_optimizer.associate_2d_tracks_loma --config "${CONFIG_PATH}"
  status="$?"
  echo "[loma] finished $(date '+%F %T') status=${status}"
  exit "${status}"
) >> "${RUN_LOG}" 2>&1 &

pid="$!"
echo "${pid}" > "${PID_FILE}"
echo "[launcher] started $(date '+%F %T') pid=${pid}" >> "${LAUNCHER_LOG}"
echo "STARTED pid=${pid} log=${RUN_LOG}"
