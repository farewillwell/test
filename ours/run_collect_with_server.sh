#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI0_ROOT="${PI0_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OPENPI_ROOT="${OPENPI_ROOT:-${PI0_ROOT}/openpi}"

MODEL_PATH="${MODEL_PATH:-${1:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${2:-}}"
CRITIC_PATH="${CRITIC_PATH:-}"
CONFIG_NAME="${CONFIG_NAME:-pi0_libero_low_mem_finetune}"
HOST="${HOST:-127.0.0.1}"
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-}"
NUM_ACTION_SAMPLES="${NUM_ACTION_SAMPLES:-8}"
SAMPLE_MODE="${SAMPLE_MODE:-qselect}"
Q_VIEWS="${Q_VIEWS:-image,wrist_image}"
TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_goal}"
TASK_ID="${TASK_ID:-6}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
VIDEO_OUT_PATH="${VIDEO_OUT_PATH:-}"
SEED="${SEED:-7}"
OVERWRITE="${OVERWRITE:-true}"
SAVE_CANDIDATE_TRACES="${SAVE_CANDIDATE_TRACES:-false}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/data/huangdi/heliqun/openvla-oft/openvla-oft/LIBERO/.libero}"
MUJOCO_GL="${MUJOCO_GL:-egl}"

if [[ -z "${MODEL_PATH}" || -z "${OUTPUT_DIR}" ]]; then
  echo "Usage: MODEL_PATH=/path/to/final OUTPUT_DIR=/path/to/lerobot bash ours/run_collect_with_server.sh" >&2
  exit 1
fi
if [[ "${SAMPLE_MODE}" == "qselect" || "${SAMPLE_MODE}" == "best" ]]; then
  if [[ -z "${CRITIC_PATH}" ]]; then
    echo "CRITIC_PATH is required when SAMPLE_MODE=${SAMPLE_MODE}." >&2
    exit 1
  fi
fi

SERVER_PY="${OPENPI_ROOT}/pi_env/bin/python"
COLLECT_PY="${OPENPI_ROOT}/examples/libero/libero_env/bin/python"
if [[ ! -x "${SERVER_PY}" ]]; then
  echo "Missing pi_env python: ${SERVER_PY}" >&2
  exit 1
fi
if [[ ! -x "${COLLECT_PY}" ]]; then
  echo "Missing libero_env python: ${COLLECT_PY}" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

export PI0_CACHE_ROOT="${PI0_CACHE_ROOT:-${PI0_ROOT}/cache}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${PI0_CACHE_ROOT}/jax}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${PI0_CACHE_ROOT}/cuda}"
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-2147483648}"
export JAX_ENABLE_COMPILATION_CACHE="${JAX_ENABLE_COMPILATION_CACHE:-true}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"
mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${CUDA_CACHE_PATH}" "$(dirname "${OUTPUT_DIR}")"

export PYTHONPATH="${OPENPI_ROOT}/packages/openpi-client/src:${OPENPI_ROOT}/third_party/libero:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH
export MUJOCO_GL

if ! "${COLLECT_PY}" - <<'PY'
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import pandas
import pyarrow
PY
then
  echo "libero_env cannot write LeRobot yet. Install writer deps, for example:" >&2
  echo "uv pip install --python ${COLLECT_PY} lerobot pandas pyarrow datasets huggingface-hub filelock" >&2
  exit 1
fi

(
  cd "${OPENPI_ROOT}"
  exec "${SERVER_PY}" -u "${PI0_ROOT}/ours/qselect_policy_server.py" \
    --config "${CONFIG_NAME}" \
    --checkpoint-dir "${MODEL_PATH}" \
    --critic-path "${CRITIC_PATH}" \
    --default-prompt "${DEFAULT_PROMPT}" \
    --host "${SERVER_HOST}" \
    --port "${PORT}" \
    --num-action-samples "${NUM_ACTION_SAMPLES}" \
    --sample-mode "${SAMPLE_MODE}" \
    --q-views "${Q_VIEWS}" \
    --seed "${SEED}"
) &
SERVER_PID="$!"

SERVER_READY=false
for _ in $(seq 1 120); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "qselect policy server exited before becoming ready." >&2
    exit 1
  fi
  if "${SERVER_PY}" - <<PY >/dev/null 2>&1
import urllib.request
urllib.request.urlopen("http://${HOST}:${PORT}/healthz", timeout=1).read()
PY
  then
    SERVER_READY=true
    break
  fi
  sleep 2
done
if [[ "${SERVER_READY}" != "true" ]]; then
  echo "Timed out waiting for qselect policy server at ${HOST}:${PORT}." >&2
  exit 1
fi

COLLECT_CMD=(
  "${COLLECT_PY}" -u "${PI0_ROOT}/ours/collect_libero_lerobot.py"
  --output-dir "${OUTPUT_DIR}"
  --host "${HOST}"
  --port "${PORT}"
  --resize-size "${RESIZE_SIZE}"
  --replan-steps "${REPLAN_STEPS}"
  --task-suite-name "${TASK_SUITE_NAME}"
  --task-id "${TASK_ID}"
  --num-trials-per-task "${NUM_TRIALS_PER_TASK}"
  --seed "${SEED}"
)
if [[ "${OVERWRITE}" == "true" ]]; then
  COLLECT_CMD+=(--overwrite)
fi
if [[ -n "${VIDEO_OUT_PATH}" ]]; then
  COLLECT_CMD+=(--video-out-path "${VIDEO_OUT_PATH}")
fi
if [[ "${SAVE_CANDIDATE_TRACES}" == "true" ]]; then
  COLLECT_CMD+=(--save-candidate-traces)
fi

exec "${COLLECT_CMD[@]}"
