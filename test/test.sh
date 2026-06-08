#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI0_ROOT="${PI0_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OPENPI_ROOT="${OPENPI_ROOT:-${PI0_ROOT}/openpi}"

MODE="${MODE:-serve}"
MODEL_PATH="${MODEL_PATH:-${1:-}}"
CONFIG_NAME="${CONFIG_NAME:-pi0_libero_low_mem_finetune}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
DEFAULT_PROMPT="${DEFAULT_PROMPT:-}"
DRY_RUN="${DRY_RUN:-false}"

TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_goal}"
TASK_ID="${TASK_ID:-6}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
VIDEO_OUT_PATH="${VIDEO_OUT_PATH:-${PI0_ROOT}/sft_runs/eval_videos}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/data/huangdi/heliqun/openvla-oft/openvla-oft/LIBERO/.libero}"
MUJOCO_GL="${MUJOCO_GL:-egl}"

PI0_CACHE_ROOT="${PI0_CACHE_ROOT:-${PI0_ROOT}/cache}"
JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${PI0_CACHE_ROOT}/jax}"
CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${PI0_CACHE_ROOT}/cuda}"
CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-2147483648}"
JAX_ENABLE_COMPILATION_CACHE="${JAX_ENABLE_COMPILATION_CACHE:-true}"
JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"

export JAX_COMPILATION_CACHE_DIR
export JAX_ENABLE_COMPILATION_CACHE
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS
export CUDA_CACHE_PATH
export CUDA_CACHE_MAXSIZE
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${PI0_ROOT}/venv}"

print_summary() {
  echo "PI0 root: ${PI0_ROOT}"
  echo "OpenPI root: ${OPENPI_ROOT}"
  echo "Mode: ${MODE}"
  echo "Config: ${CONFIG_NAME}"
  echo "Model path: ${MODEL_PATH:-<required for serve>}"
  echo "Host: ${HOST}"
  echo "Port: ${PORT}"
}

run_serve() {
  if [[ -z "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH is required for MODE=serve." >&2
    exit 1
  fi
  if [[ ! -d "${MODEL_PATH}/params" || ! -d "${MODEL_PATH}/assets" ]]; then
    echo "MODEL_PATH must contain params/ and assets/: ${MODEL_PATH}" >&2
    exit 1
  fi
  if [[ ! -f "${OPENPI_ROOT}/pi_env/bin/activate" ]]; then
    echo "Missing OpenPI env: ${OPENPI_ROOT}/pi_env/bin/activate" >&2
    exit 1
  fi

  mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${CUDA_CACHE_PATH}"
  source "${OPENPI_ROOT}/pi_env/bin/activate"
  cd "${OPENPI_ROOT}"

  cmd=(
    python -u scripts/serve_policy.py
    --env LIBERO
    --port "${PORT}"
    policy:checkpoint
    --policy.config "${CONFIG_NAME}"
    --policy.dir "${MODEL_PATH}"
  )
  if [[ -n "${DEFAULT_PROMPT}" ]]; then
    cmd+=(--default-prompt "${DEFAULT_PROMPT}")
  fi

  echo "JAX compilation cache: ${JAX_COMPILATION_CACHE_DIR}"
  echo "CUDA cache path: ${CUDA_CACHE_PATH}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    printf 'Dry run serve command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return
  fi
  exec "${cmd[@]}"
}

run_eval() {
  if [[ ! -f "${OPENPI_ROOT}/examples/libero/libero_env/bin/activate" ]]; then
    echo "Missing LIBERO env: ${OPENPI_ROOT}/examples/libero/libero_env/bin/activate" >&2
    exit 1
  fi

  source "${OPENPI_ROOT}/examples/libero/libero_env/bin/activate"
  export PYTHONPATH="${PYTHONPATH:-}:${OPENPI_ROOT}/third_party/libero"
  export LIBERO_CONFIG_PATH
  export MUJOCO_GL

  cd "${OPENPI_ROOT}"
  cmd=(
    python -u examples/libero/main.py
    --host "${HOST}"
    --port "${PORT}"
    --task-suite-name "${TASK_SUITE_NAME}"
    --task-id "${TASK_ID}"
    --num-trials-per-task "${NUM_TRIALS_PER_TASK}"
    --video-out-path "${VIDEO_OUT_PATH}"
  )

  if [[ "${DRY_RUN}" == "true" ]]; then
    printf 'Dry run eval command:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return
  fi
  exec "${cmd[@]}"
}

print_summary
case "${MODE}" in
  serve)
    run_serve
    ;;
  eval)
    run_eval
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use MODE=serve or MODE=eval." >&2
    exit 1
    ;;
esac
