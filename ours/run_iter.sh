#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI0_ROOT="${PI0_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
OPENPI_ROOT="${OPENPI_ROOT:-${PI0_ROOT}/openpi}"

# Keep the lowercase name too, so this can be used like the old iter-learn entry.
TASK_ID="${TASK_ID:-${task_id:-6}}"
export task_id="${TASK_ID}"

TASK_SUITE_NAME="${TASK_SUITE_NAME:-libero_goal}"
RUN_NAME="${RUN_NAME:-goal-${TASK_ID}-qselect}"
WORKSPACE="${WORKSPACE:-/data/aoss/heliqun/pi0-ours/${RUN_NAME}}"

# Required real inputs. Defaults are the local pi0 SFT outputs used during development.
SEED_DATA_DIR="${SEED_DATA_DIR:-${PI0_ROOT}/sft_runs/lerobot_data}"
INIT_MODEL_DIR="${INIT_MODEL_DIR:-${PI0_ROOT}/sft_runs/model/final}"

ITERS="${ITERS:-4}"
IQL_STEPS="${IQL_STEPS:-4000}"
TRAIN_STEPS="${TRAIN_STEPS:-3000}"
BATCH="${BATCH:-16}"
IQL_BATCH="${IQL_BATCH:-32}"
GPUS="${GPUS:-2}"
TRAJ_COUNT_PER_TASK="${TRAJ_COUNT_PER_TASK:-50}"
NUM_ACTION_SAMPLES="${NUM_ACTION_SAMPLES:-16}"
SAMPLE_MODE="${SAMPLE_MODE:-qselect}"

CONFIG_NAME="${CONFIG_NAME:-pi0_libero_low_mem_finetune}"
VIEWS="${VIEWS:-image,wrist_image}"
HORIZON="${HORIZON:-10}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
NUM_Q="${NUM_Q:-2}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
SEED="${SEED:-7}"
NORM_MAX_FRAMES="${NORM_MAX_FRAMES:-0}"
DRY_RUN="${DRY_RUN:-false}"
LOG_FILE="${LOG_FILE:-${WORKSPACE}/iter-${TASK_ID}.log}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/data/huangdi/heliqun/openvla-oft/openvla-oft/LIBERO/.libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PI0_CACHE_ROOT="${PI0_CACHE_ROOT:-${PI0_ROOT}/cache}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${PI0_CACHE_ROOT}/jax}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${PI0_CACHE_ROOT}/cuda}"
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-2147483648}"
export JAX_ENABLE_COMPILATION_CACHE="${JAX_ENABLE_COMPILATION_CACHE:-true}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ ! -x "${OPENPI_ROOT}/pi_env/bin/python" ]]; then
  echo "Missing pi_env python: ${OPENPI_ROOT}/pi_env/bin/python" >&2
  exit 1
fi
if [[ ! -d "${SEED_DATA_DIR}/data" || ! -d "${SEED_DATA_DIR}/meta" ]]; then
  echo "Invalid SEED_DATA_DIR, expected LeRobot data/meta: ${SEED_DATA_DIR}" >&2
  exit 1
fi
if [[ ! -d "${INIT_MODEL_DIR}/params" ]]; then
  echo "Invalid INIT_MODEL_DIR, expected params/: ${INIT_MODEL_DIR}" >&2
  exit 1
fi

mkdir -p "$(dirname "${LOG_FILE}")" "${JAX_COMPILATION_CACHE_DIR}" "${CUDA_CACHE_PATH}"
cd "${PI0_ROOT}"

cmd=(
  "${OPENPI_ROOT}/pi_env/bin/python" -u "${PI0_ROOT}/ours/iter.py"
  --workspace "${WORKSPACE}"
  --seed-data-dir "${SEED_DATA_DIR}"
  --init-model-dir "${INIT_MODEL_DIR}"
  --iters "${ITERS}"
  --iql-steps "${IQL_STEPS}"
  --train-steps "${TRAIN_STEPS}"
  --config-name "${CONFIG_NAME}"
  --batch-size "${BATCH}"
  --iql-batch-size "${IQL_BATCH}"
  --fsdp-devices "${GPUS}"
  --views "${VIEWS}"
  --horizon "${HORIZON}"
  --hidden-dim "${HIDDEN_DIM}"
  --num-q "${NUM_Q}"
  --num-action-samples "${NUM_ACTION_SAMPLES}"
  --sample-mode "${SAMPLE_MODE}"
  --task-suite-name "${TASK_SUITE_NAME}"
  --task-id "${TASK_ID}"
  --num-trials-per-task "${TRAJ_COUNT_PER_TASK}"
  --replan-steps "${REPLAN_STEPS}"
  --port "${PORT}"
  --host "${HOST}"
  --seed "${SEED}"
  --norm-max-frames "${NORM_MAX_FRAMES}"
)

if [[ "${DRY_RUN}" == "true" ]]; then
  cmd+=(--dry-run)
fi

{
  echo "========== pi0 ours iter $(date '+%Y-%m-%d %H:%M:%S') =========="
  echo "PI0_ROOT=${PI0_ROOT}"
  echo "WORKSPACE=${WORKSPACE}"
  echo "SEED_DATA_DIR=${SEED_DATA_DIR}"
  echo "INIT_MODEL_DIR=${INIT_MODEL_DIR}"
  echo "TASK_SUITE_NAME=${TASK_SUITE_NAME}"
  echo "TASK_ID=${TASK_ID}"
  echo "SAMPLE_MODE=${SAMPLE_MODE}"
  echo "NUM_ACTION_SAMPLES=${NUM_ACTION_SAMPLES}"
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
} > >(tee -a "${LOG_FILE}") 2>&1
