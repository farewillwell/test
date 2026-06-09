#!/usr/bin/env bash
# Minimal shared config for data preparation and SFT training.

SFT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI0_ROOT="${PI0_ROOT:-$(cd "${SFT_SCRIPT_DIR}/.." && pwd)}"
OPENPI_ROOT="${OPENPI_ROOT:-${PI0_ROOT}/openpi}"

# 1) Data preparation: read HDF5 demos from HDF5_DIR and write LeRobot data to
#    LEROBOT_DATA_DIR.
HDF5_DIR="${HDF5_DIR:-/data/aoss/heliqun/dataset/libero-dataset/bytaskid/6}"
LEROBOT_DATA_DIR="${LEROBOT_DATA_DIR:-${PI0_ROOT}/sft_runs/lerobot_data}"
LEROBOT_DATA_DIR="${LEROBOT_DATA_DIR%/}"

# OpenPI/LeRobot internally addresses a local dataset as
# "${HF_LEROBOT_HOME}/${REPO_ID}". Derive those two values from the single
# user-facing LEROBOT_DATA_DIR.
HF_LEROBOT_HOME="$(dirname "${LEROBOT_DATA_DIR}")"
REPO_ID="$(basename "${LEROBOT_DATA_DIR}")"

# Norm stats are tied to the prepared LeRobot data. Keeping them inside the
# data directory makes training from an arbitrary LEROBOT_DATA_DIR unambiguous.
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-${LEROBOT_DATA_DIR}/openpi_assets}"
ASSET_ID="${ASSET_ID:-physical-intelligence/libero}"

# 2) Training: read LEROBOT_DATA_DIR and save checkpoints under MODEL_DIR.
MODEL_DIR="${MODEL_DIR:-${PI0_ROOT}/sft_runs/model}"
STEP_CHECKPOINT_DIR="${MODEL_DIR}/steps"
FINAL_CHECKPOINT_DIR="${MODEL_DIR}/final"

CONFIG_NAME="${CONFIG_NAME:-pi0_libero_low_mem_finetune}"
EXP_NAME="${EXP_NAME:-sft}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/data/aoss/heliqun/model/pi}"

NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
SEED="${SEED:-42}"
FSDP_DEVICES="${FSDP_DEVICES:-2}"
PROJECT_NAME="${PROJECT_NAME:-openpi}"

FPS="${FPS:-30}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
MAX_EPISODES="${MAX_EPISODES:-}"
STRICT_DATA="${STRICT_DATA:-false}"
OVERWRITE_LEROBOT="${OVERWRITE_LEROBOT:-true}"
COMPUTE_NORM_STATS="${COMPUTE_NORM_STATS:-true}"
NORM_MAX_FRAMES="${NORM_MAX_FRAMES:-}"

XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
OVERWRITE="${OVERWRITE:-false}"
RESUME="${RESUME:-false}"
DRY_RUN="${DRY_RUN:-false}"
CHECK_INPUTS="${CHECK_INPUTS:-true}"

PI0_CACHE_ROOT="${PI0_CACHE_ROOT:-${PI0_ROOT}/cache}"
JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${PI0_CACHE_ROOT}/jax}"
CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${PI0_CACHE_ROOT}/cuda}"
CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-2147483648}"
JAX_ENABLE_COMPILATION_CACHE="${JAX_ENABLE_COMPILATION_CACHE:-true}"
JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"

LOG_DIR="${LOG_DIR:-${PI0_ROOT}/sft_runs/logs}"
PREPARE_LOG_FILE="${PREPARE_LOG_FILE:-${LOG_DIR}/prepare.log}"
TRAIN_LOG_FILE="${TRAIN_LOG_FILE:-${LOG_DIR}/train.log}"

export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${PI0_ROOT}/venv}"
export OPENPI_DATA_HOME
export HF_LEROBOT_HOME
export XLA_PYTHON_CLIENT_MEM_FRACTION
export JAX_COMPILATION_CACHE_DIR
export JAX_ENABLE_COMPILATION_CACHE
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS
export CUDA_CACHE_PATH
export CUDA_CACHE_MAXSIZE
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

sft_ensure_cache_dirs() {
  mkdir -p "${JAX_COMPILATION_CACHE_DIR}" "${CUDA_CACHE_PATH}"
}

sft_activate_env() {
  if [[ ! -f "${OPENPI_ROOT}/pi_env/bin/activate" ]]; then
    echo "Missing virtualenv: ${OPENPI_ROOT}/pi_env/bin/activate" >&2
    return 1
  fi
  source "${OPENPI_ROOT}/pi_env/bin/activate"
}

sft_asset_key() {
  printf '%s\n' "${ASSET_ID}"
}

sft_assets_stats_dir() {
  printf '%s/%s/%s\n' "${ASSETS_BASE_DIR}" "${CONFIG_NAME}" "$(sft_asset_key)"
}

sft_print_common_summary() {
  echo "PI0 root: ${PI0_ROOT}"
  echo "OpenPI root: ${OPENPI_ROOT}"
  echo "Config: ${CONFIG_NAME}"
  echo "HDF5 dir: ${HDF5_DIR}"
  echo "LeRobot data dir: ${LEROBOT_DATA_DIR}"
  echo "HF_LEROBOT_HOME: ${HF_LEROBOT_HOME}"
  echo "Repo id: ${REPO_ID}"
  echo "Assets base dir: ${ASSETS_BASE_DIR}"
  echo "Norm stats dir: $(sft_assets_stats_dir)"
  echo "Model dir: ${MODEL_DIR}"
  echo "Step checkpoint dir: ${STEP_CHECKPOINT_DIR}"
  echo "Final checkpoint dir: ${FINAL_CHECKPOINT_DIR}"
  echo "OPENPI_DATA_HOME: ${OPENPI_DATA_HOME}"
}
