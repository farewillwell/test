#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/sft_config.sh"

sft_activate_env

mkdir -p "$(dirname "${PREPARE_LOG_FILE}")"
exec > >(tee -a "${PREPARE_LOG_FILE}") 2>&1

echo "========== prepare $(date '+%Y-%m-%d %H:%M:%S') =========="
echo "Log file: ${PREPARE_LOG_FILE}"
sft_print_common_summary
echo "Norm stats dir: $(sft_assets_stats_dir)"

cd "${OPENPI_ROOT}"

assets_stats_dir="$(sft_assets_stats_dir)"

if [[ ! -d "${HDF5_DIR}" ]]; then
  echo "Missing HDF5 dir: ${HDF5_DIR}" >&2
  exit 1
fi
mkdir -p "${HF_LEROBOT_HOME}"

convert_args=(
  "--input-dir=${HDF5_DIR}"
  "--repo-id=${REPO_ID}"
  "--fps=${FPS}"
  "--image-size=${IMAGE_SIZE}"
)
if [[ "${OVERWRITE_LEROBOT}" == "true" ]]; then
  convert_args+=("--overwrite")
fi
if [[ "${STRICT_DATA}" == "true" ]]; then
  convert_args+=("--strict")
fi
if [[ -n "${MAX_EPISODES}" ]]; then
  convert_args+=("--max-episodes=${MAX_EPISODES}")
fi

echo "Converting HDF5 demos to LeRobot..."
python "${PI0_ROOT}/train/convert_libero_hdf5_to_lerobot_openpi.py" "${convert_args[@]}"

if [[ "${COMPUTE_NORM_STATS}" == "true" ]]; then
  norm_args=(
    "--repo-id=${REPO_ID}"
    "--config-name=${CONFIG_NAME}"
    "--assets-base-dir=${ASSETS_BASE_DIR}"
    "--checkpoint-base-dir=${CHECKPOINT_BASE_DIR}"
    "--batch-size=${BATCH_SIZE}"
    "--num-workers=${NUM_WORKERS}"
  )
  if [[ -n "${ASSET_ID}" ]]; then
    norm_args+=("--asset-id=${ASSET_ID}")
  fi
  if [[ -n "${NORM_MAX_FRAMES}" ]]; then
    norm_args+=("--max-frames=${NORM_MAX_FRAMES}")
  fi

  echo "Computing normalization stats..."
  python "${PI0_ROOT}/train/compute_norm_stats_custom.py" "${norm_args[@]}"
else
  echo "Skip norm stats because COMPUTE_NORM_STATS=false."
fi

echo "Data preparation complete."
