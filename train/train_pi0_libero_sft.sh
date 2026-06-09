SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/sft_config.sh"

sft_ensure_cache_dirs
sft_activate_env

mkdir -p "$(dirname "${TRAIN_LOG_FILE}")"
exec > >(tee -a "${TRAIN_LOG_FILE}") 2>&1

echo "========== train $(date '+%Y-%m-%d %H:%M:%S') =========="
echo "Log file: ${TRAIN_LOG_FILE}"
sft_print_common_summary
echo "Steps: ${NUM_TRAIN_STEPS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Num workers: ${NUM_WORKERS}"
echo "Model dir: ${MODEL_DIR}"
echo "Step checkpoint dir: ${STEP_CHECKPOINT_DIR}"
echo "Final checkpoint dir: ${FINAL_CHECKPOINT_DIR}"
echo "JAX compilation cache: ${JAX_COMPILATION_CACHE_DIR}"
echo "CUDA cache path: ${CUDA_CACHE_PATH}"

cd "${OPENPI_ROOT}"

assets_stats_dir="$(sft_assets_stats_dir)"

if [[ "${CHECK_INPUTS}" == "true" ]]; then
  if [[ ! -d "${LEROBOT_DATA_DIR}" ]]; then
    echo "Missing LeRobot dataset: ${LEROBOT_DATA_DIR}" >&2
    echo "Run: bash ${PI0_ROOT}/train/prepare_pi0_libero_data.sh" >&2
    exit 1
  fi
  if [[ ! -d "${LEROBOT_DATA_DIR}/data" || ! -d "${LEROBOT_DATA_DIR}/meta" ]]; then
    echo "Invalid LeRobot dataset: ${LEROBOT_DATA_DIR}" >&2
    echo "Expected ${LEROBOT_DATA_DIR}/data and ${LEROBOT_DATA_DIR}/meta" >&2
    exit 1
  fi

  if [[ ! -d "${assets_stats_dir}" ]]; then
    echo "Missing normalization stats: ${assets_stats_dir}" >&2
    echo "Run: COMPUTE_NORM_STATS=true bash ${PI0_ROOT}/train/prepare_pi0_libero_data.sh" >&2
    exit 1
  fi
else
  echo "Skip input checks. Set CHECK_INPUTS=true to require prepared data and norm stats."
fi

train_args=(
  "${CONFIG_NAME}"
  "--exp-name=${EXP_NAME}"
  "--project-name=${PROJECT_NAME}"
  "--assets-base-dir=${ASSETS_BASE_DIR}"
  "--checkpoint-base-dir=${MODEL_DIR}"
  "--checkpoint-dir-override=${STEP_CHECKPOINT_DIR}"
  "--published-checkpoint-dir=${FINAL_CHECKPOINT_DIR}"
  "--batch-size=${BATCH_SIZE}"
  "--num-workers=${NUM_WORKERS}"
  "--num-train-steps=${NUM_TRAIN_STEPS}"
  "--save-interval=${SAVE_INTERVAL}"
  "--log-interval=${LOG_INTERVAL}"
  "--seed=${SEED}"
  "--fsdp-devices=${FSDP_DEVICES}"
  "--data.repo-id=${REPO_ID}"
)

if [[ -n "${ASSET_ID}" ]]; then
  train_args+=("--data.assets.asset-id=${ASSET_ID}")
fi

if [[ "${KEEP_PERIOD}" == "none" || "${KEEP_PERIOD}" == "None" ]]; then
  train_args+=("--keep-period=None")
else
  train_args+=("--keep-period=${KEEP_PERIOD}")
fi

if [[ "${WANDB_ENABLED}" == "false" ]]; then
  train_args+=("--no-wandb-enabled")
fi
if [[ "${OVERWRITE}" == "true" ]]; then
  train_args+=("--overwrite")
fi
if [[ "${RESUME}" == "true" ]]; then
  train_args+=("--resume")
fi

if [[ "${DRY_RUN}" == "true" ]]; then
  printf 'Dry run command: python -u scripts/train.py'
  printf ' %q' "${train_args[@]}"
  printf '\n'
  exit 0
fi

echo "Starting OpenPI training..."
python -u scripts/train.py "${train_args[@]}"
