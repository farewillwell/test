#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DATA_PREPARE="${RUN_DATA_PREPARE:-true}"
RUN_TRAIN="${RUN_TRAIN:-true}"

if [[ "${RUN_DATA_PREPARE}" == "true" ]]; then
  bash "${SCRIPT_DIR}/prepare_pi0_libero_data.sh"
else
  echo "Skip data preparation. Set RUN_DATA_PREPARE=true to enable it."
fi

if [[ "${RUN_TRAIN}" == "true" ]]; then
  bash "${SCRIPT_DIR}/train_pi0_libero_sft.sh"
else
  echo "Skip training. Set RUN_TRAIN=true to enable it."
fi
