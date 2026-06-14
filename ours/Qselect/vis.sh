export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/data/huangdi/heliqun/pi0/venv}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/data/huangdi/heliqun/openvla-oft/openvla-oft/LIBERO/.libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

PI0_ROOT="${PI0_ROOT:-/data/huangdi/heliqun/pi0}"
OPENPI_ROOT="${OPENPI_ROOT:-${PI0_ROOT}/openpi}"
PI_PYTHON="${PI_PYTHON:-${OPENPI_ROOT}/pi_env/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-${OPENPI_ROOT}/examples/libero/libero_env/bin/python}"

export PI0_CACHE_ROOT="${PI0_CACHE_ROOT:-${PI0_ROOT}/cache}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${PI0_CACHE_ROOT}/jax}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${PI0_CACHE_ROOT}/cuda}"
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-2147483648}"
export JAX_ENABLE_COMPILATION_CACHE="${JAX_ENABLE_COMPILATION_CACHE:-true}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"

task_id="${task_id:-6}"
task_suite="${task_suite:-libero_goal}"
action_horizon="${action_horizon:-10}"
export action_horizon

POLICY_DIR="${POLICY_DIR:-/data/aoss/heliqun/pi05-ours/goal-6/iter3/awbc_model/final}"
POLICY_CONFIG_NAME="${POLICY_CONFIG_NAME:-pi05_libero_awbc}"
POLICY_NAME="${POLICY_NAME:-pi05_iter3_awbc}"
PORT="${PORT:-8000}"
SEED="${SEED:-7}"
NUM_TRIALS="${NUM_TRIALS:-50}"
NUM_ACTION_SAMPLES="${NUM_ACTION_SAMPLES:-16}"
NOISE_STRATEGY="${NOISE_STRATEGY:-base}"
NOISE_SCALE="${NOISE_SCALE:-2}"
FIXED_INIT_STATE_IDX=0
SUCCESS_TARGET_COUNT="${SUCCESS_TARGET_COUNT:-50}"
MAX_TRIALS_FOR_SUCCESS_TARGET="${MAX_TRIALS_FOR_SUCCESS_TARGET:-300}"

VIS_MODE="${VIS_MODE:-${NOISE_STRATEGY}}"
RUN_TAG="${RUN_TAG:-${POLICY_NAME}-${NOISE_STRATEGY}-n${NUM_ACTION_SAMPLES}}"
DEBUG_ROOT="${DEBUG_ROOT:-${PI0_ROOT}/ours/Qselect/policy-vis-${RUN_TAG}}"

unset LD_LIBRARY_PATH
mkdir -p "${DEBUG_ROOT}/logs"

if [[ ! -d "${POLICY_DIR}/params" ]]; then
  echo "POLICY_DIR must point to a checkpoint directory containing params/: ${POLICY_DIR}" >&2
  exit 1
fi

SERVER_LOG="${DEBUG_ROOT}/logs/policy_server.log"
COLLECT_LOG="${DEBUG_ROOT}/logs/diverse_vis.log"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

cd "${OPENPI_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
"${PI_PYTHON}" -u "${PI0_ROOT}/ours/Qselect/server.py" \
  --policy-config "${POLICY_CONFIG_NAME}" \
  --policy-dir "${POLICY_DIR}" \
  --sample-mode random \
  --num-action-samples "${NUM_ACTION_SAMPLES}" \
  --noise-strategy "${NOISE_STRATEGY}" \
  --noise-scale "${NOISE_SCALE}" \
  --seed "${SEED}" \
  --port "${PORT}" \
  > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

echo "[server] pid=${SERVER_PID} log=${SERVER_LOG}"

READY=0
for _ in $(seq 1 900); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Policy server exited before becoming ready. See ${SERVER_LOG}" >&2
    exit 1
  fi
  if "${PI_PYTHON}" -c "import socket; s=socket.create_connection(('127.0.0.1', int('${PORT}')), timeout=1); s.close()" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done

if [[ "${READY}" != "1" ]]; then
  echo "Timed out waiting for policy server on 127.0.0.1:${PORT}. See ${SERVER_LOG}" >&2
  exit 1
fi

PYTHONPATH="${OPENPI_ROOT}/third_party/libero:${PYTHONPATH:-}" \
"${LIBERO_PYTHON}" -u "${PI0_ROOT}/ours/Qselect/diverse_vis.py" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --policy_names "${POLICY_NAME}" \
  --pretrained_checkpoints "${POLICY_DIR}" \
  --task_suite_name "${task_suite}" \
  --target_task "${task_id}" \
  --fixed_init_state_idx "${FIXED_INIT_STATE_IDX}" \
  --num_trials_per_task "${NUM_TRIALS}" \
  --success_target_count "${SUCCESS_TARGET_COUNT}" \
  --max_trials_for_success_target "${MAX_TRIALS_FOR_SUCCESS_TARGET}" \
  --sample_modes "${VIS_MODE}" \
  --selection_mode random \
  --num_action_samples "${NUM_ACTION_SAMPLES}" \
  --noise_strategy "${NOISE_STRATEGY}" \
  --noise_scale "${NOISE_SCALE}" \
  --replan_steps "${action_horizon}" \
  --plot_x_min "${PLOT_X_MIN:--0.22}" \
  --plot_x_max "${PLOT_X_MAX:-0.06}" \
  --plot_y_min "${PLOT_Y_MIN:--0.06}" \
  --plot_y_max "${PLOT_Y_MAX:-0.18}" \
  --traj_dir "${DEBUG_ROOT}" \
  --local_log_dir "${DEBUG_ROOT}/logs" \
  --save_rollout_hdf5 "${SAVE_ROLLOUT_HDF5:-False}" \
  --save_video "${SAVE_VIDEO:-True}" \
  --seed "${SEED}" \
  > "${COLLECT_LOG}" 2>&1

echo "[done] saved visualization to ${DEBUG_ROOT}"
echo "[done] collect log: ${COLLECT_LOG}"
