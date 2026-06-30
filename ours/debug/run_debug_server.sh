#!/usr/bin/env bash
# Only edit the variables in this block.
MODEL_PATH="/data/aoss/heliqun/pi05-ours-ent-2/goal-6/iter3/awbc_model/final"
CRITIC_PATH="/data/aoss/heliqun/pi05-ours-ent-2/goal-6/iter3/iql/final.pt"
OUT_DIR="/data/aoss/heliqun/qselect_debug/ours-select-ent1"
SAMPLE_MODE="qselect"        # simple / random / qselect
NUM_ACTION_SAMPLES=16
NOISE_SCALE=1.0
NOISE_STRATEGY="base"
LOG_EVERY=20
DIAG_SAVE_CANDIDATES_EVERY=0
TASK_SUITE="libero_goal"
TASK_ID=6
NUM_TRIALS=50
INITIAL_STATE_OFFSET=0
PI0_ROOT="/data/huangdi/heliqun/pi0"
OPENPI_ROOT="/data/huangdi/heliqun/pi0/openpi"
PI_PYTHON="${OPENPI_ROOT}/pi_env/bin/python"
LIBERO_PYTHON="${OPENPI_ROOT}/examples/libero/libero_env/bin/python"
POLICY_CONFIG="pi05_libero_awbc"
PORT=8000
SEED=7
DIAG_EXEC_HORIZON=0

export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${PI0_ROOT}/venv}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/data/huangdi/heliqun/openvla-oft/openvla-oft/LIBERO/.libero}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export action_horizon=10

export PI0_CACHE_ROOT="${PI0_CACHE_ROOT:-${PI0_ROOT}/cache}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${PI0_CACHE_ROOT}/jax}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${PI0_CACHE_ROOT}/cuda}"
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-2147483648}"
export JAX_ENABLE_COMPILATION_CACHE="${JAX_ENABLE_COMPILATION_CACHE:-true}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export JAX_LOG_COMPILES="${JAX_LOG_COMPILES:-1}"
unset LD_LIBRARY_PATH

if [[ ! -x "${PI_PYTHON}" ]]; then
  echo "OpenPI Python is not executable: ${PI_PYTHON}" >&2
  exit 1
fi
if [[ ! -x "${LIBERO_PYTHON}" ]]; then
  echo "LIBERO Python is not executable: ${LIBERO_PYTHON}" >&2
  exit 1
fi
if [[ -z "${MODEL_PATH}" || ! -d "${MODEL_PATH}/params" ]]; then
  echo "MODEL_PATH must point to a pi05 checkpoint containing params/: ${MODEL_PATH}" >&2
  exit 1
fi
case "${SAMPLE_MODE}" in
  simple|random)
    ;;
  qselect)
    if [[ -z "${CRITIC_PATH}" || ! -f "${CRITIC_PATH}" ]]; then
      echo "SAMPLE_MODE=qselect requires an existing CRITIC_PATH: ${CRITIC_PATH}" >&2
      exit 1
    fi
    ;;
  *)
    echo "SAMPLE_MODE must be simple, random, or qselect: ${SAMPLE_MODE}" >&2
    exit 1
    ;;
esac
case "${NOISE_STRATEGY}" in
  base|hubu|zhengjiao|guocaiyang)
    ;;
  *)
    echo "Unsupported NOISE_STRATEGY: ${NOISE_STRATEGY}" >&2
    exit 1
    ;;
esac
if (( NUM_ACTION_SAMPLES <= 0 )); then
  echo "NUM_ACTION_SAMPLES must be positive: ${NUM_ACTION_SAMPLES}" >&2
  exit 1
fi
if (( LOG_EVERY < 0 )); then
  echo "LOG_EVERY must be non-negative: ${LOG_EVERY}" >&2
  exit 1
fi
if (( DIAG_SAVE_CANDIDATES_EVERY < 0 )); then
  echo "DIAG_SAVE_CANDIDATES_EVERY must be non-negative" >&2
  exit 1
fi
if (( NUM_TRIALS <= 0 )); then
  echo "NUM_TRIALS must be positive: ${NUM_TRIALS}" >&2
  exit 1
fi
if (( INITIAL_STATE_OFFSET < 0 )); then
  echo "INITIAL_STATE_OFFSET must be non-negative: ${INITIAL_STATE_OFFSET}" >&2
  exit 1
fi

if "${PI_PYTHON}" -c \
  "import socket; s=socket.create_connection(('127.0.0.1', ${PORT}), timeout=1); s.close()" \
  >/dev/null 2>&1; then
  echo "Port ${PORT} is already in use. Stop the existing server (Ctrl+C) before running this one-click workflow." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}/records"
SERVER_LOG="${OUT_DIR}/policy_server.log"
DIAG_JSONL="${OUT_DIR}/diag.jsonl"
DIAG_RECORD_DIR="${OUT_DIR}/records"
EVAL_LOG="${OUT_DIR}/eval.log"
EVAL_METRICS="${OUT_DIR}/eval_metrics.jsonl"
SUMMARY_JSON="${OUT_DIR}/summary.json"
SUMMARY_CSV="${OUT_DIR}/summary.csv"
SUMMARY_STDOUT="${OUT_DIR}/summary.stdout"

: > "${SERVER_LOG}"
: > "${EVAL_LOG}"
: > "${EVAL_METRICS}"
if [[ "${SAMPLE_MODE}" != "simple" ]]; then
  : > "${DIAG_JSONL}"
fi

COMMAND=(
  "${PI_PYTHON}" -u "${PI0_ROOT}/ours/debug/qselect_debug_server.py"
  --policy-config "${POLICY_CONFIG}"
  --policy-dir "${MODEL_PATH}"
  --sample-mode "${SAMPLE_MODE}"
  --num-action-samples "${NUM_ACTION_SAMPLES}"
  --noise-strategy "${NOISE_STRATEGY}"
  --noise-scale "${NOISE_SCALE}"
  --seed "${SEED}"
  --port "${PORT}"
  --log-every "${LOG_EVERY}"
  --diag-jsonl "${DIAG_JSONL}"
  --diag-save-candidates-every "${DIAG_SAVE_CANDIDATES_EVERY}"
  --diag-record-dir "${DIAG_RECORD_DIR}"
  --diag-exec-horizon "${DIAG_EXEC_HORIZON}"
)
if [[ "${SAMPLE_MODE}" == "qselect" ]]; then
  COMMAND+=(--critic-path "${CRITIC_PATH}")
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[cleanup] stopping server pid=${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "[server] starting mode=${SAMPLE_MODE} port=${PORT}"
echo "[server] log=${SERVER_LOG}"
cd "${OPENPI_ROOT}"
"${COMMAND[@]}" > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

READY=0
for attempt in $(seq 1 900); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[server] exited before becoming ready; log tail:" >&2
    tail -n 80 "${SERVER_LOG}" >&2 || true
    wait "${SERVER_PID}" || true
    exit 1
  fi
  if "${PI_PYTHON}" -c \
    "import socket; s=socket.create_connection(('127.0.0.1', ${PORT}), timeout=1); s.close()" \
    >/dev/null 2>&1; then
    READY=1
    break
  fi
  if (( attempt == 1 || attempt % 10 == 0 )); then
    latest_log=$(tail -n 1 "${SERVER_LOG}" 2>/dev/null || true)
    echo "[server] waiting ${attempt}s; ${latest_log:-no log output yet}"
  fi
  sleep 1
done

if [[ "${READY}" != "1" ]]; then
  echo "[server] timed out after 900s; log tail:" >&2
  tail -n 80 "${SERVER_LOG}" >&2 || true
  exit 1
fi
echo "[server] ready pid=${SERVER_PID}"

echo "[eval] suite=${TASK_SUITE} task=${TASK_ID} trials=${NUM_TRIALS}"
PYTHONPATH="${OPENPI_ROOT}/third_party/libero:${PYTHONPATH:-}" \
HF_LEROBOT_HOME="${OUT_DIR}" \
"${LIBERO_PYTHON}" -u "${PI0_ROOT}/ours/debug/qselect_debug_eval.py" \
  --task-suite-name "${TASK_SUITE}" \
  --task-id "${TASK_ID}" \
  --num-trials-per-task "${NUM_TRIALS}" \
  --initial-state-offset "${INITIAL_STATE_OFFSET}" \
  --seed "${SEED}" \
  --repo-id "eval_tmp" \
  --overwrite \
  --video-out-path "${OUT_DIR}/videos" \
  --metrics-path "${EVAL_METRICS}" \
  2>&1 | tee "${EVAL_LOG}"

cleanup
trap - EXIT

if [[ "${SAMPLE_MODE}" == "simple" ]]; then
  echo "[summary] simple mode has no candidate diagnostics; skipping candidate summary"
else
  if [[ ! -s "${DIAG_JSONL}" ]]; then
    echo "[summary] expected non-empty diagnostics at ${DIAG_JSONL}" >&2
    exit 1
  fi
  "${PI_PYTHON}" "${PI0_ROOT}/ours/debug/summarize_qselect_diag.py" \
    --diag-jsonl "${DIAG_JSONL}" \
    --out-json "${SUMMARY_JSON}" \
    --out-csv "${SUMMARY_CSV}" \
    | tee "${SUMMARY_STDOUT}"
fi

echo "[done] server_log=${SERVER_LOG}"
echo "[done] eval_log=${EVAL_LOG}"
echo "[done] eval_metrics=${EVAL_METRICS}"
if [[ "${SAMPLE_MODE}" != "simple" ]]; then
  echo "[done] diag=${DIAG_JSONL}"
  echo "[done] summary_json=${SUMMARY_JSON}"
  echo "[done] summary_csv=${SUMMARY_CSV}"
fi
