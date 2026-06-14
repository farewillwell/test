export UV_PYTHON_INSTALL_DIR=/data/huangdi/heliqun/pi0/venv
source /data/huangdi/heliqun/pi0/openpi/pi_env/bin/activate
git config --global --add safe.directory /data/huangdi/heliqun/pi0

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export LIBERO_CONFIG_PATH=/data/huangdi/heliqun/openvla-oft/openvla-oft/LIBERO/.libero
export MUJOCO_GL=egl

export task_id=6
export action_horizon=10

cd /data/huangdi/heliqun/pi0

PI0_ROOT=/data/huangdi/heliqun/pi0
OPENPI_ROOT=${PI0_ROOT}/openpi
PI_PYTHON=${OPENPI_ROOT}/pi_env/bin/python

export PI0_CACHE_ROOT="${PI0_CACHE_ROOT:-${PI0_ROOT}/cache}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${PI0_CACHE_ROOT}/jax}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${PI0_CACHE_ROOT}/cuda}"
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-2147483648}"
export JAX_ENABLE_COMPILATION_CACHE="${JAX_ENABLE_COMPILATION_CACHE:-true}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"
export PYTHONUNBUFFERED=1

# ====== serve settings ======
MODEL_PATH=/data/aoss/heliqun/pi05-ours/goal-6/iter3/awbc_model/final
CONFIG_NAME=pi05_libero_awbc
HOST=0.0.0.0
PORT=8000

unset LD_LIBRARY_PATH

echo "PI0 root: ${PI0_ROOT}"
echo "OpenPI root: ${OPENPI_ROOT}"
echo "Config: ${CONFIG_NAME}"
echo "Model path: ${MODEL_PATH}"
echo "Host: ${HOST}"
echo "Port: ${PORT}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required." >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}/params" || ! -d "${MODEL_PATH}/assets" ]]; then
  echo "MODEL_PATH must contain params/ and assets/: ${MODEL_PATH}" >&2
  exit 1
fi

cd "${OPENPI_ROOT}"

${PI_PYTHON} -u scripts/serve_policy.py \
  --env LIBERO \
  --port "${PORT}" \
  policy:checkpoint \
  --policy.config "${CONFIG_NAME}" \
  --policy.dir "${MODEL_PATH}"