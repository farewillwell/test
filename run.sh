export UV_PYTHON_INSTALL_DIR=/data/huangdi/heliqun/pi0/venv
source /data/huangdi/heliqun/pi0/openpi/pi_env/bin/activate
git config --global --add safe.directory /data/huangdi/heliqun/pi0

export WANDB_MODE=offline
export WANDB_API_KEY=fb160cb6ca8fb120eeb3ce568a89ae77677a01f6
export TOKENIZERS_PARALLELISM=false
export LIBERO_CONFIG_PATH=/data/huangdi/heliqun/openvla-oft/openvla-oft/LIBERO/.libero
export MUJOCO_GL=egl

export task_id=6
export action_horizon=10

cd /data/huangdi/heliqun/pi0

PI0_ROOT=/data/huangdi/heliqun/pi0
OPENPI_ROOT=${PI0_ROOT}/openpi
PI_PYTHON=${OPENPI_ROOT}/pi_env/bin/python
LIBERO_PYTHON=${OPENPI_ROOT}/examples/libero/libero_env/bin/python
export PI0_CACHE_ROOT="${PI0_CACHE_ROOT:-${PI0_ROOT}/cache}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${PI0_CACHE_ROOT}/jax}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${PI0_CACHE_ROOT}/cuda}"
WORKSPACE=/data/aoss/heliqun/pi05-ours/goal-6
SRC_DIR=/data/aoss/heliqun/dataset/pi-src/pi05
BASE_MODEL=/data/aoss/heliqun/model/pi/openpi-assets/checkpoints/pi05_base
unset LD_LIBRARY_PATH
${PI_PYTHON} -u ours/iter.py \
  --workspace "${WORKSPACE}" \
  --src-dir "${SRC_DIR}" \
  --base-model "${BASE_MODEL}" \
  --pi0-root "${PI0_ROOT}" \
  --openpi-root "${OPENPI_ROOT}" \
  --pi-python "${PI_PYTHON}" \
  --libero-python "${LIBERO_PYTHON}" \
  --iters 4 \
  --gpus 4 \
  --iql-batch-size 64 \
  --awbc-batch-size 64 \
  --iql-encoder-name /data/aoss/heliqun/model/clip/clip-vit-base-patch32 \
  --task-id ${task_id} \
  --task-suite-name libero_goal \
  --num-trials-per-task 50 \
  --iql-use-q-aug \
  --policy-config-name pi05_libero_awbc \
  > "iter-${task_id}-rank-qselect5.log" 2>&1