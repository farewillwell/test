export UV_PYTHON_INSTALL_DIR=/data/huangdi/heliqun/pi0/venv
source /data/huangdi/heliqun/pi0/openpi/pi_env/bin/activate
git config --global --add safe.directory /data/huangdi/heliqun/pi0

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export LIBERO_CONFIG_PATH=/data/huangdi/heliqun/openvla-oft/openvla-oft/LIBERO/.libero
export MUJOCO_GL=egl

export task_id=-1
export action_horizon=10

cd /data/huangdi/heliqun/pi0

PI0_ROOT=/data/huangdi/heliqun/pi0
OPENPI_ROOT=${PI0_ROOT}/openpi
PI_PYTHON=${OPENPI_ROOT}/pi_env/bin/python
LIBERO_PYTHON=${OPENPI_ROOT}/examples/libero/libero_env/bin/python
export PI0_CACHE_ROOT="${PI0_CACHE_ROOT:-${PI0_ROOT}/cache}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${PI0_CACHE_ROOT}/jax}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${PI0_CACHE_ROOT}/cuda}"
WORKSPACE=/data/aoss/heliqun/iter-pi/pi05-ours-no-ent/goal
SRC_DIR=/data/aoss/heliqun/dataset/pi-src/libero-goal
BASE_MODEL=/data/aoss/heliqun/model/pi/openpi-assets/checkpoints/pi05_base
unset LD_LIBRARY_PATH
${PI_PYTHON} -u ours/BC/train_awbc.py \
  --data-dir /data/aoss/heliqun/iter-pi/pi05-ours-ent/goal/iter3/data/labeled \
  --model-dir /data/aoss/heliqun/iter-pi/debug/awbc_data_awbc \
  --base-policy-dir ${BASE_MODEL} \
  --pi0-root ${PI0_ROOT} \
  --openpi-root ${OPENPI_ROOT} \
  --python-bin ${PI_PYTHON} \
  --config-name pi05_awbc \
  --steps 20000 \
  --batch-size 64 \
  --num-workers 16 \
  --gpus 4 \
  --seed 7 > awbc_data_awbc.log 2>&1

