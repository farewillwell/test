export UV_PYTHON_INSTALL_DIR=/data/huangdi/heliqun/pi0/venv
source /data/huangdi/heliqun/pi0/openpi/pi_env/bin/activate
git config --global --add safe.directory /data/huangdi/heliqun/pi0

export WANDB_MODE=offline
export WANDB_API_KEY=fb160cb6ca8fb120eeb3ce568a89ae77677a01f6
export TOKENIZERS_PARALLELISM=false
export LIBERO_CONFIG_PATH=/data/huangdi/heliqun/openvla-oft/openvla-oft/LIBERO/.libero
export MUJOCO_GL=egl

export task_id=6

cd /data/huangdi/heliqun/pi0

PI0_ROOT=/data/huangdi/heliqun/pi0
OPENPI_ROOT=${PI0_ROOT}/openpi
PI_PYTHON=${OPENPI_ROOT}/pi_env/bin/python
LIBERO_PYTHON=${OPENPI_ROOT}/examples/libero/libero_env/bin/python

WORKSPACE=/data/aoss/heliqun/pi0-ours/goal-6
SRC_DIR=/data/aoss/heliqun/dataset/libero-dataset/bytaskid/${task_id}
BASE_MODEL=/data/aoss/heliqun/model/pi

${PI_PYTHON} -u ours/iter.py \
  --workspace "${WORKSPACE}" \
  --src-dir "${SRC_DIR}" \
  --base-model "${BASE_MODEL}" \
  --pi0-root "${PI0_ROOT}" \
  --openpi-root "${OPENPI_ROOT}" \
  --pi-python "${PI_PYTHON}" \
  --libero-python "${LIBERO_PYTHON}" \
  --iters 4 \
  --horizon 5 \
  --replan-steps 5 \
  --awbc-config-name pi0_libero_awbc \
  --policy-config-name pi0_libero_awbc \
  --iql-encoder-name /data/aoss/heliqun/model/clip/clip-vit-base-patch32 \
  --iql-steps 4000 \
  --iql-batch-size 64 \
  --iql-num-workers 4 \
  --iql-hidden-dim 512 \
  --iql-num-q 2 \
  --iql-lr 1e-4 \
  --iql-use-q-aug \
  --awbc-steps 30000 \
  --awbc-batch-size 16 \
  --awbc-num-workers 4 \
  --awbc-save-interval 1000 \
  --awbc-log-interval 100 \
  --awbc-keep-period 5000 \
  --awbc-fsdp-devices 2 \
  --asset-id physical-intelligence/libero \
  --project-name openpi \
  --no-wandb-enabled \
  --use-q-select \
  --num-action-samples 16 \
  --qselect-num-steps 10 \
  --noise-scale 1.0 \
  --host 127.0.0.1 \
  --port 8000 \
  --task-suite-name libero_goal \
  --task-id "${task_id}" \
  --num-trials-per-task 50 \
  --initial-state-offset 0 \
  --max-steps-override -1 \
  --save-videos \
  --seed 7 \
  "$@" \
  > "iter-${task_id}-rank-qselect.log" 2>&1