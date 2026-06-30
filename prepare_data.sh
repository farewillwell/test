export UV_PYTHON_INSTALL_DIR=/data/huangdi/heliqun/pi0/venv
source /data/huangdi/heliqun/pi0/openpi/pi_env/bin/activate
HDF5_DIR="/data/aoss/heliqun/dataset/libero-dataset/libero_goal_select_50"
LEROBOT_DATA_DIR="/data/aoss/heliqun/dataset/pi-src/libero-goal"
export HF_LEROBOT_HOME="/data/aoss/heliqun/dataset/pi-src"
REPO_ID="libero-goal"
convert_args=(
  "--input-dir=${HDF5_DIR}"
  "--repo-id=${REPO_ID}"
  "--fps=10"
  "--image-size=256"
)
python "/data/huangdi/heliqun/pi0/train/convert_libero_hdf5_to_lerobot_openpi.py" "${convert_args[@]}"
