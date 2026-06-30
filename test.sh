kill_port_if_occupied() {
  local port="$1"

  echo "[precheck] check port ${port} via /proc"

  python - "${port}" <<'PY'
import os
import sys
import time
import signal

port = int(sys.argv[1])
port_hex = f"{port:04X}"

def listening_socket_inodes(port_hex):
    inodes = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                next(f)
                for line in f:
                    parts = line.split()
                    if len(parts) < 10:
                        continue
                    local_addr = parts[1]
                    state = parts[3]
                    inode = parts[9]

                    # TCP_LISTEN = 0A
                    if state != "0A":
                        continue

                    local_port_hex = local_addr.rsplit(":", 1)[-1].upper()
                    if local_port_hex == port_hex:
                        inodes.add(inode)
        except FileNotFoundError:
            pass
    return inodes

def pids_for_inodes(inodes):
    pids = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        fd_dir = f"/proc/{pid}/fd"
        try:
            for fd in os.listdir(fd_dir):
                fd_path = os.path.join(fd_dir, fd)
                try:
                    target = os.readlink(fd_path)
                except OSError:
                    continue
                if target.startswith("socket:[") and target.endswith("]"):
                    inode = target[len("socket:["):-1]
                    if inode in inodes:
                        pids.append(pid)
                        break
        except OSError:
            continue
    return sorted(set(pids))

def cmdline(pid):
    try:
        raw = open(f"/proc/{pid}/cmdline", "rb").read()
        return raw.replace(b"\0", b" ").decode("utf-8", errors="ignore").strip()
    except OSError:
        return ""

inodes = listening_socket_inodes(port_hex)
pids = pids_for_inodes(inodes)

if not pids:
    print(f"[precheck] port {port} is free")
    sys.exit(0)

print(f"[precheck] port {port} occupied by pids: {pids}")

policy_pids = []
for pid in pids:
    cmd = cmdline(pid)
    print(f"[precheck] pid={pid} cmd={cmd}")

    if "serve_policy.py" in cmd:
        policy_pids.append(pid)
    else:
        print(f"[error] port {port} is occupied by non-policy process, refuse to kill pid={pid}")
        sys.exit(2)

for pid in policy_pids:
    try:
        print(f"[precheck] SIGTERM old policy server pid={pid}")
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

time.sleep(3)

for pid in policy_pids:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        continue

    try:
        print(f"[precheck] SIGKILL old policy server pid={pid}")
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass

time.sleep(1)

# Final check.
inodes = listening_socket_inodes(port_hex)
pids = pids_for_inodes(inodes)
if pids:
    print(f"[error] port {port} still occupied after kill: {pids}")
    sys.exit(3)

print(f"[precheck] port {port} is free after cleanup")
PY
}
# ============================================================
# User settings
# ============================================================
MODEL_PATH="/data/aoss/heliqun/iter-pi/debug/awbc_data_awbc/final"
CONFIG_NAME="${CONFIG_NAME:-pi05_libero_awbc}"

TASK_ID=-1
ACTION_HORIZON="${ACTION_HORIZON:-10}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

PI0_ROOT="${PI0_ROOT:-/data/huangdi/heliqun/pi0}"
OPENPI_ROOT="${OPENPI_ROOT:-${PI0_ROOT}/openpi}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/data/huangdi/heliqun/openvla-oft/openvla-oft/LIBERO/.libero}"

LOG_DIR="${LOG_DIR:-${MODEL_PATH}/logs}"
SERVER_LOG="${SERVER_LOG:-${LOG_DIR}/policy_server.log}"
EVAL_LOG="${EVAL_LOG:-${LOG_DIR}/libero_eval.log}"

# Cache settings
PI0_CACHE_ROOT="${PI0_CACHE_ROOT:-${PI0_ROOT}/cache}"
JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${PI0_CACHE_ROOT}/jax}"
CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${PI0_CACHE_ROOT}/cuda}"
CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE:-2147483648}"

PI_PYTHON="${PI_PYTHON:-${OPENPI_ROOT}/pi_env/bin/python}"
LIBERO_PYTHON="${LIBERO_PYTHON:-${OPENPI_ROOT}/examples/libero/libero_env/bin/python}"

SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[cleanup] kill policy server pid=${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# ============================================================
# Basic checks
# ============================================================
mkdir -p "${LOG_DIR}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[error] MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}/params" || ! -d "${MODEL_PATH}/assets" ]]; then
  echo "[error] MODEL_PATH must contain params/ and assets/: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ ! -x "${PI_PYTHON}" ]]; then
  echo "[error] PI_PYTHON not executable: ${PI_PYTHON}" >&2
  exit 1
fi

if [[ ! -x "${LIBERO_PYTHON}" ]]; then
  echo "[error] LIBERO_PYTHON not executable: ${LIBERO_PYTHON}" >&2
  exit 1
fi

# ============================================================
# Environment
# ============================================================
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}"
export MUJOCO_GL=egl

export task_id="${TASK_ID}"
export action_horizon="${ACTION_HORIZON}"

export PI0_CACHE_ROOT="${PI0_CACHE_ROOT}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH}"
export CUDA_CACHE_MAXSIZE="${CUDA_CACHE_MAXSIZE}"
export JAX_ENABLE_COMPILATION_CACHE="${JAX_ENABLE_COMPILATION_CACHE:-true}"
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="${JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS:-0}"
export PYTHONUNBUFFERED=1

# Avoid some CUDA / conda library conflicts.
unset LD_LIBRARY_PATH || true

echo "============================================================"
echo "[config]"
echo "PI0_ROOT        = ${PI0_ROOT}"
echo "OPENPI_ROOT     = ${OPENPI_ROOT}"
echo "MODEL_PATH      = ${MODEL_PATH}"
echo "CONFIG_NAME     = ${CONFIG_NAME}"
echo "TASK_ID         = ${TASK_ID}"
echo "ACTION_HORIZON  = ${ACTION_HORIZON}"
echo "HOST            = ${HOST}"
echo "PORT            = ${PORT}"
echo "SERVER_LOG      = ${SERVER_LOG}"
echo "EVAL_LOG        = ${EVAL_LOG}"
echo "============================================================"

# ============================================================
# Start policy server directly, no test/test.sh dependency
# ============================================================
kill_port_if_occupied "${PORT}"
echo "[server] starting policy server..."
(
  cd "${OPENPI_ROOT}"

  "${PI_PYTHON}" -u scripts/serve_policy.py \
    --env LIBERO \
    --port "${PORT}" \
    policy:checkpoint \
    --policy.config "${CONFIG_NAME}" \
    --policy.dir "${MODEL_PATH}"
) > "${SERVER_LOG}" 2>&1 &

SERVER_PID=$!

echo "[server] pid=${SERVER_PID}"
echo "[server] log=${SERVER_LOG}"

# ============================================================
# Wait for server port
# ============================================================
echo "[wait] waiting for server port ${PORT}..."

python - <<PY
import socket
import time
import sys

host = "127.0.0.1"
port = int("${PORT}")
timeout = 240
start = time.time()

while time.time() - start < timeout:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.0)
    try:
        s.connect((host, port))
        s.close()
        print(f"[wait] server is ready: {host}:{port}")
        sys.exit(0)
    except OSError:
        s.close()
        time.sleep(2)

print(f"[wait error] server not ready after {timeout}s: {host}:{port}", file=sys.stderr)
sys.exit(1)
PY

# If server died during startup, fail early.
if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
  echo "[error] policy server exited before eval. Check log: ${SERVER_LOG}" >&2
  tail -200 "${SERVER_LOG}" >&2 || true
  exit 1
fi

# ============================================================
# Run LIBERO eval
# ============================================================
echo "[libero] starting examples/libero/main.py..."

(
  cd "${OPENPI_ROOT}"

  export PYTHONPATH="${PYTHONPATH:-}:${OPENPI_ROOT}/third_party/libero"
  export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}"
  export task_id="${TASK_ID}"
  export action_horizon="${ACTION_HORIZON}"

  "${LIBERO_PYTHON}" -u examples/libero/main.py \
  --args.task-id "${TASK_ID}" \
  --args.task-suite-name libero_goal \
  --args.port "${PORT}"
) 2>&1 | tee "${EVAL_LOG}"

echo "[done] eval log saved to ${EVAL_LOG}"
echo "[done] server log saved to ${SERVER_LOG}"