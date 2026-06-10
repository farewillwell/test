#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-resumable pi0 iterative learning driver with workspace-local data pool.

Workspace layout:

workspace/
├── save.json
├── logs/main.log
├── pool/raw              # accumulated collector-style data: success + failure
├── pool/meta/sources.jsonl
├── iter0/
│   ├── logs/
│   ├── iql/final.pt
│   ├── data/labeled      # success-only AWBC data labeled by iter0 critic
│   ├── data/collect      # newly collected data from iter0 policy
│   ├── awbc_model/final/params
│   ├── collect_videos/
│   └── collect_metrics.jsonl
└── iter1/...
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np


STAGE_TRAIN_IQL = "train_iql"
STAGE_LABEL = "label"
STAGE_TRAIN_AWBC = "train_awbc"
STAGE_COLLECT = "collect"
STAGE_APPEND_POOL = "append_pool"
STAGE_FINISHED = "finished"

STAGES = (STAGE_TRAIN_IQL, STAGE_LABEL, STAGE_TRAIN_AWBC, STAGE_COLLECT, STAGE_APPEND_POOL)
VALID_STAGES = set(STAGES) | {STAGE_FINISHED}

IQL_REQUIRED_KEYS = {
    "image", "wrist_image", "state", "actions", "reward", "done", "success", "task", "episode_index",
}



def _task_mode(task_id: int) -> str:
    return "single" if int(task_id) >= 0 else "multi"


def get_iql_steps(iter_index: int, task_id: int) -> int:
    base = 4000
    iter_task_add = 2000
    iter_scale = max(int(iter_index) + 1, 1)
    num_tasks = 1 if task_id >=0 else 10
    return int(base+ iter_task_add * num_tasks * iter_scale)


def get_awbc_steps(iter_index: int, task_id: int) -> int:
    base = 3000
    iter_task_add = 2000
    iter_scale = max(int(iter_index) + 1, 1)
    num_tasks = 1 if task_id >=0 else 10
    return int(base+ iter_task_add * num_tasks * iter_scale)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pi0 iterative IQL/AWBC/Q-select with workspace-local data pool.")

    # Only three core user paths.
    p.add_argument("--workspace", required=True, help="Experiment root. All generated data/checkpoints/logs stay under it.")
    p.add_argument("--src-dir", required=True, help="Initial collector-style LeRobot dataset directory.")
    p.add_argument("--base-model", required=True, help="Initial pi0 checkpoint directory containing params/.")

    # Repository / environment roots.
    p.add_argument("--pi0-root", default="", help="Defaults to parent of this script.")
    p.add_argument("--openpi-root", default="", help="Defaults to <pi0-root>/openpi.")
    p.add_argument("--pi-python", default="", help="Defaults to <openpi-root>/pi_env/bin/python.")
    p.add_argument("--libero-python", default="", help="Defaults to <openpi-root>/examples/libero/libero_env/bin/python.")

    # Iteration / resume.
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--restart", action="store_true", help="Delete workspace before starting.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--gpus", type=int, default=2)
    p.add_argument("--overwrite-repos", action=argparse.BooleanOptionalAction, default=True)

    # Time scale. These are intentionally fixed to 5.
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--replan-steps", type=int, default=5)

    # IQL.
    p.add_argument("--iql-encoder-name", default="/data/aoss/heliqun/model/clip/clip-vit-base-patch32")
    p.add_argument("--iql-steps", type=int, default=4000)
    p.add_argument("--iql-batch-size", type=int, default=64)
    p.add_argument("--iql-num-workers", type=int, default=4)
    p.add_argument("--iql-devices", type=int, default=1)
    p.add_argument("--iql-hidden-dim", type=int, default=512)
    p.add_argument("--iql-num-q", type=int, default=2)
    p.add_argument("--iql-action-layers", type=int, default=2)
    p.add_argument("--iql-q-layers", type=int, default=2)
    p.add_argument("--iql-dropout", type=float, default=0.1)
    p.add_argument("--iql-lr", type=float, default=1e-4)
    p.add_argument("--iql-weight-decay", type=float, default=1e-4)
    p.add_argument("--iql-gamma", type=float, default=0.99)
    p.add_argument("--iql-expectile", type=float, default=0.7)
    p.add_argument("--iql-tau", type=float, default=0.02)
    p.add_argument("--iql-grad-clip", type=float, default=1.0)
    p.add_argument("--iql-q-l2-coef", type=float, default=1e-4)
    p.add_argument("--iql-use-q-aug", action="store_true")
    p.add_argument("--iql-log-every", type=int, default=50)
    p.add_argument("--iql-debug-every", type=int, default=500)
    p.add_argument("--iql-save-every", type=int, default=1000)

    # Label.
    p.add_argument("--label-batch-size", type=int, default=128)
    p.add_argument("--label-normalize-adv", action="store_true")
    p.add_argument("--label-clamp-adv", type=float, default=0.0)
    p.add_argument("--label-drop-incomplete-tail", action="store_true")
    p.add_argument("--label-max-episodes", type=int, default=0)

    # AWBC.
    p.add_argument("--awbc-config-name", default="pi0_libero_awbc")
    p.add_argument("--policy-config-name", default="pi0_libero_awbc")
    p.add_argument("--awbc-batch-size", type=int, default=16)
    p.add_argument("--awbc-num-workers", type=int, default=4)
    p.add_argument("--awbc-save-interval", type=int, default=1000)
    p.add_argument("--awbc-log-interval", type=int, default=100)
    p.add_argument("--awbc-keep-period", default="5000")
    p.add_argument("--norm-max-frames", type=int, default=0)
    p.add_argument("--asset-id", default="physical-intelligence/libero")
    p.add_argument("--project-name", default="openpi")
    p.add_argument("--wandb-enabled", action=argparse.BooleanOptionalAction, default=False)

    # Q-select / collect.
    p.add_argument("--use-q-select", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-action-samples", type=int, default=16)
    p.add_argument("--qselect-num-steps", type=int, default=10)
    p.add_argument("--noise-scale", type=float, default=1.0)
    p.add_argument("--score-horizon", type=int, default=0)
    p.add_argument("--selector-device", default="")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--server-wait-timeout", type=float, default=180.0)

    p.add_argument("--task-suite-name", default="libero_goal")
    p.add_argument("--task-id", type=int, default=6)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--num-trials-per-task", type=int, default=50)
    p.add_argument("--initial-state-offset", type=int, default=0)
    p.add_argument("--max-steps-override", type=int, default=-1)
    p.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-success", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-failure", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    workspace = Path(args.workspace).resolve()
    src_dir = Path(args.src_dir).resolve()
    base_model = Path(args.base_model).resolve()

    if args.restart and workspace.exists():
        shutil.rmtree(workspace)

    if not src_dir.exists():
        raise FileNotFoundError(f"--src-dir does not exist: {src_dir}")
    if not base_model.exists():
        raise FileNotFoundError(f"--base-model does not exist: {base_model}")
    if not (base_model / "params").exists():
        raise FileNotFoundError(f"--base-model must contain params/: {base_model}")
    if args.horizon != 5 or args.replan_steps != 5:
        raise ValueError("pi0 action_horizon, IQL horizon and replan_steps must all be 5.")

    args.workspace = str(workspace)
    args.src_dir = str(src_dir)
    args.base_model = str(base_model)

    this_file = Path(__file__).resolve()
    pi0_root = Path(args.pi0_root).resolve() if args.pi0_root else this_file.parents[1]
    args.pi0_root = str(pi0_root)

    openpi_root = Path(args.openpi_root).resolve() if args.openpi_root else pi0_root / "openpi"
    args.openpi_root = str(openpi_root)
    args.pi_python = str(Path(args.pi_python).resolve() if args.pi_python else openpi_root / "pi_env" / "bin" / "python")
    args.libero_python = str(
        Path(args.libero_python).resolve()
        if args.libero_python
        else openpi_root / "examples" / "libero" / "libero_env" / "bin" / "python"
    )
    args.cache_root = str(workspace / "cache")
    return args


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def quote_cmd(cmd: list[Any]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def log_line(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def main_log(args: argparse.Namespace) -> Path:
    return Path(args.workspace) / "logs" / "main.log"


def workspace_paths(args: argparse.Namespace, iter_index: int) -> dict[str, Path]:
    root = Path(args.workspace) / f"iter{iter_index}"
    return {
        "root": root,
        "logs": root / "logs",
        "iql_dir": root / "iql",
        "data": root / "data",
        "labeled_data": root / "data" / "labeled",
        "collect_data": root / "data" / "collect",
        "model_dir": root / "awbc_model",
        "videos": root / "collect_videos",
        "metrics": root / "collect_metrics.jsonl",
    }


def pool_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = Path(args.workspace) / "pool"
    return {
        "root": root,
        "raw": root / "raw",
        "raw_tmp": root / "raw_tmp",
        "raw_backup": root / "raw_backup",
        "meta": root / "meta",
        "sources": root / "meta" / "sources.jsonl",
    }


def mkdir_stage_dirs(p: dict[str, Path]) -> None:
    for key in ("root", "logs", "iql_dir", "data", "model_dir", "videos"):
        p[key].mkdir(parents=True, exist_ok=True)


def base_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = env.get("TOKENIZERS_PARALLELISM", "false")
    env["WANDB_MODE"] = env.get("WANDB_MODE", "offline")
    if "OPENPI_DATA_HOME" not in env or not env["OPENPI_DATA_HOME"]:
        raise RuntimeError("OPENPI_DATA_HOME must be set in the shell environment.")
    env["JAX_COMPILATION_CACHE_DIR"] = env.get("JAX_COMPILATION_CACHE_DIR", str(Path(args.cache_root) / "jax"))
    env["CUDA_CACHE_PATH"] = env.get("CUDA_CACHE_PATH", str(Path(args.cache_root) / "cuda"))
    env["CUDA_CACHE_MAXSIZE"] = env.get("CUDA_CACHE_MAXSIZE", "2147483648")
    env["JAX_ENABLE_COMPILATION_CACHE"] = env.get("JAX_ENABLE_COMPILATION_CACHE", "true")
    env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = env.get("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
    return env


def libero_env(args: argparse.Namespace, data_root: Path) -> dict[str, str]:
    env = base_env(args)
    if "LIBERO_CONFIG_PATH" not in env or not env["LIBERO_CONFIG_PATH"]:
        raise RuntimeError("LIBERO_CONFIG_PATH must be set in the shell environment.")
    if "MUJOCO_GL" not in env or not env["MUJOCO_GL"]:
        raise RuntimeError("MUJOCO_GL must be set in the shell environment, e.g. export MUJOCO_GL=egl")
    env["HF_LEROBOT_HOME"] = str(data_root)
    third_party_libero = str(Path(args.openpi_root) / "third_party" / "libero")
    env["PYTHONPATH"] = third_party_libero + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run_cmd(cmd: list[Any], *, log_path: Path, cwd: Path, env: dict[str, str], dry_run: bool) -> None:
    log_line(log_path, f"[cmd] {quote_cmd(cmd)}")
    log_line(log_path, f"[cwd] {cwd}")
    if dry_run:
        return
    with subprocess.Popen(
        [str(x) for x in cmd],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line)
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)


def wait_for_port(host: str, port: int, proc: subprocess.Popen[Any], timeout: float, log_path: Path) -> None:
    connect_host = "127.0.0.1" if host in {"0.0.0.0", ""} else host
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Policy server exited early with code {proc.returncode}. See {log_path}")
        try:
            with socket.create_connection((connect_host, int(port)), timeout=2.0):
                log_line(log_path, f"[server] ready at {connect_host}:{port}")
                return
        except OSError:
            time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for policy server at {connect_host}:{port}. See {log_path}")


def to_scalar(x: Any) -> Any:
    try:
        import torch
        if torch.is_tensor(x):
            if x.numel() == 1:
                return x.detach().cpu().item()
            return x.detach().cpu().numpy()
    except Exception:
        pass
    if isinstance(x, (int, float, bool, str, bytes)):
        return x
    arr = np.asarray(x)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr


def to_numpy(x: Any) -> np.ndarray:
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    try:
        from PIL import Image
        if isinstance(x, Image.Image):
            return np.asarray(x)
    except Exception:
        pass
    return np.asarray(x)


def decode_task(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if isinstance(x, str):
        return x
    try:
        import torch
        if torch.is_tensor(x):
            if x.numel() == 1:
                return decode_task(x.detach().cpu().item())
            return str(x.detach().cpu().numpy())
    except Exception:
        pass
    arr = np.asarray(x)
    if arr.shape == ():
        return decode_task(arr.item())
    if arr.size == 1:
        return decode_task(arr.reshape(-1)[0])
    return str(x)


def image_hwc_uint8(x: Any) -> np.ndarray:
    arr = to_numpy(x)
    if arr.ndim != 3:
        raise ValueError(f"Expected image [H,W,3] or [3,H,W], got {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got {arr.shape}")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def require_keys(sample: dict[str, Any], keys: set[str], *, repo: str) -> None:
    missing = sorted(k for k in keys if k not in sample)
    if missing:
        raise KeyError(f"Repo {repo} missing required keys: {missing}. keys={sorted(sample.keys())}")


def open_lerobot_dataset(repo_dir: Path):
    repo_dir = repo_dir.resolve()
    repo_id = repo_dir.name
    root = repo_dir.parent
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    try:
        return LeRobotDataset(repo_id, root=root)
    except TypeError:
        return LeRobotDataset(repo_id, root)


def remove_path(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def merge_collector_repos(input_dirs: list[Path], output_dir: Path, *, overwrite: bool, log_path: Path) -> dict[str, int]:
    """Materialize a collector-style LeRobot repo by concatenating whole episodes."""
    openpi_src = Path(__file__).resolve().parents[1] / "openpi" / "src"
    if openpi_src.exists() and str(openpi_src) not in sys.path:
        sys.path.insert(0, str(openpi_src))
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    input_dirs = [Path(p).resolve() for p in input_dirs]
    output_dir = Path(output_dir).resolve()
    output_parent = output_dir.parent
    output_repo_id = output_dir.name

    for src in input_dirs:
        if not src.exists():
            raise FileNotFoundError(f"Input LeRobot repo does not exist: {src}")

    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"Output repo exists: {output_dir}")

    output_parent.mkdir(parents=True, exist_ok=True)

    first_sample = None
    first_repo = ""
    for src in input_dirs:
        ds = open_lerobot_dataset(src)
        if len(ds) > 0:
            first_sample = ds[0]
            first_repo = str(src)
            break
    if first_sample is None:
        raise RuntimeError(f"All input repos are empty: {input_dirs}")

    require_keys(first_sample, IQL_REQUIRED_KEYS, repo=first_repo)
    image_shape = image_hwc_uint8(first_sample["image"]).shape
    wrist_shape = image_hwc_uint8(first_sample["wrist_image"]).shape
    state_dim = int(to_numpy(first_sample["state"]).reshape(-1).shape[0])
    action_dim = int(to_numpy(first_sample["actions"]).reshape(-1).shape[0])

    old_home = os.environ.get("HF_LEROBOT_HOME", "")
    os.environ["HF_LEROBOT_HOME"] = str(output_parent)

    out = LeRobotDataset.create(
        repo_id=output_repo_id,
        robot_type="libero",
        fps=10,
        features={
            "image": {"dtype": "image", "shape": tuple(image_shape), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": tuple(wrist_shape), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (state_dim,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (action_dim,), "names": ["actions"]},
            "reward": {"dtype": "float32", "shape": (1,), "names": ["reward"]},
            "done": {"dtype": "int64", "shape": (1,), "names": ["done"]},
            "success": {"dtype": "int64", "shape": (1,), "names": ["success"]},
        },
        use_videos=False,
    )

    total_frames = 0
    total_episodes = 0
    try:
        for src in input_dirs:
            repo_name = str(src)
            ds = open_lerobot_dataset(src)
            by_ep: dict[int, list[int]] = defaultdict(list)
            for idx in range(len(ds)):
                sample = ds[idx]
                require_keys(sample, IQL_REQUIRED_KEYS, repo=repo_name)
                ep = int(to_scalar(sample["episode_index"]))
                by_ep[ep].append(idx)
            log_line(log_path, f"[merge] reading {src}: frames={len(ds)} episodes={len(by_ep)}")
            for ep in sorted(by_ep):
                for idx in by_ep[ep]:
                    sample = ds[idx]
                    out.add_frame({
                        "image": image_hwc_uint8(sample["image"]),
                        "wrist_image": image_hwc_uint8(sample["wrist_image"]),
                        "state": to_numpy(sample["state"]).astype(np.float32).reshape(-1),
                        "actions": to_numpy(sample["actions"]).astype(np.float32).reshape(-1),
                        "reward": np.asarray([float(to_scalar(sample["reward"]))], dtype=np.float32),
                        "done": np.asarray([int(bool(to_scalar(sample["done"])))], dtype=np.int64),
                        "success": np.asarray([int(bool(to_scalar(sample["success"])))], dtype=np.int64),
                        "task": decode_task(sample["task"]),
                    })
                    total_frames += 1
                out.save_episode()
                total_episodes += 1
    finally:
        if hasattr(out, "stop_image_writer"):
            out.stop_image_writer()
        if old_home:
            os.environ["HF_LEROBOT_HOME"] = old_home
        else:
            os.environ.pop("HF_LEROBOT_HOME", None)

    log_line(log_path, f"[merge] wrote {output_dir}: episodes={total_episodes} frames={total_frames}")
    return {"episodes": int(total_episodes), "frames": int(total_frames)}


def append_sources_record(args: argparse.Namespace, record: dict[str, Any]) -> None:
    pools = pool_paths(args)
    pools["meta"].mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record.setdefault("time", now())
    with pools["sources"].open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def atomic_replace_dir(src_tmp: Path, dst: Path, backup: Path) -> None:
    remove_path(backup)
    if dst.exists():
        dst.rename(backup)
    src_tmp.rename(dst)
    remove_path(backup)


def ensure_pool_initialized(args: argparse.Namespace, log_path: Path) -> None:
    pools = pool_paths(args)
    raw = pools["raw"]
    if raw.exists():
        return
    if args.dry_run:
        log_line(log_path, f"[dry-run] init pool: {args.src_dir} -> {raw}")
        return
    pools["root"].mkdir(parents=True, exist_ok=True)
    stats = merge_collector_repos([Path(args.src_dir).resolve()], pools["raw_tmp"], overwrite=True, log_path=log_path)
    atomic_replace_dir(pools["raw_tmp"], raw, pools["raw_backup"])
    append_sources_record(args, {"event": "init_pool", "src_dir": str(Path(args.src_dir).resolve()), "pool_raw": str(raw), **stats})


def build_initial_state(args: argparse.Namespace) -> dict[str, Any]:
    pools = pool_paths(args)
    return {
        "version": 4,
        "iter_index": 0,
        "next_stage": STAGE_TRAIN_IQL,
        "pool_raw_dir": str(pools["raw"]),
        "current_policy_dir": str(Path(args.base_model).resolve()),
        "current_head_path": "",
        "current_awbc_data_dir": "",
        "last_collect_data_dir": "",
        "history": [],
    }


def append_history(state: dict[str, Any], iter_index: int, stage: str, extra: dict[str, Any] | None = None) -> None:
    item = {"iter": int(iter_index), "stage": stage, "time": now()}
    if extra:
        item.update(extra)
    state.setdefault("history", []).append(item)


def save_state(args: argparse.Namespace, state: dict[str, Any]) -> None:
    path = Path(args.workspace) / "save.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_or_init_state(args: argparse.Namespace) -> dict[str, Any]:
    save_path = Path(args.workspace) / "save.json"
    log = main_log(args)
    if save_path.exists() and args.resume and not args.restart:
        state = json.loads(save_path.read_text(encoding="utf-8"))
        if state.get("next_stage") not in VALID_STAGES:
            raise ValueError(f"Invalid next_stage in save.json: {state.get('next_stage')}")
        log_line(log, f"[resume] {save_path}: iter={state.get('iter_index')} next_stage={state.get('next_stage')}")
        return state
    if save_path.exists() and not args.resume:
        raise FileExistsError(f"{save_path} exists. Use --resume or --restart.")
    ensure_pool_initialized(args, log)
    state = build_initial_state(args)
    save_state(args, state)
    return state


def stage_train_iql(args: argparse.Namespace, state: dict[str, Any], p: dict[str, Path]) -> dict[str, Any]:
    log_path = p["logs"] / f"{STAGE_TRAIN_IQL}.log"
    pools = pool_paths(args)
    iter_index = int(state["iter_index"])
    iql_steps = get_iql_steps(iter_index, args.task_id)
    log_line(
        log_path,
        f"[steps] IQL max_steps={iql_steps} mode={_task_mode(args.task_id)} "
        f"iter={iter_index}",
    )
    cmd = [
        args.pi_python, "-u", "-m", "torch.distributed.run",
        "--standalone", "--nnodes", "1", "--nproc-per-node", str(args.gpus),
        str("IQL/train.py"),
        "--repo-id", "raw", "--root", str(pools["root"]), "--output-dir", str(p["iql_dir"]),
        "--encoder-name", args.iql_encoder_name, "--horizon", str(args.horizon),
        "--batch-size", str(args.iql_batch_size), "--num-workers", str(args.iql_num_workers),
        "--max-steps", str(iql_steps), "--lr", str(args.iql_lr),
        "--weight-decay", str(args.iql_weight_decay), "--gamma", str(args.iql_gamma),
        "--expectile", str(args.iql_expectile), "--tau", str(args.iql_tau),
        "--grad-clip", str(args.iql_grad_clip), "--hidden-dim", str(args.iql_hidden_dim),
        "--num-q", str(args.iql_num_q), "--action-layers", str(args.iql_action_layers),
        "--q-layers", str(args.iql_q_layers), "--dropout", str(args.iql_dropout),
        "--q-l2-coef", str(args.iql_q_l2_coef), "--log-every", str(args.iql_log_every),
        "--debug-every", str(args.iql_debug_every), "--save-every", str(args.iql_save_every),
        "--seed", str(args.seed),
    ]
    if args.iql_use_q_aug:
        cmd.append("--use-q-aug")
    log_line(
        log_path,
        f"[steps] IQL max_steps={iql_steps} iter={iter_index} "
        f"gpus={args.gpus} per_gpu_batch_size={args.iql_batch_size} "
        f"effective_global_batch_size={int(args.gpus) * int(args.iql_batch_size)}",
    )
    run_cmd(cmd, log_path=log_path, cwd=Path(args.pi0_root) / "ours" / "IQL", env=base_env(args), dry_run=args.dry_run)
    head = p["iql_dir"] / "final.pt"
    if not args.dry_run and not head.exists():
        raise RuntimeError(f"Expected IQL checkpoint not found: {head}")
    state["current_head_path"] = str(head)
    state["next_stage"] = STAGE_LABEL
    append_history(state, int(state["iter_index"]), STAGE_TRAIN_IQL, {"pool_raw_dir": str(pools["raw"]), "head_path": str(head), "iql_steps": int(iql_steps), "gpus": int(args.gpus)})
    return state


def stage_label(args: argparse.Namespace, state: dict[str, Any], p: dict[str, Path]) -> dict[str, Any]:
    log_path = p["logs"] / f"{STAGE_LABEL}.log"
    pools = pool_paths(args)
    env = base_env(args)
    env["HF_LEROBOT_HOME"] = str(p["data"])
    cmd = [
        args.pi_python, "-u", str(Path(args.pi0_root) / "ours" / "IQL" / "label.py"),
        "--input-repo-id", "raw", "--input-root", str(pools["root"]),
        "--output-repo-id", "labeled", "--critic-path", state["current_head_path"],
        "--horizon", str(args.horizon), "--batch-size", str(args.label_batch_size),
        "--seed", str(args.seed),
    ]
    if args.label_normalize_adv:
        cmd.append("--normalize-adv")
    if args.label_clamp_adv > 0:
        cmd.extend(["--clamp-adv", str(args.label_clamp_adv)])
    if args.label_drop_incomplete_tail:
        cmd.append("--drop-incomplete-tail")
    if args.label_max_episodes > 0:
        cmd.extend(["--max-episodes", str(args.label_max_episodes)])
    if args.overwrite_repos:
        cmd.append("--overwrite")
    run_cmd(cmd, log_path=log_path, cwd=Path(args.pi0_root) / "ours" / "IQL", env=env, dry_run=args.dry_run)
    state["current_awbc_data_dir"] = str(p["labeled_data"])
    state["next_stage"] = STAGE_TRAIN_AWBC
    append_history(state, int(state["iter_index"]), STAGE_LABEL, {"input_pool_raw_dir": str(pools["raw"]), "labeled_data_dir": str(p["labeled_data"])})
    return state


def stage_train_awbc(args: argparse.Namespace, state: dict[str, Any], p: dict[str, Path]) -> dict[str, Any]:
    log_path = p["logs"] / f"{STAGE_TRAIN_AWBC}.log"
    iter_index = int(state["iter_index"])
    awbc_steps = get_awbc_steps(iter_index, args.task_id)
    log_line(
        log_path,
        f"[steps] AWBC num_train_steps={awbc_steps} mode={_task_mode(args.task_id)} "
        f"iter={iter_index}",
    )
    cmd = [
        args.pi_python, "-u", str(Path(args.pi0_root) / "ours" / "AWBC" / "train_awbc.py"),
        "--awbc-repo-id", "labeled", "--hf-lerobot-home", str(p["data"]),
        "--model-dir", str(p["model_dir"]), "--base-policy-dir", state["current_policy_dir"],
        "--pi0-root", str(Path(args.pi0_root)), "--openpi-root", str(Path(args.openpi_root)),
        "--python-bin", args.pi_python, "--config-name", args.awbc_config_name,
        "--asset-id", args.asset_id, "--project-name", args.project_name,
        "--exp-name", f"iter{iter_index}_awbc", "--num-train-steps", str(awbc_steps),
        "--batch-size", str(args.awbc_batch_size), "--num-workers", str(args.awbc_num_workers),
        "--save-interval", str(args.awbc_save_interval), "--log-interval", str(args.awbc_log_interval),
        "--keep-period", args.awbc_keep_period, "--seed", str(args.seed),
        "--fsdp-devices", str(args.gpus),
        "--cache-root", args.cache_root, "--log-file", str(log_path),
    ]
    if args.norm_max_frames > 0:
        cmd.extend(["--norm-max-frames", str(args.norm_max_frames)])
    cmd.append("--wandb-enabled" if args.wandb_enabled else "--no-wandb-enabled")
    cmd.append("--overwrite" if args.overwrite_repos else "--no-overwrite")
    if args.dry_run:
        cmd.append("--dry-run")
    run_cmd(cmd, log_path=log_path, cwd=Path(args.pi0_root), env=base_env(args), dry_run=args.dry_run)
    next_policy = p["model_dir"] / "final"
    if not args.dry_run and not (next_policy / "params").exists():
        raise RuntimeError(f"Expected AWBC final checkpoint with params/ not found: {next_policy}")
    state["current_policy_dir"] = str(next_policy)
    state["next_stage"] = STAGE_COLLECT
    append_history(state, iter_index, STAGE_TRAIN_AWBC, {"policy_dir": str(next_policy), "awbc_data_dir": str(p["labeled_data"])})
    return state


def stage_collect(args: argparse.Namespace, state: dict[str, Any], p: dict[str, Path]) -> dict[str, Any]:
    iter_index = int(state["iter_index"])
    log_path = p["logs"] / f"{STAGE_COLLECT}.log"
    server_log = p["logs"] / "policy_server.log"
    mode = "qselect" if args.use_q_select else "simple"

    server_cmd = [
        args.pi_python, "-u", str(Path(args.pi0_root) / "ours" / "Qselect" / "server.py"),
        "--policy-config", args.policy_config_name, "--policy-dir", state["current_policy_dir"],
        "--sample-mode", mode, "--num-action-samples", str(args.num_action_samples),
        "--num-steps", str(args.qselect_num_steps), "--noise-scale", str(args.noise_scale),
        "--seed", str(args.seed), "--score-horizon", str(args.score_horizon),
        "--port", str(args.port),
    ]
    if args.selector_device:
        server_cmd.extend(["--selector-device", args.selector_device])
    if mode == "qselect":
        server_cmd.extend(["--critic-path", state["current_head_path"]])

    collect_cmd = [
        args.libero_python, "-u", str(Path(args.pi0_root) / "ours" / "Qselect" / "collect.py"),
        "--host", args.host, "--port", str(args.port), "--resize-size", "224",
        "--replan-steps", str(args.replan_steps), "--task-suite-name", args.task_suite_name,
        "--task-id", str(args.task_id), "--num-steps-wait", str(args.num_steps_wait),
        "--num-trials-per-task", str(args.num_trials_per_task),
        "--initial-state-offset", str(args.initial_state_offset), "--seed", str(args.seed),
        "--max-steps-override", str(args.max_steps_override), "--repo-id", "collect",
        "--video-out-path", str(p["videos"]), "--metrics-path", str(p["metrics"]),
    ]
    if args.overwrite_repos:
        collect_cmd.append("--overwrite")
    collect_cmd.append("--save-videos" if args.save_videos else "--no-save-videos")
    collect_cmd.append("--save-success" if args.save_success else "--no-save-success")
    collect_cmd.append("--save-failure" if args.save_failure else "--no-save-failure")

    log_line(log_path, f"[server] {quote_cmd(server_cmd)}")
    log_line(log_path, f"[collect] {quote_cmd(collect_cmd)}")

    if args.dry_run:
        state["last_collect_data_dir"] = str(p["collect_data"])
        state["next_stage"] = STAGE_APPEND_POOL
        return state

    env_server = base_env(args)
    env_collect = libero_env(args, p["data"])

    server_log.parent.mkdir(parents=True, exist_ok=True)
    with server_log.open("a", encoding="utf-8") as server_fp:
        server_fp.write(f"\n[{now()}] [server start] {quote_cmd(server_cmd)}\n")
        server_fp.flush()
        proc = subprocess.Popen(
            [str(x) for x in server_cmd],
            cwd=str(Path(args.openpi_root)),
            env=env_server,
            stdout=server_fp,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_for_port(args.host, args.port, proc, args.server_wait_timeout, server_log)
            run_cmd(collect_cmd, log_path=log_path, cwd=Path(args.openpi_root), env=env_collect, dry_run=False)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=20)

    if not p["collect_data"].exists():
        raise RuntimeError(f"Expected collect data not found: {p['collect_data']}")

    state["last_collect_data_dir"] = str(p["collect_data"])
    state["next_stage"] = STAGE_APPEND_POOL
    append_history(state, iter_index, STAGE_COLLECT, {"collect_data_dir": str(p["collect_data"]), "sample_mode": mode})
    return state


def stage_append_pool(args: argparse.Namespace, state: dict[str, Any], p: dict[str, Path]) -> dict[str, Any]:
    log_path = p["logs"] / f"{STAGE_APPEND_POOL}.log"
    pools = pool_paths(args)
    iter_index = int(state["iter_index"])
    collect_dir = Path(state["last_collect_data_dir"]).resolve()
    if args.dry_run:
        log_line(log_path, f"[dry-run] append {collect_dir} -> {pools['raw']}")
        state["iter_index"] = iter_index + 1
        state["next_stage"] = STAGE_FINISHED if state["iter_index"] >= int(args.iters) else STAGE_TRAIN_IQL
        return state
    if not pools["raw"].exists():
        raise FileNotFoundError(f"Pool raw does not exist: {pools['raw']}")
    if not collect_dir.exists():
        raise FileNotFoundError(f"Collect data does not exist: {collect_dir}")
    stats = merge_collector_repos([pools["raw"], collect_dir], pools["raw_tmp"], overwrite=True, log_path=log_path)
    atomic_replace_dir(pools["raw_tmp"], pools["raw"], pools["raw_backup"])
    append_sources_record(args, {"event": "append_collect", "iter": iter_index, "collect_dir": str(collect_dir), "pool_raw": str(pools["raw"]), **stats})
    append_history(state, iter_index, STAGE_APPEND_POOL, {"collect_data_dir": str(collect_dir), "pool_raw_dir": str(pools["raw"]), **stats})
    state["iter_index"] = iter_index + 1
    state["next_stage"] = STAGE_FINISHED if state["iter_index"] >= int(args.iters) else STAGE_TRAIN_IQL
    return state


STAGE_EXECUTORS = {
    STAGE_TRAIN_IQL: stage_train_iql,
    STAGE_LABEL: stage_label,
    STAGE_TRAIN_AWBC: stage_train_awbc,
    STAGE_COLLECT: stage_collect,
    STAGE_APPEND_POOL: stage_append_pool,
}


def iter_train(args: argparse.Namespace) -> None:
    args = resolve_args(args)
    workspace = Path(args.workspace)
    (workspace / "logs").mkdir(parents=True, exist_ok=True)
    Path(args.cache_root).mkdir(parents=True, exist_ok=True)

    log_line(main_log(args), "=" * 80)
    log_line(main_log(args), f"workspace={args.workspace}")
    log_line(main_log(args), f"src_dir={args.src_dir}")
    log_line(main_log(args), f"base_model={args.base_model}")
    log_line(main_log(args), f"iters={args.iters} task={args.task_suite_name}:{args.task_id}")
    log_line(main_log(args), f"gpus={args.gpus}")
    log_line(main_log(args), f"use_q_select={args.use_q_select} num_action_samples={args.num_action_samples}")
    log_line(main_log(args), f"iql_use_q_aug={args.iql_use_q_aug}")
    log_line(main_log(args), f"horizon={args.horizon} replan_steps={args.replan_steps}")

    state = load_or_init_state(args)

    while state.get("next_stage") != STAGE_FINISHED:
        stage = str(state["next_stage"])
        if stage not in STAGE_EXECUTORS:
            raise ValueError(f"Unknown stage: {stage}")
        iter_index = int(state["iter_index"])
        p = workspace_paths(args, iter_index)
        mkdir_stage_dirs(p)
        log_line(main_log(args), f"[start] iter={iter_index} stage={stage}")
        state = STAGE_EXECUTORS[stage](args, state, p)
        save_state(args, state)
        log_line(main_log(args), f"[done] iter={iter_index} stage={stage} next={state.get('next_stage')}")

    log_line(main_log(args), "[finished]")


def main() -> None:
    iter_train(parse_args())


if __name__ == "__main__":
    main()
