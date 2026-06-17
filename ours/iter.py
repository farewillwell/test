#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-resumable pi0 iterative learning driver with a virtual data pool.

Key design:
    - pool/raw is only the initial success-demo dataset converted to IQL style.
    - New collect datasets are NOT physically merged into pool/raw.
    - pool/meta/sources.jsonl records all data sources.
    - IQL/train_pool.py and IQL/label_pool.py receive the current source list via --repo-dirs.

Workspace layout:

workspace/
├── save.json
├── logs/main.log
├── pool/
│   ├── raw                  # initial converted success-demo IQL-style LeRobot repo
│   └── meta/sources.jsonl   # virtual pool source list
├── iter0/
│   ├── logs/
│   ├── iql/final.pt
│   ├── data/labeled         # success-only AWBC data labeled by iter0 critic
│   ├── data/collect         # newly collected success+failure data
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
import time
from typing import Any
from Qselect.raw_collect_to_lerobot import convert_raw_collect_to_lerobot

STAGE_TRAIN_IQL = "train_iql"
STAGE_LABEL = "label"
STAGE_TRAIN_AWBC = "train_awbc"
STAGE_COLLECT = "collect"
STAGE_APPEND_POOL = "append_pool"
STAGE_FINISHED = "finished"

STAGES = (STAGE_TRAIN_IQL, STAGE_LABEL, STAGE_TRAIN_AWBC, STAGE_COLLECT, STAGE_APPEND_POOL)
VALID_STAGES = set(STAGES) | {STAGE_FINISHED}
LIBERO_REPLAN_STEPS = int(os.environ['action_horizon'])
STEP_REWARD = -1.0
SUCCESS_TERMINAL_REWARD = 10.0
FAILURE_TERMINAL_REWARD = -100.0
QSELECT_HOST = "127.0.0.1"
QSELECT_PORT = 8000
QSELECT_SERVER_WAIT_TIMEOUT = 900.0


def _task_mode(task_id: int) -> str:
    return "single" if int(task_id) >= 0 else "multi"


def get_iql_steps(iter_index: int, task_id: int) -> int:
    base = 4000
    iter_task_add = 1000
    iter_scale = max(int(iter_index) + 1, 1)
    num_tasks = 1 if int(task_id) >= 0 else 10
    return  int(base + iter_task_add * num_tasks * iter_scale)


def get_awbc_steps(iter_index: int, task_id: int) -> int:
    base = 3500
    iter_task_add = 1000
    iter_scale = max(int(iter_index) + 1, 1)
    num_tasks = 1 if int(task_id) >= 0 else 10
    return  int(base + iter_task_add * num_tasks * iter_scale)

def check_env_for_subprocess(env: dict[str, Any], *, log_path: Path) -> None:
    bad = {k: v for k, v in env.items() if v is None}
    if bad:
        log_line(log_path, "[env-error] subprocess env contains None values:")
        for k in sorted(bad):
            log_line(log_path, f"[env-error] {k}=None")
        raise RuntimeError(
            "subprocess env contains None values: "
            + ", ".join(sorted(bad.keys()))
        )

    bad_type = {
        k: type(v).__name__
        for k, v in env.items()
        if not isinstance(v, (str, bytes, os.PathLike))
    }
    if bad_type:
        log_line(log_path, "[env-error] subprocess env contains non-string values:")
        for k in sorted(bad_type):
            log_line(log_path, f"[env-error] {k}: type={bad_type[k]} value={env[k]!r}")
        raise RuntimeError(
            "subprocess env contains non-string values: "
            + ", ".join(f"{k}:{bad_type[k]}" for k in sorted(bad_type))
        )

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pi0 iterative IQL/AWBC/Q-select with a virtual data pool.")

    # Core paths.
    p.add_argument("--workspace", required=True, help="Experiment root. All generated data/checkpoints/logs stay under it.")
    p.add_argument("--src-dir", required=True, help="Initial success-only OpenPI/LeRobot demo dataset directory.")
    p.add_argument("--base-model", required=True, help="Initial pi0 checkpoint directory containing params/.")

    # Repository / environment roots.
    p.add_argument("--pi0-root", default="", help="Defaults to parent of this script.")
    p.add_argument("--openpi-root", default="", help="Defaults to <pi0-root>/openpi.")
    p.add_argument("--pi-python", default="", help="Defaults to <openpi-root>/pi_env/bin/python.")
    p.add_argument("--libero-python", default="", help="Defaults to <openpi-root>/examples/libero/libero_env/bin/python.")

    # Iteration / resume.
    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--gpus", type=int, default=2)
    p.add_argument("--overwrite-repos", action=argparse.BooleanOptionalAction, default=True)

    # Time scale. These are intentionally fixed to 5.
    p.add_argument("--horizon", type=int, default=LIBERO_REPLAN_STEPS)
    p.add_argument("--replan-steps", type=int, default=LIBERO_REPLAN_STEPS)

    # Initial demo conversion reward.
    p.add_argument("--demo-step-reward", type=float, default=STEP_REWARD)
    p.add_argument("--demo-success-terminal-reward", type=float, default=SUCCESS_TERMINAL_REWARD)

    # IQL.
    p.add_argument("--iql-encoder-name", default="/data/aoss/heliqun/model/clip/clip-vit-base-patch32")
    p.add_argument("--iql-batch-size", type=int, default=64)
    p.add_argument("--iql-num-workers", type=int, default=4)
    p.add_argument("--iql-use-q-aug", action="store_true")
    # Label.
    p.add_argument("--label-batch-size", type=int, default=128)
    p.add_argument("--label-normalize-adv", action="store_true")
    p.add_argument("--label-clamp-adv", type=float, default=0.0)
    p.add_argument("--label-drop-incomplete-tail", action="store_true")
    p.add_argument("--label-max-episodes", type=int, default=0)

    # AWBC.
    p.add_argument("--awbc-batch-size", type=int, default=16)
    p.add_argument("--awbc-num-workers", type=int, default=4)
    p.add_argument("--norm-max-frames", type=int, default=0)
    p.add_argument("--asset-id", default="physical-intelligence/libero")
    p.add_argument("--project-name", default="openpi")
    p.add_argument("--policy-config-name", default="")

    # Q-select / collect.
    p.add_argument("--use-q-select", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-action-samples", type=int, default=16)

    p.add_argument("--task-suite-name", default="libero_goal")
    p.add_argument("--task-id", type=int, default=-1)
    p.add_argument("--num-trials-per-task", type=int, default=50)
    p.add_argument("--initial-state-offset", type=int, default=0)
    p.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-success", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-failure", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--proposal-noise-scale", type=float, default=1.0)
    p.add_argument(
        "--proposal-noise-strategy",
        default="base",
        choices=("base", "hubu", "zhengjiao", "guocaiyang"),
    )
    return p.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    workspace = Path(args.workspace)
    src_dir = Path(args.src_dir)
    base_model = Path(args.base_model)

    if not src_dir.exists():
        raise FileNotFoundError(f"--src-dir does not exist: {src_dir}")
    if not base_model.exists():
        raise FileNotFoundError(f"--base-model does not exist: {base_model}")
    if not (base_model / "params").exists():
        raise FileNotFoundError(f"--base-model must contain params/: {base_model}")
    if args.horizon != LIBERO_REPLAN_STEPS or args.replan_steps != LIBERO_REPLAN_STEPS:
        raise ValueError(f"pi action_horizon{args.horizon}, IQL horizon and replan_steps{args.replan_steps} must all be {LIBERO_REPLAN_STEPS}.")

    args.workspace = str(workspace)
    args.src_dir = str(src_dir)
    args.base_model = str(base_model)

    this_file = Path(__file__)
    pi0_root = Path(args.pi0_root) if args.pi0_root else this_file.parents[1]
    args.pi0_root = str(pi0_root)

    openpi_root = Path(args.openpi_root) if args.openpi_root else pi0_root / "openpi"
    args.openpi_root = str(openpi_root)
    args.pi_python = str(Path(args.pi_python) if args.pi_python else openpi_root / "pi_env" / "bin" / "python")
    args.libero_python = str(
        Path(args.libero_python)
        if args.libero_python
        else openpi_root / "examples" / "libero" / "libero_env" / "bin" / "python"
    )

    # Required helper scripts.
    for script in (
        Path(args.pi0_root) / "ours" / "convert_demo.py",
        Path(args.pi0_root) / "ours" / "IQL" / "train.py",
        Path(args.pi0_root) / "ours" / "IQL" / "label.py",
    ):
        if not script.exists():
            raise FileNotFoundError(f"Required script not found: {script}")

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
    env["JAX_COMPILATION_CACHE_DIR"] = env.get("JAX_COMPILATION_CACHE_DIR")
    env["CUDA_CACHE_PATH"] = env.get("CUDA_CACHE_PATH")
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


def run_cmd(cmd: list[Any], *, log_path: Path, cwd: Path, env: dict[str, str]) -> None:
    log_line(log_path, f"[cmd] {quote_cmd(cmd)}")
    log_line(log_path, f"[cwd] {cwd}")
    check_env_for_subprocess(env, log_path=log_path)
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
    deadline = time.time() + float(timeout)
    last_log_time = 0.0
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Policy server exited early with code {proc.returncode}. See {log_path}")
        try:
            with socket.create_connection((connect_host, int(port)), timeout=2.0):
                log_line(log_path, f"[server] ready at {connect_host}:{port}")
                return
        except OSError:
            now_t = time.time()
            if now_t - last_log_time > 30:
                log_line(
                    log_path,
                    f"[server] still waiting for {connect_host}:{port}; "
                    f"server_pid={proc.pid}; elapsed={timeout - (deadline - now_t):.1f}s"
                )
                last_log_time = now_t
            time.sleep(1.0)

    if proc.poll() is None:
        raise TimeoutError(
            f"Timed out waiting for policy server at {connect_host}:{port} after {timeout}s. "
            f"Server process is still alive, pid={proc.pid}. See {log_path}"
        )

    raise RuntimeError(f"Policy server exited with code {proc.returncode}. See {log_path}")

def remove_path(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def append_sources_record(args: argparse.Namespace, record: dict[str, Any]) -> None:
    pools = pool_paths(args)
    pools["meta"].mkdir(parents=True, exist_ok=True)

    record = dict(record)
    record.setdefault("time", now())

    with pools["sources"].open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def pool_source_dirs(args: argparse.Namespace) -> list[Path]:
    """
    Return virtual pool source repos in append order.

    sources.jsonl records absolute repo directories. The first source is normally
    workspace/pool/raw. Later sources are iterN/data/collect.
    """
    pools = pool_paths(args)
    sources = pools["sources"]

    if not sources.exists():
        if pools["raw"].exists():
            return [pools["raw"]]
        raise FileNotFoundError(f"Pool sources file not found: {sources}")

    repo_dirs: list[Path] = []
    seen: set[str] = set()

    with sources.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            repo_dir = record.get("repo_dir") or record.get("collect_dir") or record.get("pool_raw")

            if not repo_dir:
                raise KeyError(f"{sources}:{line_no} missing repo_dir/collect_dir/pool_raw: {record}")

            path = Path(repo_dir)
            if not path.exists():
                raise FileNotFoundError(f"Pool source from {sources}:{line_no} does not exist: {path}")

            key = str(path)
            if key not in seen:
                repo_dirs.append(path)
                seen.add(key)

    if not repo_dirs:
        raise RuntimeError(f"No valid source repos found in {sources}")

    return repo_dirs


def ensure_pool_initialized(args: argparse.Namespace, log_path: Path) -> None:
    """
    Initialize the virtual pool.

    Because load_or_init_state() only calls this when save.json does not exist,
    this function directly overwrites workspace/pool/raw.
    """
    pools = pool_paths(args)
    raw = pools["raw"]

    pools["root"].mkdir(parents=True, exist_ok=True)
    pools["meta"].mkdir(parents=True, exist_ok=True)

    # Reset stale source manifest for a fresh run.
    remove_path(pools["sources"])

    cmd = [
        args.pi_python,
        "-u",
        str(Path(args.pi0_root) / "ours" / "convert_demo.py"),
        "--input-dir",
        str(Path(args.src_dir)),
        "--output-dir",
        str(raw),
        "--fps",
        "10",
        "--task-id",
        str(max(int(args.task_id), 0)),
        "--step-reward",
        str(args.demo_step_reward),
        "--success-terminal-reward",
        str(args.demo_success_terminal_reward),
        "--overwrite",
    ]

    run_cmd(
        cmd,
        log_path=log_path,
        cwd=Path(args.pi0_root) / "ours",
        env=base_env(args),
    )

    if not raw.exists():
        raise RuntimeError(f"Expected initialized pool/raw not found: {raw}")

    append_sources_record(
        args,
        {
            "event": "init_pool_from_success_demo",
            "repo_dir": str(raw),
            "src_dir": str(Path(args.src_dir)),
            "pool_raw": str(raw),
        },
    )


def build_initial_state(args: argparse.Namespace) -> dict[str, Any]:
    pools = pool_paths(args)
    return {
        "iter_index": 0,
        "next_stage": STAGE_TRAIN_IQL,
        "pool_raw_dir": str(pools["raw"]),
        "pool_sources_path": str(pools["sources"]),
        "current_policy_dir": str(Path(args.base_model)),
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

    if save_path.exists():
        state = json.loads(save_path.read_text(encoding="utf-8"))
        if state.get("next_stage") not in VALID_STAGES:
            raise ValueError(f"Invalid next_stage in save.json: {state.get('next_stage')}")
        log_line(log, f"[resume] {save_path}: iter={state.get('iter_index')} next_stage={state.get('next_stage')}")
        return state

    ensure_pool_initialized(args, log)
    state = build_initial_state(args)
    save_state(args, state)
    return state


def stage_train_iql(args: argparse.Namespace, state: dict[str, Any], p: dict[str, Path]) -> dict[str, Any]:
    log_path = p["logs"] / f"{STAGE_TRAIN_IQL}.log"
    iter_index = int(state["iter_index"])
    iql_steps = get_iql_steps(iter_index, args.task_id)
    repo_dirs = pool_source_dirs(args)

    log_line(
        log_path,
        f"[steps] IQL max_steps={iql_steps} mode={_task_mode(args.task_id)} iter={iter_index}",
    )
    log_line(log_path, f"[pool] repo_dirs={json.dumps([str(x) for x in repo_dirs], ensure_ascii=False)}")

    cmd = [
        args.pi_python,
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes", "1",
        "--nproc-per-node", str(args.gpus),

        str(Path(args.pi0_root) / "ours" / "IQL" / "train.py"),

        "--repo-dirs",
        *[str(x) for x in repo_dirs],

        "--output-dir",
        str(p["iql_dir"]),

        "--encoder-name",
        args.iql_encoder_name,

        "--horizon",
        str(args.horizon),

        "--batch-size",
        str(args.iql_batch_size),

        "--num-workers",
        str(args.iql_num_workers),

        "--max-steps",
        str(iql_steps),

        "--seed",
        str(args.seed),
    ]

    if args.iql_use_q_aug:
        cmd.append("--use-q-aug")

    log_line(
        log_path,
        f"[batch] gpus={args.gpus} per_gpu_batch_size={args.iql_batch_size} "
        f"effective_global_batch_size={int(args.gpus) * int(args.iql_batch_size)}",
    )

    run_cmd(cmd, log_path=log_path, cwd=Path(args.pi0_root) / "ours" / "IQL", env=base_env(args))

    head = p["iql_dir"] / "final.pt"
    if not head.exists():
        raise RuntimeError(f"Expected IQL checkpoint not found: {head}")

    state["current_head_path"] = str(head)
    state["next_stage"] = STAGE_LABEL

    append_history(
        state,
        iter_index,
        STAGE_TRAIN_IQL,
        {
            "repo_dirs": [str(x) for x in repo_dirs],
            "head_path": str(head),
            "iql_steps": int(iql_steps),
            "gpus": int(args.gpus),
        },
    )
    return state


def stage_label(args: argparse.Namespace, state: dict[str, Any], p: dict[str, Path]) -> dict[str, Any]:
    log_path = p["logs"] / f"{STAGE_LABEL}.log"
    repo_dirs = pool_source_dirs(args)

    env = base_env(args)
    env["HF_LEROBOT_HOME"] = str(p["data"])

    log_line(log_path, f"[pool] repo_dirs={json.dumps([str(x) for x in repo_dirs], ensure_ascii=False)}")

    cmd = [
        args.pi_python,
        "-u",
        str(Path(args.pi0_root) / "ours" / "IQL" / "label.py"),
        "--repo-dirs",
        *[str(x) for x in repo_dirs],
        "--output-repo-id",
        "labeled",
        "--critic-path",
        state["current_head_path"],
        "--horizon",
        str(args.horizon),
        "--batch-size",
        str(args.label_batch_size),
        "--seed",
        str(args.seed),
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

    run_cmd(cmd, log_path=log_path, cwd=Path(args.pi0_root) / "ours" / "IQL", env=env)

    labeled_data_dir = p["labeled_data"]
    if not labeled_data_dir.exists():
        raise RuntimeError(f"Expected labeled data not found: {labeled_data_dir}")

    state["current_awbc_data_dir"] = str(labeled_data_dir)
    state["next_stage"] = STAGE_TRAIN_AWBC

    append_history(
        state,
        int(state["iter_index"]),
        STAGE_LABEL,
        {
            "repo_dirs": [str(x) for x in repo_dirs],
            "labeled_data_dir": str(labeled_data_dir),
        },
    )
    return state


def stage_train_awbc(args: argparse.Namespace, state: dict[str, Any], p: dict[str, Path]) -> dict[str, Any]:
    log_path = p["logs"] / f"{STAGE_TRAIN_AWBC}.log"
    iter_index = int(state["iter_index"])
    awbc_steps = get_awbc_steps(iter_index, args.task_id)

    log_line(
        log_path,
        f"[steps] AWBC num_train_steps={awbc_steps} mode={_task_mode(args.task_id)} iter={iter_index}",
    )

    cmd = [
        args.pi_python,
        "-u",
        str(Path(args.pi0_root) / "ours" / "BC" / "train_awbc.py"),
        "--data-dir",
        str(p["labeled_data"]),
        "--model-dir",
        str(p["model_dir"]),
        "--base-policy-dir",
        state["current_policy_dir"],
        "--pi0-root",
        str(Path(args.pi0_root)),
        "--openpi-root",
        str(Path(args.openpi_root)),
        "--python-bin",
        args.pi_python,

        "--steps",
        str(awbc_steps),
        "--batch-size",
        str(args.awbc_batch_size),
        "--num-workers",
        str(args.awbc_num_workers),
        "--gpus",
        str(args.gpus),
        "--seed",
        str(args.seed),
        "--log-file",
        str(log_path),
        "--config-name",
        str(args.policy_config_name),
    ]
    if args.norm_max_frames > 0:
        cmd.extend(["--norm-max-frames", str(args.norm_max_frames)])
    run_cmd(cmd, log_path=log_path, cwd=Path(args.pi0_root), env=base_env(args))

    next_policy = p["model_dir"] / "final"
    if not (next_policy / "params").exists():
        raise RuntimeError(f"Expected AWBC final checkpoint with params/ not found: {next_policy}")

    state["current_policy_dir"] = str(next_policy)
    state["next_stage"] = STAGE_COLLECT

    append_history(
        state,
        iter_index,
        STAGE_TRAIN_AWBC,
        {
            "policy_dir": str(next_policy),
            "awbc_data_dir": str(p["labeled_data"]),
            "awbc_steps": int(awbc_steps),
            "batch_size": int(args.awbc_batch_size),
            "num_workers": int(args.awbc_num_workers),
            "gpus": int(args.gpus),
            "seed": int(args.seed),
        },
    )
    return state


def stage_collect(args: argparse.Namespace, state: dict[str, Any], p: dict[str, Path]) -> dict[str, Any]:
    iter_index = int(state["iter_index"])
    log_path = p["logs"] / f"{STAGE_COLLECT}.log"
    server_log = p["logs"] / "policy_server.log"
    mode = "qselect" if args.use_q_select else "simple"
    tmp_collect_dir = p["data"] / "collect_tmp"
    final_collect_dir = p["collect_data"]
    server_cmd = [
        args.pi_python,
        "-u",
        str(Path(args.pi0_root) / "ours" / "Qselect" / "server.py"),
        "--policy-dir",
        state["current_policy_dir"],
        "--sample-mode",
        mode,
        "--num-action-samples",
        str(args.num_action_samples),
        "--seed",
        str(args.seed),
        "--policy-config",
        str(args.policy_config_name),
        "--noise-scale",
        str(args.proposal_noise_scale),
        "--noise-strategy",
        str(args.proposal_noise_strategy),
    ]

    if mode == "qselect":
        server_cmd.extend(["--critic-path", state["current_head_path"]])

    collect_cmd = [
        args.libero_python,
        "-u",
        str(Path(args.pi0_root) / "ours" / "Qselect" / "collect.py"),
        "--task-suite-name",
        args.task_suite_name,
        "--task-id",
        str(args.task_id),
        "--num-trials-per-task",
        str(args.num_trials_per_task),
        "--initial-state-offset",
        str(args.initial_state_offset),
        "--seed",
        str(args.seed),
        "--repo-id",
        "collect_tmp",
        "--video-out-path",
        str(p["videos"]),
        "--metrics-path",
        str(p["metrics"]),
    ]

    if args.overwrite_repos:
        collect_cmd.append("--overwrite")
    if args.save_videos:
        collect_cmd.append("--save-videos")
    if args.save_success:
        collect_cmd.append("--save-success")
    if args.save_failure:
        collect_cmd.append("--save-failure")

    log_line(log_path, f"[server] {quote_cmd(server_cmd)}")
    log_line(log_path, f"[collect_raw] {quote_cmd(collect_cmd)}")
    log_line(log_path, f"[collect_raw_dir] {tmp_collect_dir}")
    log_line(log_path, f"[collect_final_dir] {final_collect_dir}")

    env_server = base_env(args)
    env_server["CUDA_VISIBLE_DEVICES"] = "0"

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
            wait_for_port(QSELECT_HOST, QSELECT_PORT, proc, QSELECT_SERVER_WAIT_TIMEOUT, server_log)
            run_cmd(collect_cmd, log_path=log_path, cwd=Path(args.openpi_root), env=env_collect)
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=20)

    # Convert raw tmp data -> final LeRobot repo.
    #
    # This direct import assumes iter.py itself is running under pi_env.
    # If iter.py is run under libero_env, this will fail because lerobot is unavailable.
    log_line(log_path, f"[convert] raw tmp -> LeRobot: {tmp_collect_dir} -> {final_collect_dir}")

    convert_summary = convert_raw_collect_to_lerobot(
        input_dir=tmp_collect_dir,
        output_dir=final_collect_dir,
        fps=10,
        resize_size=224,
        overwrite=bool(args.overwrite_repos),
        cleanup_input_on_success=True,
        log_fn=lambda msg: log_line(log_path, msg),
    )

    log_line(log_path, f"[convert_summary] {json.dumps(convert_summary, ensure_ascii=False)}")

    if not final_collect_dir.exists():
        raise RuntimeError(f"Expected collect data not found after conversion: {final_collect_dir}")

    state["last_collect_data_dir"] = str(final_collect_dir)
    state["next_stage"] = STAGE_APPEND_POOL

    append_history(
        state,
        iter_index,
        STAGE_COLLECT,
        {
            "collect_data_dir": str(final_collect_dir),
            "raw_tmp_collect_dir": str(tmp_collect_dir),
            "sample_mode": mode,
            "convert_summary": convert_summary,
        },
    )
    return state


def stage_append_pool(args: argparse.Namespace, state: dict[str, Any], p: dict[str, Path]) -> dict[str, Any]:
    """
    O(1) virtual append.

    This stage does not copy/merge LeRobot data. It only records the new collect repo
    path in pool/meta/sources.jsonl. Later train/label stages consume the full
    repo list through --repo-dirs.
    """
    log_path = p["logs"] / f"{STAGE_APPEND_POOL}.log"
    pools = pool_paths(args)
    iter_index = int(state["iter_index"])

    collect_dir = Path(state["last_collect_data_dir"])

    if not pools["raw"].exists():
        raise FileNotFoundError(f"Pool raw does not exist: {pools['raw']}")
    if not collect_dir.exists():
        raise FileNotFoundError(f"Collect data does not exist: {collect_dir}")

    append_sources_record(
        args,
        {
            "event": "append_collect",
            "iter": iter_index,
            "repo_dir": str(collect_dir),
            "collect_dir": str(collect_dir),
        },
    )

    repo_dirs = pool_source_dirs(args)
    log_line(log_path, f"[append_pool] virtual sources now={json.dumps([str(x) for x in repo_dirs], ensure_ascii=False)}")

    append_history(
        state,
        iter_index,
        STAGE_APPEND_POOL,
        {
            "collect_data_dir": str(collect_dir),
            "pool_sources_path": str(pools["sources"]),
            "num_pool_sources": int(len(repo_dirs)),
        },
    )

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
