#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage-resumable pi0 iterative learning driver.

This is the pi0/OpenPI counterpart of the old OpenVLA-OFT iter-learn script.
It uses the current ``ours`` modules directly:

    prepare_iql  -> merge seed + collected LeRobot repos for IQL
    train_iql    -> ours/IQL/train.py
    label        -> ours/IQL/label.py, success-only AWBC repo with adv
    train_awbc   -> ours/AWBC/train_awbc.py, OpenPI scripts/train.py underneath
    collect      -> ours/Qselect/server.py + ours/Qselect/collect.py

The important data contract is kept explicit:
    - collector/IQL repos contain success + failure and reward/done/success
    - AWBC repos are label outputs only and contain success trajectories + adv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import time
from typing import Any

import numpy as np


STAGE_PREPARE_IQL = "prepare_iql"
STAGE_TRAIN_IQL = "train_iql"
STAGE_LABEL = "label"
STAGE_TRAIN_AWBC = "train_awbc"
STAGE_COLLECT = "collect"
STAGE_FINISHED = "finished"

STAGES = (STAGE_PREPARE_IQL, STAGE_TRAIN_IQL, STAGE_LABEL, STAGE_TRAIN_AWBC, STAGE_COLLECT)
VALID_STAGES = set(STAGES) | {STAGE_FINISHED}

IQL_REQUIRED_KEYS = {
    "image",
    "wrist_image",
    "state",
    "actions",
    "reward",
    "done",
    "success",
    "task",
    "episode_index",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run pi0 iterative IQL/AWBC/Q-select on LIBERO.")
    p.add_argument("--workspace", required=True)
    p.add_argument("--seed-iql-repo-id", default="", help="Initial collector-style LeRobot repo for IQL.")
    p.add_argument("--seed-data-dir", default="", help="Backward-compatible alias for --seed-iql-repo-id.")
    p.add_argument("--init-policy-dir", default="", help="Initial pi0 checkpoint dir with params/.")
    p.add_argument("--init-model-dir", default="", help="Backward-compatible alias for --init-policy-dir.")
    p.add_argument("--repo-prefix", default="", help="Prefix for generated local repo ids, e.g. heliqun/libero_goal.")
    p.add_argument("--hf-lerobot-home", default="", help="Parent directory for all local LeRobot repos.")

    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--merge-iql-history", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite-repos", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--collect-after-final", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--pi0-root", default="")
    p.add_argument("--openpi-root", default="")
    p.add_argument("--pi-python", default="")
    p.add_argument("--libero-python", default="")
    p.add_argument("--libero-config-path", default="/data/huangdi/heliqun/openvla-oft/openvla-oft/LIBERO/.libero")
    p.add_argument("--mujoco-gl", default="egl")

    p.add_argument("--iql-encoder-name", default="openai/clip-vit-base-patch32")
    p.add_argument("--iql-steps", type=int, default=4000)
    p.add_argument("--iql-batch-size", type=int, default=64)
    p.add_argument("--iql-num-workers", type=int, default=4)
    p.add_argument("--iql-hidden-dim", type=int, default=512)
    p.add_argument("--iql-num-q", type=int, default=2)
    p.add_argument("--iql-lr", type=float, default=1e-4)
    p.add_argument("--iql-use-q-aug", action="store_true")

    p.add_argument("--awbc-config-name", default="pi0_libero_awbc")
    p.add_argument("--policy-config-name", default="pi0_libero_awbc")
    p.add_argument("--awbc-steps", type=int, default=30000)
    p.add_argument("--awbc-batch-size", type=int, default=16)
    p.add_argument("--awbc-num-workers", type=int, default=4)
    p.add_argument("--awbc-save-interval", type=int, default=1000)
    p.add_argument("--awbc-log-interval", type=int, default=100)
    p.add_argument("--awbc-keep-period", default="5000")
    p.add_argument("--awbc-fsdp-devices", type=int, default=2)
    p.add_argument("--asset-id", default="physical-intelligence/libero")
    p.add_argument("--project-name", default="openpi")
    p.add_argument("--norm-max-frames", type=int, default=0)
    p.add_argument("--wandb-enabled", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--use-q-select", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num-action-samples", type=int, default=16)
    p.add_argument("--qselect-num-steps", type=int, default=10)
    p.add_argument("--noise-scale", type=float, default=1.0)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--server-wait-timeout", type=float, default=120.0)

    p.add_argument("--task-suite-name", default="libero_goal")
    p.add_argument("--task-id", type=int, default=6)
    p.add_argument("--num-trials-per-task", type=int, default=50)
    p.add_argument("--initial-state-offset", type=int, default=0)
    p.add_argument("--max-steps-override", type=int, default=-1)
    p.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--openpi-data-home", default="/data/aoss/heliqun/model/pi")
    p.add_argument("--cache-root", default="")
    p.add_argument("--start-stage", choices=tuple(VALID_STAGES), default="")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log_line(path: Path, message: str, *, also_print: bool = True) -> None:
    line = f"[{now()}] {message}"
    if also_print:
        print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("next_stage") not in VALID_STAGES:
        raise ValueError(f"Invalid next_stage in {path}: {state.get('next_stage')}")
    return state


def repo_basename(repo_id: str) -> str:
    return repo_id.rstrip("/").split("/")[-1]


def default_repo_prefix(seed_repo_id: str) -> str:
    if "/" in seed_repo_id:
        owner, name = seed_repo_id.split("/", 1)
        return f"{owner}/{repo_basename(name)}_iterlearn"
    return f"{repo_basename(seed_repo_id)}_iterlearn"


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    if not args.seed_iql_repo_id:
        args.seed_iql_repo_id = args.seed_data_dir
    if not args.init_policy_dir:
        args.init_policy_dir = args.init_model_dir
    if not args.seed_iql_repo_id:
        raise ValueError("--seed-iql-repo-id is required.")
    if not args.init_policy_dir:
        raise ValueError("--init-policy-dir is required.")
    if args.horizon != 5 or args.replan_steps != 5:
        raise ValueError("pi0 action_horizon, IQL horizon and replan_steps must all be 5.")
    if not args.repo_prefix:
        args.repo_prefix = default_repo_prefix(args.seed_iql_repo_id)

    this_file = Path(__file__).resolve()
    pi0_root = Path(args.pi0_root).resolve() if args.pi0_root else this_file.parents[1]
    args.pi0_root = str(pi0_root)
    args.openpi_root = str(Path(args.openpi_root).resolve() if args.openpi_root else pi0_root / "openpi")
    args.pi_python = str(Path(args.pi_python).resolve() if args.pi_python else Path(args.openpi_root) / "pi_env" / "bin" / "python")
    args.libero_python = str(
        Path(args.libero_python).resolve()
        if args.libero_python
        else Path(args.openpi_root) / "examples" / "libero" / "libero_env" / "bin" / "python"
    )
    args.cache_root = str(Path(args.cache_root).resolve() if args.cache_root else pi0_root / "cache")
    return args


def workspace_paths(args: argparse.Namespace, iter_index: int) -> dict[str, Path]:
    root = Path(args.workspace).resolve() / f"iter{iter_index}"
    return {
        "root": root,
        "logs": root / "logs",
        "iql_dir": root / "iql",
        "model_dir": root / "awbc_model",
        "videos": root / "collect_videos",
        "metrics": root / "collect_metrics.jsonl",
    }


def repo_id(args: argparse.Namespace, iter_index: int, kind: str) -> str:
    return f"{args.repo_prefix}_iter{iter_index}_{kind}"


def main_log(args: argparse.Namespace) -> Path:
    return Path(args.workspace).resolve() / "logs" / "main.log"


def state_path(args: argparse.Namespace) -> Path:
    return Path(args.workspace).resolve() / "save.json"


def base_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.hf_lerobot_home:
        env["HF_LEROBOT_HOME"] = str(Path(args.hf_lerobot_home).resolve())
    env["OPENPI_DATA_HOME"] = args.openpi_data_home
    env["PYTHONUNBUFFERED"] = "1"
    env["JAX_COMPILATION_CACHE_DIR"] = env.get("JAX_COMPILATION_CACHE_DIR", str(Path(args.cache_root) / "jax"))
    env["CUDA_CACHE_PATH"] = env.get("CUDA_CACHE_PATH", str(Path(args.cache_root) / "cuda"))
    env["CUDA_CACHE_MAXSIZE"] = env.get("CUDA_CACHE_MAXSIZE", "2147483648")
    env["JAX_ENABLE_COMPILATION_CACHE"] = env.get("JAX_ENABLE_COMPILATION_CACHE", "true")
    env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = env.get("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    return env


def libero_env(args: argparse.Namespace) -> dict[str, str]:
    env = base_env(args)
    env["LIBERO_CONFIG_PATH"] = args.libero_config_path
    env["MUJOCO_GL"] = args.mujoco_gl
    old_pp = env.get("PYTHONPATH", "")
    openpi_root = Path(args.openpi_root)
    pieces = [
        str(openpi_root / "third_party" / "libero"),
        str(openpi_root / "packages" / "openpi-client" / "src"),
        str(openpi_root / "src"),
        old_pp,
    ]
    env["PYTHONPATH"] = ":".join(p for p in pieces if p)
    return env


def run_cmd(cmd: list[str], *, log_path: Path, cwd: Path | None, env: dict[str, str], dry_run: bool) -> None:
    log_line(log_path, "=" * 100)
    log_line(log_path, f"[cmd] {quote_cmd(cmd)}")
    if cwd is not None:
        log_line(log_path, f"[cwd] {cwd}")
    if dry_run:
        return

    with subprocess.Popen(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        with log_path.open("a", encoding="utf-8") as f:
            for line in proc.stdout:
                print(line, end="", flush=True)
                f.write(line)
                f.flush()
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


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


def to_scalar(x: Any) -> Any:
    arr = to_numpy(x)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return x


def decode_task(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if isinstance(x, str):
        return x
    value = to_scalar(x)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def image_hwc_uint8(x: Any) -> np.ndarray:
    arr = to_numpy(x)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got {arr.shape}")
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
        raise KeyError(f"{repo} missing required keys {missing}. keys={sorted(sample.keys())}")


def create_merged_iql_repo(args: argparse.Namespace, source_repo_ids: list[str], output_repo_id: str, log_path: Path) -> str:
    if args.dry_run:
        log_line(log_path, f"[dry-run] merge IQL repos {source_repo_ids} -> {output_repo_id}")
        return output_repo_id

    if args.hf_lerobot_home:
        os.environ["HF_LEROBOT_HOME"] = str(Path(args.hf_lerobot_home).resolve())
    sys.path.insert(0, str(Path(args.openpi_root) / "src"))
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    output_path = HF_LEROBOT_HOME / output_repo_id
    if output_path.exists():
        if args.overwrite_repos:
            shutil.rmtree(output_path)
        else:
            log_line(log_path, f"[prepare_iql] reuse existing merged repo: {output_repo_id}")
            return output_repo_id

    first_ds = LeRobotDataset(source_repo_ids[0])
    if len(first_ds) == 0:
        raise RuntimeError(f"Empty source repo: {source_repo_ids[0]}")
    first = first_ds[0]
    require_keys(first, IQL_REQUIRED_KEYS, repo=source_repo_ids[0])
    image_shape = image_hwc_uint8(first["image"]).shape
    wrist_shape = image_hwc_uint8(first["wrist_image"]).shape

    out = LeRobotDataset.create(
        repo_id=output_repo_id,
        robot_type="libero",
        fps=10,
        features={
            "image": {"dtype": "image", "shape": tuple(image_shape), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": tuple(wrist_shape), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
            "reward": {"dtype": "float32", "shape": (1,), "names": ["reward"]},
            "done": {"dtype": "int64", "shape": (1,), "names": ["done"]},
            "success": {"dtype": "int64", "shape": (1,), "names": ["success"]},
        },
        use_videos=False,
    )

    total_frames = 0
    total_episodes = 0
    try:
        for repo in source_repo_ids:
            ds = LeRobotDataset(repo)
            by_ep: dict[int, list[int]] = defaultdict(list)
            for idx in range(len(ds)):
                sample = ds[idx]
                require_keys(sample, IQL_REQUIRED_KEYS, repo=repo)
                by_ep[int(to_scalar(sample["episode_index"]))].append(idx)

            for ep in sorted(by_ep):
                for idx in by_ep[ep]:
                    sample = ds[idx]
                    out.add_frame(
                        {
                            "image": image_hwc_uint8(sample["image"]),
                            "wrist_image": image_hwc_uint8(sample["wrist_image"]),
                            "state": to_numpy(sample["state"]).astype(np.float32).reshape(-1),
                            "actions": to_numpy(sample["actions"]).astype(np.float32).reshape(-1),
                            "reward": np.asarray([float(to_scalar(sample["reward"]))], dtype=np.float32),
                            "done": np.asarray([int(bool(to_scalar(sample["done"])))], dtype=np.int64),
                            "success": np.asarray([int(bool(to_scalar(sample["success"])))], dtype=np.int64),
                            "task": decode_task(sample["task"]),
                        }
                    )
                    total_frames += 1
                out.save_episode()
                total_episodes += 1
    finally:
        if hasattr(out, "stop_image_writer"):
            out.stop_image_writer()

    log_line(log_path, f"[prepare_iql] wrote {output_repo_id}: episodes={total_episodes} frames={total_frames}")
    return output_repo_id


def build_initial_state(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "version": 2,
        "iter_index": 0,
        "next_stage": STAGE_PREPARE_IQL,
        "current_policy_dir": str(Path(args.init_policy_dir).resolve()),
        "seed_iql_repo_id": args.seed_iql_repo_id,
        "collected_repo_ids": [],
        "current_iql_repo_id": args.seed_iql_repo_id,
        "current_head_path": "",
        "current_awbc_repo_id": "",
        "history": [],
    }


def append_history(state: dict[str, Any], iter_index: int, stage: str, extra: dict[str, Any] | None = None) -> None:
    item = {"iter": iter_index, "stage": stage, "time": now()}
    if extra:
        item.update(extra)
    state.setdefault("history", []).append(item)


def stage_prepare_iql(args: argparse.Namespace, state: dict[str, Any], p: dict[str, Path]) -> dict[str, Any]:
    log_path = p["logs"] / f"{STAGE_PREPARE_IQL}.log"
    iter_index = int(state["iter_index"])
    sources = [state["seed_iql_repo_id"], *state.get("collected_repo_ids", [])]
    if args.merge_iql_history:
        current_iql = create_merged_iql_repo(args, sources, repo_id(args, iter_index, "iql"), log_path)
    else:
        current_iql = sources[-1]
        log_line(log_path, f"[prepare_iql] merge disabled; using latest repo: {current_iql}")
    state["current_iql_repo_id"] = current_iql
    state["next_stage"] = STAGE_TRAIN_IQL
    append_history(state, iter_index, STAGE_PREPARE_IQL, {"iql_repo_id": current_iql, "sources": sources})
    return state


def stage_train_iql(args: argparse.Namespace, state: dict[str, Any], p: dict[str, Path]) -> dict[str, Any]:
    log_path = p["logs"] / f"{STAGE_TRAIN_IQL}.log"
    cmd = [
        args.pi_python,
        "-u",
        str(Path(args.pi0_root) / "ours" / "IQL" / "train.py"),
        "--repo-id",
        state["current_iql_repo_id"],
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
        str(args.iql_steps),
        "--hidden-dim",
        str(args.iql_hidden_dim),
        "--num-q",
        str(args.iql_num_q),
        "--lr",
        str(args.iql_lr),
        "--seed",
        str(args.seed),
    ]
    if args.iql_use_q_aug:
        cmd.append("--use-q-aug")
    run_cmd(cmd, log_path=log_path, cwd=Path(args.pi0_root) / "ours" / "IQL", env=base_env(args), dry_run=args.dry_run)

    head = p["iql_dir"] / "final.pt"
    if not args.dry_run and not head.exists():
        raise RuntimeError(f"Expected IQL checkpoint not found: {head}")
    state["current_head_path"] = str(head)
    state["next_stage"] = STAGE_LABEL
    append_history(state, int(state["iter_index"]), STAGE_TRAIN_IQL, {"head_path": str(head)})
    return state


def stage_label(args: argparse.Namespace, state: dict[str, Any], p: dict[str, Path]) -> dict[str, Any]:
    log_path = p["logs"] / f"{STAGE_LABEL}.log"
    iter_index = int(state["iter_index"])
    awbc_repo = repo_id(args, iter_index, "awbc_labeled")
    cmd = [
        args.pi_python,
        "-u",
        str(Path(args.pi0_root) / "ours" / "IQL" / "label.py"),
        "--input-repo-id",
        state["current_iql_repo_id"],
        "--output-repo-id",
        awbc_repo,
        "--critic-path",
        state["current_head_path"],
        "--horizon",
        str(args.horizon),
        "--batch-size",
        "128",
        "--seed",
        str(args.seed),
    ]
    if args.overwrite_repos:
        cmd.append("--overwrite")
    run_cmd(cmd, log_path=log_path, cwd=Path(args.pi0_root) / "ours" / "IQL", env=base_env(args), dry_run=args.dry_run)
    state["current_awbc_repo_id"] = awbc_repo
    state["next_stage"] = STAGE_TRAIN_AWBC
    append_history(state, iter_index, STAGE_LABEL, {"awbc_repo_id": awbc_repo})
    return state


def stage_train_awbc(args: argparse.Namespace, state: dict[str, Any], p: dict[str, Path]) -> dict[str, Any]:
    log_path = p["logs"] / f"{STAGE_TRAIN_AWBC}.log"
    iter_index = int(state["iter_index"])
    cmd = [
        args.pi_python,
        "-u",
        str(Path(args.pi0_root) / "ours" / "AWBC" / "train_awbc.py"),
        "--awbc-repo-id",
        state["current_awbc_repo_id"],
        "--model-dir",
        str(p["model_dir"]),
        "--base-policy-dir",
        state["current_policy_dir"],
        "--config-name",
        args.awbc_config_name,
        "--asset-id",
        args.asset_id,
        "--project-name",
        args.project_name,
        "--exp-name",
        f"iter{iter_index}_awbc",
        "--num-train-steps",
        str(args.awbc_steps),
        "--batch-size",
        str(args.awbc_batch_size),
        "--num-workers",
        str(args.awbc_num_workers),
        "--save-interval",
        str(args.awbc_save_interval),
        "--log-interval",
        str(args.awbc_log_interval),
        "--keep-period",
        args.awbc_keep_period,
        "--seed",
        str(args.seed),
        "--fsdp-devices",
        str(args.awbc_fsdp_devices),
        "--openpi-data-home",
        args.openpi_data_home,
        "--cache-root",
        args.cache_root,
        "--log-file",
        str(log_path),
    ]
    if args.hf_lerobot_home:
        cmd.extend(["--hf-lerobot-home", args.hf_lerobot_home])
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
    is_last = iter_index >= int(args.iters) - 1
    state["next_stage"] = STAGE_COLLECT if (not is_last or args.collect_after_final) else STAGE_FINISHED
    append_history(state, iter_index, STAGE_TRAIN_AWBC, {"policy_dir": str(next_policy)})
    return state


def wait_for_port(host: str, port: int, proc: subprocess.Popen, timeout: float, log_path: Path) -> None:
    connect_host = "127.0.0.1" if host in {"0.0.0.0", ""} else host
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Policy server exited early with code {proc.returncode}. Check {log_path}")
        try:
            with socket.create_connection((connect_host, int(port)), timeout=2.0):
                return
        except OSError:
            time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for policy server at {connect_host}:{port}")


def stage_collect(args: argparse.Namespace, state: dict[str, Any], p: dict[str, Path]) -> dict[str, Any]:
    iter_index = int(state["iter_index"])
    log_path = p["logs"] / f"{STAGE_COLLECT}.log"
    server_log = p["logs"] / "policy_server.log"
    collect_repo = repo_id(args, iter_index, "collect")
    mode = "qselect" if args.use_q_select else "simple"

    server_cmd = [
        args.pi_python,
        "-u",
        str(Path(args.pi0_root) / "ours" / "Qselect" / "server.py"),
        "--policy-config",
        args.policy_config_name,
        "--policy-dir",
        state["current_policy_dir"],
        "--sample-mode",
        mode,
        "--num-action-samples",
        str(args.num_action_samples),
        "--num-steps",
        str(args.qselect_num_steps),
        "--noise-scale",
        str(args.noise_scale),
        "--seed",
        str(args.seed),
        "--port",
        str(args.port),
    ]
    if mode == "qselect":
        server_cmd.extend(["--critic-path", state["current_head_path"]])

    collect_cmd = [
        args.libero_python,
        "-u",
        str(Path(args.pi0_root) / "ours" / "Qselect" / "collect.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
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
        "--max-steps-override",
        str(args.max_steps_override),
        "--repo-id",
        collect_repo,
        "--replan-steps",
        str(args.replan_steps),
        "--video-out-path",
        str(p["videos"]),
        "--metrics-path",
        str(p["metrics"]),
    ]
    if args.overwrite_repos:
        collect_cmd.append("--overwrite")
    collect_cmd.append("--save-videos" if args.save_videos else "--no-save-videos")

    log_line(log_path, f"[server] {quote_cmd(server_cmd)}")
    log_line(log_path, f"[collect] {quote_cmd(collect_cmd)}")
    if args.dry_run:
        state.setdefault("collected_repo_ids", []).append(collect_repo)
        state["iter_index"] = iter_index + 1
        state["current_iql_repo_id"] = collect_repo
        state["next_stage"] = STAGE_FINISHED if state["iter_index"] >= int(args.iters) else STAGE_PREPARE_IQL
        return state

    env_server = base_env(args)
    env_collect = libero_env(args)
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

    state.setdefault("collected_repo_ids", []).append(collect_repo)
    state["current_iql_repo_id"] = collect_repo
    state["iter_index"] = iter_index + 1
    state["next_stage"] = STAGE_FINISHED if state["iter_index"] >= int(args.iters) else STAGE_PREPARE_IQL
    append_history(state, iter_index, STAGE_COLLECT, {"collect_repo_id": collect_repo, "sample_mode": mode})
    return state


STAGE_EXECUTORS = {
    STAGE_PREPARE_IQL: stage_prepare_iql,
    STAGE_TRAIN_IQL: stage_train_iql,
    STAGE_LABEL: stage_label,
    STAGE_TRAIN_AWBC: stage_train_awbc,
    STAGE_COLLECT: stage_collect,
}


def iter_train(args: argparse.Namespace) -> None:
    args = resolve_args(args)
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    Path(args.cache_root, "jax").mkdir(parents=True, exist_ok=True)
    Path(args.cache_root, "cuda").mkdir(parents=True, exist_ok=True)

    state_file = state_path(args)
    state = load_json(state_file) or build_initial_state(args)
    if args.start_stage:
        state["next_stage"] = args.start_stage
    save_json(state_file, state)

    log_line(main_log(args), f"workspace={workspace}")
    log_line(main_log(args), f"repo_prefix={args.repo_prefix}")
    log_line(main_log(args), f"seed_iql_repo_id={args.seed_iql_repo_id}")
    log_line(main_log(args), f"init_policy_dir={args.init_policy_dir}")

    while state["next_stage"] != STAGE_FINISHED:
        iter_index = int(state["iter_index"])
        if iter_index >= int(args.iters):
            state["next_stage"] = STAGE_FINISHED
            save_json(state_file, state)
            break

        stage = str(state["next_stage"])
        paths = workspace_paths(args, iter_index)
        paths["root"].mkdir(parents=True, exist_ok=True)
        paths["logs"].mkdir(parents=True, exist_ok=True)

        state["running_stage"] = stage
        state["running_iter"] = iter_index
        save_json(state_file, state)
        log_line(main_log(args), f"[start] iter={iter_index} stage={stage}")

        state = STAGE_EXECUTORS[stage](args, state, paths)

        state.pop("running_stage", None)
        state.pop("running_iter", None)
        state["last_completed_stage"] = stage
        save_json(state_file, state)
        log_line(main_log(args), f"[done] iter={iter_index} stage={stage} next={state['next_stage']}")

    log_line(main_log(args), f"finished. save file: {state_file}")


def main() -> None:
    iter_train(parse_args())


if __name__ == "__main__":
    main()
