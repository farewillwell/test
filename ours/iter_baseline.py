#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage-resumable iterative SFT baseline for pi0/OpenPI on LIBERO.

This baseline deliberately excludes IQL, advantage labeling, AWBC, and Q-select.
The only state-machine stages are:

    train_sft -> collect -> append_success_sft

Per iteration:
  1. materialize the virtual success-only SFT pool into iterN/data/sft
  2. train ordinary SFT into iterN/sft_model/final/params
  3. collect success + failure rollouts through the existing server/collector in
     server sample-mode=simple, without selector-side scoring
  4. extract only successful episodes into iterN/data/collect_success_sft and add
     that repo to pool/meta/sft_sources.jsonl

The pool manifest records success-only ordinary SFT repos. Failures can remain in
iterN/data/collect for metrics/statistics, but they never enter the SFT pool.
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
import time
from typing import Any

import numpy as np
import torch
from PIL import Image


STAGE_TRAIN_SFT = "train_sft"
STAGE_COLLECT = "collect"
STAGE_APPEND_SUCCESS_SFT = "append_success_sft"
STAGE_FINISHED = "finished"

STAGES = (STAGE_TRAIN_SFT, STAGE_COLLECT, STAGE_APPEND_SUCCESS_SFT)
VALID_STAGES = set(STAGES) | {STAGE_FINISHED}

CONFIG_NAME = "pi0_libero"
ASSET_ID = "physical-intelligence/libero"
PROJECT_NAME = "openpi"

IMAGE_KEY = "image"
WRIST_IMAGE_KEY = "wrist_image"
STATE_KEY = "state"
ACTION_KEY = "actions"
TASK_KEY = "task"
EPISODE_KEY = "episode_index"
SUCCESS_KEY = "success"
FPS = 10


def get_sft_steps(iter_index: int, task_id: int) -> int:
    base = 3000
    iter_task_add = 2000
    iter_scale = max(int(iter_index) + 1, 1)
    num_tasks = 1 if int(task_id) >= 0 else 10
    return int(base + iter_task_add * num_tasks * iter_scale)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a pi0/OpenPI iterative SFT baseline.")

    p.add_argument("--workspace", required=True)
    p.add_argument("--src-dir", required=True)
    p.add_argument("--base-model", required=True)

    p.add_argument("--pi0-root", default="")
    p.add_argument("--openpi-root", default="")
    p.add_argument("--pi-python", default="")
    p.add_argument("--libero-python", default="")

    p.add_argument("--iters", type=int, default=4)
    p.add_argument("--gpus", type=int, default=2)
    p.add_argument("--sft-batch-size", type=int, default=16)
    p.add_argument("--sft-num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--task-suite-name", default="libero_goal")
    p.add_argument("--task-id", type=int, default=-1)
    p.add_argument("--num-trials-per-task", type=int, default=50)
    p.add_argument("--initial-state-offset", type=int, default=0)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--replan-steps", type=int, default=5)

    p.add_argument("--num-action-samples", type=int, default=16)
    p.add_argument("--qselect-num-steps", type=int, default=10)
    p.add_argument("--noise-scale", type=float, default=1.0)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--server-wait-timeout", type=float, default=180.0)

    p.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--save-success", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-failure", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite-repos", action=argparse.BooleanOptionalAction, default=True)

    return p.parse_args()


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
        "data": root / "data",
        "sft_data": root / "data" / "sft",
        "collect_data": root / "data" / "collect",
        "collect_success_sft": root / "data" / "collect_success_sft",
        "model_dir": root / "sft_model",
        "videos": root / "collect_videos",
        "metrics": root / "collect_metrics.jsonl",
    }


def pool_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = Path(args.workspace) / "pool"
    return {
        "root": root,
        "meta": root / "meta",
        "sft_sources": root / "meta" / "sft_sources.jsonl",
    }


def mkdir_stage_dirs(p: dict[str, Path]) -> None:
    for key in ("root", "logs", "data", "model_dir", "videos"):
        p[key].mkdir(parents=True, exist_ok=True)


def check_env_for_subprocess(env: dict[str, Any], *, log_path: Path) -> None:
    bad_none = {k: v for k, v in env.items() if v is None}
    if bad_none:
        for k in sorted(bad_none):
            log_line(log_path, f"[env-error] {k}=None")
        raise RuntimeError("subprocess env contains None values: " + ", ".join(sorted(bad_none)))

    bad_type = {
        k: type(v).__name__
        for k, v in env.items()
        if not isinstance(v, (str, bytes, os.PathLike))
    }
    if bad_type:
        for k in sorted(bad_type):
            log_line(log_path, f"[env-error] {k}: type={bad_type[k]} value={env[k]!r}")
        raise RuntimeError("subprocess env contains non-string values: " + ", ".join(sorted(bad_type)))


def base_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = env.get("TOKENIZERS_PARALLELISM", "false")
    env["WANDB_MODE"] = env.get("WANDB_MODE", "offline")
    env.setdefault("CUDA_CACHE_MAXSIZE", "2147483648")
    env.setdefault("JAX_ENABLE_COMPILATION_CACHE", "true")
    env.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
    return env


def libero_env(args: argparse.Namespace, data_root: Path) -> dict[str, str]:
    env = base_env(args)

    if not env.get("LIBERO_CONFIG_PATH", ""):
        raise RuntimeError("LIBERO_CONFIG_PATH must be set in the shell environment.")
    if not env.get("MUJOCO_GL", ""):
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


def remove_path(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, Image.Image):
        return np.asarray(x)
    return np.asarray(x)


def to_scalar(x: Any) -> Any:
    if torch.is_tensor(x):
        if x.numel() == 1:
            return x.detach().cpu().item()
        return x.detach().cpu().numpy()
    if isinstance(x, (int, float, bool, str, bytes)):
        return x
    arr = np.asarray(x)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr


def decode_task(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if isinstance(x, str):
        return x
    if torch.is_tensor(x):
        if x.numel() == 1:
            return decode_task(x.detach().cpu().item())
        return str(x.detach().cpu().numpy())
    arr = np.asarray(x)
    if arr.shape == ():
        return decode_task(arr.item())
    if arr.size == 1:
        return decode_task(arr.reshape(-1)[0])
    return str(x)


def require_key(sample: dict[str, Any], key: str) -> Any:
    if key not in sample:
        raise KeyError(f"Missing `{key}`. Available keys={sorted(sample.keys())}")
    return sample[key]


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


def float_vec(x: Any, name: str) -> np.ndarray:
    arr = to_numpy(x).astype(np.float32).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


def int_success(x: Any) -> int:
    v = to_scalar(x)
    if isinstance(v, np.ndarray):
        v = v.reshape(-1)[0]
    return int(v)


def assert_lerobot_repo_dir(repo_dir: Path, *, name: str) -> None:
    info_path = repo_dir / "meta" / "info.json"
    if not repo_dir.exists():
        raise FileNotFoundError(f"{name} directory does not exist: {repo_dir}")
    if not info_path.exists():
        raise FileNotFoundError(
            f"{name} is not a valid local LeRobot repo: {repo_dir}\nExpected metadata file: {info_path}"
        )


def open_lerobot_dataset(repo_dir: Path):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    assert_lerobot_repo_dir(repo_dir, name="repo")
    return LeRobotDataset(repo_dir.name, root=repo_dir)


def _create_lerobot_dataset_with_root(
    *,
    repo_id: str,
    root: Path,
    robot_type: str,
    fps: int,
    features: dict[str, Any],
):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    create_kwargs = dict(
        repo_id=repo_id,
        robot_type=robot_type,
        fps=fps,
        features=features,
        use_videos=False,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    try:
        return LeRobotDataset.create(root=root, **create_kwargs)
    except TypeError as e:
        if "root" not in str(e):
            raise
        os.environ["HF_LEROBOT_HOME"] = str(root.parent)
        return LeRobotDataset.create(**create_kwargs)


def create_sft_output_dataset(
    output_dir: Path,
    *,
    overwrite: bool,
    fps: int,
    image_shape: tuple[int, int, int],
    wrist_shape: tuple[int, int, int],
    state_dim: int,
    action_dim: int,
):
    output_dir = Path(output_dir)
    output_repo_id = output_dir.name

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite-repos.")
        shutil.rmtree(output_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    os.environ["HF_LEROBOT_HOME"] = str(output_dir.parent)

    features = {
        "image": {
            "dtype": "image",
            "shape": tuple(image_shape),
            "names": ["height", "width", "channel"],
        },
        "wrist_image": {
            "dtype": "image",
            "shape": tuple(wrist_shape),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["actions"],
        },
    }

    return _create_lerobot_dataset_with_root(
        repo_id=output_repo_id,
        root=output_dir,
        robot_type="libero",
        fps=fps,
        features=features,
    )


def stop_image_writer(dataset: Any) -> None:
    if hasattr(dataset, "stop_image_writer"):
        dataset.stop_image_writer()


def assert_output_created(output_dir: Path) -> None:
    info_path = output_dir / "meta" / "info.json"
    if not output_dir.exists() or not info_path.exists():
        raise RuntimeError(f"Expected LeRobot output repo not created: {output_dir}; missing {info_path}")


def group_indices_by_episode(ds: Any) -> dict[int, list[int]]:
    by_ep: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(ds)):
        sample = ds[idx]
        ep = int(to_scalar(require_key(sample, EPISODE_KEY)))
        by_ep[ep].append(idx)
    return dict(by_ep)


def first_sample_from_sources(source_dirs: list[Path]) -> dict[str, Any]:
    for source_dir in source_dirs:
        ds = open_lerobot_dataset(source_dir)
        if len(ds) > 0:
            return ds[0]
    raise RuntimeError(f"All source datasets are empty: {[str(x) for x in source_dirs]}")


def infer_sft_shapes(sample: dict[str, Any]) -> tuple[tuple[int, int, int], tuple[int, int, int], int, int]:
    image_shape = image_hwc_uint8(require_key(sample, IMAGE_KEY)).shape
    wrist_shape = image_hwc_uint8(require_key(sample, WRIST_IMAGE_KEY)).shape
    state_dim = float_vec(require_key(sample, STATE_KEY), STATE_KEY).shape[0]
    action_dim = float_vec(require_key(sample, ACTION_KEY), ACTION_KEY).shape[0]

    if state_dim != 8:
        raise ValueError(f"Expected state dim 8, got {state_dim}")
    if action_dim != 7:
        raise ValueError(f"Expected action dim 7, got {action_dim}")

    return image_shape, wrist_shape, state_dim, action_dim


def add_sft_frame(dst: Any, sample: dict[str, Any], *, src_desc: str, frame_idx: int) -> None:
    task = decode_task(require_key(sample, TASK_KEY)).strip()
    if not task:
        raise ValueError(f"Empty task at {src_desc} frame {frame_idx}")

    dst.add_frame(
        {
            "image": image_hwc_uint8(require_key(sample, IMAGE_KEY)),
            "wrist_image": image_hwc_uint8(require_key(sample, WRIST_IMAGE_KEY)),
            "state": float_vec(require_key(sample, STATE_KEY), STATE_KEY).astype(np.float32),
            "actions": float_vec(require_key(sample, ACTION_KEY), ACTION_KEY).astype(np.float32),
            "task": task,
        }
    )


def materialize_sft_pool(source_dirs: list[Path], output_dir: Path, *, overwrite: bool = True) -> dict[str, int]:
    """Merge success-only ordinary SFT LeRobot repos into one ordinary SFT repo."""
    if not source_dirs:
        raise ValueError("materialize_sft_pool requires at least one source repo")

    source_dirs = [Path(x) for x in source_dirs]
    for source_dir in source_dirs:
        assert_lerobot_repo_dir(source_dir, name="SFT source")

    first_sample = first_sample_from_sources(source_dirs)
    image_shape, wrist_shape, state_dim, action_dim = infer_sft_shapes(first_sample)

    dst = create_sft_output_dataset(
        output_dir,
        overwrite=overwrite,
        fps=FPS,
        image_shape=image_shape,
        wrist_shape=wrist_shape,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    total_episodes = 0
    total_frames = 0

    try:
        for source_dir in source_dirs:
            src = open_lerobot_dataset(source_dir)
            by_ep = group_indices_by_episode(src)
            for src_ep in sorted(by_ep):
                indices = by_ep[src_ep]
                if not indices:
                    continue
                for idx in indices:
                    add_sft_frame(dst, src[idx], src_desc=str(source_dir), frame_idx=idx)
                    total_frames += 1
                dst.save_episode()
                total_episodes += 1
    finally:
        stop_image_writer(dst)

    assert_output_created(output_dir)
    return {"episodes": int(total_episodes), "frames": int(total_frames)}


def episode_is_success(src: Any, indices: list[int]) -> bool:
    if not indices:
        return False
    flags = []
    for idx in indices:
        sample = src[idx]
        if SUCCESS_KEY not in sample:
            raise KeyError(f"Collect repo sample lacks `{SUCCESS_KEY}`. Available keys={sorted(sample.keys())}")
        flags.append(int_success(sample[SUCCESS_KEY]))
    return bool(max(flags))


def extract_success_sft_repo(collect_dir: Path, output_dir: Path, *, overwrite: bool = True) -> dict[str, int]:
    """Extract only successful episodes from a collector-style repo into an ordinary SFT repo."""
    collect_dir = Path(collect_dir)
    assert_lerobot_repo_dir(collect_dir, name="collect-dir")

    src = open_lerobot_dataset(collect_dir)
    if len(src) == 0:
        raise RuntimeError(f"Collect dataset is empty: {collect_dir}")

    first_sample = src[0]
    image_shape, wrist_shape, state_dim, action_dim = infer_sft_shapes(first_sample)

    dst = create_sft_output_dataset(
        output_dir,
        overwrite=overwrite,
        fps=FPS,
        image_shape=image_shape,
        wrist_shape=wrist_shape,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    by_ep = group_indices_by_episode(src)
    total_success_episodes = 0
    total_frames = 0
    total_seen_episodes = 0

    try:
        for src_ep in sorted(by_ep):
            indices = by_ep[src_ep]
            if not indices:
                continue
            total_seen_episodes += 1
            if not episode_is_success(src, indices):
                continue
            for idx in indices:
                add_sft_frame(dst, src[idx], src_desc=str(collect_dir), frame_idx=idx)
                total_frames += 1
            dst.save_episode()
            total_success_episodes += 1
    finally:
        stop_image_writer(dst)

    assert_output_created(output_dir)
    return {
        "seen_episodes": int(total_seen_episodes),
        "success_episodes": int(total_success_episodes),
        "frames": int(total_frames),
    }


def append_sft_source_record(args: argparse.Namespace, record: dict[str, Any]) -> None:
    pools = pool_paths(args)
    pools["meta"].mkdir(parents=True, exist_ok=True)
    record = dict(record)
    record.setdefault("time", now())
    with pools["sft_sources"].open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def sft_pool_source_dirs(args: argparse.Namespace) -> list[Path]:
    pools = pool_paths(args)
    path = pools["sft_sources"]
    if not path.exists():
        raise FileNotFoundError(f"SFT pool source manifest not found: {path}")

    repo_dirs: list[Path] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            repo_dir = record.get("repo_dir") or record.get("sft_repo_dir") or record.get("src_dir")
            if not repo_dir:
                raise KeyError(f"{path}:{line_no} missing repo_dir/sft_repo_dir/src_dir: {record}")
            repo_path = Path(repo_dir)
            assert_lerobot_repo_dir(repo_path, name=f"SFT source from {path}:{line_no}")
            key = str(repo_path)
            if key not in seen:
                repo_dirs.append(repo_path)
                seen.add(key)

    if not repo_dirs:
        raise RuntimeError(f"No valid SFT sources found in {path}")
    return repo_dirs


def ensure_pool_initialized(args: argparse.Namespace, log_path: Path) -> None:
    pools = pool_paths(args)
    pools["root"].mkdir(parents=True, exist_ok=True)
    pools["meta"].mkdir(parents=True, exist_ok=True)

    remove_path(pools["sft_sources"])

    src_dir = Path(args.src_dir)
    assert_lerobot_repo_dir(src_dir, name="initial --src-dir")

    append_sft_source_record(
        args,
        {
            "event": "init_success_demo_sft_pool",
            "repo_dir": str(src_dir),
            "src_dir": str(src_dir),
        },
    )
    log_line(log_path, f"[pool] initialized success-only SFT pool with {src_dir}")


def build_initial_state(args: argparse.Namespace) -> dict[str, Any]:
    pools = pool_paths(args)
    return {
        "version": 1,
        "iter_index": 0,
        "next_stage": STAGE_TRAIN_SFT,
        "current_policy_dir": str(Path(args.base_model)),
        "last_collect_data_dir": "",
        "last_success_sft_dir": "",
        "pool_sources_path": str(pools["sft_sources"]),
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


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    workspace = Path(args.workspace)
    src_dir = Path(args.src_dir)
    base_model = Path(args.base_model)

    if not src_dir.exists():
        raise FileNotFoundError(f"--src-dir does not exist: {src_dir}")
    if not (src_dir / "meta" / "info.json").exists():
        raise FileNotFoundError(f"--src-dir is not a local LeRobot repo: {src_dir}")
    if not base_model.exists():
        raise FileNotFoundError(f"--base-model does not exist: {base_model}")
    if not (base_model / "params").exists():
        raise FileNotFoundError(f"--base-model must contain params/: {base_model}")
    if int(args.replan_steps) <= 0:
        raise ValueError(f"--replan-steps must be positive, got {args.replan_steps}")
    if int(args.iters) <= 0:
        raise ValueError(f"--iters must be positive, got {args.iters}")

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

    required_scripts = (
        Path(args.pi0_root) / "ours" / "BC" / "train_sft.py",
        Path(args.pi0_root) / "ours" / "Qselect" / "server.py",
        Path(args.pi0_root) / "ours" / "Qselect" / "collect.py",
    )
    for script in required_scripts:
        if not script.exists():
            raise FileNotFoundError(f"Required script not found: {script}")

    return args


def stage_train_sft(args: argparse.Namespace, state: dict[str, Any], p: dict[str, Path]) -> dict[str, Any]:
    log_path = p["logs"] / f"{STAGE_TRAIN_SFT}.log"
    iter_index = int(state["iter_index"])
    sft_steps = get_sft_steps(iter_index, args.task_id)
    source_dirs = sft_pool_source_dirs(args)

    log_line(log_path, f"[pool] source_dirs={json.dumps([str(x) for x in source_dirs], ensure_ascii=False)}")
    mat_stats = materialize_sft_pool(source_dirs, p["sft_data"], overwrite=bool(args.overwrite_repos))
    log_line(log_path, f"[materialize_sft_data] output={p['sft_data']} stats={mat_stats}")
    log_line(log_path, f"[steps] SFT num_train_steps={sft_steps} iter={iter_index} task_id={args.task_id}")

    cmd = [
        args.pi_python,
        "-u",
        str(Path(args.pi0_root) / "ours" / "BC" / "train_sft.py"),
        "--data-dir",
        str(p["sft_data"]),
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
        str(sft_steps),
        "--batch-size",
        str(args.sft_batch_size),
        "--num-workers",
        str(args.sft_num_workers),
        "--gpus",
        str(args.gpus),
        "--seed",
        str(args.seed),
        "--log-file",
        str(log_path),
    ]

    run_cmd(cmd, log_path=log_path, cwd=Path(args.pi0_root), env=base_env(args))

    next_policy = p["model_dir"] / "final"
    if not (next_policy / "params").exists():
        raise RuntimeError(f"Expected SFT final checkpoint with params/ not found: {next_policy}")

    state["current_policy_dir"] = str(next_policy)
    state["next_stage"] = STAGE_COLLECT

    append_history(
        state,
        iter_index,
        STAGE_TRAIN_SFT,
        {
            "sft_data_dir": str(p["sft_data"]),
            "source_dirs": [str(x) for x in source_dirs],
            "materialize_stats": mat_stats,
            "policy_dir": str(next_policy),
            "sft_steps": int(sft_steps),
            "batch_size": int(args.sft_batch_size),
            "num_workers": int(args.sft_num_workers),
            "gpus": int(args.gpus),
            "seed": int(args.seed),
        },
    )
    return state


def stage_collect(args: argparse.Namespace, state: dict[str, Any], p: dict[str, Path]) -> dict[str, Any]:
    iter_index = int(state["iter_index"])
    log_path = p["logs"] / f"{STAGE_COLLECT}.log"
    server_log = p["logs"] / "policy_server.log"

    server_cmd = [
        args.pi_python,
        "-u",
        str(Path(args.pi0_root) / "ours" / "Qselect" / "server.py"),
        "--policy-config",
        CONFIG_NAME,
        "--policy-dir",
        state["current_policy_dir"],
        "--sample-mode",
        "simple",
        "--num-action-samples",
        str(args.num_action_samples),
        "--num-steps",
        str(args.qselect_num_steps),
        "--noise-scale",
        str(args.noise_scale),
        "--seed",
        str(args.seed),
        "--score-horizon",
        "0",
        "--port",
        str(args.port),
    ]

    collect_cmd = [
        args.libero_python,
        "-u",
        str(Path(args.pi0_root) / "ours" / "Qselect" / "collect.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--resize-size",
        "224",
        "--replan-steps",
        str(args.replan_steps),
        "--task-suite-name",
        args.task_suite_name,
        "--task-id",
        str(args.task_id),
        "--num-steps-wait",
        str(args.num_steps_wait),
        "--num-trials-per-task",
        str(args.num_trials_per_task),
        "--initial-state-offset",
        str(args.initial_state_offset),
        "--seed",
        str(args.seed),
        "--max-steps-override",
        str(args.max_steps_override),
        "--repo-id",
        "collect",
        "--video-out-path",
        str(p["videos"]),
        "--metrics-path",
        str(p["metrics"]),
    ]

    if args.overwrite_repos:
        collect_cmd.append("--overwrite")
    collect_cmd.append("--save-videos" if args.save_videos else "--no-save-videos")
    collect_cmd.append("--save-success" if args.save_success else "--no-save-success")
    collect_cmd.append("--save-failure" if args.save_failure else "--no-save-failure")

    log_line(log_path, f"[server] {quote_cmd(server_cmd)}")
    log_line(log_path, f"[collect] {quote_cmd(collect_cmd)}")

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
            wait_for_port("127.0.0.1", args.port, proc, args.server_wait_timeout, server_log)
            run_cmd(collect_cmd, log_path=log_path, cwd=Path(args.openpi_root), env=env_collect)
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
    state["next_stage"] = STAGE_APPEND_SUCCESS_SFT

    append_history(
        state,
        iter_index,
        STAGE_COLLECT,
        {
            "collect_data_dir": str(p["collect_data"]),
            "sample_mode": "simple",
            "policy_config": CONFIG_NAME,
            "policy_dir": state["current_policy_dir"],
        },
    )
    return state


def stage_append_success_sft(args: argparse.Namespace, state: dict[str, Any], p: dict[str, Path]) -> dict[str, Any]:
    log_path = p["logs"] / f"{STAGE_APPEND_SUCCESS_SFT}.log"
    iter_index = int(state["iter_index"])
    collect_dir = Path(state["last_collect_data_dir"])

    if not collect_dir.exists():
        raise FileNotFoundError(f"Collect data does not exist: {collect_dir}")

    stats = extract_success_sft_repo(
        collect_dir,
        p["collect_success_sft"],
        overwrite=bool(args.overwrite_repos),
    )
    log_line(log_path, f"[extract_success_sft_repo] output={p['collect_success_sft']} stats={stats}")

    appended = False
    if int(stats["success_episodes"]) > 0:
        append_sft_source_record(
            args,
            {
                "event": "append_collect_success_sft",
                "iter": iter_index,
                "repo_dir": str(p["collect_success_sft"]),
                "collect_dir": str(collect_dir),
                "success_episodes": int(stats["success_episodes"]),
                "frames": int(stats["frames"]),
            },
        )
        appended = True
        state["last_success_sft_dir"] = str(p["collect_success_sft"])
    else:
        state["last_success_sft_dir"] = ""
        log_line(log_path, "[append] no successful episodes; SFT pool unchanged")

    source_dirs = sft_pool_source_dirs(args)
    log_line(log_path, f"[pool] sources now={json.dumps([str(x) for x in source_dirs], ensure_ascii=False)}")

    append_history(
        state,
        iter_index,
        STAGE_APPEND_SUCCESS_SFT,
        {
            "collect_data_dir": str(collect_dir),
            "success_sft_dir": str(p["collect_success_sft"]),
            "extract_stats": stats,
            "appended": bool(appended),
            "pool_sources_path": str(pool_paths(args)["sft_sources"]),
            "num_pool_sources": int(len(source_dirs)),
        },
    )

    state["iter_index"] = iter_index + 1
    state["next_stage"] = STAGE_FINISHED if state["iter_index"] >= int(args.iters) else STAGE_TRAIN_SFT
    return state


STAGE_EXECUTORS = {
    STAGE_TRAIN_SFT: stage_train_sft,
    STAGE_COLLECT: stage_collect,
    STAGE_APPEND_SUCCESS_SFT: stage_append_success_sft,
}


def iter_train(args: argparse.Namespace) -> None:
    args = resolve_args(args)

    workspace = Path(args.workspace)
    (workspace / "logs").mkdir(parents=True, exist_ok=True)

    log_line(main_log(args), "=" * 80)
    log_line(main_log(args), "pi0/OpenPI iterative SFT baseline")
    log_line(main_log(args), f"workspace={args.workspace}")
    log_line(main_log(args), f"src_dir={args.src_dir}")
    log_line(main_log(args), f"base_model={args.base_model}")
    log_line(main_log(args), f"iters={args.iters} task={args.task_suite_name}:{args.task_id}")
    log_line(main_log(args), f"gpus={args.gpus}")
    log_line(main_log(args), f"policy_config={CONFIG_NAME}")
    log_line(main_log(args), "collect_sample_mode=simple")
    log_line(main_log(args), f"replan_steps={args.replan_steps}")

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
