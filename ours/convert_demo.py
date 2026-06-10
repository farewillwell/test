#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert success-only OpenPI/LeRobot demo data into IQL-ready collector-style data.

Input LeRobot fields:
    image, wrist_image, state, actions, task, episode_index

Output LeRobot fields:
    image, wrist_image, state, actions, reward, terminal, success,
    task_id, trial_id, step_index, task

Assumption:
    All input episodes are successful demonstrations.

Reward:
    normal step: -1
    final success terminal: 10

Important local-path rule:
    This script treats --input-dir and --output-dir as actual local LeRobot repo
    directories. It does not rely on HuggingFace Hub lookup, and it does not rely
    on HF_LEROBOT_HOME to decide the output path.
"""

from __future__ import annotations

import argparse
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


IMAGE_KEY = "image"
WRIST_IMAGE_KEY = "wrist_image"
STATE_KEY = "state"
ACTION_KEY = "actions"
TASK_KEY = "task"
EPISODE_KEY = "episode_index"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, help="Input LeRobot repo directory, e.g. /path/to/demo_repo")
    p.add_argument("--output-dir", required=True, help="Output LeRobot repo directory, e.g. workspace/pool/raw")
    p.add_argument("--overwrite", action="store_true")

    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--step-reward", type=float, default=-1.0)
    p.add_argument("--success-terminal-reward", type=float, default=10.0)
    p.add_argument("--max-episodes", type=int, default=0)
    return p.parse_args()


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


def require_key(sample: dict[str, Any], key: str) -> Any:
    if key not in sample:
        raise KeyError(f"Missing `{key}`. Available keys={sorted(sample.keys())}")
    return sample[key]


def assert_lerobot_repo_dir(repo_dir: Path, *, name: str) -> None:
    info_path = repo_dir / "meta" / "info.json"
    if not repo_dir.exists():
        raise FileNotFoundError(f"{name} directory does not exist: {repo_dir}")
    if not info_path.exists():
        raise FileNotFoundError(
            f"{name} is not a valid local LeRobot repo: {repo_dir}\n"
            f"Expected metadata file: {info_path}"
        )


def open_lerobot_dataset(repo_dir: Path):
    """Open a local LeRobot repo whose root is exactly repo_dir."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    repo_dir = Path(repo_dir).expanduser().absolute()
    assert_lerobot_repo_dir(repo_dir, name="input-dir")

    # In this user's LeRobot version, root must be the actual dataset root
    # containing meta/info.json. Passing repo_dir.parent causes Hub fallback.
    return LeRobotDataset(repo_dir.name, root=repo_dir)


def _create_lerobot_dataset_with_root(
    *,
    repo_id: str,
    root: Path,
    robot_type: str,
    fps: int,
    features: dict[str, Any],
):
    """Create a LeRobot dataset at an explicit local root.

    Some LeRobot versions support root= in create(); some older variants do not.
    We prefer root=output_dir. If unsupported, we set HF_LEROBOT_HOME to the
    parent and verify that the actual output path is still output_dir.
    """
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
        # Compatibility fallback for LeRobot versions whose create() has no
        # root= argument. This is intentionally guarded and checked after use.
        if "root" not in str(e):
            raise

        os.environ["HF_LEROBOT_HOME"] = str(root.parent)
        return LeRobotDataset.create(**create_kwargs)


def create_output_dataset(
    output_dir: Path,
    *,
    overwrite: bool,
    fps: int,
    image_shape: tuple[int, int, int],
    wrist_shape: tuple[int, int, int],
    state_dim: int,
    action_dim: int,
):
    output_dir = Path(output_dir).expanduser().absolute()
    output_parent = output_dir.parent
    output_repo_id = output_dir.name

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite.")
        shutil.rmtree(output_dir)

    output_parent.mkdir(parents=True, exist_ok=True)

    # Keep this for compatibility with code paths that still consult it, but do
    # not rely on it as the primary path binding.
    os.environ["HF_LEROBOT_HOME"] = str(output_parent)

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
        "reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["reward"],
        },
        "terminal": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["terminal"],
        },
        "success": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["success"],
        },
        "task_id": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["task_id"],
        },
        "trial_id": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["trial_id"],
        },
        "step_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["step_index"],
        },
    }

    ds = _create_lerobot_dataset_with_root(
        repo_id=output_repo_id,
        root=output_dir,
        robot_type="libero",
        fps=fps,
        features=features,
    )

    actual_root = Path(getattr(ds, "root", output_dir)).expanduser().absolute()
    print(f"[convert] requested output root: {output_dir}", flush=True)
    print(f"[convert] actual dataset root: {actual_root}", flush=True)

    if actual_root != output_dir:
        print(
            f"[convert][warn] LeRobotDataset root differs from requested output-dir: "
            f"actual={actual_root} requested={output_dir}",
            flush=True,
        )

    return ds


def assert_output_created(output_dir: Path) -> None:
    output_dir = Path(output_dir).expanduser().absolute()
    info_path = output_dir / "meta" / "info.json"
    if not output_dir.exists():
        raise RuntimeError(
            f"Output directory was not created: {output_dir}\n"
            "This means LeRobotDataset.create() did not bind to --output-dir."
        )
    if not info_path.exists():
        candidates = [str(p) for p in output_dir.parent.glob("*/meta/info.json")]
        raise RuntimeError(
            f"Output directory exists but is not a valid LeRobot repo: {output_dir}\n"
            f"Expected metadata: {info_path}\n"
            f"Nearby LeRobot repos: {candidates}"
        )


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().absolute()
    output_dir = Path(args.output_dir).expanduser().absolute()

    src = open_lerobot_dataset(input_dir)

    by_ep: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(src)):
        sample = src[idx]
        ep = int(to_scalar(require_key(sample, EPISODE_KEY)))
        by_ep[ep].append(idx)

    if not by_ep:
        raise RuntimeError(f"Empty source dataset: {input_dir}")

    first_idx = next(iter(next(iter(by_ep.values()))))
    first_sample = src[first_idx]

    image_shape = image_hwc_uint8(require_key(first_sample, IMAGE_KEY)).shape
    wrist_shape = image_hwc_uint8(require_key(first_sample, WRIST_IMAGE_KEY)).shape
    state_dim = float_vec(require_key(first_sample, STATE_KEY), STATE_KEY).shape[0]
    action_dim = float_vec(require_key(first_sample, ACTION_KEY), ACTION_KEY).shape[0]

    if state_dim != 8:
        raise ValueError(f"Expected state dim 8, got {state_dim}")
    if action_dim != 7:
        raise ValueError(f"Expected action dim 7, got {action_dim}")

    dst = create_output_dataset(
        output_dir,
        overwrite=args.overwrite,
        fps=args.fps,
        image_shape=image_shape,
        wrist_shape=wrist_shape,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    total_episodes = 0
    total_frames = 0

    for out_ep, src_ep in enumerate(sorted(by_ep)):
        if args.max_episodes > 0 and total_episodes >= args.max_episodes:
            break

        indices = by_ep[src_ep]
        ep_len = len(indices)
        if ep_len <= 0:
            continue

        for local_pos, global_idx in enumerate(indices):
            sample = src[global_idx]

            image = image_hwc_uint8(require_key(sample, IMAGE_KEY))
            wrist_image = image_hwc_uint8(require_key(sample, WRIST_IMAGE_KEY))
            state = float_vec(require_key(sample, STATE_KEY), STATE_KEY)
            action = float_vec(require_key(sample, ACTION_KEY), ACTION_KEY)
            task = decode_task(require_key(sample, TASK_KEY)).strip()

            if not task:
                raise ValueError(f"Empty task at source frame {global_idx}")

            is_last = local_pos == ep_len - 1
            reward = args.success_terminal_reward if is_last else args.step_reward
            terminal = 1 if is_last else 0

            dst.add_frame(
                {
                    "image": image,
                    "wrist_image": wrist_image,
                    "state": state.astype(np.float32),
                    "actions": action.astype(np.float32),
                    "reward": np.asarray([reward], dtype=np.float32),
                    "terminal": np.asarray([terminal], dtype=np.int64),
                    "success": np.asarray([1], dtype=np.int64),
                    "task_id": np.asarray([int(args.task_id)], dtype=np.int64),
                    "trial_id": np.asarray([int(src_ep)], dtype=np.int64),
                    "step_index": np.asarray([int(local_pos)], dtype=np.int64),
                    "task": task,
                }
            )
            total_frames += 1

        dst.save_episode()
        total_episodes += 1
        print(f"[convert] src_ep={src_ep} out_ep={out_ep} len={ep_len} success=1", flush=True)

    assert_output_created(output_dir)

    print(f"[convert] saved: {output_dir}", flush=True)
    print(f"[convert] episodes={total_episodes} frames={total_frames}", flush=True)
    print(f"[convert] reward step={args.step_reward} success_terminal={args.success_terminal_reward}", flush=True)


if __name__ == "__main__":
    main()
