#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tool utilities for converting raw LIBERO collect tmp data to a LeRobot dataset.

Input format is produced by ours/Qselect/collect.py:

input_dir/
├── meta.json
├── episodes.jsonl
└── episodes/
    ├── episode_task6_trial0.npz
    ├── episode_task6_trial1.npz
    └── ...

Output is a LeRobot repo directory, usually:

iterN/data/collect/

This module is intentionally importable and has no argparse/main entrypoint.
It should be imported by iter.py running under pi_env.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np


DEFAULT_FPS = 10
DEFAULT_RESIZE_SIZE = 224
DEFAULT_IMAGE_WRITER_THREADS = 10
DEFAULT_IMAGE_WRITER_PROCESSES = 5


LogFn = Callable[[str], None] | None


def _log(log_fn: LogFn, msg: str) -> None:
    if log_fn is not None:
        log_fn(msg)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid jsonl at {path}:{line_no}: {e}") from e

    return records


def discover_episode_files(input_dir: Path) -> list[Path]:
    manifest = input_dir / "episodes.jsonl"
    records = load_jsonl(manifest)

    files: list[Path] = []
    if records:
        for rec in records:
            rel = rec.get("file")
            if not rel:
                raise KeyError(f"Manifest record missing 'file': {rec}")

            path = input_dir / str(rel)
            if not path.exists():
                raise FileNotFoundError(f"Episode listed in manifest does not exist: {path}")

            files.append(path)
    else:
        files = sorted((input_dir / "episodes").glob("*.npz"))

    if not files:
        raise RuntimeError(f"No episode npz files found under {input_dir}")

    return files


def scalar_str(x: Any) -> str:
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return str(x.item())
        if x.size == 1:
            return str(x.reshape(-1)[0].item())
        return str(x.tolist())
    return str(x)


def _ensure_shape(path: Path, key: str, actual: tuple[int, ...], expected_tail: tuple[int, ...]) -> None:
    if actual[1:] != expected_tail:
        raise ValueError(f"{path} {key} shape should be [T,{','.join(map(str, expected_tail))}], got {actual}")


def validate_episode(ep: dict[str, np.ndarray], path: Path, resize_size: int) -> int:
    required = [
        "image",
        "wrist_image",
        "state",
        "actions",
        "reward",
        "terminal",
        "success",
        "task_id",
        "trial_id",
        "step_index",
        "task",
    ]
    missing = [k for k in required if k not in ep]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")

    n = int(ep["actions"].shape[0])
    if n <= 0:
        raise ValueError(f"{path} has empty actions")

    expected_first_dim_keys = [
        "image",
        "wrist_image",
        "state",
        "actions",
        "reward",
        "terminal",
        "success",
        "task_id",
        "trial_id",
        "step_index",
    ]
    for key in expected_first_dim_keys:
        if int(ep[key].shape[0]) != n:
            raise ValueError(
                f"{path} key={key} has inconsistent first dim: "
                f"{ep[key].shape[0]} vs actions={n}"
            )

    _ensure_shape(path, "image", tuple(ep["image"].shape), (resize_size, resize_size, 3))
    _ensure_shape(path, "wrist_image", tuple(ep["wrist_image"].shape), (resize_size, resize_size, 3))
    _ensure_shape(path, "state", tuple(ep["state"].shape), (8,))
    _ensure_shape(path, "actions", tuple(ep["actions"].shape), (7,))

    # These are saved as [T, 1] by collect.py. Be strict here, because IQL expects this layout.
    for key in ("reward", "terminal", "success", "task_id", "trial_id", "step_index"):
        _ensure_shape(path, key, tuple(ep[key].shape), (1,))

    return n


def should_convert_episode(
    ep: dict[str, np.ndarray],
    *,
    success_only: bool = False,
    failure_only: bool = False,
) -> bool:
    if success_only and failure_only:
        raise ValueError("success_only and failure_only cannot both be True.")

    success = bool(int(np.asarray(ep["success"])[0].reshape(-1)[0]))

    if success_only:
        return success
    if failure_only:
        return not success
    return True


def create_lerobot_dataset(
    *,
    repo_id: str,
    fps: int,
    resize_size: int,
    image_writer_threads: int,
    image_writer_processes: int,
):
    # Import lazily. This lets the module be imported without immediately requiring lerobot.
    # The actual conversion still must run under pi_env or another env with lerobot installed.
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="libero",
        fps=fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (resize_size, resize_size, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (resize_size, resize_size, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
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
        },
        use_videos=False,
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )


def add_episode_to_lerobot(
    dataset: Any,
    ep: dict[str, np.ndarray],
    path: Path,
    *,
    resize_size: int,
) -> int:
    n = validate_episode(ep, path, resize_size)
    task = scalar_str(ep["task"])

    for i in range(n):
        dataset.add_frame(
            {
                "image": np.asarray(ep["image"][i], dtype=np.uint8),
                "wrist_image": np.asarray(ep["wrist_image"][i], dtype=np.uint8),
                "state": np.asarray(ep["state"][i], dtype=np.float32),
                "actions": np.asarray(ep["actions"][i], dtype=np.float32),
                "reward": np.asarray(ep["reward"][i], dtype=np.float32).reshape(1),
                "terminal": np.asarray(ep["terminal"][i], dtype=np.int64).reshape(1),
                "success": np.asarray(ep["success"][i], dtype=np.int64).reshape(1),
                "task_id": np.asarray(ep["task_id"][i], dtype=np.int64).reshape(1),
                "trial_id": np.asarray(ep["trial_id"][i], dtype=np.int64).reshape(1),
                "step_index": np.asarray(ep["step_index"][i], dtype=np.int64).reshape(1),
                "task": task,
            }
        )

    dataset.save_episode()
    return n


def validate_raw_collect_dir(input_dir: str | Path) -> Path:
    input_dir = Path(input_dir).expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Raw collect input_dir does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Raw collect input_dir is not a directory: {input_dir}")

    episodes_dir = input_dir / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Raw collect input must contain episodes/: {episodes_dir}")

    if not episodes_dir.is_dir():
        raise NotADirectoryError(f"Raw collect episodes path is not a directory: {episodes_dir}")

    return input_dir


def prepare_output_dir(output_dir: str | Path, *, overwrite: bool) -> Path:
    output_dir = Path(output_dir).expanduser().resolve()

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output LeRobot dir already exists: {output_dir}. Use overwrite=True.")
        shutil.rmtree(output_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return output_dir


def convert_raw_collect_to_lerobot(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    fps: int = DEFAULT_FPS,
    resize_size: int = DEFAULT_RESIZE_SIZE,
    overwrite: bool = False,
    image_writer_threads: int = DEFAULT_IMAGE_WRITER_THREADS,
    image_writer_processes: int = DEFAULT_IMAGE_WRITER_PROCESSES,
    success_only: bool = False,
    failure_only: bool = False,
    cleanup_input_on_success: bool = False,
    log_fn: LogFn = None,
) -> dict[str, Any]:
    """
    Convert raw tmp collect data to a LeRobot repo.

    This function performs all checks internally:
      - validates raw input layout;
      - validates every episode's required keys and shapes;
      - handles output overwrite;
      - sets HF_LEROBOT_HOME so LeRobotDataset.create writes exactly to output_dir;
      - writes a conversion summary into output_dir;
      - optionally deletes input_dir after successful conversion.

    Args:
        input_dir: Raw tmp dir, e.g. iterN/data/.collect_tmp.
        output_dir: Final LeRobot repo dir, e.g. iterN/data/collect.
        cleanup_input_on_success: If True, delete input_dir only after output_dir is successfully created.

    Returns:
        A summary dict suitable for logging into iter history.
    """
    if int(fps) <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    if int(resize_size) <= 0:
        raise ValueError(f"resize_size must be positive, got {resize_size}")

    if success_only and failure_only:
        raise ValueError("success_only and failure_only cannot both be True.")

    input_dir = validate_raw_collect_dir(input_dir)
    output_dir = prepare_output_dir(output_dir, overwrite=overwrite)

    episode_files = discover_episode_files(input_dir)
    _log(log_fn, f"[raw_to_lerobot] input_dir={input_dir}")
    _log(log_fn, f"[raw_to_lerobot] output_dir={output_dir}")
    _log(log_fn, f"[raw_to_lerobot] discovered_episodes={len(episode_files)}")

    # Make LeRobotDataset.create(repo_id=...) write exactly to output_dir.
    #
    # Example:
    #   output_dir = iter0/data/collect
    #   HF_LEROBOT_HOME = iter0/data
    #   repo_id = collect
    os.environ["HF_LEROBOT_HOME"] = str(output_dir.parent)
    repo_id = output_dir.name

    dataset = create_lerobot_dataset(
        repo_id=repo_id,
        fps=int(fps),
        resize_size=int(resize_size),
        image_writer_threads=int(image_writer_threads),
        image_writer_processes=int(image_writer_processes),
    )

    num_input_episodes = 0
    num_output_episodes = 0
    num_output_frames = 0
    success_episodes = 0
    failure_episodes = 0
    skipped_episodes = 0

    try:
        for ep_path in episode_files:
            num_input_episodes += 1

            with np.load(ep_path, allow_pickle=True) as data:
                ep = {k: data[k] for k in data.files}

            if not should_convert_episode(ep, success_only=success_only, failure_only=failure_only):
                skipped_episodes += 1
                continue

            success = bool(int(np.asarray(ep["success"])[0].reshape(-1)[0]))
            if success:
                success_episodes += 1
            else:
                failure_episodes += 1

            frames = add_episode_to_lerobot(
                dataset,
                ep,
                ep_path,
                resize_size=int(resize_size),
            )
            num_output_frames += int(frames)
            num_output_episodes += 1
    finally:
        if hasattr(dataset, "stop_image_writer"):
            dataset.stop_image_writer()

    if num_output_episodes <= 0:
        raise RuntimeError(
            "Conversion produced zero episodes. "
            f"input_dir={input_dir}, success_only={success_only}, failure_only={failure_only}"
        )

    if not output_dir.exists():
        raise RuntimeError(f"Expected output LeRobot repo not found after conversion: {output_dir}")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "repo_id": str(repo_id),
        "num_input_episodes": int(num_input_episodes),
        "num_output_episodes": int(num_output_episodes),
        "num_output_frames": int(num_output_frames),
        "success_episodes": int(success_episodes),
        "failure_episodes": int(failure_episodes),
        "skipped_episodes": int(skipped_episodes),
        "fps": int(fps),
        "resize_size": int(resize_size),
        "cleanup_input_on_success": bool(cleanup_input_on_success),
    }

    summary_path = output_dir / "raw_collect_conversion_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _log(log_fn, f"[raw_to_lerobot] summary={json.dumps(summary, ensure_ascii=False)}")

    if cleanup_input_on_success:
        shutil.rmtree(input_dir)
        _log(log_fn, f"[raw_to_lerobot] removed input_dir={input_dir}")

    return summary