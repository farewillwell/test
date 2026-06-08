#!/usr/bin/env python3
"""Convert LIBERO HDF5 demos to the LeRobot layout expected by OpenPI.

The resulting dataset contains the fields used by
openpi.training.config.LeRobotLiberoDataConfig:
  image, wrist_image, state, actions, task
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing LIBERO .hdf5/.h5 files.")
    parser.add_argument("--repo-id", required=True, help="Local LeRobot repo id, e.g. heliqun/libero_goal_select_50.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require wrist image and 8D state fields instead of using OpenPI-compatible fallbacks.",
    )
    return parser.parse_args()


def iter_hdf5_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.rglob("*.hdf5")) + sorted(input_dir.rglob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5/.h5 files found under {input_dir}")
    return files


def task_from_path(path: Path) -> str:
    name = path.stem
    if name.endswith("_demo"):
        name = name[: -len("_demo")]
    return name.replace("_", " ")


def ensure_uint8_hwc(image: np.ndarray, image_size: int) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {image.shape}")

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.shape[:2] != (image_size, image_size):
        image = np.asarray(Image.fromarray(image).resize((image_size, image_size), Image.BILINEAR))
    return image


def read_state(obs: h5py.Group, length: int, *, strict: bool) -> np.ndarray:
    """Return the 8D LIBERO state used by OpenPI: ee state plus gripper state."""
    if "state" in obs:
        state = obs["state"][:length].astype(np.float32)
        if state.shape[-1] >= 8:
            return state[:, :8]

    if "ee_states" in obs and "gripper_states" in obs:
        ee_states = obs["ee_states"][:length].astype(np.float32)
        gripper_states = obs["gripper_states"][:length].astype(np.float32)
        return np.concatenate([ee_states[:, :6], gripper_states[:, :2]], axis=-1)

    if "robot_states" in obs:
        robot_states = obs["robot_states"][:length].astype(np.float32)
        if robot_states.shape[-1] >= 8:
            return robot_states[:, :8]

    if strict:
        raise KeyError("Could not build 8D state from obs/state, obs/{ee_states,gripper_states}, or obs/robot_states")

    return np.zeros((length, 8), dtype=np.float32)


def read_wrist_images(obs: h5py.Group, base_images: h5py.Dataset, length: int, *, strict: bool) -> tuple[object, bool]:
    for key in ("eye_in_hand_rgb", "wrist_image", "robot0_eye_in_hand_image"):
        if key in obs:
            return obs[key], False

    if strict:
        raise KeyError("Could not find wrist image in obs/eye_in_hand_rgb, obs/wrist_image, or obs/robot0_eye_in_hand_image")

    return np.zeros((length, *base_images.shape[1:]), dtype=base_images.dtype), True


def finish_dataset(dataset: LeRobotDataset) -> None:
    if hasattr(dataset, "stop_image_writer"):
        dataset.stop_image_writer()


def main() -> None:
    args = parse_args()
    output_path = HF_LEROBOT_HOME / args.repo_id

    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite to recreate it.")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="libero",
        fps=args.fps,
        features={
            "image": {"dtype": "image", "shape": (args.image_size, args.image_size, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (args.image_size, args.image_size, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        use_videos=False,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    total_episodes = 0
    total_frames = 0
    for h5_path in iter_hdf5_files(args.input_dir):
        task = task_from_path(h5_path)
        print(f"Converting {h5_path} | task='{task}'")
        with h5py.File(h5_path, "r") as h5_file:
            if "data" not in h5_file:
                print(f"[skip] {h5_path}: missing data group")
                continue
            for demo_name in sorted(h5_file["data"].keys()):
                demo = h5_file["data"][demo_name]
                if "obs" not in demo or "actions" not in demo:
                    print(f"[skip] {h5_path}:{demo_name}: missing obs/actions")
                    continue

                obs = demo["obs"]
                if "agentview_rgb" in obs:
                    base_images = obs["agentview_rgb"]
                elif "image" in obs:
                    base_images = obs["image"]
                elif "robot0_agentview_image" in obs:
                    base_images = obs["robot0_agentview_image"]
                else:
                    print(f"[skip] {h5_path}:{demo_name}: missing base image key")
                    continue

                actions = demo["actions"][:].astype(np.float32)
                wrist_images, used_zero_wrist = read_wrist_images(obs, base_images, len(actions), strict=args.strict)
                length = min(len(actions), len(base_images), len(wrist_images))
                actions = actions[:length, :7]
                state = read_state(obs, length, strict=args.strict)
                used_zero_state = bool(np.all(state == 0))
                if used_zero_wrist:
                    print(f"[fallback] {h5_path}:{demo_name}: missing wrist image, using zero wrist images")
                if used_zero_state:
                    print(f"[fallback] {h5_path}:{demo_name}: missing 8D state, using zero states")

                for i in range(length):
                    dataset.add_frame(
                        {
                            "image": ensure_uint8_hwc(base_images[i], args.image_size),
                            "wrist_image": ensure_uint8_hwc(wrist_images[i], args.image_size),
                            "state": state[i],
                            "actions": actions[i],
                            "task": task,
                        }
                    )
                dataset.save_episode()
                total_episodes += 1
                total_frames += length

                if args.max_episodes is not None and total_episodes >= args.max_episodes:
                    finish_dataset(dataset)
                    print(f"Reached --max-episodes={args.max_episodes}")
                    print(f"Saved to: {output_path}")
                    print(f"episodes={total_episodes} frames={total_frames}")
                    return

    finish_dataset(dataset)
    print(f"Saved to: {output_path}")
    print(f"episodes={total_episodes} frames={total_frames}")


if __name__ == "__main__":
    main()
