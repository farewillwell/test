#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


def resize_image(img: np.ndarray, size: int) -> np.ndarray:
    """Convert image to uint8 HWC RGB and resize to [size, size, 3]."""
    img = np.asarray(img)

    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected image shape [H, W, 3], got {img.shape}")

    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            if img.max() <= 1.0:
                img = img * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)

    if img.shape[0] != size or img.shape[1] != size:
        img = np.asarray(Image.fromarray(img).resize((size, size), Image.BILINEAR))

    return img


def iter_hdf5_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    files = sorted(input_path.rglob("*.hdf5")) + sorted(input_path.rglob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5/.h5 files found under {input_path}")
    return files


def infer_task_name(h5_path: Path) -> str:
    """
    Your files look like:
      success/<task_name>_demo.hdf5
      fail/<task_name>_demo.hdf5
    Use <task_name> as task text.
    """
    name = h5_path.stem
    if name.endswith("_demo"):
        name = name[: -len("_demo")]
    return name.replace("_", " ")


def load_demo(demo_group: h5py.Group) -> dict[str, np.ndarray]:
    if "obs" not in demo_group:
        raise KeyError(f"{demo_group.name} does not contain group 'obs'")

    obs = demo_group["obs"]

    if "agentview_rgb" not in obs:
        raise KeyError(f"{demo_group.name}/obs does not contain 'agentview_rgb'")

    required = ["actions", "rewards", "is_terminal", "from_success"]
    for key in required:
        if key not in demo_group:
            raise KeyError(f"{demo_group.name} does not contain '{key}'")

    agentview_rgb = obs["agentview_rgb"][:]
    actions = demo_group["actions"][:].astype(np.float32)
    rewards = demo_group["rewards"][:].astype(np.float32)
    is_terminal = demo_group["is_terminal"][:].astype(np.bool_)
    from_success = demo_group["from_success"][:].astype(np.bool_)

    T = min(
        len(agentview_rgb),
        len(actions),
        len(rewards),
        len(is_terminal),
        len(from_success),
    )

    agentview_rgb = agentview_rgb[:T]
    actions = actions[:T]
    rewards = rewards[:T]
    is_terminal = is_terminal[:T]
    from_success = from_success[:T]

    if actions.ndim != 2:
        raise ValueError(f"{demo_group.name}: actions should be [T, A], got {actions.shape}")

    return {
        "agentview_rgb": agentview_rgb,
        "actions": actions,
        "rewards": rewards,
        "is_terminal": is_terminal,
        "from_success": from_success,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="HDF5 file or directory containing success/fail hdf5 files.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="LeRobot repo id, e.g. heliqun/simple_libero_rollouts.",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = HF_LEROBOT_HOME / args.repo_id

    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_path} already exists. Use --overwrite to remove it.")
        shutil.rmtree(output_path)

    h5_files = iter_hdf5_files(input_path)
    print(f"Found {len(h5_files)} HDF5 files.")

    # Infer action dim from the first valid demo.
    action_dim = None
    for h5_path in h5_files:
        with h5py.File(h5_path, "r") as f:
            if "data" not in f:
                continue
            for demo_name in sorted(f["data"].keys()):
                demo = f["data"][demo_name]
                if "actions" in demo:
                    action_dim = int(demo["actions"].shape[-1])
                    break
        if action_dim is not None:
            break

    if action_dim is None:
        raise RuntimeError("Could not infer action_dim from any HDF5 file.")

    print(f"Inferred action_dim = {action_dim}")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="libero",
        fps=args.fps,
        features={
            "agentview_rgb": {
                "dtype": "image",
                "shape": (args.image_size, args.image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": ["actions"],
            },
            "rewards": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["rewards"],
            },
            "is_terminal": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["is_terminal"],
            },
            "from_success": {
                "dtype": "bool",
                "shape": (1,),
                "names": ["from_success"],
            },
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )

    total_episodes = 0
    total_frames = 0

    for h5_path in h5_files:
        task_text = infer_task_name(h5_path)
        split_name = h5_path.parent.name  # usually success or fail

        print(f"Converting: {h5_path} | task='{task_text}' | split='{split_name}'")

        with h5py.File(h5_path, "r") as f:
            if "data" not in f:
                print(f"[SKIP] {h5_path}: no 'data' group")
                continue

            data_group = f["data"]
            for demo_name in sorted(data_group.keys()):
                demo_group = data_group[demo_name]

                try:
                    ep = load_demo(demo_group)
                except Exception as e:
                    print(f"[SKIP] {h5_path}:{demo_name}: {e}")
                    continue

                T = len(ep["actions"])
                for t in range(T):
                    dataset.add_frame(
                        {
                            "agentview_rgb": resize_image(ep["agentview_rgb"][t], args.image_size),
                            "actions": ep["actions"][t].astype(np.float32),
                            "rewards": np.asarray([ep["rewards"][t]], dtype=np.float32),
                            "is_terminal": np.asarray([ep["is_terminal"][t]], dtype=np.bool_),
                            "from_success": np.asarray([ep["from_success"][t]], dtype=np.bool_),
                            # LeRobot uses this as episode/task metadata.
                            # This is inferred from file name, not an extra observation field.
                            "task": task_text,
                        }
                    )

                dataset.save_episode()
                total_episodes += 1
                total_frames += T

    print("Done.")
    print(f"Saved to: {output_path}")
    print(f"episodes = {total_episodes}")
    print(f"frames = {total_frames}")


if __name__ == "__main__":
    main()