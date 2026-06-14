#!/usr/bin/env python3
"""Convert LIBERO HDF5 demos to the LeRobot layout expected by OpenPI.

The resulting dataset contains the fields used by
openpi.training.config.LeRobotLiberoDataConfig:
  image, wrist_image, state, actions, task

Expected OpenPI LIBERO fields:
  image:        uint8 HWC RGB, shape=(H, W, 3)
  wrist_image:  uint8 HWC RGB, shape=(H, W, 3)
  state:        float32, shape=(8,), [eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]
  actions:      float32, shape=(7,), [eef_delta(6), gripper_action(1)]
  task:         language instruction
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from PIL import Image


BASE_IMAGE_KEYS = (
    "obs/agentview_rgb",
    "obs/image",
    "obs/robot0_agentview_image",
    "obs/agentview_image",
    "agentview_rgb",
    "image",
    "robot0_agentview_image",
)

WRIST_IMAGE_KEYS = (
    "obs/eye_in_hand_rgb",
    "obs/wrist_image",
    "obs/robot0_eye_in_hand_image",
    "obs/robot0_eye_in_hand_rgb",
    "obs/eye_in_hand_image",
    "eye_in_hand_rgb",
    "wrist_image",
    "robot0_eye_in_hand_image",
    "robot0_eye_in_hand_rgb",
)

STATE_KEYS = (
    "obs/state",
    "state",
    "observation/state",
)

EE_STATE_KEYS = (
    "obs/ee_states",
    "ee_states",
)

EE_POS_KEYS = (
    "obs/ee_pos",
    "obs/eef_pos",
    "obs/robot0_eef_pos",
    "ee_pos",
    "eef_pos",
    "robot0_eef_pos",
)

EE_ORI_KEYS = (
    "obs/ee_ori",
    "obs/eef_ori",
    "obs/robot0_eef_axis_angle",
    "obs/robot0_eef_quat",
    "ee_ori",
    "eef_ori",
    "robot0_eef_axis_angle",
    "robot0_eef_quat",
)

GRIPPER_STATE_KEYS = (
    "obs/gripper_states",
    "obs/robot0_gripper_qpos",
    "gripper_states",
    "robot0_gripper_qpos",
)

ROBOT_STATE_KEYS = (
    "robot_states",
    "obs/robot_states",
)

TASK_DATASET_KEYS = (
    "language_instruction",
    "task",
    "prompt",
)

TASK_ATTR_KEYS = (
    "language_instruction",
    "task",
    "prompt",
    "task_description",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing LIBERO .hdf5/.h5 files.")
    parser.add_argument("--repo-id", required=True, help="Local LeRobot repo id, e.g. heliqun/libero_goal_select_50.")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional smoke-test limit.")

    parser.add_argument(
        "--missing-wrist-policy",
        choices=("error", "base", "zero"),
        default="error",
        help=(
            "What to do if no real wrist image exists. "
            "'error' is recommended for real OpenPI training. "
            "'base' duplicates base image as wrist image. "
            "'zero' writes black wrist images; only use for smoke tests."
        ),
    )
    parser.add_argument(
        "--missing-state-policy",
        choices=("error", "zero"),
        default="error",
        help=(
            "What to do if no 8D OpenPI-compatible state can be built. "
            "'error' is recommended for real OpenPI training. "
            "'zero' only for smoke tests."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Alias for --missing-wrist-policy=error and --missing-state-policy=error.",
    )
    parser.add_argument(
        "--allow-direct-robot-state",
        action="store_true",
        help=(
            "If robot_states has >=8 dims but is not recognized as [gripper2,pos3,quat4], "
            "allow using robot_states[:, :8] directly. Disabled by default because this can be wrong."
        ),
    )
    parser.add_argument(
        "--quat-format",
        choices=("xyzw", "wxyz"),
        default="xyzw",
        help="Quaternion convention when converting robot_states or ee_ori quaternion to axis-angle.",
    )
    parser.add_argument(
        "--no-rotate-180",
        action="store_true",
        help="Disable 180-degree image rotation. Default keeps the previous LIBERO/OpenVLA-style rotation.",
    )
    parser.add_argument(
        "--print-schema",
        action="store_true",
        help="Print HDF5 dataset keys for each file before conversion.",
    )
    return parser.parse_args()


def iter_hdf5_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.rglob("*.hdf5")) + sorted(input_dir.rglob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5/.h5 files found under {input_dir}")
    return files


def demo_sort_key(name: str) -> tuple[int, str]:
    match = re.match(r"demo_(\d+)$", name)
    if match:
        return int(match.group(1)), name
    return 10**12, name


def task_from_path(path: Path) -> str:
    name = path.stem
    if name.endswith("_demo"):
        name = name[: -len("_demo")]
    return name.replace("_", " ")


def decode_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return decode_text(value.item())
        if value.size == 1:
            return decode_text(value.reshape(-1)[0])
        return None
    if isinstance(value, str):
        return value
    return str(value)


def get_dataset(root: h5py.Group, key: str) -> h5py.Dataset | None:
    try:
        obj = root[key]
    except KeyError:
        return None
    if isinstance(obj, h5py.Dataset):
        return obj
    return None


def get_first_dataset(root: h5py.Group, keys: tuple[str, ...]) -> tuple[h5py.Dataset | None, str | None]:
    for key in keys:
        ds = get_dataset(root, key)
        if ds is not None:
            return ds, key
    return None, None


def list_dataset_paths(group: h5py.Group) -> list[str]:
    paths: list[str] = []

    def visitor(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset):
            paths.append(name)

    group.visititems(visitor)
    return sorted(paths)


def print_h5_schema(h5_file: h5py.File, h5_path: Path) -> None:
    print(f"\n========== HDF5 schema: {h5_path} ==========")
    for key in list_dataset_paths(h5_file):
        obj = h5_file[key]
        print(f"{key}: shape={obj.shape}, dtype={obj.dtype}")
    print("===========================================\n")


def ensure_uint8_hwc(image: np.ndarray, image_size: int) -> np.ndarray:
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected 3D RGB image, got shape {image.shape}")

    # Accept CHW and convert to HWC.
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))

    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {image.shape}")

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.shape[:2] != (image_size, image_size):
        image = np.asarray(Image.fromarray(image).resize((image_size, image_size), Image.BILINEAR))

    return np.ascontiguousarray(image)


def rotate_180(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(image)[::-1, ::-1])


def maybe_rotate(image: np.ndarray, *, rotate: bool) -> np.ndarray:
    return rotate_180(image) if rotate else np.asarray(image)


def quat_to_axis_angle(quat: np.ndarray, *, quat_format: str) -> np.ndarray:
    """Convert quaternion batch to axis-angle.

    Args:
        quat: shape (..., 4)
        quat_format:
            xyzw: [x, y, z, w], common in robosuite observations.
            wxyz: [w, x, y, z].
    """
    quat = np.asarray(quat, dtype=np.float32)
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion shape (..., 4), got {quat.shape}")

    if quat_format == "xyzw":
        xyz = quat[..., :3]
        w = quat[..., 3:4]
    else:
        w = quat[..., :1]
        xyz = quat[..., 1:4]

    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    xyz = xyz / norm
    w = w / norm

    # Normalize sign for shorter rotation.
    sign = np.where(w < 0, -1.0, 1.0).astype(np.float32)
    xyz = xyz * sign
    w = w * sign

    sin_half = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half, np.clip(w, -1.0, 1.0))

    axis = np.zeros_like(xyz, dtype=np.float32)
    valid = sin_half[..., 0] > 1e-8
    axis[valid] = xyz[valid] / sin_half[valid]

    return (axis * angle).astype(np.float32)


def ensure_gripper_2d(gripper: np.ndarray, source: str) -> np.ndarray:
    gripper = np.asarray(gripper, dtype=np.float32)

    if gripper.ndim == 1:
        gripper = gripper[:, None]

    if gripper.shape[-1] >= 2:
        return gripper[:, :2].astype(np.float32)

    if gripper.shape[-1] == 1:
        print(f"[warn] {source}: gripper state is 1D; duplicating it to 2D for OpenPI state.")
        return np.repeat(gripper[:, :1], 2, axis=-1).astype(np.float32)

    raise ValueError(f"{source}: invalid gripper state shape {gripper.shape}")


def read_task(h5_file: h5py.File, demo: h5py.Group, h5_path: Path) -> str:
    for key in TASK_ATTR_KEYS:
        text = decode_text(demo.attrs.get(key))
        if text:
            return text

    for key in TASK_DATASET_KEYS:
        ds = get_dataset(demo, key)
        if ds is not None:
            text = decode_text(ds[()])
            if text:
                return text

    for key in TASK_ATTR_KEYS:
        text = decode_text(h5_file.attrs.get(key))
        if text:
            return text

    return task_from_path(h5_path)


def read_base_images(demo: h5py.Group) -> tuple[h5py.Dataset, str]:
    base_images, source = get_first_dataset(demo, BASE_IMAGE_KEYS)
    if base_images is None or source is None:
        available = "\n  ".join(list_dataset_paths(demo))
        raise KeyError(
            "Could not find base image. Tried:\n  "
            + "\n  ".join(BASE_IMAGE_KEYS)
            + "\nAvailable demo datasets:\n  "
            + available
        )
    return base_images, source


def read_wrist_images(
    demo: h5py.Group,
    base_images: h5py.Dataset,
    length: int,
    *,
    missing_policy: str,
) -> tuple[Any, str]:
    wrist_images, source = get_first_dataset(demo, WRIST_IMAGE_KEYS)
    if wrist_images is not None and source is not None:
        return wrist_images, source

    if missing_policy == "base":
        print("[fallback] Missing wrist image; using base image as wrist image.")
        return base_images, "fallback:base_image_as_wrist"

    if missing_policy == "zero":
        print("[fallback] Missing wrist image; using zero wrist images.")
        return np.zeros((length, *base_images.shape[1:]), dtype=base_images.dtype), "fallback:zero_wrist"

    available = "\n  ".join(list_dataset_paths(demo))
    raise KeyError(
        "Could not find wrist image. Tried:\n  "
        + "\n  ".join(WRIST_IMAGE_KEYS)
        + "\nAvailable demo datasets:\n  "
        + available
    )


def read_state_from_ee_and_gripper(
    demo: h5py.Group,
    length: int,
    *,
    quat_format: str,
) -> tuple[np.ndarray | None, str | None]:
    ee_states_ds, ee_source = get_first_dataset(demo, EE_STATE_KEYS)
    grip_ds, grip_source = get_first_dataset(demo, GRIPPER_STATE_KEYS)

    if ee_states_ds is not None and grip_ds is not None:
        ee_states = ee_states_ds[:length].astype(np.float32)
        gripper = ensure_gripper_2d(grip_ds[:length].astype(np.float32), grip_source or "gripper")

        if ee_states.shape[-1] >= 6:
            state = np.concatenate([ee_states[:, :6], gripper], axis=-1)
            return state.astype(np.float32), f"{ee_source}+{grip_source}"

    ee_pos_ds, ee_pos_source = get_first_dataset(demo, EE_POS_KEYS)
    ee_ori_ds, ee_ori_source = get_first_dataset(demo, EE_ORI_KEYS)

    if ee_pos_ds is not None and ee_ori_ds is not None and grip_ds is not None:
        ee_pos = ee_pos_ds[:length].astype(np.float32)[:, :3]
        ee_ori = ee_ori_ds[:length].astype(np.float32)
        gripper = ensure_gripper_2d(grip_ds[:length].astype(np.float32), grip_source or "gripper")

        if ee_ori.shape[-1] >= 3 and ee_ori.shape[-1] != 4:
            axis_angle = ee_ori[:, :3]
        elif ee_ori.shape[-1] == 4:
            axis_angle = quat_to_axis_angle(ee_ori[:, :4], quat_format=quat_format)
        else:
            raise ValueError(f"{ee_ori_source}: invalid ee orientation shape {ee_ori.shape}")

        state = np.concatenate([ee_pos, axis_angle, gripper], axis=-1)
        return state.astype(np.float32), f"{ee_pos_source}+{ee_ori_source}+{grip_source}"

    return None, None


def read_state_from_robot_states(
    demo: h5py.Group,
    length: int,
    *,
    quat_format: str,
    allow_direct_robot_state: bool,
) -> tuple[np.ndarray | None, str | None]:
    robot_states_ds, source = get_first_dataset(demo, ROBOT_STATE_KEYS)
    if robot_states_ds is None or source is None:
        return None, None

    robot_states = robot_states_ds[:length].astype(np.float32)

    # Regenerated LIBERO script commonly saves:
    #   robot_states = concat([robot0_gripper_qpos(2), robot0_eef_pos(3), robot0_eef_quat(4)])
    # Convert it to OpenPI state:
    #   [eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]
    if robot_states.shape[-1] >= 9:
        gripper = robot_states[:, :2]
        ee_pos = robot_states[:, 2:5]
        ee_quat = robot_states[:, 5:9]
        axis_angle = quat_to_axis_angle(ee_quat, quat_format=quat_format)
        state = np.concatenate([ee_pos, axis_angle, gripper], axis=-1)
        return state.astype(np.float32), f"{source} interpreted as [gripper2,pos3,quat4]"

    if robot_states.shape[-1] >= 8 and allow_direct_robot_state:
        print(f"[warn] Using {source}[:, :8] directly. Make sure its order is [pos3, axis_angle3, gripper2].")
        return robot_states[:, :8].astype(np.float32), f"{source} direct first 8 dims"

    raise ValueError(
        f"{source} has shape {robot_states.shape}. "
        "It is not recognized as [gripper2,pos3,quat4]. "
        "If you are sure it is already [pos3,axis_angle3,gripper2], pass --allow-direct-robot-state."
    )


def read_state(
    demo: h5py.Group,
    length: int,
    *,
    missing_policy: str,
    quat_format: str,
    allow_direct_robot_state: bool,
) -> tuple[np.ndarray, str]:
    # 1. Direct OpenPI-compatible state.
    state_ds, source = get_first_dataset(demo, STATE_KEYS)
    if state_ds is not None and source is not None:
        state = state_ds[:length].astype(np.float32)
        if state.shape[-1] >= 8:
            return state[:, :8].astype(np.float32), source
        raise ValueError(f"{source} exists but has shape {state.shape}; expected last dim >= 8.")

    # 2. Preferred regenerated-LIBERO format: ee_states + gripper_states.
    state, source = read_state_from_ee_and_gripper(demo, length, quat_format=quat_format)
    if state is not None and source is not None:
        return state, source

    # 3. Root-level robot_states from regenerated scripts.
    state, source = read_state_from_robot_states(
        demo,
        length,
        quat_format=quat_format,
        allow_direct_robot_state=allow_direct_robot_state,
    )
    if state is not None and source is not None:
        return state, source

    if missing_policy == "zero":
        print("[fallback] Missing 8D state; using zero states.")
        return np.zeros((length, 8), dtype=np.float32), "fallback:zero_state"

    available = "\n  ".join(list_dataset_paths(demo))
    raise KeyError(
        "Could not build OpenPI 8D state. Tried direct state, ee/gripper state, and robot_states.\n"
        "Available demo datasets:\n  "
        + available
    )


def validate_actions(actions: np.ndarray, h5_path: Path, demo_name: str) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)

    if actions.ndim != 2:
        raise ValueError(f"{h5_path}:{demo_name}: expected actions to be 2D, got shape {actions.shape}")
    if actions.shape[-1] < 7:
        raise ValueError(f"{h5_path}:{demo_name}: expected action dim >= 7, got shape {actions.shape}")

    actions = actions[:, :7].astype(np.float32)

    if not np.all(np.isfinite(actions)):
        raise ValueError(f"{h5_path}:{demo_name}: actions contain NaN or Inf")

    return actions


def finish_dataset(dataset: LeRobotDataset) -> None:
    if hasattr(dataset, "stop_image_writer"):
        dataset.stop_image_writer()


def main() -> None:
    args = parse_args()

    if args.strict:
        args.missing_wrist_policy = "error"
        args.missing_state_policy = "error"

    rotate = not args.no_rotate_180
    output_path = HF_LEROBOT_HOME / args.repo_id

    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite to recreate it.")
        shutil.rmtree(output_path)
    print(f"fps={args.fps}")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="libero",
        fps=args.fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (args.image_size, args.image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (args.image_size, args.image_size, 3),
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
        },
        use_videos=False,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    total_episodes = 0
    total_frames = 0

    try:
        for h5_path in iter_hdf5_files(args.input_dir):
            print(f"Converting {h5_path}")
            with h5py.File(h5_path, "r") as h5_file:
                if args.print_schema:
                    print_h5_schema(h5_file, h5_path)

                if "data" not in h5_file:
                    print(f"[skip] {h5_path}: missing data group")
                    continue

                demo_names = sorted(h5_file["data"].keys(), key=demo_sort_key)

                for demo_name in demo_names:
                    demo = h5_file["data"][demo_name]
                    if "actions" not in demo:
                        print(f"[skip] {h5_path}:{demo_name}: missing actions")
                        continue

                    base_images, base_source = read_base_images(demo)

                    actions = validate_actions(demo["actions"][:], h5_path, demo_name)
                    initial_length = min(len(actions), len(base_images))

                    wrist_images, wrist_source = read_wrist_images(
                        demo,
                        base_images,
                        initial_length,
                        missing_policy=args.missing_wrist_policy,
                    )

                    length = min(initial_length, len(wrist_images))
                    actions = actions[:length]

                    state, state_source = read_state(
                        demo,
                        length,
                        missing_policy=args.missing_state_policy,
                        quat_format=args.quat_format,
                        allow_direct_robot_state=args.allow_direct_robot_state,
                    )

                    task = read_task(h5_file, demo, h5_path)

                    grip = actions[:, -1]
                    print(
                        f"[episode] {h5_path.name}:{demo_name} "
                        f"task='{task}' len={length} "
                        f"image={base_source} wrist={wrist_source} state={state_source} "
                        f"grip_min={float(grip.min()):.3f} grip_max={float(grip.max()):.3f}"
                    )

                    for i in range(length):
                        image = ensure_uint8_hwc(maybe_rotate(base_images[i], rotate=rotate), args.image_size)
                        wrist_image = ensure_uint8_hwc(maybe_rotate(wrist_images[i], rotate=rotate), args.image_size)

                        dataset.add_frame(
                            {
                                "image": image,
                                "wrist_image": wrist_image,
                                "state": state[i].astype(np.float32),
                                "actions": actions[i].astype(np.float32),
                                "task": task,
                            }
                        )

                    dataset.save_episode()
                    total_episodes += 1
                    total_frames += length

                    if args.max_episodes is not None and total_episodes >= args.max_episodes:
                        print(f"Reached --max-episodes={args.max_episodes}")
                        return

    finally:
        finish_dataset(dataset)
        print(f"Saved to: {output_path}")
        print(f"episodes={total_episodes} frames={total_frames}")


if __name__ == "__main__":
    main()