#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed-initial-state trajectory diversity visualization for pi0/OpenPI policies.

This is the pi0 counterpart of OpenVLA Qselect/diverse_vis.py.  It keeps the
same output layout and visualization artifacts, but obtains actions by directly
querying a running OpenPI websocket policy server:

    server.py --sample-mode random --noise-strategy <base|hubu|zhengjiao|guocaiyang>

Because pi0 is an implicit-distribution policy, this script does not estimate
explicit action probabilities.  The policy server samples several action chunks
from the configured noise strategy, randomly returns one chunk, and this script
executes that chunk in LIBERO.
"""

from __future__ import annotations

import argparse
import collections
import csv
from dataclasses import dataclass
import json
import logging
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import imageio
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import tqdm
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy


logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_RESIZE_SIZE = 224
DEFAULT_REPLAN_STEPS = int(os.environ.get("action_horizon", "10"))
DEFAULT_FPS = 10
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclass
class GenerateConfig:
    # Policy server.
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    resize_size: int = DEFAULT_RESIZE_SIZE
    replan_steps: int = DEFAULT_REPLAN_STEPS

    # Metadata only.  The actual policy is already loaded by the server.
    policy_name: str = "pi0_policy"
    policy_names: str = ""
    policy_dir: str = ""
    pretrained_checkpoint: str = ""
    pretrained_checkpoints: str = ""

    # LIBERO.
    task_suite_name: str = "libero_goal"
    target_task: int = -1
    fixed_init_state_idx: int = 0
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    initial_states_path: str = "DEFAULT"
    env_img_res: int = LIBERO_ENV_RESOLUTION
    max_steps_override: int = -1

    # Rollout label.  For pi0 debug visualization this is usually the active
    # noise strategy name: base, hubu, zhengjiao, or guocaiyang.
    sample_modes: str = "random"
    selection_mode: str = "random"
    num_action_samples: int = 16
    noise_strategy: str = "base"
    noise_scale: float = 1.0

    # Output / logging.
    traj_dir: str = "./fixed_init_rollout_vis"
    local_log_dir: str = "./experiments/logs"
    seed: int = 7
    save_rollout_hdf5: bool = False
    save_video: bool = True
    video_fps: int = DEFAULT_FPS
    ghost_frame_stride: int = 2
    voxel_size: float = 0.01

    # Fixed-success-budget analysis.
    success_target_count: int = -1
    max_trials_for_success_target: int = 1000
    fixed_success_count: int = -1
    fixed_success_seed: int = 0

    # Plot options.
    plot_x_min: float = -999.0
    plot_x_max: float = -999.0
    plot_y_min: float = -999.0
    plot_y_max: float = -999.0


def str2bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {x!r}")


def parse_args() -> GenerateConfig:
    d = GenerateConfig()
    p = argparse.ArgumentParser(description="Visualize pi0 fixed-initial-state rollout diversity.")

    p.add_argument("--host", default=d.host)
    p.add_argument("--port", type=int, default=d.port)
    p.add_argument("--resize_size", "--resize-size", dest="resize_size", type=int, default=d.resize_size)
    p.add_argument("--replan_steps", "--replan-steps", dest="replan_steps", type=int, default=d.replan_steps)

    p.add_argument("--policy_name", "--policy-name", dest="policy_name", default=d.policy_name)
    p.add_argument("--policy_names", "--policy-names", dest="policy_names", default=d.policy_names)
    p.add_argument("--policy_dir", "--policy-dir", dest="policy_dir", default=d.policy_dir)
    p.add_argument("--pretrained_checkpoint", "--pretrained-checkpoint", dest="pretrained_checkpoint", default=d.pretrained_checkpoint)
    p.add_argument("--pretrained_checkpoints", "--pretrained-checkpoints", dest="pretrained_checkpoints", default=d.pretrained_checkpoints)

    p.add_argument("--task_suite_name", "--task-suite-name", dest="task_suite_name", default=d.task_suite_name)
    p.add_argument("--target_task", "--task-id", "--target-task", dest="target_task", type=int, default=d.target_task)
    p.add_argument("--fixed_init_state_idx", "--fixed-init-state-idx", dest="fixed_init_state_idx", type=int, default=d.fixed_init_state_idx)
    p.add_argument("--num_steps_wait", "--num-steps-wait", dest="num_steps_wait", type=int, default=d.num_steps_wait)
    p.add_argument("--num_trials_per_task", "--num-trials-per-task", dest="num_trials_per_task", type=int, default=d.num_trials_per_task)
    p.add_argument("--initial_states_path", "--initial-states-path", dest="initial_states_path", default=d.initial_states_path)
    p.add_argument("--env_img_res", "--env-img-res", dest="env_img_res", type=int, default=d.env_img_res)
    p.add_argument("--max_steps_override", "--max-steps-override", dest="max_steps_override", type=int, default=d.max_steps_override)

    p.add_argument("--sample_modes", "--sample-modes", dest="sample_modes", default=d.sample_modes)
    p.add_argument("--selection_mode", "--selection-mode", dest="selection_mode", default=d.selection_mode)
    p.add_argument("--num_action_samples", "--num-action-samples", dest="num_action_samples", type=int, default=d.num_action_samples)
    p.add_argument("--noise_strategy", "--noise-strategy", dest="noise_strategy", default=d.noise_strategy)
    p.add_argument("--noise_scale", "--noise-scale", dest="noise_scale", type=float, default=d.noise_scale)

    p.add_argument("--traj_dir", "--traj-dir", dest="traj_dir", default=d.traj_dir)
    p.add_argument("--local_log_dir", "--local-log-dir", dest="local_log_dir", default=d.local_log_dir)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--save_rollout_hdf5", "--save-rollout-hdf5", dest="save_rollout_hdf5", type=str2bool, nargs="?", const=True, default=d.save_rollout_hdf5)
    p.add_argument("--save_video", "--save-video", dest="save_video", type=str2bool, nargs="?", const=True, default=d.save_video)
    p.add_argument("--video_fps", "--video-fps", dest="video_fps", type=int, default=d.video_fps)
    p.add_argument("--ghost_frame_stride", "--ghost-frame-stride", dest="ghost_frame_stride", type=int, default=d.ghost_frame_stride)
    p.add_argument("--voxel_size", "--voxel-size", dest="voxel_size", type=float, default=d.voxel_size)

    p.add_argument("--success_target_count", "--success-target-count", dest="success_target_count", type=int, default=d.success_target_count)
    p.add_argument("--max_trials_for_success_target", "--max-trials-for-success-target", dest="max_trials_for_success_target", type=int, default=d.max_trials_for_success_target)
    p.add_argument("--fixed_success_count", "--fixed-success-count", dest="fixed_success_count", type=int, default=d.fixed_success_count)
    p.add_argument("--fixed_success_seed", "--fixed-success-seed", dest="fixed_success_seed", type=int, default=d.fixed_success_seed)

    p.add_argument("--plot_x_min", "--plot-x-min", dest="plot_x_min", type=float, default=d.plot_x_min)
    p.add_argument("--plot_x_max", "--plot-x-max", dest="plot_x_max", type=float, default=d.plot_x_max)
    p.add_argument("--plot_y_min", "--plot-y-min", dest="plot_y_min", type=float, default=d.plot_y_min)
    p.add_argument("--plot_y_max", "--plot-y-max", dest="plot_y_max", type=float, default=d.plot_y_max)

    return GenerateConfig(**vars(p.parse_args()))


def parse_sample_modes(sample_modes: str) -> List[str]:
    modes = [m.strip().lower() for m in str(sample_modes).split(",") if m.strip()]
    if len(modes) == 0:
        raise ValueError("sample_modes is empty")
    valid = {"random", "simple", "direct", "base", "hubu", "zhengjiao", "guocaiyang"}
    bad = [m for m in modes if m not in valid]
    if bad:
        raise ValueError(f"Unsupported sample_modes={bad}; valid={sorted(valid)}")
    return modes


def parse_csv_list(x: Any) -> List[str]:
    if x is None:
        return []
    return [s.strip() for s in str(x).split(",") if s.strip()]


def infer_policy_name(policy_path: str) -> str:
    if not policy_path:
        return "pi0_policy"
    p = Path(policy_path)
    if p.name == "final" and p.parent.name:
        return p.parent.parent.name + "_" + p.parent.name
    return p.name or "pi0_policy"


def build_policy_specs(cfg: GenerateConfig) -> List[Dict[str, str]]:
    checkpoints = parse_csv_list(cfg.pretrained_checkpoints)
    if not checkpoints:
        if cfg.pretrained_checkpoint:
            checkpoints = [cfg.pretrained_checkpoint]
        elif cfg.policy_dir:
            checkpoints = [cfg.policy_dir]
        else:
            checkpoints = [""]

    names = parse_csv_list(cfg.policy_names)
    if not names:
        names = [cfg.policy_name] if cfg.policy_name else []
    if not names:
        names = [infer_policy_name(checkpoints[0])]

    if len(checkpoints) != 1 or len(names) != 1:
        raise ValueError(
            "pi0 diverse_vis connects to one already-running policy server. "
            "Pass exactly one policy name/checkpoint per run."
        )

    return [{"policy_name": names[0], "policy_checkpoint": checkpoints[0]}]


def get_success_target_count(cfg: GenerateConfig) -> int:
    if int(cfg.success_target_count) > 0:
        return int(cfg.success_target_count)
    if int(cfg.fixed_success_count) > 0:
        return int(cfg.fixed_success_count)
    return -1


def validate_config(cfg: GenerateConfig) -> None:
    if cfg.task_suite_name not in TASK_MAX_STEPS:
        raise ValueError(f"Unsupported task_suite_name={cfg.task_suite_name}")
    if cfg.target_task < 0:
        raise ValueError("--target_task must be set to a non-negative LIBERO task id")
    if cfg.fixed_init_state_idx < 0:
        raise ValueError("fixed_init_state_idx must be >= 0")
    if cfg.num_trials_per_task <= 0:
        raise ValueError("num_trials_per_task must be > 0")
    if cfg.replan_steps <= 0:
        raise ValueError("replan_steps must be > 0")
    if cfg.num_action_samples <= 0:
        raise ValueError("num_action_samples must be > 0")
    if cfg.selection_mode not in {"random", "simple", "qselect"}:
        raise ValueError("selection_mode must be one of: random, simple, qselect")
    if cfg.noise_strategy not in {"base", "hubu", "zhengjiao", "guocaiyang"}:
        raise ValueError("noise_strategy must be one of: base, hubu, zhengjiao, guocaiyang")
    success_target = get_success_target_count(cfg)
    if success_target > 0 and cfg.max_trials_for_success_target < cfg.num_trials_per_task:
        raise ValueError("max_trials_for_success_target must be >= num_trials_per_task")
    modes = parse_sample_modes(cfg.sample_modes)
    if len(modes) != 1:
        raise ValueError(
            "pi0 noise debug visualization expects one sample/noise mode per running server. "
            "Run vis.sh once per noise_strategy to compare base/hubu/zhengjiao/guocaiyang."
        )
    build_policy_specs(cfg)


def log_message(message: str, log_file=None) -> None:
    print(message, flush=True)
    if log_file is not None:
        log_file.write(message + "\n")
        log_file.flush()


def setup_logging(cfg: GenerateConfig):
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(cfg.local_log_dir, f"pi0_diverse_vis_{stamp}.log")
    log_file = open(log_path, "w", encoding="utf-8")
    log_message(f"Logging to {log_path}", log_file)
    return log_file, log_path


def set_seed_everywhere(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_max_steps(task_suite_name: str, override: int = -1) -> int:
    if override > 0:
        return int(override)
    return TASK_MAX_STEPS[task_suite_name]


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)

    den = np.sqrt(max(1.0 - float(quat[3] * quat[3]), 0.0))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def preprocess_obs_for_openpi(obs: dict[str, Any], resize_size: int) -> dict[str, np.ndarray]:
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, resize_size, resize_size))

    state = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        ),
        axis=0,
    ).astype(np.float32)

    if state.shape != (8,):
        raise ValueError(f"Expected LIBERO OpenPI state shape (8,), got {state.shape}")

    return {
        "image": np.ascontiguousarray(img),
        "wrist_image": np.ascontiguousarray(wrist_img),
        "state": state,
    }


def build_policy_element(processed: dict[str, np.ndarray], task_description: str) -> dict[str, Any]:
    return {
        "observation/image": processed["image"],
        "observation/wrist_image": processed["wrist_image"],
        "observation/state": processed["state"],
        "prompt": str(task_description),
    }


def query_action_chunk(
    client: _websocket_client_policy.WebsocketClientPolicy,
    element: dict[str, Any],
    replan_steps: int,
) -> np.ndarray:
    result = client.infer(element)
    if "actions" not in result:
        raise KeyError(f"Policy server response does not contain 'actions'. Keys: {list(result.keys())}")

    action_chunk = np.asarray(result["actions"], dtype=np.float32)
    if action_chunk.ndim != 2:
        raise ValueError(f"Expected action chunk [H, A], got shape={action_chunk.shape}")
    if action_chunk.shape[-1] < 7:
        raise ValueError(f"Expected action dim >= 7, got shape={action_chunk.shape}")
    action_chunk = action_chunk[:, :7]

    if len(action_chunk) < replan_steps:
        raise ValueError(
            f"We want to replan every {replan_steps} steps, "
            f"but policy only predicts {len(action_chunk)} steps."
        )
    return action_chunk


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, task_description: str, log_file=None):
    default_initial_states = task_suite.get_task_init_states(task_id)

    if cfg.initial_states_path == "DEFAULT":
        idx = cfg.fixed_init_state_idx % len(default_initial_states)
        log_message(f"Using DEFAULT initial state idx={idx}", log_file)
        return np.asarray(default_initial_states[idx])

    with open(cfg.initial_states_path, "r", encoding="utf-8") as f:
        all_initial_states = json.load(f)
    task_key = task_description.replace(" ", "_")
    demo_key = f"demo_{cfg.fixed_init_state_idx}"
    if task_key not in all_initial_states:
        raise KeyError(f"Task key {task_key} not found in {cfg.initial_states_path}")
    if demo_key not in all_initial_states[task_key]:
        raise KeyError(f"{demo_key} not found for {task_key}")
    log_message(f"Using custom initial state {task_key}/{demo_key}", log_file)
    return np.asarray(all_initial_states[task_key][demo_key]["initial_state"])


def _empty_rollout_array(width: int) -> np.ndarray:
    return np.zeros((0, width), dtype=np.float32)


def run_episode_collect(
    cfg: GenerateConfig,
    env,
    task_description: str,
    client: _websocket_client_policy.WebsocketClientPolicy,
    initial_state=None,
    sample_mode: str = "random",
) -> Dict[str, Any]:
    env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else env.get_observation()

    action_queue: collections.deque[np.ndarray] = collections.deque(maxlen=cfg.replan_steps)
    max_steps = get_max_steps(cfg.task_suite_name, cfg.max_steps_override)

    replay_images = []
    agentview_images = []
    data_actions = []
    eef_positions = []
    eef_axisangles = []
    gripper_qpos = []

    t = 0
    success = False

    while t < max_steps + cfg.num_steps_wait:
        if t < cfg.num_steps_wait:
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        processed = preprocess_obs_for_openpi(obs, cfg.resize_size)
        replay_images.append(processed["image"])
        agentview_images.append(np.asarray(obs["agentview_image"], dtype=np.uint8))
        eef_positions.append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
        eef_axisangles.append(np.asarray(quat2axisangle(obs["robot0_eef_quat"]), dtype=np.float32))
        gripper_qpos.append(np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32))

        if len(action_queue) == 0:
            element = build_policy_element(processed, task_description)
            action_chunk = query_action_chunk(client, element, cfg.replan_steps)
            action_queue.extend(action_chunk[: cfg.replan_steps])

        action = np.asarray(action_queue.popleft(), dtype=np.float32)
        data_actions.append(action)

        obs, _, done, _ = env.step(action.tolist())
        if done:
            success = True
            break
        t += 1

    actions = np.asarray(data_actions, dtype=np.float32) if data_actions else _empty_rollout_array(7)
    positions = np.asarray(eef_positions, dtype=np.float32) if eef_positions else _empty_rollout_array(3)
    axisangles = np.asarray(eef_axisangles, dtype=np.float32) if eef_axisangles else _empty_rollout_array(3)
    grippers = np.asarray(gripper_qpos, dtype=np.float32) if gripper_qpos else _empty_rollout_array(2)

    return {
        "success": bool(success),
        "images": replay_images,
        "agentview_images": agentview_images,
        "actions": actions,
        "eef_positions": positions,
        "eef_axisangles": axisangles,
        "gripper_qpos": grippers,
        "length": int(len(data_actions)),
        "sample_mode": sample_mode,
    }


def _safe_concat(arrays: List[np.ndarray], axis=0):
    clean = [np.asarray(a) for a in arrays if a is not None and len(a) > 0]
    if not clean:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(clean, axis=axis)


def pairwise_l2(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 2:
        return np.zeros((0,), dtype=np.float32)
    vals = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            vals.append(float(np.linalg.norm(points[i] - points[j])))
    return np.asarray(vals, dtype=np.float32)


def filter_rollouts_by_subset(rollouts: List[Dict[str, Any]], subset: str) -> List[Dict[str, Any]]:
    subset = subset.lower()
    if subset == "all":
        return list(rollouts)
    if subset == "success":
        return [r for r in rollouts if bool(r.get("success", False))]
    if subset == "failure":
        return [r for r in rollouts if not bool(r.get("success", False))]
    raise ValueError(f"Unknown subset={subset}")


def _maybe_axis_limits_from_cfg(cfg: GenerateConfig):
    xlim = None
    ylim = None
    if float(cfg.plot_x_min) != -999.0 and float(cfg.plot_x_max) != -999.0:
        xlim = (float(cfg.plot_x_min), float(cfg.plot_x_max))
    if float(cfg.plot_y_min) != -999.0 and float(cfg.plot_y_max) != -999.0:
        ylim = (float(cfg.plot_y_min), float(cfg.plot_y_max))
    return xlim, ylim


def compute_rollout_stats(sample_mode: str, rollouts: List[Dict[str, Any]], voxel_size: float) -> Dict[str, Any]:
    num_rollouts = len(rollouts)
    successes = np.asarray([bool(r.get("success", False)) for r in rollouts], dtype=np.float32)
    lengths = np.asarray([int(r.get("length", 0)) for r in rollouts], dtype=np.float32)

    trajs = [r["eef_positions"] for r in rollouts if len(r.get("eef_positions", [])) > 0]
    endpoints = (
        np.asarray([tr[-1] for tr in trajs], dtype=np.float32)
        if trajs
        else np.zeros((0, 3), dtype=np.float32)
    )
    all_points = (
        _safe_concat(trajs, axis=0).reshape(-1, 3)
        if trajs
        else np.zeros((0, 3), dtype=np.float32)
    )

    path_lens = []
    path_len_per_step = []
    for tr in trajs:
        if len(tr) >= 2:
            cur_path_len = float(np.linalg.norm(np.diff(tr, axis=0), axis=-1).sum())
        else:
            cur_path_len = 0.0
        path_lens.append(cur_path_len)
        path_len_per_step.append(cur_path_len / max(len(tr), 1))

    path_lens = np.asarray(path_lens, dtype=np.float32)
    path_len_per_step = np.asarray(path_len_per_step, dtype=np.float32)

    if len(all_points) > 0:
        xyz_min = all_points.min(axis=0)
        xyz_max = all_points.max(axis=0)
        xyz_span = xyz_max - xyz_min
        xyz_bbox_volume = float(np.prod(xyz_span))
        vox = np.floor(all_points / max(voxel_size, 1e-8)).astype(np.int64)
        voxel_count = int(len(np.unique(vox, axis=0)))
        total_points = int(len(all_points))
    else:
        xyz_span = np.zeros(3, dtype=np.float32)
        xyz_bbox_volume = 0.0
        voxel_count = 0
        total_points = 0

    if len(endpoints) > 0:
        endpoint_std = endpoints.std(axis=0)
        endpoint_span = endpoints.max(axis=0) - endpoints.min(axis=0)
        endpoint_pairwise = pairwise_l2(endpoints)
    else:
        endpoint_std = endpoint_span = np.zeros(3, dtype=np.float32)
        endpoint_pairwise = np.zeros((0,), dtype=np.float32)

    actions_all = _safe_concat([r["actions"] for r in rollouts], axis=0)
    if actions_all.ndim == 2 and len(actions_all) > 0:
        action_std = actions_all.std(axis=0)
        trans_action_std_mean = float(action_std[:3].mean()) if actions_all.shape[1] >= 3 else 0.0
        rot_action_std_mean = float(action_std[3:6].mean()) if actions_all.shape[1] >= 6 else 0.0
        gripper_action_std = float(action_std[-1])
        action_std_mean = float(action_std.mean())
    else:
        action_std_mean = trans_action_std_mean = rot_action_std_mean = gripper_action_std = 0.0

    traj_len_mean = float(lengths.mean()) if len(lengths) else 0.0
    traj_len_std = float(lengths.std()) if len(lengths) else 0.0
    path_len_xyz_mean = float(path_lens.mean()) if len(path_lens) else 0.0
    path_len_xyz_std = float(path_lens.std()) if len(path_lens) else 0.0

    voxel_per_point = float(voxel_count / max(total_points, 1))
    voxel_per_rollout = float(voxel_count / max(num_rollouts, 1))

    return {
        "sample_mode": sample_mode,
        "num_rollouts": num_rollouts,
        "success_rate": float(successes.mean()) if len(successes) else 0.0,
        "num_success": int(successes.sum()) if len(successes) else 0,
        "num_trajs_with_points": int(len(trajs)),
        "traj_len_mean": traj_len_mean,
        "traj_len_std": traj_len_std,
        "path_len_xyz_mean": path_len_xyz_mean,
        "path_len_xyz_std": path_len_xyz_std,
        "path_len_per_step_mean": float(path_len_per_step.mean()) if len(path_len_per_step) else 0.0,
        "path_len_per_step_std": float(path_len_per_step.std()) if len(path_len_per_step) else 0.0,
        "x_span": float(xyz_span[0]),
        "y_span": float(xyz_span[1]),
        "z_span": float(xyz_span[2]),
        "xyz_bbox_volume": xyz_bbox_volume,
        "total_points": total_points,
        "voxel_count": voxel_count,
        "voxel_per_point": voxel_per_point,
        "voxel_per_rollout": voxel_per_rollout,
        "endpoint_x_std": float(endpoint_std[0]),
        "endpoint_y_std": float(endpoint_std[1]),
        "endpoint_z_std": float(endpoint_std[2]),
        "endpoint_spread_norm": float(np.linalg.norm(endpoint_std)),
        "endpoint_x_span": float(endpoint_span[0]),
        "endpoint_y_span": float(endpoint_span[1]),
        "endpoint_z_span": float(endpoint_span[2]),
        "endpoint_pairwise_l2_mean": float(endpoint_pairwise.mean()) if len(endpoint_pairwise) else 0.0,
        "action_std_mean": action_std_mean,
        "trans_action_std_mean": trans_action_std_mean,
        "rot_action_std_mean": rot_action_std_mean,
        "gripper_action_std": gripper_action_std,
    }


def add_common_metadata(
    stats: Dict[str, Any],
    *,
    rollout_budget: str,
    rollout_subset: str,
    source_rollouts: List[Dict[str, Any]],
    fixed_total_rollouts: List[Dict[str, Any]],
    success_target_count: int,
    success_target_reached: bool,
    attempts_to_success_target: int,
    policy_name: str,
    policy_checkpoint: str,
    value_head_path: str,
    task_id: int,
    task_name: str,
    task_description: str,
    fixed_init_state_idx: int,
    num_action_samples: int,
    selection_mode: str = "",
    noise_strategy: str = "",
    noise_scale: float = 1.0,
) -> Dict[str, Any]:
    source_num_rollouts = len(source_rollouts)
    source_num_success = sum(bool(r.get("success", False)) for r in source_rollouts)
    source_success_rate = float(source_num_success / max(source_num_rollouts, 1))

    fixed_total_num_rollouts = len(fixed_total_rollouts)
    fixed_total_num_success = sum(bool(r.get("success", False)) for r in fixed_total_rollouts)
    fixed_total_success_rate = float(fixed_total_num_success / max(fixed_total_num_rollouts, 1))

    stats.update({
        "rollout_budget": rollout_budget,
        "rollout_subset": rollout_subset,
        "parent_num_rollouts": int(source_num_rollouts),
        "parent_num_success": int(source_num_success),
        "parent_success_rate": float(source_success_rate),
        "subset_num_rollouts": int(stats.get("num_rollouts", 0)),
        "fixed_total_num_rollouts": int(fixed_total_num_rollouts),
        "fixed_total_num_success": int(fixed_total_num_success),
        "fixed_total_success_rate": float(fixed_total_success_rate),
        "success_target_count": int(success_target_count),
        "success_target_reached": bool(success_target_reached),
        "attempts_to_success_target": int(attempts_to_success_target),
        "failures_to_success_target": int(max(attempts_to_success_target - success_target_count, 0))
        if success_target_reached and success_target_count > 0
        else -1,
        "policy_name": policy_name,
        "policy_checkpoint": policy_checkpoint,
        "value_head_path": value_head_path,
        "task_id": task_id,
        "task_name": task_name,
        "task_description": task_description,
        "fixed_init_state_idx": fixed_init_state_idx,
        "num_action_samples": num_action_samples,
        "selection_mode": selection_mode,
        "noise_strategy": noise_strategy,
        "noise_scale": float(noise_scale),
    })
    return stats


def make_fixed_total_rows(
    sample_mode: str,
    fixed_total_rollouts: List[Dict[str, Any]],
    voxel_size: float,
    metadata_kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = []
    for subset in ["all", "success", "failure"]:
        subset_rollouts = filter_rollouts_by_subset(fixed_total_rollouts, subset)
        stats = compute_rollout_stats(sample_mode, subset_rollouts, voxel_size)
        row = add_common_metadata(
            stats,
            rollout_budget="fixed_total",
            rollout_subset=subset,
            source_rollouts=fixed_total_rollouts,
            fixed_total_rollouts=fixed_total_rollouts,
            **metadata_kwargs,
        )
        rows.append(row)
    return rows


def collect_until_success_target(
    cfg: GenerateConfig,
    env,
    task_description: str,
    client: _websocket_client_policy.WebsocketClientPolicy,
    initial_state,
    sample_mode: str,
    rollouts: List[Dict[str, Any]],
    log_file=None,
) -> Tuple[List[Dict[str, Any]], bool, int]:
    success_target = get_success_target_count(cfg)
    if success_target <= 0:
        return list(rollouts), False, len(rollouts)

    rollouts_until_target = list(rollouts)
    success_count = sum(bool(r.get("success", False)) for r in rollouts_until_target)
    next_trial_idx = len(rollouts_until_target)

    log_message(
        f"[{sample_mode}] fixed_total successes={success_count}/{len(rollouts_until_target)}; "
        f"target_success_count={success_target}; max_trials={cfg.max_trials_for_success_target}",
        log_file,
    )

    pbar = tqdm.tqdm(
        total=max(0, success_target - success_count),
        initial=0,
        desc=f"{sample_mode} extra successes",
        dynamic_ncols=True,
        leave=False,
    )

    while success_count < success_target and next_trial_idx < int(cfg.max_trials_for_success_target):
        rollout = run_episode_collect(
            cfg=cfg,
            env=env,
            task_description=task_description,
            client=client,
            initial_state=initial_state,
            sample_mode=sample_mode,
        )
        rollout["trial_idx"] = next_trial_idx
        rollout["collection_phase"] = "extra_until_success_target"
        rollouts_until_target.append(rollout)

        if bool(rollout["success"]):
            success_count += 1
            pbar.update(1)

        log_message(
            f"[{sample_mode}/extra] trial={next_trial_idx} "
            f"success={rollout['success']} len={rollout['length']} "
            f"success_count={success_count}/{success_target}",
            log_file,
        )

        next_trial_idx += 1

    pbar.close()

    success_target_reached = success_count >= success_target
    attempts_to_success_target = len(rollouts_until_target)

    if not success_target_reached:
        log_message(
            f"[{sample_mode}] WARNING: success target not reached. "
            f"successes={success_count}/{success_target}, attempts={attempts_to_success_target}",
            log_file,
        )

    return rollouts_until_target, success_target_reached, attempts_to_success_target


def first_attempts_until_k_successes(
    rollouts: List[Dict[str, Any]],
    k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    attempts_prefix = []
    first_k_successes = []
    k = int(k)

    for r in rollouts:
        attempts_prefix.append(r)
        if bool(r.get("success", False)):
            first_k_successes.append(r)
            if len(first_k_successes) >= k:
                return attempts_prefix, first_k_successes[:k], True

    return attempts_prefix, first_k_successes, False


def make_until_success_rows(
    sample_mode: str,
    rollouts_until_target: List[Dict[str, Any]],
    fixed_total_rollouts: List[Dict[str, Any]],
    success_target_count: int,
    voxel_size: float,
    metadata_kwargs: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = []
    attempts_prefix, success_fixed, reached = first_attempts_until_k_successes(
        rollouts_until_target,
        k=success_target_count,
    )
    attempts_to_target = len(attempts_prefix)

    stats_all = compute_rollout_stats(sample_mode, attempts_prefix, voxel_size)
    rows.append(add_common_metadata(
        stats_all,
        rollout_budget=f"until_success_{success_target_count}",
        rollout_subset=f"all_until_success_{success_target_count}",
        source_rollouts=attempts_prefix,
        fixed_total_rollouts=fixed_total_rollouts,
        success_target_count=success_target_count,
        success_target_reached=reached,
        attempts_to_success_target=attempts_to_target,
        **metadata_kwargs,
    ))

    stats_success = compute_rollout_stats(sample_mode, success_fixed, voxel_size)
    rows.append(add_common_metadata(
        stats_success,
        rollout_budget=f"until_success_{success_target_count}",
        rollout_subset=f"success_fixed_{success_target_count}",
        source_rollouts=attempts_prefix,
        fixed_total_rollouts=fixed_total_rollouts,
        success_target_count=success_target_count,
        success_target_reached=reached,
        attempts_to_success_target=attempts_to_target,
        **metadata_kwargs,
    ))

    failures_prefix = [r for r in attempts_prefix if not bool(r.get("success", False))]
    stats_failure = compute_rollout_stats(sample_mode, failures_prefix, voxel_size)
    rows.append(add_common_metadata(
        stats_failure,
        rollout_budget=f"until_success_{success_target_count}",
        rollout_subset=f"failure_until_success_{success_target_count}",
        source_rollouts=attempts_prefix,
        fixed_total_rollouts=fixed_total_rollouts,
        success_target_count=success_target_count,
        success_target_reached=reached,
        attempts_to_success_target=attempts_to_target,
        **metadata_kwargs,
    ))

    return rows


def save_rollouts_hdf5(path: str, sample_mode: str, rollouts: List[Dict[str, Any]]):
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "h5py is not installed in this Python environment. "
            "Run with --save_rollout_hdf5 False, or install h5py in the LIBERO env."
        ) from exc

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["sample_mode"] = sample_mode
        data = f.create_group("data")
        for i, r in enumerate(rollouts):
            g = data.create_group(f"demo_{i}")
            g.attrs["success"] = bool(r["success"])
            g.attrs["length"] = int(r["length"])
            g.attrs["trial_idx"] = int(r.get("trial_idx", i))
            g.attrs["collection_phase"] = str(r.get("collection_phase", "unknown"))
            obs = g.create_group("obs")
            if len(r["agentview_images"]) > 0:
                obs.create_dataset("agentview_rgb", data=np.stack(r["agentview_images"], axis=0), compression="gzip")
            g.create_dataset("actions", data=r["actions"], compression="gzip")
            g.create_dataset("eef_positions", data=r["eef_positions"], compression="gzip")
            g.create_dataset("eef_axisangles", data=r["eef_axisangles"], compression="gzip")
            g.create_dataset("gripper_qpos", data=r["gripper_qpos"], compression="gzip")


def save_temporal_ghost_image(rollouts: List[Dict[str, Any]], out_path: str, frame_stride: int = 2, max_frames: int = 1200):
    frames = []
    for r in rollouts:
        imgs = r["images"]
        for img in imgs[::max(1, frame_stride)]:
            frames.append(np.asarray(img, dtype=np.float32))
            if len(frames) >= max_frames:
                break
        if len(frames) >= max_frames:
            break
    if not frames:
        return
    ghost = np.mean(np.stack(frames, axis=0), axis=0)
    ghost = np.clip(ghost, 0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    imageio.imwrite(out_path, ghost)


def save_rollout_mean_video(rollouts: List[Dict[str, Any]], out_path: str, fps: int = 10):
    max_len = max((len(r["images"]) for r in rollouts), default=0)
    if max_len == 0:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
    try:
        for t in range(max_len):
            frames_t = []
            for r in rollouts:
                imgs = r["images"]
                if len(imgs) == 0:
                    continue
                idx = min(t, len(imgs) - 1)
                frames_t.append(np.asarray(imgs[idx], dtype=np.float32))
            if frames_t:
                frame = np.mean(np.stack(frames_t, axis=0), axis=0)
                writer.append_data(np.clip(frame, 0, 255).astype(np.uint8))
    finally:
        writer.close()


def _red_to_blue_cmap():
    return LinearSegmentedColormap.from_list(
        "trajectory_time_red_to_blue",
        [
            (0.0, "#d62728"),
            (0.5, "#f7f7f7"),
            (1.0, "#1f77b4"),
        ],
    )


def save_xy_trajectory_plot(
    rollouts: List[Dict[str, Any]],
    out_path: str,
    title: str,
    xlim=None,
    ylim=None,
    alpha: float = 0.45,
    linewidth: float = 1.15,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=180)
    cmap = _red_to_blue_cmap()
    norm = plt.Normalize(0.0, 1.0)
    has_any = False

    for r in rollouts:
        tr = r.get("eef_positions", None)
        if tr is None or len(tr) < 2:
            continue

        xy = np.asarray(tr[:, :2], dtype=np.float32)
        points = xy.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        tvals = np.linspace(0.0, 1.0, len(segments), dtype=np.float32)

        lc = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidths=linewidth,
            alpha=alpha,
        )
        lc.set_array(tvals)
        ax.add_collection(lc)
        has_any = True

        ax.scatter(xy[0, 0], xy[0, 1], marker="o", s=12, c="black", alpha=0.25, linewidths=0)
        ax.scatter(xy[-1, 0], xy[-1, 1], marker="x", s=18, c="black", alpha=0.35, linewidths=0.8)

    ax.set_xlabel("EEF x")
    ax.set_ylabel("EEF y")
    ax.set_title(title)
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if has_any:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("normalized trajectory time: red=start, blue=end")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_xyz_trajectory_plot(rollouts: List[Dict[str, Any]], out_path: str, title: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(figsize=(7, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    for r in rollouts:
        tr = r["eef_positions"]
        if len(tr) == 0:
            continue
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], linewidth=1.0, alpha=0.65)
        ax.scatter(tr[0, 0], tr[0, 1], tr[0, 2], marker="o", s=18)
        ax.scatter(tr[-1, 0], tr[-1, 1], tr[-1, 2], marker="x", s=24)
    ax.set_xlabel("EEF x")
    ax.set_ylabel("EEF y")
    ax.set_zlabel("EEF z")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def save_combined_xy_plot(mode_to_rollouts: Dict[str, List[Dict[str, Any]]], out_path: str, title: str, xlim=None, ylim=None):
    all_rollouts = []
    for _, rollouts in mode_to_rollouts.items():
        all_rollouts.extend(rollouts)
    save_xy_trajectory_plot(all_rollouts, out_path, title, xlim=xlim, ylim=ylim)


def save_combined_xy_plot_light(label_to_trajs: Dict[str, List[np.ndarray]], out_path: str, title: str, xlim=None, ylim=None):
    rollouts = []
    for _, trajs in label_to_trajs.items():
        for tr in trajs:
            if tr is None or len(tr) == 0:
                continue
            rollouts.append({"eef_positions": np.asarray(tr, dtype=np.float32)})
    save_xy_trajectory_plot(rollouts, out_path, title, xlim=xlim, ylim=ylim)


def write_summary_csv(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return

    keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_standard_plots(
    rollouts: List[Dict[str, Any]],
    out_dir: str,
    basename: str,
    title: str,
    cfg: GenerateConfig,
    plot_xlim=None,
    plot_ylim=None,
    save_video: bool = False,
):
    os.makedirs(out_dir, exist_ok=True)
    save_temporal_ghost_image(
        rollouts,
        os.path.join(out_dir, f"ghost_overlay_{basename}.png"),
        frame_stride=cfg.ghost_frame_stride,
    )
    save_xy_trajectory_plot(
        rollouts,
        os.path.join(out_dir, f"xy_trajectories_time_gradient_{basename}.png"),
        title=title,
        xlim=plot_xlim,
        ylim=plot_ylim,
    )
    save_xyz_trajectory_plot(
        rollouts,
        os.path.join(out_dir, f"xyz_trajectories_{basename}.png"),
        title=title,
    )
    if save_video:
        save_rollout_mean_video(
            rollouts,
            os.path.join(out_dir, f"mean_rollout_ghost_video_{basename}.mp4"),
            fps=cfg.video_fps,
        )


def run_fixed_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    client: _websocket_client_policy.WebsocketClientPolicy,
    log_file=None,
    policy_name: str = "policy",
    policy_checkpoint: str = "",
):
    task = task_suite.get_task(task_id)
    env, task_description = get_libero_env(task, resolution=cfg.env_img_res, seed=cfg.seed)
    initial_state = load_initial_states(cfg, task_suite, task_id, task_description, log_file)
    modes = parse_sample_modes(cfg.sample_modes)

    task_out_dir = os.path.join(cfg.traj_dir, f"task_{task_id}_{task.name}")
    policy_out_dir = os.path.join(task_out_dir, policy_name)
    os.makedirs(policy_out_dir, exist_ok=True)

    summary_rows = []
    mode_to_rollouts = {}
    mode_to_trajs = {}

    log_message(f"Fixed task_id={task_id}, task.name={task.name}", log_file)
    log_message(f"Policy name={policy_name}", log_file)
    log_message(f"Policy checkpoint={policy_checkpoint}", log_file)
    log_message(f"Task description: {task_description}", log_file)
    log_message(f"Fixed init state idx={cfg.fixed_init_state_idx}", log_file)
    log_message(f"Sample modes={modes}", log_file)

    plot_xlim, plot_ylim = _maybe_axis_limits_from_cfg(cfg)
    success_target = get_success_target_count(cfg)

    try:
        for mode in modes:
            mode_out_dir = os.path.join(policy_out_dir, mode)
            os.makedirs(mode_out_dir, exist_ok=True)

            fixed_total_rollouts = []
            for trial_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc=f"{mode} fixed-total rollouts", dynamic_ncols=True):
                rollout = run_episode_collect(
                    cfg=cfg,
                    env=env,
                    task_description=task_description,
                    client=client,
                    initial_state=initial_state,
                    sample_mode=mode,
                )
                rollout["trial_idx"] = trial_idx
                rollout["collection_phase"] = "fixed_total"
                fixed_total_rollouts.append(rollout)
                log_message(
                    f"[{mode}/fixed_total] trial={trial_idx} success={rollout['success']} len={rollout['length']}",
                    log_file,
                )

            metadata_kwargs_base = {
                "success_target_count": success_target,
                "success_target_reached": False,
                "attempts_to_success_target": len(fixed_total_rollouts),
                "policy_name": policy_name,
                "policy_checkpoint": policy_checkpoint,
                "value_head_path": "",
                "task_id": task_id,
                "task_name": task.name,
                "task_description": task_description,
                "fixed_init_state_idx": cfg.fixed_init_state_idx,
                "num_action_samples": cfg.num_action_samples,
                "selection_mode": cfg.selection_mode,
                "noise_strategy": cfg.noise_strategy,
                "noise_scale": cfg.noise_scale,
            }

            fixed_rows = make_fixed_total_rows(
                sample_mode=mode,
                fixed_total_rollouts=fixed_total_rollouts,
                voxel_size=cfg.voxel_size,
                metadata_kwargs=metadata_kwargs_base,
            )
            summary_rows.extend(fixed_rows)

            if cfg.save_rollout_hdf5:
                save_rollouts_hdf5(os.path.join(mode_out_dir, "rollouts_fixed_total.hdf5"), mode, fixed_total_rollouts)

            save_standard_plots(
                fixed_total_rollouts,
                mode_out_dir,
                basename="all_fixed_total",
                title=f"{task.name} | {mode} | fixed-total all | init {cfg.fixed_init_state_idx}",
                cfg=cfg,
                plot_xlim=plot_xlim,
                plot_ylim=plot_ylim,
                save_video=cfg.save_video,
            )

            for subset_name in ["success", "failure"]:
                subset_rollouts = filter_rollouts_by_subset(fixed_total_rollouts, subset_name)
                subset_out_dir = os.path.join(mode_out_dir, f"{subset_name}_fixed_total")
                if len(subset_rollouts) == 0:
                    log_message(f"[{mode}/{subset_name}_fixed_total] no rollouts, skip plots", log_file)
                    continue
                save_standard_plots(
                    subset_rollouts,
                    subset_out_dir,
                    basename=f"{subset_name}_fixed_total",
                    title=f"{task.name} | {mode} | fixed-total {subset_name} | init {cfg.fixed_init_state_idx}",
                    cfg=cfg,
                    plot_xlim=plot_xlim,
                    plot_ylim=plot_ylim,
                    save_video=False,
                )

            if success_target > 0:
                rollouts_until_target, target_reached, _ = collect_until_success_target(
                    cfg=cfg,
                    env=env,
                    task_description=task_description,
                    client=client,
                    initial_state=initial_state,
                    sample_mode=mode,
                    rollouts=fixed_total_rollouts,
                    log_file=log_file,
                )

                attempts_prefix, success_fixed, reached_by_prefix = first_attempts_until_k_successes(
                    rollouts_until_target,
                    k=success_target,
                )
                target_reached = bool(target_reached and reached_by_prefix)

                metadata_kwargs_until = {
                    "policy_name": policy_name,
                    "policy_checkpoint": policy_checkpoint,
                    "value_head_path": "",
                    "task_id": task_id,
                    "task_name": task.name,
                    "task_description": task_description,
                    "fixed_init_state_idx": cfg.fixed_init_state_idx,
                    "num_action_samples": cfg.num_action_samples,
                    "selection_mode": cfg.selection_mode,
                    "noise_strategy": cfg.noise_strategy,
                    "noise_scale": cfg.noise_scale,
                }

                until_rows = make_until_success_rows(
                    sample_mode=mode,
                    rollouts_until_target=rollouts_until_target,
                    fixed_total_rollouts=fixed_total_rollouts,
                    success_target_count=success_target,
                    voxel_size=cfg.voxel_size,
                    metadata_kwargs=metadata_kwargs_until,
                )
                summary_rows.extend(until_rows)

                until_dir = os.path.join(mode_out_dir, f"until_success_{success_target}")
                success_fixed_dir = os.path.join(mode_out_dir, f"success_fixed_{success_target}")
                failure_until_dir = os.path.join(mode_out_dir, f"failure_until_success_{success_target}")

                if cfg.save_rollout_hdf5:
                    save_rollouts_hdf5(
                        os.path.join(until_dir, f"rollouts_until_success_{success_target}.hdf5"),
                        mode,
                        attempts_prefix,
                    )
                    save_rollouts_hdf5(
                        os.path.join(success_fixed_dir, f"rollouts_success_fixed_{success_target}.hdf5"),
                        mode,
                        success_fixed,
                    )

                save_standard_plots(
                    attempts_prefix,
                    until_dir,
                    basename=f"all_until_success_{success_target}",
                    title=f"{task.name} | {mode} | attempts until {success_target} successes | init {cfg.fixed_init_state_idx}",
                    cfg=cfg,
                    plot_xlim=plot_xlim,
                    plot_ylim=plot_ylim,
                    save_video=False,
                )

                if len(success_fixed) > 0:
                    save_standard_plots(
                        success_fixed,
                        success_fixed_dir,
                        basename=f"success_fixed_{success_target}",
                        title=f"{task.name} | {mode} | first {success_target} successes | init {cfg.fixed_init_state_idx}",
                        cfg=cfg,
                        plot_xlim=plot_xlim,
                        plot_ylim=plot_ylim,
                        save_video=False,
                    )

                failures_prefix = [r for r in attempts_prefix if not bool(r.get("success", False))]
                if len(failures_prefix) > 0:
                    save_standard_plots(
                        failures_prefix,
                        failure_until_dir,
                        basename=f"failure_until_success_{success_target}",
                        title=f"{task.name} | {mode} | failures before {success_target} successes | init {cfg.fixed_init_state_idx}",
                        cfg=cfg,
                        plot_xlim=plot_xlim,
                        plot_ylim=plot_ylim,
                        save_video=False,
                    )

                mode_to_rollouts[mode] = attempts_prefix
                mode_to_trajs[mode] = [r["eef_positions"] for r in attempts_prefix if len(r["eef_positions"]) > 0]
            else:
                mode_to_rollouts[mode] = fixed_total_rollouts
                mode_to_trajs[mode] = [r["eef_positions"] for r in fixed_total_rollouts if len(r["eef_positions"]) > 0]
    finally:
        if hasattr(env, "close"):
            env.close()

    save_combined_xy_plot(
        mode_to_rollouts,
        os.path.join(policy_out_dir, "combined_xy_trajectories.png"),
        title=f"{task.name} | {policy_name} | init {cfg.fixed_init_state_idx} | mode comparison",
        xlim=plot_xlim,
        ylim=plot_ylim,
    )
    write_summary_csv(os.path.join(policy_out_dir, "summary.csv"), summary_rows)

    print("\n" + "=" * 100)
    print("[FIXED INIT ROLLOUT SUMMARY]")
    print("=" * 100)
    for row in summary_rows:
        print(
            f"budget={row.get('rollout_budget', 'unknown')} "
            f"subset={row.get('rollout_subset', 'all')} "
            f"mode={row['sample_mode']} "
            f"n={row['num_rollouts']} "
            f"fixed_total_success={row.get('fixed_total_success_rate', row['success_rate']):.3f} "
            f"success={row['success_rate']:.3f} "
            f"target_reached={row.get('success_target_reached', False)} "
            f"attempts_to_target={row.get('attempts_to_success_target', -1)} "
            f"len={row['traj_len_mean']:.1f}+/-{row['traj_len_std']:.1f} "
            f"path={row['path_len_xyz_mean']:.4f} "
            f"bbox={row['xyz_bbox_volume']:.6f} "
            f"endpoint_spread={row['endpoint_spread_norm']:.4f} "
            f"voxel_count={row['voxel_count']} "
            f"voxel_per_point={row['voxel_per_point']:.5f}"
        )
    print(f"\nSaved to: {policy_out_dir}")

    lightweight_trajs = {f"{policy_name}/{mode}": trajs for mode, trajs in mode_to_trajs.items()}
    return summary_rows, lightweight_trajs


def eval_libero(cfg: GenerateConfig) -> float:
    validate_config(cfg)
    policy_specs = build_policy_specs(cfg)
    set_seed_everywhere(cfg.seed)

    log_file, _ = setup_logging(cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    task = task_suite.get_task(cfg.target_task)
    task_out_dir = os.path.join(cfg.traj_dir, f"task_{cfg.target_task}_{task.name}")
    os.makedirs(task_out_dir, exist_ok=True)

    log_message(f"Task suite={cfg.task_suite_name}", log_file)
    log_message(f"Target task={cfg.target_task}, task.name={task.name}", log_file)
    log_message(f"Fixed init state idx={cfg.fixed_init_state_idx}", log_file)
    log_message(f"Sample modes={parse_sample_modes(cfg.sample_modes)}", log_file)
    log_message(
        f"Selection mode={cfg.selection_mode}, noise_strategy={cfg.noise_strategy}, "
        f"noise_scale={cfg.noise_scale}, num_action_samples={cfg.num_action_samples}",
        log_file,
    )
    log_message(f"Number of policies={len(policy_specs)}", log_file)
    log_message(f"Connecting to policy server at {cfg.host}:{cfg.port}", log_file)

    success_target = get_success_target_count(cfg)
    if success_target > 0:
        log_message(
            f"Fixed-success collection enabled: target={success_target}, "
            f"max_trials={cfg.max_trials_for_success_target}",
            log_file,
        )

    all_summary_rows: List[Dict[str, Any]] = []
    all_light_trajs: Dict[str, List[np.ndarray]] = {}
    plot_xlim, plot_ylim = _maybe_axis_limits_from_cfg(cfg)
    client = _websocket_client_policy.WebsocketClientPolicy(cfg.host, cfg.port)

    try:
        for policy_idx, spec in enumerate(policy_specs):
            policy_name = spec["policy_name"]
            policy_checkpoint = spec["policy_checkpoint"]

            print("\n" + "=" * 100)
            print(f"[POLICY {policy_idx + 1}/{len(policy_specs)}] {policy_name}")
            print(f"checkpoint: {policy_checkpoint}")
            print("=" * 100)

            set_seed_everywhere(cfg.seed)

            summary_rows, light_trajs = run_fixed_task(
                cfg=cfg,
                task_suite=task_suite,
                task_id=cfg.target_task,
                client=client,
                log_file=log_file,
                policy_name=policy_name,
                policy_checkpoint=policy_checkpoint,
            )
            all_summary_rows.extend(summary_rows)
            all_light_trajs.update(light_trajs)

        all_summary_path = os.path.join(task_out_dir, "summary_all_policies.csv")
        write_summary_csv(all_summary_path, all_summary_rows)
        save_combined_xy_plot_light(
            all_light_trajs,
            os.path.join(task_out_dir, "combined_xy_all_policies_modes.png"),
            title=f"{task.name} | fixed init {cfg.fixed_init_state_idx} | policy/mode comparison",
            xlim=plot_xlim,
            ylim=plot_ylim,
        )

        print("\n" + "=" * 100)
        print("[ALL POLICY SUMMARY]")
        print("=" * 100)
        for row in all_summary_rows:
            print(
                f"policy={row['policy_name']} "
                f"mode={row['sample_mode']} "
                f"budget={row.get('rollout_budget', 'unknown')} "
                f"subset={row.get('rollout_subset', 'all')} "
                f"n={row['num_rollouts']} "
                f"fixed_total_success={row.get('fixed_total_success_rate', row['success_rate']):.3f} "
                f"success={row['success_rate']:.3f} "
                f"target_reached={row.get('success_target_reached', False)} "
                f"attempts_to_target={row.get('attempts_to_success_target', -1)} "
                f"len={row['traj_len_mean']:.1f}+/-{row['traj_len_std']:.1f} "
                f"path={row['path_len_xyz_mean']:.4f} "
                f"bbox={row['xyz_bbox_volume']:.6f} "
                f"endpoint_spread={row['endpoint_spread_norm']:.4f} "
                f"voxel_count={row['voxel_count']} "
                f"voxel_per_point={row['voxel_per_point']:.5f}"
            )
        print(f"\nSaved all-policy summary to: {all_summary_path}")
        print(f"Saved all-policy XY comparison to: {os.path.join(task_out_dir, 'combined_xy_all_policies_modes.png')}")

        for row in all_summary_rows:
            if row.get("rollout_budget") == "fixed_total" and row.get("rollout_subset") == "all":
                return float(row["success_rate"])
        return float(all_summary_rows[0]["success_rate"]) if all_summary_rows else 0.0
    finally:
        if log_file:
            log_file.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = parse_args()
    eval_libero(cfg)


if __name__ == "__main__":
    main()
