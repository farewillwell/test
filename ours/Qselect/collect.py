#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect LIBERO rollouts through an OpenPI websocket policy server and save them
as a LIBERO LeRobot dataset.

This is the data-collection counterpart of openpi/examples/libero/main.py:
    - this process runs LIBERO env;
    - a separate process runs openpi/scripts/serve_policy.py;
    - this process calls WebsocketClientPolicy(host, port).infer(element);
    - the returned action chunk is executed for --replan-steps steps;
    - transitions are saved to LeRobot.

Expected two-process usage:

Terminal 1, policy server:
    cd /data/huangdi/heliqun/pi0/openpi
    source pi_env/bin/activate
    python -u scripts/serve_policy.py \
      --env LIBERO \
      --port 8000 \
      policy:checkpoint \
      --policy.config pi0_libero_low_mem_finetune \
      --policy.dir /path/to/model

Terminal 2, collector:
    cd /data/huangdi/heliqun/pi0/openpi
    source examples/libero/libero_env/bin/activate
    export PYTHONPATH="${PYTHONPATH}:$(pwd)/third_party/libero"
    export LIBERO_CONFIG_PATH=/data/huangdi/heliqun/openvla-oft/openvla-oft/LIBERO/.libero
    export MUJOCO_GL=egl
    python /data/huangdi/heliqun/pi0/ours/collect_libero_lerobot.py \
      --host 0.0.0.0 \
      --port 8000 \
      --task-suite-name libero_goal \
      --task-id 6 \
      --num-trials-per-task 50 \
      --repo-id heliqun/libero_goal_iter0_collect \
      --overwrite \
      --replan-steps 5
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import logging
import math
import pathlib
import shutil
import time
from typing import Any

import imageio
import numpy as np
import tqdm
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy


from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset



LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
STEP_REWARD = -1.0
SUCCESS_TERMINAL_REWARD = 10.0
FAILURE_TERMINAL_REWARD = -100.0


@dataclasses.dataclass
class Args:
    ###########################################################################
    # Policy server parameters. Same role as openpi/examples/libero/main.py.
    ###########################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    ###########################################################################
    # LIBERO environment parameters.
    ###########################################################################
    task_suite_name: str = "libero_goal"
    task_id: int = 6
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    initial_state_offset: int = 0
    seed: int = 7
    max_steps_override: int = -1

    ###########################################################################
    # LeRobot output.
    ###########################################################################
    repo_id: str = "heliqun/libero_collect"
    overwrite: bool = False
    fps: int = 10
    image_writer_threads: int = 10
    image_writer_processes: int = 5

    # Save successes/failures. IQL generally benefits from keeping both.
    save_success: bool = True
    save_failure: bool = True

    ###########################################################################
    # Logging / videos.
    ###########################################################################
    video_out_path: str = "data/libero/collect_videos"
    save_videos: bool = True
    metrics_path: str = "data/libero/collect_metrics.jsonl"


@dataclasses.dataclass
class RolloutFrame:
    image: np.ndarray
    wrist_image: np.ndarray
    state: np.ndarray
    action: np.ndarray
    reward: float
    terminal: int      # RL terminal mask，用于 IQL target
    step_index: int


@dataclasses.dataclass
class RolloutResult:
    task_id: int
    task_description: str
    trial_id: int
    success: bool
    frames: list[RolloutFrame]
    replay_images: list[np.ndarray]
    error: str = ""


def parse_args() -> Args:
    p = argparse.ArgumentParser()

    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--resize-size", type=int, default=224)
    p.add_argument("--replan-steps", type=int, default=5)

    p.add_argument("--task-suite-name", default="libero_goal")
    p.add_argument("--task-id", type=int, default=6)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--num-trials-per-task", type=int, default=50)
    p.add_argument("--initial-state-offset", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-steps-override", type=int, default=-1)

    p.add_argument("--repo-id", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--image-writer-threads", type=int, default=10)
    p.add_argument("--image-writer-processes", type=int, default=5)

    p.add_argument("--save-success", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-failure", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--video-out-path", default="data/libero/collect_videos")
    p.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--metrics-path", default="data/libero/collect_metrics.jsonl")

    return Args(**vars(p.parse_args()))


def get_max_steps(task_suite_name: str, override: int = -1) -> int:
    if override > 0:
        return int(override)
    return TASK_MAX_STEPS[task_suite_name]


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Same convention as openpi/examples/libero/main.py and robosuite: xyzw."""
    quat = np.asarray(quat, dtype=np.float32).copy()

    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(max(1.0 - float(quat[3] * quat[3]), 0.0))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)

    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def preprocess_obs_for_openpi(obs: dict[str, Any], resize_size: int) -> dict[str, np.ndarray]:
    """
    Matches openpi/examples/libero/main.py:
        - rotate agentview and wrist image by 180 degrees;
        - resize_with_pad;
        - convert_to_uint8;
        - state = eef_pos + eef_axis_angle + gripper_qpos.
    """
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
    """Exactly the element shape expected by the LIBERO OpenPI websocket policy."""
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
    """Query policy server exactly like openpi/examples/libero/main.py."""
    result = client.infer(element)
    if "actions" not in result:
        raise KeyError(f"Policy server response does not contain 'actions'. Keys: {list(result.keys())}")

    action_chunk = np.asarray(result["actions"], dtype=np.float32)
    if action_chunk.ndim != 2:
        raise ValueError(f"Expected action chunk [H, A], got shape={action_chunk.shape}")
    if action_chunk.shape[-1] < 7:
        raise ValueError(f"Expected action dim >= 7, got shape={action_chunk.shape}")

    # Server may return padded / extra dims; env only expects LIBERO 7D action.
    action_chunk = action_chunk[:, :7]

    if len(action_chunk) < replan_steps:
        raise ValueError(
            f"We want to replan every {replan_steps} steps, "
            f"but policy only predicts {len(action_chunk)} steps."
        )
    return action_chunk


def create_lerobot_dataset(args: Args) -> LeRobotDataset:
    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite to recreate it.")
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="libero",
        fps=args.fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (args.resize_size, args.resize_size, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (args.resize_size, args.resize_size, 3),
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
            # Extra fields used by IQL. OpenPI SFT/AWBC data config can ignore them
            # because it only repacks image/wrist_image/state/actions/task/adv.
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
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )


def finish_dataset(dataset: LeRobotDataset) -> None:
    if hasattr(dataset, "stop_image_writer"):
        dataset.stop_image_writer()


def should_save_episode(args: Args, success: bool) -> bool:
    if success:
        return bool(args.save_success)
    return bool(args.save_failure)


def write_episode_to_lerobot(dataset: LeRobotDataset, result: RolloutResult) -> int:
    for frame in result.frames:
        dataset.add_frame(
            {
                "image": frame.image,
                "wrist_image": frame.wrist_image,
                "state": frame.state.astype(np.float32),
                "actions": frame.action.astype(np.float32),
                "reward": np.asarray([frame.reward], dtype=np.float32),
                "terminal": np.asarray([int(frame.terminal)], dtype=np.int64),
                "success": np.asarray([int(result.success)], dtype=np.int64),
                "task_id": np.asarray([int(result.task_id)], dtype=np.int64),
                "trial_id": np.asarray([int(result.trial_id)], dtype=np.int64),
                "step_index": np.asarray([int(frame.step_index)], dtype=np.int64),
                "task": result.task_description,
            }
        )
    dataset.save_episode()
    return len(result.frames)


def save_rollout_video(args: Args, result: RolloutResult) -> None:
    if not args.save_videos or not result.replay_images:
        return

    video_dir = pathlib.Path(args.video_out_path)
    video_dir.mkdir(parents=True, exist_ok=True)

    suffix = "success" if result.success else "failure"
    task_segment = result.task_description.replace(" ", "_")
    path = video_dir / f"rollout_task{result.task_id}_trial{result.trial_id}_{task_segment}_{suffix}.mp4"

    imageio.mimwrite(path, [np.asarray(x) for x in result.replay_images], fps=args.fps)


def append_metrics(args: Args, record: dict[str, Any]) -> None:
    path = pathlib.Path(args.metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_one_episode(
    *,
    env,
    initial_state,
    client: _websocket_client_policy.WebsocketClientPolicy,
    task_id: int,
    task_description: str,
    trial_id: int,
    args: Args,
    max_steps: int,
) -> RolloutResult:
    env.reset()
    obs = env.set_init_state(initial_state)

    action_plan: collections.deque[np.ndarray] = collections.deque()
    frames: list[RolloutFrame] = []
    replay_images: list[np.ndarray] = []

    t = 0
    env_action_step = 0
    success = False
    error = ""

    while t < max_steps + args.num_steps_wait:

        # Same stabilization stage as OpenPI eval.
        if t < args.num_steps_wait:
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            t += 1
            continue

        processed = preprocess_obs_for_openpi(obs, args.resize_size)
        replay_images.append(processed["image"])

        if not action_plan:
            element = build_policy_element(processed, task_description)
            action_chunk = query_action_chunk(client, element, args.replan_steps)
            action_plan.extend(action_chunk[: args.replan_steps])

        action = np.asarray(action_plan.popleft(), dtype=np.float32)
        next_obs, _, done, info = env.step(action.tolist())

        frames.append(
            RolloutFrame(
                image=processed["image"],
                wrist_image=processed["wrist_image"],
                state=processed["state"],
                action=action,
                reward=float(STEP_REWARD),
                terminal=False,
                step_index=env_action_step,
            )
        )

        env_action_step += 1
        obs = next_obs
        t += 1

        if done:
            success = True
            break
    frames[-1].terminal=True
    if success:
        frames[-1].reward = SUCCESS_TERMINAL_REWARD
    else:
        frames[-1].reward = FAILURE_TERMINAL_REWARD
    return RolloutResult(
        task_id=int(task_id),
        task_description=str(task_description),
        trial_id=int(trial_id),
        success=bool(success),
        frames=frames,
        replay_images=replay_images,
        error=error,
    )


def collect(args: Args) -> None:
    if args.replan_steps <= 0:
        raise ValueError(f"--replan-steps must be positive, got {args.replan_steps}")

    np.random.seed(args.seed)

    logging.info("Args: %s", args)
    logging.info("Connecting to policy server at %s:%s", args.host, args.port)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    max_steps = get_max_steps(args.task_suite_name, args.max_steps_override)

    logging.info("Task suite: %s, num_tasks=%s, max_steps=%s", args.task_suite_name, num_tasks_in_suite, max_steps)

    dataset = create_lerobot_dataset(args)

    total_episodes = 0
    total_successes = 0
    total_saved_episodes = 0
    total_saved_frames = 0

    for cur_task_id in tqdm.tqdm(range(num_tasks_in_suite), desc="tasks"):
        if int(args.task_id) != -1 and int(args.task_id) != cur_task_id:
            continue

        task = task_suite.get_task(cur_task_id)
        initial_states = task_suite.get_task_init_states(cur_task_id)
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes = 0
        task_successes = 0

        for episode_idx in tqdm.tqdm(
            range(args.num_trials_per_task),
            desc=f"task{cur_task_id}",
            leave=False,
        ):
            state_index = args.initial_state_offset + episode_idx
            if state_index >= len(initial_states):
                raise IndexError(
                    f"Initial state index {state_index} >= {len(initial_states)}. "
                    f"Reduce --num-trials-per-task or --initial-state-offset."
                )

            logging.info("\nTask: %s", task_description)
            logging.info("Starting episode %s...", task_episodes + 1)

            result = run_one_episode(
                env=env,
                initial_state=initial_states[state_index],
                client=client,
                task_id=cur_task_id,
                task_description=task_description,
                trial_id=state_index,
                args=args,
                max_steps=max_steps,
            )

            total_episodes += 1
            task_episodes += 1

            if result.success:
                total_successes += 1
                task_successes += 1

            saved_frames = 0
            if result.frames and should_save_episode(args, result.success):
                saved_frames = write_episode_to_lerobot(dataset, result)
                total_saved_episodes += 1
                total_saved_frames += saved_frames

            # save_rollout_video(args, result)

            record = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "task_suite_name": args.task_suite_name,
                "task_id": int(cur_task_id),
                "task_description": task_description,
                "trial_id": int(state_index),
                "success": bool(result.success),
                "num_frames": int(len(result.frames)),
                "saved_frames": int(saved_frames),
                "error": result.error,
                "task_episodes": int(task_episodes),
                "task_successes": int(task_successes),
                "task_success_rate": float(task_successes / max(task_episodes, 1)),
                "total_episodes": int(total_episodes),
                "total_successes": int(total_successes),
                "total_success_rate": float(total_successes / max(total_episodes, 1)),
                "total_saved_episodes": int(total_saved_episodes),
                "total_saved_frames": int(total_saved_frames),
            }
            append_metrics(args, record)

            logging.info("Success: %s", result.success)
            logging.info("# episodes completed so far: %s", total_episodes)
            logging.info(
                "# successes: %s (%.1f%%)",
                total_successes,
                total_successes / max(total_episodes, 1) * 100.0,
            )
            logging.info(
                "Current task success rate: %.3f",
                task_successes / max(task_episodes, 1),
            )
            logging.info(
                "Current total success rate: %.3f",
                total_successes / max(total_episodes, 1),
            )
            logging.info(
                "Saved episodes=%s frames=%s",
                total_saved_episodes,
                total_saved_frames,
            )
    logging.info("Saved LeRobot dataset to: %s", HF_LEROBOT_HOME / args.repo_id)
    logging.info(
        "Total episodes=%s successes=%s success_rate=%.3f",
        total_episodes,
        total_successes,
        total_successes / max(total_episodes, 1),
    )
    logging.info("Saved episodes=%s frames=%s", total_saved_episodes, total_saved_frames)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    collect(args)


if __name__ == "__main__":
    main()
