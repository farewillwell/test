from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import shutil
from pathlib import Path

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def _import_lerobot_dataset():
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:  # pragma: no cover - only hit when env is missing writer deps.
        raise RuntimeError(
            "collect_libero_lerobot.py runs inside libero_env and writes LeRobot directly, "
            "so that env must have lerobot + parquet writer deps installed. Example:\n"
            "  uv pip install --python /data/huangdi/heliqun/pi0/openpi/examples/libero/libero_env/bin/python "
            "lerobot pandas pyarrow datasets huggingface-hub filelock\n"
            f"Original import error: {type(exc).__name__}: {exc}"
        ) from exc
    return LeRobotDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--task-suite-name", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=6, help="-1 means all tasks in suite.")
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--video-out-path", default="")
    parser.add_argument("--save-candidate-traces", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def max_steps_for_suite(task_suite_name: str) -> int:
    if task_suite_name == "libero_spatial":
        return 220
    if task_suite_name == "libero_object":
        return 280
    if task_suite_name == "libero_goal":
        return 300
    if task_suite_name == "libero_10":
        return 520
    if task_suite_name == "libero_90":
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def make_lerobot_dataset(output_dir: Path, fps: int, overwrite: bool):
    LeRobotDataset = _import_lerobot_dataset()
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"{output_dir} already exists. Pass --overwrite to replace it.")
    return LeRobotDataset.create(
        repo_id=output_dir.name,
        root=output_dir,
        robot_type="libero",
        fps=fps,
        features={
            "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        use_videos=False,
        image_writer_threads=10,
        image_writer_processes=5,
    )


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def preprocess_obs(obs: dict, resize_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    record_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    record_wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    model_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(record_img, resize_size, resize_size))
    model_wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(record_wrist, resize_size, resize_size))
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return record_img, record_wrist, model_img, model_wrist, state


def collect_libero(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    dataset = make_lerobot_dataset(output_dir, args.fps, args.overwrite)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    max_steps = max_steps_for_suite(args.task_suite_name)
    video_out = Path(args.video_out_path) if args.video_out_path else None
    if video_out:
        video_out.mkdir(parents=True, exist_ok=True)

    total_episodes = 0
    total_successes = 0
    for task_id in tqdm.tqdm(range(task_suite.n_tasks), desc="tasks"):
        if args.task_id != -1 and args.task_id != task_id:
            continue
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        task_episodes = 0
        task_successes = 0

        for rollout_index in tqdm.tqdm(range(args.num_trials_per_task), desc=f"task {task_id}"):
            env.reset()
            action_plan = collections.deque()
            action_meta_plan = collections.deque()
            init_state = initial_states[rollout_index % len(initial_states)]
            obs = env.set_init_state(init_state)
            replay_images = []
            episode_index = int(dataset.meta.total_episodes)
            frames_written = 0
            success = False

            for t in range(max_steps + args.num_steps_wait):
                try:
                    if t < args.num_steps_wait:
                        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                        continue

                    record_img, record_wrist, model_img, model_wrist, state = preprocess_obs(obs, args.resize_size)
                    replay_images.append(record_img)

                    if not action_plan:
                        element = {
                            "observation/image": model_img,
                            "observation/wrist_image": model_wrist,
                            "observation/state": state,
                            "prompt": str(task_description),
                        }
                        result = client.infer(element)
                        action_chunk = np.asarray(result["actions"], dtype=np.float32)
                        if len(action_chunk) < args.replan_steps:
                            raise ValueError(
                                f"replan_steps={args.replan_steps}, but server returned only {len(action_chunk)} actions"
                            )
                        selected_index = int(np.asarray(result.get("selected_index", 0)).item())
                        q_values = np.asarray(result.get("q_values", []), dtype=np.float32).tolist()
                        policy_timing = result.get("policy_timing", {})
                        server_timing = result.get("server_timing", {})
                        for offset, action in enumerate(action_chunk[: args.replan_steps]):
                            action_plan.append(np.asarray(action, dtype=np.float32))
                            action_meta_plan.append(
                                {
                                    "selected_index": selected_index,
                                    "q_values": q_values,
                                    "offset": offset,
                                    "policy_timing": policy_timing,
                                    "server_timing": server_timing,
                                }
                            )

                    action = action_plan.popleft()
                    action_meta = action_meta_plan.popleft()
                    dataset.add_frame(
                        {
                            "image": record_img,
                            "wrist_image": record_wrist,
                            "state": state,
                            "actions": action,
                            "task": str(task_description),
                        }
                    )
                    if args.save_candidate_traces:
                        append_jsonl(
                            output_dir / "meta" / "ours_qselect_steps.jsonl",
                            {
                                "episode_index": episode_index,
                                "frame_index": frames_written,
                                "task_id": task_id,
                                **action_meta,
                            },
                        )
                    frames_written += 1

                    obs, _, done, _ = env.step(action.tolist())
                    if done:
                        success = True
                        break
                except Exception as exc:
                    logging.exception("Rollout failed: %s", exc)
                    break

            if frames_written > 0:
                dataset.save_episode()
                append_jsonl(
                    output_dir / "meta" / "ours_rollouts.jsonl",
                    {
                        "episode_index": episode_index,
                        "success": success,
                        "reward": 1.0 if success else 0.0,
                        "task_suite_name": args.task_suite_name,
                        "task_id": task_id,
                        "rollout_index": rollout_index,
                        "length": frames_written,
                    },
                )

            if video_out and replay_images:
                suffix = "success" if success else "failure"
                task_segment = str(task_description).replace(" ", "_")
                imageio.mimwrite(
                    video_out / f"rollout_t{task_id}_{rollout_index}_{task_segment}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=args.fps,
                )

            task_episodes += 1
            total_episodes += 1
            if success:
                task_successes += 1
                total_successes += 1
            logging.info("episode=%s success=%s frames=%s", total_episodes, success, frames_written)

        if task_episodes:
            logging.info("Task %s success rate: %.4f", task_id, task_successes / task_episodes)
    if hasattr(dataset, "stop_image_writer"):
        dataset.stop_image_writer()
    if total_episodes:
        logging.info("Total success rate: %.4f (%s/%s)", total_successes / total_episodes, total_successes, total_episodes)


def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    collect_libero(parse_args())


if __name__ == "__main__":
    main()
