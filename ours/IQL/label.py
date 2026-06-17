#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pool-aware strict-schema labeler.

Input:
    One or more collector/IQL-style LeRobot repos passed through --repo-dirs.

Each input repo must contain:
    image, wrist_image, state, actions, reward, terminal, success, task, episode_index

Only successful episodes are labeled and written.

Output:
    A single AWBC LeRobot repo containing:
        image, wrist_image, state, actions, adv, task

Usage:
    python label_pool.py \
      --repo-dirs /path/to/pool/raw /path/to/iter0/data/collect \
      --output-repo-id labeled \
      --critic-path /path/to/iql/final.pt \
      --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from model import LightIQLCritic


IMAGE_KEY = "image"
WRIST_IMAGE_KEY = "wrist_image"
STATE_KEY = "state"
ACTION_KEY = "actions"
TASK_KEY = "task"
EPISODE_KEY = "episode_index"
SUCCESS_KEY = "success"
ADV_KEY = "adv"


@dataclass
class Args:
    repo_dirs: list[str]
    output_repo_id: str
    critic_path: str

    encoder_name: str = ""
    horizon: int = 5
    image_size: int = 224
    batch_size: int = 128
    device: str = "cuda"
    seed: int = 7

    overwrite: bool = False
    fps: int = 10
    image_writer_threads: int = 10
    image_writer_processes: int = 5

    drop_incomplete_tail: bool = False
    normalize_adv: bool = False
    clamp_adv: float = 0.0
    max_episodes: int = 0


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--repo-dirs",
        nargs="+",
        required=True,
        help="List of local collector/IQL-style LeRobot repo directories.",
    )
    p.add_argument("--output-repo-id", required=True)
    p.add_argument("--critic-path", required=True)

    p.add_argument("--encoder-name", default="")
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--image-writer-threads", type=int, default=10)
    p.add_argument("--image-writer-processes", type=int, default=5)

    p.add_argument("--drop-incomplete-tail", action="store_true")
    p.add_argument("--normalize-adv", action="store_true")
    p.add_argument("--clamp-adv", type=float, default=0.0)
    p.add_argument("--max-episodes", type=int, default=0)

    return Args(**vars(p.parse_args()))


def require_key(sample: dict[str, Any], key: str) -> Any:
    if key not in sample:
        raise KeyError(f"Required key `{key}` not found. Available keys: {sorted(sample.keys())}")
    return sample[key]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def open_lerobot_dataset_from_dir(repo_dir: str | Path) -> LeRobotDataset:
    """
    Open a local LeRobot repo directory.

    For repo:
        /x/workspace/pool/raw
    use:
        repo_id = raw
        root    = /x/workspace/pool
    """
    repo_dir = Path(repo_dir).resolve()
    if not repo_dir.exists():
        raise FileNotFoundError(f"LeRobot repo dir does not exist: {repo_dir}")
    return LeRobotDataset(repo_dir.name, root=repo_dir)


def to_scalar(x: Any) -> Any:
    if isinstance(x, (int, float, bool, str)):
        return x
    if torch.is_tensor(x):
        if x.numel() == 1:
            return x.detach().cpu().item()
        return x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr


def to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, Image.Image):
        return np.asarray(x)
    return np.asarray(x)


def to_float_array(x: Any) -> np.ndarray:
    return to_numpy(x).astype(np.float32)


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


def to_pil_rgb(x: Any) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGB")

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

    return Image.fromarray(np.ascontiguousarray(arr)).convert("RGB")


def ensure_uint8_hwc(x: Any, image_size: int) -> np.ndarray:
    img = to_pil_rgb(x).resize((image_size, image_size), Image.BILINEAR)
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


def move_to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    return x


EpisodeKey = tuple[int, int]  # (repo_idx, episode_index_inside_repo)


class PoolCollectorIndex:
    """
    Index multiple collector/IQL-style LeRobot repos as one virtual dataset.

    Episodes are keyed by (repo_idx, episode_index) to avoid collisions across repos.
    """

    def __init__(self, repo_dirs: list[str]) -> None:
        if not repo_dirs:
            raise ValueError("repo_dirs must be non-empty.")

        self.repo_dirs = [Path(x).resolve() for x in repo_dirs]
        self.datasets: list[LeRobotDataset] = [
            open_lerobot_dataset_from_dir(x) for x in self.repo_dirs
        ]

        # internal_idx -> (repo_idx, frame_idx_inside_repo)
        self.frame_refs: list[tuple[int, int]] = []

        self.episode_of: list[EpisodeKey] = []
        self.local_pos_of: list[int] = []

        self.actions: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self.successes: list[bool] = []
        self.tasks: list[str] = []

        self.episodes: dict[EpisodeKey, list[int]] = defaultdict(list)

        self._scan()

    def _scan(self) -> None:
        total_raw_frames = sum(len(ds) for ds in self.datasets)
        print(
            f"[label] pool scan: repos={len(self.datasets)} raw_frames={total_raw_frames}",
            flush=True,
        )

        for repo_idx, ds in enumerate(self.datasets):
            repo_dir = self.repo_dirs[repo_idx]
            print(f"[label] repo[{repo_idx}] {repo_dir}: frames={len(ds)}", flush=True)

            for frame_idx in tqdm(range(len(ds)), desc=f"scan repo{repo_idx}", leave=False):
                sample = ds[frame_idx]

                ep_raw = int(to_scalar(require_key(sample, EPISODE_KEY)))
                ep_key = (repo_idx, ep_raw)
                local_pos = len(self.episodes[ep_key])

                action = to_float_array(require_key(sample, ACTION_KEY)).reshape(-1)
                state = to_float_array(require_key(sample, STATE_KEY)).reshape(-1)
                success = bool(to_scalar(require_key(sample, SUCCESS_KEY)))
                task = decode_task(require_key(sample, TASK_KEY)).strip()

                if not task:
                    raise ValueError(f"Empty `{TASK_KEY}` at repo={repo_dir}, frame={frame_idx}")

                internal_idx = len(self.frame_refs)

                self.frame_refs.append((repo_idx, frame_idx))
                self.episode_of.append(ep_key)
                self.local_pos_of.append(local_pos)

                self.actions.append(action.astype(np.float32))
                self.states.append(state.astype(np.float32))
                self.successes.append(success)
                self.tasks.append(task)

                self.episodes[ep_key].append(internal_idx)

        if not self.frame_refs:
            raise RuntimeError("Input pool is empty.")

        success_eps = self.successful_episode_ids()

        print(
            f"[label] frames={len(self.frame_refs)} "
            f"episodes={len(self.episodes)} "
            f"success_episodes={len(success_eps)} "
            f"success_frames={int(sum(bool(x) for x in self.successes))}",
            flush=True,
        )

    def successful_episode_ids(self) -> list[EpisodeKey]:
        return sorted(
            ep
            for ep, indices in self.episodes.items()
            if any(self.successes[i] for i in indices)
        )

    def read_frame_for_output(self, internal_idx: int, image_size: int) -> dict[str, Any]:
        repo_idx, frame_idx = self.frame_refs[internal_idx]
        ds = self.datasets[repo_idx]
        sample = ds[frame_idx]

        return {
            "image": ensure_uint8_hwc(require_key(sample, IMAGE_KEY), image_size),
            "wrist_image": ensure_uint8_hwc(require_key(sample, WRIST_IMAGE_KEY), image_size),
            "state": self.states[internal_idx].astype(np.float32),
            "actions": self.actions[internal_idx].astype(np.float32),
            "task": self.tasks[internal_idx],
        }

    def make_action_chunk(self, ep_indices: list[int], local_pos: int, horizon: int) -> np.ndarray:
        ep_len = len(ep_indices)
        action_dim = self.actions[ep_indices[0]].shape[-1]
        out = np.zeros((horizon, action_dim), dtype=np.float32)

        for k in range(horizon):
            pos = min(local_pos + k, ep_len - 1)
            out[k] = self.actions[ep_indices[pos]]

        return out


class Pi0StyleCriticCollator:
    def __init__(self, processor: CLIPProcessor, image_size: int) -> None:
        self.processor = processor
        self.image_size = int(image_size)

    def _resize_pil(self, x: Any) -> Image.Image:
        return to_pil_rgb(x).resize((self.image_size, self.image_size), Image.BILINEAR)

    def _process_images(self, images: list[Image.Image]) -> torch.Tensor:
        return self.processor(images=images, return_tensors="pt")["pixel_values"]

    def build_observation(self, frames: list[dict[str, Any]]) -> dict[str, Any]:
        base_images = [self._resize_pil(f["image"]) for f in frames]
        wrist_images = [self._resize_pil(f["wrist_image"]) for f in frames]
        zero_images = [Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0)) for _ in frames]

        prompts = [str(f["task"]) for f in frames]
        tokens = self.processor.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

        bsz = len(frames)
        return {
            "image": {
                "base_0_rgb": self._process_images(base_images),
                "left_wrist_0_rgb": self._process_images(wrist_images),
                "right_wrist_0_rgb": self._process_images(zero_images),
            },
            "image_mask": {
                "base_0_rgb": torch.ones(bsz, dtype=torch.bool),
                "left_wrist_0_rgb": torch.ones(bsz, dtype=torch.bool),
                "right_wrist_0_rgb": torch.zeros(bsz, dtype=torch.bool),
            },
            "state": torch.as_tensor(np.stack([f["state"] for f in frames]), dtype=torch.float32),
            "tokenized_prompt": tokens["input_ids"].long(),
            "tokenized_prompt_mask": tokens["attention_mask"].long(),
        }

    def build_batch(self, frames: list[dict[str, Any]], chunks: list[np.ndarray]) -> dict[str, Any]:
        return {
            "observation": self.build_observation(frames),
            "actions": torch.as_tensor(np.stack(chunks), dtype=torch.float32),
        }


def build_model_from_ckpt(args: Args, state_dim: int, action_dim: int, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(args.critic_path, map_location="cpu")
    ckpt_args = dict(ckpt.get("args", {}))

    encoder_name = args.encoder_name or ckpt_args.get("encoder_name", "openai/clip-vit-base-patch32")
    hidden_dim = int(ckpt_args.get("hidden_dim", ckpt_args.get("d_model", 512)))
    num_q = int(ckpt_args.get("num_q", 2))
    action_layers = int(ckpt_args.get("action_layers", 2))
    q_layers = int(ckpt_args.get("q_layers", ckpt_args.get("fusion_layers", 2)))
    dropout = float(ckpt_args.get("dropout", 0.1))
    q_l2_coef = float(ckpt_args.get("q_l2_coef", 1e-4))
    encoder_mode = str(
        ckpt_args.get("critic_encoder_mode", ckpt_args.get("encoder_mode", "full_pi0"))
    )
    # Use the LightIQLCritic compatibility signature from model.py.
    model = LightIQLCritic(
        encoder_name=encoder_name,
        state_dim=state_dim,
        action_dim=action_dim,
        horizon=args.horizon,
        hidden_dim=hidden_dim,
        num_q=num_q,
        action_layers=action_layers,
        q_layers=q_layers,
        dropout=dropout,
        q_l2_coef=q_l2_coef,
        encoder_mode=encoder_mode,
    )

    model.load_state_dict(ckpt.get("model", ckpt))

    if hasattr(model, "freeze_backbone"):
        model.freeze_backbone()

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def infer_adv(
    model: torch.nn.Module,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch = move_to_device(batch, device)

    if not hasattr(model, "infer_batch"):
        raise AttributeError("Model must expose infer_batch(batch) for label script.")

    # model.infer_batch accepts `device`, not `device_id`.
    out = model.infer_batch(batch, device=device)

    if isinstance(out, dict):
        adv = out["adv"]
        q = out.get("target_q", out.get("q", torch.zeros_like(adv)))
        v = out.get("v", torch.zeros_like(adv))
    else:
        adv, q, v = out[:3]

    return (
        adv.detach().cpu().float().numpy().reshape(-1),
        q.detach().cpu().float().numpy().reshape(-1),
        v.detach().cpu().float().numpy().reshape(-1),
    )


def create_output_dataset(args: Args) -> LeRobotDataset:
    output_path = HF_LEROBOT_HOME / args.output_repo_id

    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_path} already exists. Pass --overwrite.")
        shutil.rmtree(output_path)

    return LeRobotDataset.create(
        repo_id=args.output_repo_id,
        robot_type="libero",
        fps=args.fps,
        features={
            IMAGE_KEY: {
                "dtype": "image",
                "shape": (args.image_size, args.image_size, 3),
                "names": ["height", "width", "channel"],
            },
            WRIST_IMAGE_KEY: {
                "dtype": "image",
                "shape": (args.image_size, args.image_size, 3),
                "names": ["height", "width", "channel"],
            },
            STATE_KEY: {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            ACTION_KEY: {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
            ADV_KEY: {
                "dtype": "float32",
                "shape": (1,),
                "names": ["adv"],
            },
        },
        use_videos=False,
        image_writer_threads=args.image_writer_threads,
        image_writer_processes=args.image_writer_processes,
    )


def finish_dataset(dataset: LeRobotDataset) -> None:
    if hasattr(dataset, "stop_image_writer"):
        dataset.stop_image_writer()


def write_labeled_episode(dataset: LeRobotDataset, frames: list[dict[str, Any]], adv_values: np.ndarray) -> int:
    for frame, adv in zip(frames, adv_values):
        dataset.add_frame(
            {
                IMAGE_KEY: frame["image"],
                WRIST_IMAGE_KEY: frame["wrist_image"],
                STATE_KEY: frame["state"].astype(np.float32),
                ACTION_KEY: frame["actions"].astype(np.float32),
                ADV_KEY: np.asarray([float(adv)], dtype=np.float32),
                TASK_KEY: frame["task"],
            }
        )

    dataset.save_episode()
    return len(frames)


def label_episode(
    index: PoolCollectorIndex,
    ep: EpisodeKey,
    collator: Pi0StyleCriticCollator,
    model: torch.nn.Module,
    device: torch.device,
    args: Args,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    ep_indices = index.episodes[ep]
    ep_len = len(ep_indices)
    usable_len = max(0, ep_len - args.horizon + 1) if args.drop_incomplete_tail else ep_len

    frames_all: list[dict[str, Any]] = []
    adv_all: list[float] = []
    q_all: list[float] = []
    v_all: list[float] = []

    pos = 0
    while pos < usable_len:
        end = min(pos + args.batch_size, usable_len)

        frames = [
            index.read_frame_for_output(ep_indices[i], args.image_size)
            for i in range(pos, end)
        ]
        chunks = [
            index.make_action_chunk(ep_indices, i, args.horizon)
            for i in range(pos, end)
        ]

        adv, q, v = infer_adv(model, collator.build_batch(frames, chunks), device)

        frames_all.extend(frames)
        adv_all.extend([float(x) for x in adv])
        q_all.extend([float(x) for x in q])
        v_all.extend([float(x) for x in v])

        pos = end

    repo_idx, ep_raw = ep
    stats = {
        "repo_idx": int(repo_idx),
        "episode_id": int(ep_raw),
        "frames": int(ep_len),
        "usable_frames": int(usable_len),
        "adv_mean": float(np.mean(adv_all)) if adv_all else 0.0,
        "adv_std": float(np.std(adv_all)) if adv_all else 0.0,
        "adv_min": float(np.min(adv_all)) if adv_all else 0.0,
        "adv_max": float(np.max(adv_all)) if adv_all else 0.0,
        "q_mean": float(np.mean(q_all)) if q_all else 0.0,
        "v_mean": float(np.mean(v_all)) if v_all else 0.0,
    }

    return frames_all, np.asarray(adv_all, dtype=np.float32), stats


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    index = PoolCollectorIndex(args.repo_dirs)

    success_eps = index.successful_episode_ids()
    if args.max_episodes > 0:
        success_eps = success_eps[: args.max_episodes]

    if not success_eps:
        raise RuntimeError("No successful episodes found.")

    first = index.episodes[success_eps[0]][0]
    state_dim = int(index.states[first].shape[-1])
    action_dim = int(index.actions[first].shape[-1])

    if state_dim != 8:
        raise ValueError(f"Expected state_dim=8, got {state_dim}")
    if action_dim != 7:
        raise ValueError(f"Expected action_dim=7, got {action_dim}")

    ckpt = torch.load(args.critic_path, map_location="cpu")
    ckpt_args = dict(ckpt.get("args", {}))
    encoder_name = args.encoder_name or ckpt_args.get("encoder_name", "openai/clip-vit-base-patch32")

    model = build_model_from_ckpt(args, state_dim, action_dim, device)
    processor = CLIPProcessor.from_pretrained(encoder_name)
    collator = Pi0StyleCriticCollator(processor, args.image_size)
    out_ds = create_output_dataset(args)

    all_frames: list[list[dict[str, Any]]] = []
    all_advs: list[np.ndarray] = []

    stats: dict[str, Any] = {
        "args": asdict(args),
        "repo_dirs": [str(Path(x).resolve()) for x in args.repo_dirs],
        "input_frames": len(index.frame_refs),
        "input_episodes": len(index.episodes),
        "successful_episodes": len(success_eps),
        "episode_stats": [],
    }

    for ep in tqdm(success_eps, desc="label"):
        frames, adv, ep_stats = label_episode(index, ep, collator, model, device, args)
        all_frames.append(frames)
        all_advs.append(adv)
        stats["episode_stats"].append(ep_stats)

    valid_advs = [a for a in all_advs if len(a) > 0]
    if not valid_advs:
        raise RuntimeError("No labeled frames produced. Check --drop-incomplete-tail and episode lengths.")

    global_adv = np.concatenate(valid_advs, axis=0)
    stats["raw_adv"] = {
        "mean": float(global_adv.mean()),
        "std": float(global_adv.std()),
        "min": float(global_adv.min()),
        "max": float(global_adv.max()),
    }

    if args.normalize_adv:
        mean = float(global_adv.mean())
        std = float(global_adv.std())
        all_advs = [
            ((adv - mean) / max(std, 1e-6)).astype(np.float32)
            for adv in all_advs
        ]

    if args.clamp_adv and args.clamp_adv > 0:
        c = float(args.clamp_adv)
        all_advs = [
            np.clip(adv, -c, c).astype(np.float32)
            for adv in all_advs
        ]

    written_episodes = 0
    written_frames = 0
    written_adv: list[float] = []

    try:
        for frames, adv in tqdm(list(zip(all_frames, all_advs)), desc="write"):
            if len(frames) == 0:
                continue

            n = write_labeled_episode(out_ds, frames, adv)
            written_episodes += 1
            written_frames += n
            written_adv.extend([float(x) for x in adv])

    finally:
        finish_dataset(out_ds)

    written_adv_arr = np.asarray(written_adv, dtype=np.float32)
    stats["written"] = {
        "episodes": int(written_episodes),
        "frames": int(written_frames),
        "adv_mean": float(written_adv_arr.mean()) if written_adv_arr.size else 0.0,
        "adv_std": float(written_adv_arr.std()) if written_adv_arr.size else 0.0,
        "adv_min": float(written_adv_arr.min()) if written_adv_arr.size else 0.0,
        "adv_max": float(written_adv_arr.max()) if written_adv_arr.size else 0.0,
    }

    output_path = HF_LEROBOT_HOME / args.output_repo_id
    (output_path / "iql_label_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[label] saved: {output_path}")
    print(json.dumps(stats["written"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
