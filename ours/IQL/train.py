#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a lightweight IQL critic for pi0-style LIBERO LeRobot data.

This script is intentionally independent from OpenPI's training stack.
It assumes a LIBERO LeRobot dataset with frame fields:
    image, wrist_image, state, actions, task
and constructs pi0-style observation batches:
    observation = {
        image: {
            base_0_rgb,
            left_wrist_0_rgb,
            right_wrist_0_rgb,
        },
        image_mask: {...},
        state,
        tokenized_prompt,
        tokenized_prompt_mask,
    }

The model is expected to consume:
    batch["observation"], batch["next_observation"],
    batch["actions"], batch["rewards"], batch["is_terminal"], batch["from_success"].

The action chunk length is fixed by --horizon. For your current setting,
use --horizon 5, matching pi0 action_horizon == replan_steps.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPProcessor

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except Exception as exc:
    raise ImportError(
        "Could not import LeRobotDataset. Run this script in the openpi/lerobot environment."
    ) from exc

# train.py is expected to live in ours/IQL/.
# Import the local model.py, not OpenPI.
try:
    from model import LightIQLCritic
except ImportError:
    try:
        from model import LightVLValueModel as LightIQLCritic
    except ImportError as exc:
        raise ImportError(
            "Could not import LightIQLCritic or LightVLValueModel from local model.py."
        ) from exc


IMAGE_KEYS = ("image", "observation/image")
WRIST_IMAGE_KEYS = ("wrist_image", "observation/wrist_image")
STATE_KEYS = ("state", "observation/state")
ACTION_KEYS = ("actions", "action")
TASK_KEYS = ("task", "prompt", "language_instruction")
EPISODE_KEYS = ("episode_index", "episode")
FRAME_KEYS = ("frame_index", "index")
REWARD_KEYS = ("reward", "rewards")
DONE_KEYS = ("done", "is_terminal", "terminal")
SUCCESS_KEYS = ("success", "episode_success", "from_success")


@dataclass
class TrainArgs:
    repo_id: str
    output_dir: str

    root: str = ""
    encoder_name: str = "openai/clip-vit-base-patch32"

    horizon: int = 5
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    max_steps: int = 4000

    lr: float = 1e-4
    weight_decay: float = 1e-4
    gamma: float = 0.99
    expectile: float = 0.7
    tau: float = 0.02
    grad_clip: float = 1.0

    hidden_dim: int = 512
    num_q: int = 2
    action_layers: int = 2
    q_layers: int = 2
    dropout: float = 0.1
    q_l2_coef: float = 1e-4

    step_reward: float = 0.0
    success_terminal_reward: float = 1.0
    failure_terminal_reward: float = 0.0
    default_episode_success: str = "success"

    use_wrist_image: bool = True
    use_q_aug: bool = False

    log_every: int = 50
    debug_every: int = 500
    save_every: int = 1000
    seed: int = 7
    device: str = "cuda"

    resume: str = ""


def parse_args() -> TrainArgs:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="Local or HF LeRobot repo id.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--root", default="", help="Optional local root for LeRobotDataset.")
    p.add_argument("--encoder-name", default="openai/clip-vit-base-patch32")

    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=4000)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--expectile", type=float, default=0.7)
    p.add_argument("--tau", type=float, default=0.02)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--num-q", type=int, default=2)
    p.add_argument("--action-layers", type=int, default=2)
    p.add_argument("--q-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--q-l2-coef", type=float, default=1e-4)

    p.add_argument("--step-reward", type=float, default=0.0)
    p.add_argument("--success-terminal-reward", type=float, default=1.0)
    p.add_argument("--failure-terminal-reward", type=float, default=0.0)
    p.add_argument("--default-episode-success", choices=("success", "failure"), default="success")

    p.add_argument("--use-wrist-image", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-q-aug", action="store_true")

    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--debug-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda")

    p.add_argument("--resume", default="")
    return TrainArgs(**vars(p.parse_args()))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def first_present(sample: dict[str, Any], keys: tuple[str, ...], required: bool = True) -> Any:
    for key in keys:
        if key in sample:
            return sample[key]
    if required:
        raise KeyError(f"None of keys {keys} found. Available keys: {sorted(sample.keys())}")
    return None


def to_scalar(x: Any, default: int | float | bool | None = None) -> Any:
    if x is None:
        return default
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


def to_numpy(x: Any, dtype: np.dtype | None = None) -> np.ndarray:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    elif isinstance(x, Image.Image):
        x = np.asarray(x)
    else:
        x = np.asarray(x)
    if dtype is not None:
        x = x.astype(dtype)
    return x


def to_float_array(x: Any) -> np.ndarray:
    arr = to_numpy(x).astype(np.float32)
    return arr


def to_pil_rgb(x: Any) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGB")

    arr = to_numpy(x)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape={arr.shape}")

    # Accept CHW from LeRobot and convert to HWC.
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))

    if arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got shape={arr.shape}")

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(np.ascontiguousarray(arr)).convert("RGB")


def decode_prompt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if isinstance(x, str):
        return x
    if torch.is_tensor(x):
        if x.numel() == 1:
            return decode_prompt(x.detach().cpu().item())
        return str(x.detach().cpu().numpy())
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return decode_prompt(x.item())
        if x.size == 1:
            return decode_prompt(x.reshape(-1)[0])
    return str(x)


def make_lerobot_dataset(repo_id: str, root: str = "") -> LeRobotDataset:
    if root:
        try:
            return LeRobotDataset(repo_id, root=Path(root))
        except TypeError:
            return LeRobotDataset(repo_id, Path(root))
    return LeRobotDataset(repo_id)


class LiberoLeRobotIQLDataset(Dataset):
    """
    A narrow dataset reader for the LIBERO LeRobot layout produced for this project.

    It deliberately does not use OpenPI data_loader/train.py. It reads single-step
    LeRobot frames and constructs fixed-length action chunks inside each episode.
    """

    def __init__(
        self,
        repo_id: str,
        root: str,
        horizon: int,
        step_reward: float,
        success_terminal_reward: float,
        failure_terminal_reward: float,
        default_episode_success: str,
        use_wrist_image: bool = True,
    ) -> None:
        super().__init__()
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")

        self.ds = make_lerobot_dataset(repo_id, root)
        self.horizon = int(horizon)
        self.step_reward = float(step_reward)
        self.success_terminal_reward = float(success_terminal_reward)
        self.failure_terminal_reward = float(failure_terminal_reward)
        self.default_success = default_episode_success == "success"
        self.use_wrist_image = bool(use_wrist_image)

        self.global_indices: list[int] = []
        self.episode_of: list[int] = []
        self.local_pos_of: list[int] = []

        self.actions: list[np.ndarray] = []
        self.reward_values: list[float | None] = []
        self.done_values: list[bool | None] = []
        self.success_values: list[bool | None] = []

        self.episodes: dict[int, list[int]] = defaultdict(list)
        self._build_index()

    def _build_index(self) -> None:
        print(f"[IQL data] Scanning LeRobot dataset: {len(self.ds)} frames")
        for global_idx in tqdm(range(len(self.ds)), desc="scan", leave=False):
            sample = self.ds[global_idx]

            ep_raw = first_present(sample, EPISODE_KEYS, required=False)
            ep = int(to_scalar(ep_raw, default=0))

            action = to_float_array(first_present(sample, ACTION_KEYS))
            if action.ndim != 1:
                action = action.reshape(-1)

            reward_raw = first_present(sample, REWARD_KEYS, required=False)
            reward = None if reward_raw is None else float(to_scalar(reward_raw, default=0.0))

            done_raw = first_present(sample, DONE_KEYS, required=False)
            done = None if done_raw is None else bool(to_scalar(done_raw, default=False))

            success_raw = first_present(sample, SUCCESS_KEYS, required=False)
            success = None if success_raw is None else bool(to_scalar(success_raw, default=self.default_success))

            local_pos = len(self.episodes[ep])

            self.global_indices.append(global_idx)
            self.episode_of.append(ep)
            self.local_pos_of.append(local_pos)
            self.actions.append(action.astype(np.float32))
            self.reward_values.append(reward)
            self.done_values.append(done)
            self.success_values.append(success)
            self.episodes[ep].append(len(self.global_indices) - 1)

        if not self.global_indices:
            raise RuntimeError("Empty dataset.")

        action_dims = {a.shape[-1] for a in self.actions}
        if len(action_dims) != 1:
            raise ValueError(f"Mixed action dims found: {sorted(action_dims)}")

        print(
            f"[IQL data] {len(self.global_indices)} frames, "
            f"{len(self.episodes)} episodes, action_dim={next(iter(action_dims))}, horizon={self.horizon}"
        )

    def __len__(self) -> int:
        return len(self.global_indices)

    def _episode_success(self, ep_internal_indices: list[int]) -> bool:
        vals = [self.success_values[i] for i in ep_internal_indices if self.success_values[i] is not None]
        if not vals:
            return self.default_success
        # If any frame says success, treat the whole episode as success.
        return bool(any(vals))

    def _chunk_arrays(self, internal_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, int]:
        ep = self.episode_of[internal_idx]
        ep_indices = self.episodes[ep]
        local_pos = self.local_pos_of[internal_idx]
        ep_len = len(ep_indices)
        ep_success = self._episode_success(ep_indices)

        action_dim = self.actions[internal_idx].shape[-1]
        actions = np.zeros((self.horizon, action_dim), dtype=np.float32)
        rewards = np.zeros((self.horizon,), dtype=np.float32)
        is_terminal = np.zeros((self.horizon,), dtype=np.bool_)

        first_terminal_seen = False
        for k in range(self.horizon):
            unclamped_pos = local_pos + k
            future_pos = min(unclamped_pos, ep_len - 1)
            future_internal_idx = ep_indices[future_pos]

            actions[k] = self.actions[future_internal_idx]

            if self.reward_values[future_internal_idx] is not None:
                if unclamped_pos <= ep_len - 1:
                    rewards[k] = float(self.reward_values[future_internal_idx])
                else:
                    rewards[k] = 0.0
            else:
                rewards[k] = self.step_reward
                if unclamped_pos == ep_len - 1:
                    rewards[k] = (
                        self.success_terminal_reward if ep_success else self.failure_terminal_reward
                    )
                elif unclamped_pos > ep_len - 1:
                    rewards[k] = 0.0

            if self.done_values[future_internal_idx] is not None and unclamped_pos <= ep_len - 1:
                term = bool(self.done_values[future_internal_idx])
            else:
                term = unclamped_pos >= ep_len - 1

            if first_terminal_seen:
                term = True
            if term:
                first_terminal_seen = True
            is_terminal[k] = term

        next_pos = min(local_pos + self.horizon, ep_len - 1)
        next_internal_idx = ep_indices[next_pos]
        return actions, rewards, is_terminal, ep_success, next_internal_idx

    def _read_observation(self, internal_idx: int) -> dict[str, Any]:
        sample = self.ds[self.global_indices[internal_idx]]

        base = to_pil_rgb(first_present(sample, IMAGE_KEYS))
        if self.use_wrist_image:
            wrist_raw = first_present(sample, WRIST_IMAGE_KEYS, required=False)
            wrist = to_pil_rgb(wrist_raw) if wrist_raw is not None else base.copy()
            wrist_mask = True
        else:
            wrist = Image.new("RGB", base.size, (0, 0, 0))
            wrist_mask = False

        state = to_float_array(first_present(sample, STATE_KEYS))
        if state.ndim != 1:
            state = state.reshape(-1)

        prompt_raw = first_present(sample, TASK_KEYS, required=False)
        if prompt_raw is None and "task_index" in sample:
            prompt_raw = self._task_from_index(sample["task_index"])
        prompt = decode_prompt(prompt_raw).strip()
        if not prompt:
            prompt = "do something"

        return {
            "base_image": base,
            "wrist_image": wrist,
            "wrist_mask": wrist_mask,
            "state": state.astype(np.float32),
            "prompt": prompt,
        }

    def _task_from_index(self, task_index: Any) -> str:
        idx = int(to_scalar(task_index, default=0))
        meta = getattr(self.ds, "meta", None)
        tasks = getattr(meta, "tasks", None)
        if isinstance(tasks, dict) and idx in tasks:
            return decode_prompt(tasks[idx])
        if isinstance(tasks, (list, tuple)) and 0 <= idx < len(tasks):
            return decode_prompt(tasks[idx])
        return ""

    def __getitem__(self, internal_idx: int) -> dict[str, Any]:
        actions, rewards, is_terminal, ep_success, next_internal_idx = self._chunk_arrays(internal_idx)

        obs = self._read_observation(internal_idx)
        next_obs = self._read_observation(next_internal_idx)

        # For CLIP text encoding, keep next prompt the same as current task.
        next_obs["prompt"] = obs["prompt"]

        return {
            "obs": obs,
            "next_obs": next_obs,
            "actions": actions,
            "rewards": rewards,
            "is_terminal": is_terminal,
            "from_success": np.full((self.horizon,), ep_success, dtype=np.bool_),
        }


class Pi0StyleIQLCollator:
    def __init__(self, processor: CLIPProcessor, image_size: int = 224) -> None:
        self.processor = processor
        self.image_size = int(image_size)

    def _resize(self, image: Image.Image) -> Image.Image:
        return image.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)

    def _process_images(self, images: list[Image.Image]) -> torch.Tensor:
        enc = self.processor(images=images, return_tensors="pt")
        return enc["pixel_values"]

    def _build_observation(self, obs_list: list[dict[str, Any]]) -> dict[str, Any]:
        base_images = [self._resize(o["base_image"]) for o in obs_list]
        wrist_images = [self._resize(o["wrist_image"]) for o in obs_list]
        zero_images = [Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0)) for _ in obs_list]

        prompts = [str(o["prompt"]) for o in obs_list]
        tokens = self.processor.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        state = torch.as_tensor(np.stack([o["state"] for o in obs_list]), dtype=torch.float32)
        wrist_mask = torch.as_tensor([bool(o["wrist_mask"]) for o in obs_list], dtype=torch.bool)
        batch_size = len(obs_list)

        return {
            "image": {
                "base_0_rgb": self._process_images(base_images),
                "left_wrist_0_rgb": self._process_images(wrist_images),
                "right_wrist_0_rgb": self._process_images(zero_images),
            },
            "image_mask": {
                "base_0_rgb": torch.ones(batch_size, dtype=torch.bool),
                "left_wrist_0_rgb": wrist_mask,
                "right_wrist_0_rgb": torch.zeros(batch_size, dtype=torch.bool),
            },
            "state": state,
            # Same key names as pi0 Observation, but tokenized by CLIP tokenizer.
            "tokenized_prompt": tokens["input_ids"].long(),
            "tokenized_prompt_mask": tokens["attention_mask"].long(),
        }

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        obs = self._build_observation([s["obs"] for s in samples])
        next_obs = self._build_observation([s["next_obs"] for s in samples])

        actions = torch.as_tensor(np.stack([s["actions"] for s in samples]), dtype=torch.float32)
        rewards = torch.as_tensor(np.stack([s["rewards"] for s in samples]), dtype=torch.float32)
        is_terminal = torch.as_tensor(np.stack([s["is_terminal"] for s in samples]), dtype=torch.bool)
        from_success = torch.as_tensor(np.stack([s["from_success"] for s in samples]), dtype=torch.bool)

        return {
            "observation": obs,
            "next_observation": next_obs,
            "actions": actions,
            "rewards": rewards,
            "is_terminal": is_terminal,
            "from_success": from_success,
        }


def move_to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [move_to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(move_to_device(v, device) for v in x)
    return x


def infer_dims(dataset: LiberoLeRobotIQLDataset) -> tuple[int, int]:
    sample = dataset[0]
    state_dim = int(sample["obs"]["state"].shape[-1])
    action_dim = int(sample["actions"].shape[-1])
    return state_dim, action_dim


def build_model(args: TrainArgs, state_dim: int, action_dim: int) -> torch.nn.Module:
    # Preferred constructor for the pi0-style replacement model.
    try:
        return LightIQLCritic(
            args.encoder_name,
            state_dim,
            action_dim,
            args.horizon,
            hidden_dim=args.hidden_dim,
            num_q=args.num_q,
            action_layers=args.action_layers,
            q_layers=args.q_layers,
        )
    except TypeError:
        # Fallback for a LightVLValueModel-style constructor after replacing the input interface.
        return LightIQLCritic(
            encoder_name=args.encoder_name,
            robot_state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=args.horizon,
            num_q=args.num_q,
            d_model=args.hidden_dim,
            action_layers=args.action_layers,
            fusion_layers=args.q_layers,
            dropout=args.dropout,
            q_l2_coef=args.q_l2_coef,
        )


def loss_from_model_output(output: Any) -> tuple[torch.Tensor, dict[str, float]]:
    if isinstance(output, dict):
        if "loss" in output:
            loss = output["loss"]
        else:
            loss = output["critic_loss"] + output["value_loss"]
        metrics = {
            k: float(v.detach().cpu().item()) if torch.is_tensor(v) and v.numel() == 1 else v
            for k, v in output.items()
            if k != "loss"
        }
        return loss, metrics

    if isinstance(output, tuple):
        if len(output) < 2:
            raise ValueError("compute_loss tuple output must contain critic_loss and value_loss.")
        critic_loss, value_loss = output[0], output[1]
        loss = critic_loss + value_loss
        metrics = {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "value_loss": float(value_loss.detach().cpu().item()),
        }
        if len(output) >= 3 and isinstance(output[2], dict):
            for k, v in output[2].items():
                metrics[k] = float(v.detach().cpu().item()) if torch.is_tensor(v) and v.numel() == 1 else v
        return loss, metrics

    if torch.is_tensor(output):
        return output, {}

    raise TypeError(f"Unsupported compute_loss output type: {type(output)}")


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    args: TrainArgs,
    step: int,
    state_dim: int,
    action_dim: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_args = asdict(args)
    ckpt_args.update(
        {
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
            "horizon": int(args.horizon),
            "image_size": int(args.image_size),
            "hidden_dim": int(args.hidden_dim),
            "num_q": int(args.num_q),
            "action_layers": int(args.action_layers),
            "q_layers": int(args.q_layers),
            "encoder_name": args.encoder_name,
        }
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "args": ckpt_args,
            "step": int(step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        tmp,
    )
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(asdict(args), indent=2), encoding="utf-8")

    processor = CLIPProcessor.from_pretrained(args.encoder_name)

    dataset = LiberoLeRobotIQLDataset(
        repo_id=args.repo_id,
        root=args.root,
        horizon=args.horizon,
        step_reward=args.step_reward,
        success_terminal_reward=args.success_terminal_reward,
        failure_terminal_reward=args.failure_terminal_reward,
        default_episode_success=args.default_episode_success,
        use_wrist_image=args.use_wrist_image,
    )
    state_dim, action_dim = infer_dims(dataset)
    print(f"[IQL train] state_dim={state_dim}, action_dim={action_dim}, horizon={args.horizon}")

    collator = Pi0StyleIQLCollator(processor=processor, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=collator,
        persistent_workers=(args.num_workers > 0),
    )

    model = build_model(args, state_dim=state_dim, action_dim=action_dim).to(device)
    if hasattr(model, "freeze_backbone"):
        model.freeze_backbone()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        print(f"[IQL train] Resumed from {args.resume}, step={start_step}")

    model.train()
    step = start_step
    running: dict[str, list[float]] = defaultdict(list)

    progress = tqdm(total=args.max_steps, initial=start_step, desc="iql", dynamic_ncols=True)
    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break

            batch = move_to_device(batch, device)
            debug = args.debug_every > 0 and step % args.debug_every == 0

            # The replacement pi0-style model should accept these kwargs.
            # If your local model keeps a simpler signature, remove the extra kwargs there
            # rather than changing the batch format here.
            try:
                output = model.compute_loss(
                    batch,
                    gamma=args.gamma,
                    expectile=args.expectile,
                    debug=debug,
                    use_q_aug=args.use_q_aug,
                )
            except TypeError:
                output = model.compute_loss(
                    batch,
                    gamma=args.gamma,
                    expectile=args.expectile,
                )

            loss, metrics = loss_from_model_output(output)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()

            if hasattr(model, "update_target_q"):
                model.update_target_q(tau=args.tau)

            step += 1
            progress.update(1)

            running["loss"].append(float(loss.detach().cpu().item()))
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    running[k].append(float(v))

            if args.log_every > 0 and step % args.log_every == 0:
                msg = {"step": step}
                for k, vals in running.items():
                    if vals:
                        msg[k] = sum(vals) / len(vals)
                print("[IQL train]", json.dumps(msg, ensure_ascii=False))
                running.clear()

            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(
                    output_dir / f"step_{step}.pt",
                    model,
                    optimizer,
                    args,
                    step,
                    state_dim,
                    action_dim,
                )
                save_checkpoint(
                    output_dir / "latest.pt",
                    model,
                    optimizer,
                    args,
                    step,
                    state_dim,
                    action_dim,
                )

    save_checkpoint(output_dir / "final.pt", model, optimizer, args, step, state_dim, action_dim)
    save_checkpoint(output_dir / "latest.pt", model, optimizer, args, step, state_dim, action_dim)
    progress.close()
    print(f"[IQL train] finished. final checkpoint: {output_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
