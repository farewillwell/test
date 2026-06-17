#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strict-schema IQL trainer for collected LIBERO LeRobot data.

DDP launch contract:
    torchrun --standalone --nnodes 1 --nproc-per-node N train.py ...

This follows the original Light-VL IQL style:
    PartialState -> local_process_index -> torch.cuda.set_device(device_id)
    critic = DDP(critic, device_ids=[device_id])
    critic.module.compute_loss(...)
    rank0/main process saves checkpoints

Only the data pipeline is different: this file reads a list of collector-style
LeRobot repos and treats them as one virtual pool.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import PartialState
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import CLIPProcessor

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from model import LightIQLCritic

IMAGE_KEY = "image"
WRIST_IMAGE_KEY = "wrist_image"
STATE_KEY = "state"
ACTION_KEY = "actions"
TASK_KEY = "task"
EPISODE_KEY = "episode_index"
REWARD_KEY = "reward"
TERMINAL_KEY = "terminal"
SUCCESS_KEY = "success"


@dataclass
class TrainArgs:
    repo_dirs: list[str]
    output_dir: str
    encoder_name: str = "openai/clip-vit-base-patch32"
    critic_encoder_mode: str = "full_pi0"

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

    use_wrist_image: bool = True
    use_q_aug: bool = False

    log_every: int = 50
    debug_every: int = 500
    save_every: int = 1000000
    seed: int = 7
    device: str = "cuda"
    resume: str = ""


def parse_args() -> TrainArgs:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--repo-dirs",
        nargs="+",
        required=True,
        help=(
            "List of local LeRobot repo directories. Example: "
            "workspace/pool/raw workspace/iter0/data/collect"
        ),
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--encoder-name", default="openai/clip-vit-base-patch32")
    p.add_argument(
        "--critic-encoder-mode",
        default="full_pi0",
        choices=("full_pi0", "oft_single_view"),
    )

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

    p.add_argument("--use-wrist-image", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-q-aug", action="store_true")

    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--debug-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=1000000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", default="")
    return TrainArgs(**vars(p.parse_args()))


def require_key(sample: dict[str, Any], key: str) -> Any:
    if key not in sample:
        raise KeyError(f"Required key `{key}` not found. Available keys: {sorted(sample.keys())}")
    return sample[key]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = True) -> DDP:
    return DDP(
        module,
        device_ids=[device_id],
        find_unused_parameters=find_unused,
        gradient_as_bucket_view=True,
    )


def open_lerobot_dataset_from_dir(repo_dir: str | Path) -> LeRobotDataset:
    """Open a local LeRobot repo directory.

    For a repo at /x/workspace/pool/raw, use:
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


class StrictLiberoIQLDataset(Dataset):
    """Virtual concatenation of multiple collector-style LeRobot repos.

    The physical data is not merged. Each sample is addressed by an internal
    index that maps to (repo_idx, frame_idx). Episode ids are namespaced by
    repo_idx to avoid collisions between repos that both contain episode_index=0.
    """

    def __init__(self, repo_dirs: list[str], horizon: int, use_wrist_image: bool = True) -> None:
        super().__init__()
        if not repo_dirs:
            raise ValueError("repo_dirs must be non-empty.")

        self.repo_dirs = [Path(x).resolve() for x in repo_dirs]
        self.datasets: list[LeRobotDataset] = [
            open_lerobot_dataset_from_dir(x) for x in self.repo_dirs
        ]

        self.horizon = int(horizon)
        self.use_wrist_image = bool(use_wrist_image)

        # internal_idx -> (repo_idx, frame_idx_inside_repo)
        self.frame_refs: list[tuple[int, int]] = []

        # internal_idx -> (repo_idx, episode_index_inside_repo)
        self.episode_of: list[tuple[int, int]] = []
        self.local_pos_of: list[int] = []

        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.terminals: list[bool] = []
        self.successes: list[bool] = []

        self.episodes: dict[tuple[int, int], list[int]] = defaultdict(list)

        self._build_index()

    def _build_index(self) -> None:
        total_raw_frames = sum(len(ds) for ds in self.datasets)
        print(
            f"[IQL data] pool scan: repos={len(self.datasets)} raw_frames={total_raw_frames}",
            flush=True,
        )

        for repo_idx, ds in enumerate(self.datasets):
            repo_dir = self.repo_dirs[repo_idx]
            print(f"[IQL data] repo[{repo_idx}] {repo_dir}: frames={len(ds)}", flush=True)

            for frame_idx in tqdm(range(len(ds)), desc=f"scan repo{repo_idx}", leave=False):
                sample = ds[frame_idx]

                ep_raw = int(to_scalar(require_key(sample, EPISODE_KEY)))
                ep_key = (repo_idx, ep_raw)
                local_pos = len(self.episodes[ep_key])

                action = to_float_array(require_key(sample, ACTION_KEY)).reshape(-1)
                reward = float(to_scalar(require_key(sample, REWARD_KEY)))
                is_terminal = bool(to_scalar(require_key(sample, TERMINAL_KEY)))
                success = bool(to_scalar(require_key(sample, SUCCESS_KEY)))

                internal_idx = len(self.frame_refs)

                self.frame_refs.append((repo_idx, frame_idx))
                self.episode_of.append(ep_key)
                self.local_pos_of.append(local_pos)

                self.actions.append(action.astype(np.float32))
                self.rewards.append(reward)
                self.terminals.append(is_terminal)
                self.successes.append(success)

                self.episodes[ep_key].append(internal_idx)

        if not self.frame_refs:
            raise RuntimeError("Empty dataset.")

        print(
            f"[IQL data] frames={len(self.frame_refs)} "
            f"episodes={len(self.episodes)} "
            f"repos={len(self.datasets)} "
            f"action_dim={self.actions[0].shape[-1]} "
            f"horizon={self.horizon} "
            f"success_frames={int(sum(bool(x) for x in self.successes))} "
            f"terminal_frames={int(sum(bool(x) for x in self.terminals))}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.frame_refs)

    def _chunk_arrays(self, internal_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, int]:
        ep = self.episode_of[internal_idx]
        ep_indices = self.episodes[ep]
        local_pos = self.local_pos_of[internal_idx]
        ep_len = len(ep_indices)

        action_dim = self.actions[internal_idx].shape[-1]
        actions = np.zeros((self.horizon, action_dim), dtype=np.float32)
        rewards = np.zeros((self.horizon,), dtype=np.float32)
        is_terminal = np.zeros((self.horizon,), dtype=np.bool_)

        first_terminal_seen = False

        for k in range(self.horizon):
            unclamped_pos = local_pos + k
            in_episode = unclamped_pos <= ep_len - 1

            pos = min(unclamped_pos, ep_len - 1)
            idx = ep_indices[pos]

            actions[k] = self.actions[idx]

            if in_episode:
                rewards[k] = self.rewards[idx]
                term = bool(self.terminals[idx])
            else:
                rewards[k] = 0.0
                term = True

            if first_terminal_seen:
                term = True

            if term:
                first_terminal_seen = True

            is_terminal[k] = term

        next_pos = min(local_pos + self.horizon, ep_len - 1)
        next_internal_idx = ep_indices[next_pos]

        ep_success = bool(any(self.successes[i] for i in ep_indices))

        return actions, rewards, is_terminal, ep_success, next_internal_idx

    def _read_observation(self, internal_idx: int) -> dict[str, Any]:
        repo_idx, frame_idx = self.frame_refs[internal_idx]
        ds = self.datasets[repo_idx]

        sample = ds[frame_idx]

        base = to_pil_rgb(require_key(sample, IMAGE_KEY))

        if self.use_wrist_image:
            wrist = to_pil_rgb(require_key(sample, WRIST_IMAGE_KEY))
            wrist_mask = True
        else:
            wrist = Image.new("RGB", base.size, (0, 0, 0))
            wrist_mask = False

        state = to_float_array(require_key(sample, STATE_KEY)).reshape(-1)
        task = decode_task(require_key(sample, TASK_KEY)).strip()

        if not task:
            raise ValueError(
                f"Empty `{TASK_KEY}` at internal_idx={internal_idx}, "
                f"repo_idx={repo_idx}, frame_idx={frame_idx}"
            )

        return {
            "base_image": base,
            "wrist_image": wrist,
            "wrist_mask": wrist_mask,
            "state": state.astype(np.float32),
            "task": task,
        }

    def __getitem__(self, internal_idx: int) -> dict[str, Any]:
        actions, rewards, is_terminal, ep_success, next_internal_idx = self._chunk_arrays(internal_idx)

        obs = self._read_observation(internal_idx)
        next_obs = self._read_observation(next_internal_idx)
        next_obs["task"] = obs["task"]

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
        return self.processor(images=images, return_tensors="pt")["pixel_values"]

    def _build_observation(self, obs_list: list[dict[str, Any]]) -> dict[str, Any]:
        base_images = [self._resize(o["base_image"]) for o in obs_list]
        wrist_images = [self._resize(o["wrist_image"]) for o in obs_list]
        zero_images = [Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0)) for _ in obs_list]

        prompts = [str(o["task"]) for o in obs_list]
        tokens = self.processor.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)

        bsz = len(obs_list)
        return {
            "image": {
                "base_0_rgb": self._process_images(base_images),
                "left_wrist_0_rgb": self._process_images(wrist_images),
                "right_wrist_0_rgb": self._process_images(zero_images),
            },
            "image_mask": {
                "base_0_rgb": torch.ones(bsz, dtype=torch.bool),
                "left_wrist_0_rgb": torch.as_tensor([bool(o["wrist_mask"]) for o in obs_list], dtype=torch.bool),
                "right_wrist_0_rgb": torch.zeros(bsz, dtype=torch.bool),
            },
            "state": torch.as_tensor(np.stack([o["state"] for o in obs_list]), dtype=torch.float32),
            "tokenized_prompt": tokens["input_ids"].long(),
            "tokenized_prompt_mask": tokens["attention_mask"].long(),
        }

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "observation": self._build_observation([s["obs"] for s in samples]),
            "next_observation": self._build_observation([s["next_obs"] for s in samples]),
            "actions": torch.as_tensor(np.stack([s["actions"] for s in samples]), dtype=torch.float32),
            "rewards": torch.as_tensor(np.stack([s["rewards"] for s in samples]), dtype=torch.float32),
            "is_terminal": torch.as_tensor(np.stack([s["is_terminal"] for s in samples]), dtype=torch.bool),
            "from_success": torch.as_tensor(np.stack([s["from_success"] for s in samples]), dtype=torch.bool),
        }


def move_to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    return x


def infer_dims(dataset: Dataset) -> tuple[int, int]:
    sample = dataset[0]
    return int(sample["obs"]["state"].shape[-1]), int(sample["actions"].shape[-1])


def build_model(args: TrainArgs, state_dim: int, action_dim: int) -> torch.nn.Module:
    return LightIQLCritic(
        args.encoder_name,
        state_dim,
        action_dim,
        args.horizon,
        hidden_dim=args.hidden_dim,
        num_q=args.num_q,
        action_layers=args.action_layers,
        q_layers=args.q_layers,
        dropout=args.dropout,
        q_l2_coef=args.q_l2_coef,
        encoder_mode=args.critic_encoder_mode,
    )


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
    ckpt_args.update({
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "horizon": int(args.horizon),
        "image_size": int(args.image_size),
        "hidden_dim": int(args.hidden_dim),
        "num_q": int(args.num_q),
        "action_layers": int(args.action_layers),
        "q_layers": int(args.q_layers),
        "encoder_name": args.encoder_name,
    })
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"args": ckpt_args, "step": int(step), "model": model.state_dict(), "optimizer": optimizer.state_dict()}, tmp)
    os.replace(tmp, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    device = torch.device(f"cuda:{device_id}")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    out_dir = Path(args.output_dir)
    if distributed_state.is_main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "args.json").write_text(json.dumps(asdict(args), indent=2), encoding="utf-8")

    processor = CLIPProcessor.from_pretrained(args.encoder_name)
    dataset = StrictLiberoIQLDataset(
        repo_dirs=args.repo_dirs,
        horizon=args.horizon,
        use_wrist_image=args.use_wrist_image,
    )
    state_dim, action_dim = infer_dims(dataset)
    if distributed_state.is_main_process:
        print(f"[IQL train] state_dim={state_dim}, action_dim={action_dim}, horizon={args.horizon}", flush=True)

    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=Pi0StyleIQLCollator(processor=processor, image_size=args.image_size),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    critic = build_model(args, state_dim, action_dim).to(device_id)

    step = 0
    resume_optimizer_state = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        critic.load_state_dict(ckpt["model"])
        resume_optimizer_state = ckpt["optimizer"]
        step = int(ckpt.get("step", 0))

    critic.freeze_backbone()
    critic = wrap_ddp(critic, device_id, find_unused=True)
    if distributed_state.is_main_process:
        print("[load light-vl critic] finish !!!!", flush=True)

    trainable_params = [p for p in critic.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)

    total_losses: list[float] = []
    critic_losses: list[float] = []
    value_losses: list[float] = []
    running: dict[str, list[float]] = defaultdict(list)

    progress = tqdm(total=args.max_steps, initial=step, leave=False, disable=not distributed_state.is_main_process, desc="iql")
    critic.train()
    optimizer.zero_grad(set_to_none=True)

    epoch = 0
    while step < args.max_steps:
        sampler.set_epoch(epoch)
        epoch += 1

        for batch_idx, batch in enumerate(loader):
            if step >= args.max_steps:
                break

            batch = move_to_device(batch, device)
            debug = args.debug_every > 0 and step % args.debug_every == 0 and distributed_state.is_main_process

            critic_loss, value_loss, metrics = critic.module.compute_loss(
                batch=batch,
                gamma=args.gamma,
                expectile=args.expectile,
                debug=debug,
                use_q_aug=args.use_q_aug,
            )
            loss = critic_loss + value_loss

            loss.backward()
            if args.grad_clip > 0:
                clip_grad_norm_(trainable_params, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            critic.module.update_target_q(tau=args.tau)

            step += 1

            if distributed_state.is_main_process:
                loss_value = float(loss.detach().cpu().item())
                critic_loss_value = float(critic_loss.detach().cpu().item())
                value_loss_value = float(value_loss.detach().cpu().item())
                total_losses.append(loss_value)
                critic_losses.append(critic_loss_value)
                value_losses.append(value_loss_value)
                running["loss"].append(loss_value)
                running["critic_loss"].append(critic_loss_value)
                running["value_loss"].append(value_loss_value)
                for metric_key, metric_value in metrics.items():
                    running[metric_key].append(float(metric_value))
                progress.update(1)

                if args.log_every > 0 and step % args.log_every == 0:
                    msg = {"step": step}
                    for k, vals in running.items():
                        if vals:
                            msg[k] = sum(vals) / len(vals)
                    print("[IQL train]", json.dumps(msg, ensure_ascii=False), flush=True)
                    running.clear()

                if args.save_every > 0 and step % args.save_every == 0:
                    save_checkpoint(out_dir / f"step_{step}.pt", critic.module, optimizer, args, step, state_dim, action_dim)
                    save_checkpoint(out_dir / "latest.pt", critic.module, optimizer, args, step, state_dim, action_dim)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    if distributed_state.is_main_process:
        save_checkpoint(out_dir / "final.pt", critic.module, optimizer, args, step, state_dim, action_dim)
        save_checkpoint(out_dir / "latest.pt", critic.module, optimizer, args, step, state_dim, action_dim)
        progress.close()
        print(f"[IQL train] final checkpoint: {out_dir / 'final.pt'}", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
