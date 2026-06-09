#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strict-schema IQL trainer for collected LIBERO LeRobot data.

Input dataset: collector output, and ONLY this schema is accepted:
    image
    wrist_image
    state
    actions
    reward
    done
    success
    task
    episode_index

No alias/fallback names are allowed. This is intentional to prevent silent
schema drift.

The loader constructs:
    observation, next_observation, actions[H], rewards[H], is_terminal[H],
    from_success[H]

Use --horizon 5 for:
    pi0 action_horizon == replan_steps == IQL chunk_size.
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
from PIL import Image
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPProcessor

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except Exception as exc:
    raise ImportError("Could not import LeRobotDataset. Run in the OpenPI/LeRobot env.") from exc

try:
    from model import LightIQLCritic
except ImportError:
    try:
        from model import LightVLValueModel as LightIQLCritic
    except ImportError as exc:
        raise ImportError("Could not import LightIQLCritic/LightVLValueModel from local model.py.") from exc


IMAGE_KEY = "image"
WRIST_IMAGE_KEY = "wrist_image"
STATE_KEY = "state"
ACTION_KEY = "actions"
TASK_KEY = "task"
EPISODE_KEY = "episode_index"
REWARD_KEY = "reward"
DONE_KEY = "done"
SUCCESS_KEY = "success"


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
    p.add_argument("--repo-id", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--root", default="")
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

    p.add_argument("--use-wrist-image", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use-q-aug", action="store_true")

    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--debug-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=1000)
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


def make_lerobot_dataset(repo_id: str, root: str = "") -> LeRobotDataset:
    if root:
        try:
            return LeRobotDataset(repo_id, root=Path(root))
        except TypeError:
            return LeRobotDataset(repo_id, Path(root))
    return LeRobotDataset(repo_id)


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
    def __init__(self, repo_id: str, root: str, horizon: int, use_wrist_image: bool = True) -> None:
        super().__init__()
        self.ds = make_lerobot_dataset(repo_id, root)
        self.horizon = int(horizon)
        self.use_wrist_image = bool(use_wrist_image)

        self.global_indices: list[int] = []
        self.episode_of: list[int] = []
        self.local_pos_of: list[int] = []
        self.actions: list[np.ndarray] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.successes: list[bool] = []
        self.episodes: dict[int, list[int]] = defaultdict(list)

        self._build_index()

    def _build_index(self) -> None:
        print(f"[IQL data] strict scan: {len(self.ds)} frames")
        for global_idx in tqdm(range(len(self.ds)), desc="scan", leave=False):
            sample = self.ds[global_idx]

            ep = int(to_scalar(require_key(sample, EPISODE_KEY)))
            local_pos = len(self.episodes[ep])

            action = to_float_array(require_key(sample, ACTION_KEY)).reshape(-1)
            reward = float(to_scalar(require_key(sample, REWARD_KEY)))
            done = bool(to_scalar(require_key(sample, DONE_KEY)))
            success = bool(to_scalar(require_key(sample, SUCCESS_KEY)))

            internal_idx = len(self.global_indices)
            self.global_indices.append(global_idx)
            self.episode_of.append(ep)
            self.local_pos_of.append(local_pos)
            self.actions.append(action.astype(np.float32))
            self.rewards.append(reward)
            self.dones.append(done)
            self.successes.append(success)
            self.episodes[ep].append(internal_idx)

        if not self.global_indices:
            raise RuntimeError("Empty dataset.")

        print(
            f"[IQL data] frames={len(self.global_indices)} episodes={len(self.episodes)} "
            f"action_dim={self.actions[0].shape[-1]} horizon={self.horizon}"
        )

    def __len__(self) -> int:
        return len(self.global_indices)

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
            pos = min(unclamped_pos, ep_len - 1)
            idx = ep_indices[pos]
            actions[k] = self.actions[idx]
            rewards[k] = self.rewards[idx] if unclamped_pos <= ep_len - 1 else 0.0

            term = self.dones[idx] if unclamped_pos <= ep_len - 1 else True
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
        sample = self.ds[self.global_indices[internal_idx]]

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
            raise ValueError(f"Empty `{TASK_KEY}` at frame {self.global_indices[internal_idx]}")

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


def infer_dims(dataset: StrictLiberoIQLDataset) -> tuple[int, int]:
    sample = dataset[0]
    return int(sample["obs"]["state"].shape[-1]), int(sample["actions"].shape[-1])


def build_model(args: TrainArgs, state_dim: int, action_dim: int) -> torch.nn.Module:
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


def loss_from_output(output: Any) -> tuple[torch.Tensor, dict[str, float]]:
    if isinstance(output, dict):
        loss = output["loss"] if "loss" in output else output["critic_loss"] + output["value_loss"]
        metrics = {}
        for k, v in output.items():
            if k != "loss" and torch.is_tensor(v) and v.numel() == 1:
                metrics[k] = float(v.detach().cpu().item())
        return loss, metrics

    if isinstance(output, tuple):
        critic_loss, value_loss = output[:2]
        return critic_loss + value_loss, {
            "critic_loss": float(critic_loss.detach().cpu().item()),
            "value_loss": float(value_loss.detach().cpu().item()),
        }

    if torch.is_tensor(output):
        return output, {}

    raise TypeError(f"Unsupported compute_loss output type: {type(output)}")


def save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, args: TrainArgs, step: int, state_dim: int, action_dim: int) -> None:
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

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(asdict(args), indent=2), encoding="utf-8")

    processor = CLIPProcessor.from_pretrained(args.encoder_name)
    dataset = StrictLiberoIQLDataset(
        repo_id=args.repo_id,
        root=args.root,
        horizon=args.horizon,
        use_wrist_image=args.use_wrist_image,
    )
    state_dim, action_dim = infer_dims(dataset)
    print(f"[IQL train] state_dim={state_dim}, action_dim={action_dim}, horizon={args.horizon}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        collate_fn=Pi0StyleIQLCollator(processor=processor, image_size=args.image_size),
        persistent_workers=(args.num_workers > 0),
    )

    model = build_model(args, state_dim, action_dim).to(device)
    if hasattr(model, "freeze_backbone"):
        model.freeze_backbone()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        step = int(ckpt.get("step", 0))

    running: dict[str, list[float]] = defaultdict(list)
    pbar = tqdm(total=args.max_steps, initial=step, desc="iql", dynamic_ncols=True)

    model.train()
    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break

            batch = move_to_device(batch, device)
            debug = args.debug_every > 0 and step % args.debug_every == 0

            try:
                output = model.compute_loss(
                    batch,
                    gamma=args.gamma,
                    expectile=args.expectile,
                    debug=debug,
                    use_q_aug=args.use_q_aug,
                )
            except TypeError:
                output = model.compute_loss(batch, gamma=args.gamma, expectile=args.expectile)

            loss, metrics = loss_from_output(output)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                clip_grad_norm_(params, args.grad_clip)
            optimizer.step()

            if hasattr(model, "update_target_q"):
                model.update_target_q(tau=args.tau)

            step += 1
            pbar.update(1)
            running["loss"].append(float(loss.detach().cpu().item()))
            for k, v in metrics.items():
                running[k].append(float(v))

            if args.log_every > 0 and step % args.log_every == 0:
                msg = {"step": step}
                for k, vals in running.items():
                    if vals:
                        msg[k] = sum(vals) / len(vals)
                print("[IQL train]", json.dumps(msg, ensure_ascii=False))
                running.clear()

            if args.save_every > 0 and step % args.save_every == 0:
                save_checkpoint(out_dir / f"step_{step}.pt", model, optimizer, args, step, state_dim, action_dim)
                save_checkpoint(out_dir / "latest.pt", model, optimizer, args, step, state_dim, action_dim)

    save_checkpoint(out_dir / "final.pt", model, optimizer, args, step, state_dim, action_dim)
    save_checkpoint(out_dir / "latest.pt", model, optimizer, args, step, state_dim, action_dim)
    pbar.close()
    print(f"[IQL train] final checkpoint: {out_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
