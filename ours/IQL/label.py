#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Strict-schema labeler: collected LIBERO LeRobot -> successful AWBC LeRobot.

Input dataset schema, exactly:
    image, wrist_image, state, actions, reward, done, success, task, episode_index

Output dataset schema, exactly:
    image, wrist_image, state, actions, adv, task

No alias/fallback names are allowed.
"""

from __future__ import annotations

import argparse
import json
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

try:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
except Exception as exc:
    raise ImportError("Could not import LeRobotDataset. Run in the OpenPI/LeRobot env.") from exc


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

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
SUCCESS_KEY = "success"
ADV_KEY = "adv"


@dataclass
class Args:
    input_repo_id: str
    output_repo_id: str
    critic_path: str
    input_root: str = ""
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
    p.add_argument("--input-repo-id", required=True)
    p.add_argument("--output-repo-id", required=True)
    p.add_argument("--critic-path", required=True)
    p.add_argument("--input-root", default="")
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


def ensure_uint8_hwc(x: Any, image_size: int) -> np.ndarray:
    img = to_pil_rgb(x).resize((image_size, image_size), Image.BILINEAR)
    return np.ascontiguousarray(np.asarray(img, dtype=np.uint8))


def move_to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    return x


class StrictCollectorIndex:
    def __init__(self, ds: LeRobotDataset) -> None:
        self.ds = ds
        self.global_indices: list[int] = []
        self.episode_of: list[int] = []
        self.local_pos_of: list[int] = []
        self.actions: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self.successes: list[bool] = []
        self.tasks: list[str] = []
        self.episodes: dict[int, list[int]] = defaultdict(list)
        self._scan()

    def _scan(self) -> None:
        print(f"[label] strict scan: {len(self.ds)} frames")
        for global_idx in tqdm(range(len(self.ds)), desc="scan", leave=False):
            sample = self.ds[global_idx]

            ep = int(to_scalar(require_key(sample, EPISODE_KEY)))
            local_pos = len(self.episodes[ep])
            action = to_float_array(require_key(sample, ACTION_KEY)).reshape(-1)
            state = to_float_array(require_key(sample, STATE_KEY)).reshape(-1)
            success = bool(to_scalar(require_key(sample, SUCCESS_KEY)))
            task = decode_task(require_key(sample, TASK_KEY)).strip()
            if not task:
                raise ValueError(f"Empty `{TASK_KEY}` at frame {global_idx}")

            internal_idx = len(self.global_indices)
            self.global_indices.append(global_idx)
            self.episode_of.append(ep)
            self.local_pos_of.append(local_pos)
            self.actions.append(action.astype(np.float32))
            self.states.append(state.astype(np.float32))
            self.successes.append(success)
            self.tasks.append(task)
            self.episodes[ep].append(internal_idx)

        if not self.global_indices:
            raise RuntimeError("Input dataset is empty.")

        print(f"[label] frames={len(self.global_indices)} episodes={len(self.episodes)}")

    def successful_episode_ids(self) -> list[int]:
        return sorted(ep for ep, indices in self.episodes.items() if any(self.successes[i] for i in indices))

    def read_frame_for_output(self, internal_idx: int, image_size: int) -> dict[str, Any]:
        sample = self.ds[self.global_indices[internal_idx]]
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

    try:
        model = LightIQLCritic(
            encoder_name,
            state_dim,
            action_dim,
            args.horizon,
            hidden_dim=hidden_dim,
            num_q=num_q,
            action_layers=action_layers,
            q_layers=q_layers,
        )
    except TypeError:
        model = LightIQLCritic(
            encoder_name=encoder_name,
            robot_state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=args.horizon,
            num_q=num_q,
            d_model=hidden_dim,
            action_layers=action_layers,
            fusion_layers=q_layers,
            dropout=dropout,
            q_l2_coef=q_l2_coef,
        )

    model.load_state_dict(ckpt.get("model", ckpt))
    if hasattr(model, "freeze_backbone"):
        model.freeze_backbone()
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def infer_adv(model: torch.nn.Module, batch: dict[str, Any], device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch = move_to_device(batch, device)

    if hasattr(model, "infer_batch"):
        try:
            out = model.infer_batch(batch, device_id=device)
        except TypeError:
            out = model.infer_batch(batch)

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

    raise AttributeError("Model must expose infer_batch(batch) for strict label script.")


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
            IMAGE_KEY: {"dtype": "image", "shape": (args.image_size, args.image_size, 3), "names": ["height", "width", "channel"]},
            WRIST_IMAGE_KEY: {"dtype": "image", "shape": (args.image_size, args.image_size, 3), "names": ["height", "width", "channel"]},
            STATE_KEY: {"dtype": "float32", "shape": (8,), "names": ["state"]},
            ACTION_KEY: {"dtype": "float32", "shape": (7,), "names": ["actions"]},
            ADV_KEY: {"dtype": "float32", "shape": (1,), "names": ["adv"]},
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


def label_episode(index: StrictCollectorIndex, ep: int, collator: Pi0StyleCriticCollator, model: torch.nn.Module, device: torch.device, args: Args):
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
        frames = [index.read_frame_for_output(ep_indices[i], args.image_size) for i in range(pos, end)]
        chunks = [index.make_action_chunk(ep_indices, i, args.horizon) for i in range(pos, end)]

        adv, q, v = infer_adv(model, collator.build_batch(frames, chunks), device)
        frames_all.extend(frames)
        adv_all.extend([float(x) for x in adv])
        q_all.extend([float(x) for x in q])
        v_all.extend([float(x) for x in v])
        pos = end

    stats = {
        "episode_id": int(ep),
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
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    input_ds = make_lerobot_dataset(args.input_repo_id, args.input_root)
    index = StrictCollectorIndex(input_ds)
    success_eps = index.successful_episode_ids()
    if args.max_episodes > 0:
        success_eps = success_eps[:args.max_episodes]
    if not success_eps:
        raise RuntimeError("No successful episodes found.")

    first = index.episodes[success_eps[0]][0]
    state_dim = int(index.states[first].shape[-1])
    action_dim = int(index.actions[first].shape[-1])

    model = build_model_from_ckpt(args, state_dim, action_dim, device)
    ckpt = torch.load(args.critic_path, map_location="cpu")
    ckpt_args = dict(ckpt.get("args", {}))
    encoder_name = args.encoder_name or ckpt_args.get("encoder_name", "openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained(encoder_name)
    collator = Pi0StyleCriticCollator(processor, args.image_size)
    out_ds = create_output_dataset(args)

    all_frames: list[list[dict[str, Any]]] = []
    all_advs: list[np.ndarray] = []
    stats = {
        "args": asdict(args),
        "input_frames": len(index.global_indices),
        "input_episodes": len(index.episodes),
        "successful_episodes": len(success_eps),
        "episode_stats": [],
    }

    try:
        for ep in tqdm(success_eps, desc="label"):
            frames, adv, ep_stats = label_episode(index, ep, collator, model, device, args)
            all_frames.append(frames)
            all_advs.append(adv)
            stats["episode_stats"].append(ep_stats)

        global_adv = np.concatenate([a for a in all_advs if len(a) > 0], axis=0)
        stats["raw_adv"] = {
            "mean": float(global_adv.mean()),
            "std": float(global_adv.std()),
            "min": float(global_adv.min()),
            "max": float(global_adv.max()),
        }

        if args.normalize_adv:
            mean = float(global_adv.mean())
            std = float(global_adv.std())
            all_advs = [((adv - mean) / max(std, 1e-6)).astype(np.float32) for adv in all_advs]

        if args.clamp_adv and args.clamp_adv > 0:
            c = float(args.clamp_adv)
            all_advs = [np.clip(adv, -c, c).astype(np.float32) for adv in all_advs]

        written_episodes = 0
        written_frames = 0
        written_adv = []
        for frames, adv in tqdm(list(zip(all_frames, all_advs)), desc="write"):
            if len(frames) == 0:
                continue
            n = write_labeled_episode(out_ds, frames, adv)
            written_episodes += 1
            written_frames += n
            written_adv.extend([float(x) for x in adv])

        written_adv = np.asarray(written_adv, dtype=np.float32)
        stats["written"] = {
            "episodes": int(written_episodes),
            "frames": int(written_frames),
            "adv_mean": float(written_adv.mean()) if written_adv.size else 0.0,
            "adv_std": float(written_adv.std()) if written_adv.size else 0.0,
            "adv_min": float(written_adv.min()) if written_adv.size else 0.0,
            "adv_max": float(written_adv.max()) if written_adv.size else 0.0,
        }

    finally:
        finish_dataset(out_ds)

    output_path = HF_LEROBOT_HOME / args.output_repo_id
    (output_path / "iql_label_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[label] saved: {output_path}")
    print(json.dumps(stats["written"], indent=2))


if __name__ == "__main__":
    main()
