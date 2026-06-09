from __future__ import annotations

import argparse
import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from common import build_action_chunks, episode_parquet_files, episode_reward, image_cell_to_pil, load_episode_df, load_episode_metadata, load_tasks


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return weight * diff.square()


class LeRobotIQLDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        processor: CLIPProcessor,
        horizon: int,
        image_size: int,
        views: list[str],
        default_success: bool,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.horizon = horizon
        self.image_size = image_size
        self.views = views
        self.tasks = load_tasks(data_dir)
        self.episode_meta = load_episode_metadata(data_dir)
        self.items: list[tuple[Path, int, float]] = []
        for path in episode_parquet_files(data_dir):
            episode_index = int(path.stem.split("_")[-1])
            reward = episode_reward(self.episode_meta.get(episode_index, {}), default_success=default_success)
            df = load_episode_df(path)
            for i in range(len(df)):
                self.items.append((path, i, reward if i == len(df) - 1 else 0.0))
        self._cache_path: Path | None = None
        self._cache_df = None
        self._cache_chunks = None

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, path: Path):
        if self._cache_path != path:
            df = load_episode_df(path)
            actions = np.stack(df["actions"].to_list()).astype(np.float32)
            self._cache_path = path
            self._cache_df = df
            self._cache_chunks = build_action_chunks(actions, self.horizon)
        return self._cache_df, self._cache_chunks

    def __getitem__(self, idx: int) -> dict:
        path, frame_idx, reward = self.items[idx]
        df, chunks = self._load(path)
        row = df.iloc[frame_idx]
        task = self.tasks.get(int(row["task_index"]), "")
        images = []
        for view in self.views:
            if view in row:
                images.append(image_cell_to_pil(row[view]).resize((self.image_size, self.image_size)))
        if not images:
            raise KeyError(f"No requested views {self.views} found in {path}")
        state = np.asarray(row["state"], dtype=np.float32)
        action_chunk = chunks[frame_idx]
        next_idx = min(frame_idx + 1, len(df) - 1)
        next_row = df.iloc[next_idx]
        next_state = np.asarray(next_row["state"], dtype=np.float32)
        done = float(frame_idx == len(df) - 1)
        return {
            "images": images,
            "text": task,
            "state": state,
            "actions": action_chunk,
            "next_images": [image_cell_to_pil(next_row[v]).resize((self.image_size, self.image_size)) for v in self.views if v in next_row],
            "next_state": next_state,
            "reward": np.float32(reward),
            "done": np.float32(done),
        }


def collate(batch: list[dict], processor: CLIPProcessor) -> dict:
    flat_images = []
    next_flat_images = []
    view_counts = []
    next_view_counts = []
    texts = []
    for item in batch:
        flat_images.extend(item["images"])
        next_flat_images.extend(item["next_images"])
        view_counts.append(len(item["images"]))
        next_view_counts.append(len(item["next_images"]))
        texts.append(item["text"])
    enc = processor(text=texts, images=flat_images, return_tensors="pt", padding=True, truncation=True)
    next_enc = processor(text=texts, images=next_flat_images, return_tensors="pt", padding=True, truncation=True)
    return {
        "pixel_values": enc["pixel_values"],
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "next_pixel_values": next_enc["pixel_values"],
        "next_input_ids": next_enc["input_ids"],
        "next_attention_mask": next_enc["attention_mask"],
        "view_counts": torch.tensor(view_counts),
        "next_view_counts": torch.tensor(next_view_counts),
        "state": torch.tensor(np.stack([x["state"] for x in batch]), dtype=torch.float32),
        "next_state": torch.tensor(np.stack([x["next_state"] for x in batch]), dtype=torch.float32),
        "actions": torch.tensor(np.stack([x["actions"] for x in batch]), dtype=torch.float32),
        "reward": torch.tensor([x["reward"] for x in batch], dtype=torch.float32),
        "done": torch.tensor([x["done"] for x in batch], dtype=torch.float32),
    }


class ActionChunkEncoder(nn.Module):
    def __init__(self, action_dim: int, horizon: int, hidden_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        self.input_proj = nn.Linear(action_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, horizon, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(actions.float()) + self.pos[:, : actions.shape[1]]
        return self.norm(self.encoder(x))


class TransformerQHead(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 2) -> None:
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.film = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim * 2))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fuser = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, state_feat: torch.Tensor, action_tokens: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(state_feat).chunk(2, dim=-1)
        conditioned_actions = action_tokens * (1.0 + scale[:, None]) + shift[:, None]
        cls = self.cls.expand(state_feat.shape[0], -1, -1) + state_feat[:, None]
        fused = self.fuser(torch.cat([cls, conditioned_actions], dim=1))
        return self.out(fused[:, 0])


class LightIQLCritic(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        state_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dim: int = 512,
        num_q: int = 2,
        action_layers: int = 2,
        q_layers: int = 2,
    ) -> None:
        super().__init__()
        self.num_q = num_q
        self.clip = CLIPModel.from_pretrained(encoder_name)
        for p in self.clip.parameters():
            p.requires_grad = False
        clip_dim = self.clip.config.projection_dim
        self.state_proj = nn.Sequential(
            nn.Linear(clip_dim * 2 + state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.action_encoder = ActionChunkEncoder(action_dim, horizon, hidden_dim, num_layers=action_layers)
        self.q_heads = nn.ModuleList([TransformerQHead(hidden_dim, num_layers=q_layers) for _ in range(num_q)])
        self.target_q_heads = copy.deepcopy(self.q_heads)
        for p in self.target_q_heads.parameters():
            p.requires_grad = False
        self.v = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def encode(self, pixel_values, input_ids, attention_mask, view_counts, state):
        with torch.no_grad():
            img = self.clip.get_image_features(pixel_values=pixel_values)
            txt = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        views = []
        offset = 0
        for count in view_counts.tolist():
            views.append(img[offset : offset + count].mean(dim=0))
            offset += count
        img = torch.stack(views, dim=0)
        img = F.normalize(img.float(), dim=-1)
        txt = F.normalize(txt.float(), dim=-1)
        return self.state_proj(torch.cat([img, txt, state.float()], dim=-1))

    def q_values(self, state_feat, actions, target=False):
        action_tokens = self.action_encoder(actions)
        heads = self.target_q_heads if target else self.q_heads
        return torch.stack([head(state_feat, action_tokens) for head in heads], dim=0)

    def q(self, state_feat, actions, target=False):
        qs = self.q_values(state_feat, actions, target=target)
        if qs.shape[0] == 1:
            return qs[0], qs[0]
        return qs[0], qs[1]

    def update_target(self, tau: float) -> None:
        for src, dst in zip(self.q_heads.parameters(), self.target_q_heads.parameters()):
            dst.data.mul_(1.0 - tau).add_(src.data, alpha=tau)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--encoder-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--views", default="image")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-q", type=int, default=2)
    parser.add_argument("--action-layers", type=int, default=2)
    parser.add_argument("--q-layers", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--tau", type=float, default=0.02)
    parser.add_argument("--default-success", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPProcessor.from_pretrained(args.encoder_name)
    dataset = LeRobotIQLDataset(Path(args.data_dir), processor, args.horizon, args.image_size, args.views.split(","), args.default_success)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=lambda b: collate(b, processor))
    sample = dataset[0]
    model = LightIQLCritic(
        args.encoder_name,
        len(sample["state"]),
        sample["actions"].shape[-1],
        args.horizon,
        hidden_dim=args.hidden_dim,
        num_q=args.num_q,
        action_layers=args.action_layers,
        q_layers=args.q_layers,
    ).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    pbar = tqdm(total=args.max_steps)
    while step < args.max_steps:
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            s = model.encode(batch["pixel_values"], batch["input_ids"], batch["attention_mask"], batch["view_counts"], batch["state"])
            ns = model.encode(batch["next_pixel_values"], batch["next_input_ids"], batch["next_attention_mask"], batch["next_view_counts"], batch["next_state"])
            qs = model.q_values(s, batch["actions"])
            q = qs.min(dim=0).values.squeeze(-1)
            v = model.v(s).squeeze(-1)
            with torch.no_grad():
                nv = model.v(ns).squeeze(-1)
                target = batch["reward"] + args.gamma * (1.0 - batch["done"]) * nv
            q_loss = F.mse_loss(qs.squeeze(-1), target[None].expand(qs.shape[0], -1))
            v_loss = expectile_loss(q.detach() - v, args.expectile).mean()
            loss = q_loss + v_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.update_target(args.tau)
            step += 1
            pbar.update(1)
            if step % 100 == 0:
                pbar.set_description(f"loss={loss.item():.4f} q={q_loss.item():.4f} v={v_loss.item():.4f}")
            if step >= args.max_steps:
                break
    pbar.close()
    torch.save({"model": model.state_dict(), "args": vars(args)}, save_dir / "final.pt")


if __name__ == "__main__":
    main()
