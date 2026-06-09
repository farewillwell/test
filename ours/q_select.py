from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor

from light_iql import LightIQLCritic


def _to_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


class QSelector:
    """Persistent IQL critic wrapper for scoring candidate action chunks."""

    def __init__(self, critic_path: str | Path, device: str | None = None) -> None:
        self.critic_path = Path(critic_path)
        self.ckpt = torch.load(self.critic_path, map_location="cpu")
        self.cfg = dict(self.ckpt["args"])
        self.processor = CLIPProcessor.from_pretrained(self.cfg["encoder_name"])
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: LightIQLCritic | None = None
        self.model_shape: tuple[int, int, int] | None = None

    def _ensure_model(self, state_dim: int, action_dim: int, horizon: int) -> LightIQLCritic:
        shape = (state_dim, action_dim, horizon)
        if self.model is not None and self.model_shape == shape:
            return self.model
        model = LightIQLCritic(
            self.cfg["encoder_name"],
            state_dim,
            action_dim,
            horizon,
            hidden_dim=int(self.cfg.get("hidden_dim", 512)),
            num_q=int(self.cfg.get("num_q", 2)),
            action_layers=int(self.cfg.get("action_layers", 2)),
            q_layers=int(self.cfg.get("q_layers", 2)),
        )
        model.load_state_dict(self.ckpt["model"])
        model.to(self.device).eval()
        self.model = model
        self.model_shape = shape
        return model

    @torch.no_grad()
    def score(self, images: Iterable, prompt: str, state: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        candidates = np.asarray(candidates, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        if candidates.ndim != 3:
            raise ValueError(f"Expected candidates [N,H,A], got {candidates.shape}")
        if state.ndim != 1:
            raise ValueError(f"Expected state [S], got {state.shape}")

        model = self._ensure_model(state.shape[-1], candidates.shape[-1], candidates.shape[1])
        image_size = int(self.cfg.get("image_size", 224))
        pil_images = [_to_pil(image).resize((image_size, image_size)) for image in images]
        if not pil_images:
            raise ValueError("QSelector requires at least one image")

        enc = self.processor(text=[prompt], images=pil_images, return_tensors="pt", padding=True, truncation=True)
        view_counts = torch.tensor([len(pil_images)], device=self.device)
        state_tensor = torch.tensor(state[None], dtype=torch.float32, device=self.device)
        state_feat = model.encode(
            enc["pixel_values"].to(self.device),
            enc["input_ids"].to(self.device),
            enc["attention_mask"].to(self.device),
            view_counts,
            state_tensor,
        )
        state_feat = state_feat.repeat(candidates.shape[0], 1)
        actions = torch.tensor(candidates, dtype=torch.float32, device=self.device)
        q_values = model.q_values(state_feat, actions).min(dim=0).values.squeeze(-1)
        return q_values.detach().cpu().float().numpy()

    def select(
        self,
        images: Iterable,
        prompt: str,
        state: np.ndarray,
        candidates: np.ndarray,
        mode: str = "qselect",
    ) -> tuple[int, np.ndarray]:
        candidates = np.asarray(candidates, dtype=np.float32)
        if mode == "first":
            return 0, np.zeros((candidates.shape[0],), dtype=np.float32)
        if mode == "random":
            index = random.randrange(candidates.shape[0])
            return index, np.zeros((candidates.shape[0],), dtype=np.float32)
        if mode not in {"qselect", "best"}:
            raise ValueError(f"Unknown sample mode: {mode}")
        scores = self.score(images, prompt, state, candidates)
        return int(np.argmax(scores)), scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--critic-path", required=True)
    parser.add_argument("--image", required=True, help="Main-view image path.")
    parser.add_argument("--wrist-image", default="", help="Optional wrist image path. If given, views are averaged.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--state", required=True, help="Comma-separated state vector.")
    parser.add_argument("--candidates", required=True, help=".npy file with [N,H,A] action chunks.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", default="qselect", choices=("qselect", "best", "random", "first"))
    args = parser.parse_args()

    state = np.asarray([float(x) for x in args.state.split(",")], dtype=np.float32)
    candidates = np.load(args.candidates).astype(np.float32)
    images = [Image.open(args.image).convert("RGB")]
    if args.wrist_image:
        images.append(Image.open(args.wrist_image).convert("RGB"))

    selector = QSelector(args.critic_path)
    best, scores = selector.select(images, args.prompt, state, candidates, args.mode)
    result = {"best_index": best, "best_q": float(scores[best]), "q_values": scores.tolist()}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
