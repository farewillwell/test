#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IQL selector for pi0 batched stochastic action chunks.

This module is used by Qselect/server.py.  It intentionally follows the current
ours/IQL/model.py interface:

    batch["observation"] in pi0-style format
    batch["actions"] with shape [N, chunk_size, action_dim]
    model.infer_batch(batch) -> (adv, q, v)

The selector does not depend on the old OpenVLA/OFT LightIQLCritic encode /
q_values API.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor


THIS_DIR = pathlib.Path(__file__).resolve().parent
IQL_DIR = THIS_DIR.parent / "IQL"
if str(IQL_DIR) not in sys.path:
    sys.path.insert(0, str(IQL_DIR))

try:
    from model import LightIQLCritic  # type: ignore
except ImportError:
    from model import LightVLValueModel as LightIQLCritic  # type: ignore


def _to_numpy(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, Image.Image):
        return np.asarray(x)
    return np.asarray(x)


def _to_pil_rgb(x: Any) -> Image.Image:
    if isinstance(x, Image.Image):
        return x.convert("RGB")

    arr = _to_numpy(x)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected image [H,W,3] or [3,H,W], got {tuple(arr.shape)}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got {tuple(arr.shape)}")

    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(arr)).convert("RGB")


def _move_to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: _move_to_device(v, device) for k, v in x.items()}
    return x


def _checkpoint_args(ckpt: dict[str, Any]) -> dict[str, Any]:
    args = dict(ckpt.get("args", {}))

    # Support checkpoints saved by LightVLValueModel.save_heads().
    for key in (
        "encoder_name",
        "robot_state_dim",
        "state_dim",
        "action_dim",
        "chunk_size",
        "horizon",
        "num_q",
        "d_model",
        "hidden_dim",
        "nhead",
        "action_layers",
        "fusion_layers",
        "q_layers",
        "dropout",
        "q_l2_coef",
        "encoder_amp",
    ):
        if key in ckpt and key not in args:
            args[key] = ckpt[key]

    return args


def _checkpoint_state_dict(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model", "state_dict", "model_state_dict"):
        if key in ckpt:
            return ckpt[key]
    return ckpt


def _build_model_from_checkpoint(
    critic_path: str | pathlib.Path,
    device: torch.device,
    encoder_name_override: str = "",
) -> torch.nn.Module:
    ckpt = torch.load(str(critic_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise TypeError(f"Expected torch checkpoint dict, got {type(ckpt)}")

    ckpt_args = _checkpoint_args(ckpt)
    encoder_name = encoder_name_override or str(ckpt_args.get("encoder_name", "openai/clip-vit-base-patch32"))
    state_dim = int(ckpt_args.get("state_dim", ckpt_args.get("robot_state_dim", 8)))
    action_dim = int(ckpt_args.get("action_dim", 7))
    chunk_size = int(ckpt_args.get("horizon", ckpt_args.get("chunk_size", 5)))
    hidden_dim = int(ckpt_args.get("hidden_dim", ckpt_args.get("d_model", 512)))
    num_q = int(ckpt_args.get("num_q", 2))
    nhead = int(ckpt_args.get("nhead", 8))
    action_layers = int(ckpt_args.get("action_layers", 2))
    q_layers = int(ckpt_args.get("q_layers", ckpt_args.get("fusion_layers", 2)))
    dropout = float(ckpt_args.get("dropout", 0.1))
    q_l2_coef = float(ckpt_args.get("q_l2_coef", 1e-4))
    encoder_amp = bool(ckpt_args.get("encoder_amp", True))

    try:
        model = LightIQLCritic(
            encoder_name,
            state_dim,
            action_dim,
            chunk_size,
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
            chunk_size=chunk_size,
            num_q=num_q,
            d_model=hidden_dim,
            nhead=nhead,
            action_layers=action_layers,
            fusion_layers=q_layers,
            dropout=dropout,
            q_l2_coef=q_l2_coef,
            encoder_amp=encoder_amp,
        )

    model.load_state_dict(_checkpoint_state_dict(ckpt))
    if hasattr(model, "freeze_backbone"):
        model.freeze_backbone()
    model.to(device)
    model.eval()
    return model


class QSelector:
    """Score candidate chunks with the current pi0-style IQL critic."""

    def __init__(
        self,
        critic_path: str | pathlib.Path,
        *,
        device: str | torch.device = "",
        encoder_name: str = "",
        image_size: int = 224,
    ) -> None:
        self.critic_path = str(critic_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size = int(image_size)

        self.model = _build_model_from_checkpoint(
            self.critic_path,
            device=self.device,
            encoder_name_override=encoder_name,
        )
        self.chunk_size = int(getattr(self.model, "chunk_size", 5))
        self.action_dim = int(getattr(self.model, "action_dim", 7))
        self.encoder_name = str(getattr(self.model, "encoder_name", encoder_name or "openai/clip-vit-base-patch32"))
        self.processor = CLIPProcessor.from_pretrained(self.encoder_name)

    def _process_one_image(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB").resize((self.image_size, self.image_size), Image.BILINEAR)
        return self.processor(images=[image], return_tensors="pt")["pixel_values"]

    def _build_observation(
        self,
        *,
        images: list[Any] | tuple[Any, ...],
        prompt: str,
        state: np.ndarray,
        batch_size: int,
    ) -> dict[str, Any]:
        if not images:
            raise ValueError("QSelector requires at least the base image.")

        base_image = _to_pil_rgb(images[0])
        if len(images) > 1 and images[1] is not None:
            wrist_image = _to_pil_rgb(images[1])
            wrist_mask = True
        else:
            wrist_image = Image.new("RGB", base_image.size, (0, 0, 0))
            wrist_mask = False

        base_pixels = self._process_one_image(base_image).repeat(batch_size, 1, 1, 1)
        wrist_pixels = self._process_one_image(wrist_image).repeat(batch_size, 1, 1, 1)
        zero_pixels = torch.zeros_like(base_pixels)

        state = np.asarray(state, dtype=np.float32).reshape(-1)
        state_batch = np.repeat(state[None, :], batch_size, axis=0)
        tokens = self.processor.tokenizer(
            [str(prompt)] * batch_size,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        return {
            "image": {
                "base_0_rgb": base_pixels,
                "left_wrist_0_rgb": wrist_pixels,
                "right_wrist_0_rgb": zero_pixels,
            },
            "image_mask": {
                "base_0_rgb": torch.ones(batch_size, dtype=torch.bool),
                "left_wrist_0_rgb": torch.full((batch_size,), wrist_mask, dtype=torch.bool),
                "right_wrist_0_rgb": torch.zeros(batch_size, dtype=torch.bool),
            },
            "state": torch.as_tensor(state_batch, dtype=torch.float32),
            "tokenized_prompt": tokens["input_ids"].long(),
            "tokenized_prompt_mask": tokens["attention_mask"].long(),
        }

    def build_batch(
        self,
        *,
        images: list[Any] | tuple[Any, ...],
        prompt: str,
        state: np.ndarray,
        candidates: np.ndarray,
    ) -> dict[str, Any]:
        candidates = np.asarray(candidates, dtype=np.float32)
        if candidates.ndim != 3:
            raise ValueError(f"Expected candidates [N,H,A], got {tuple(candidates.shape)}")
        if candidates.shape[0] <= 0:
            raise ValueError("Expected at least one candidate action chunk.")
        if candidates.shape[1] != self.chunk_size:
            raise ValueError(
                f"Candidate chunk length must match IQL chunk_size. "
                f"got H={candidates.shape[1]}, expected {self.chunk_size}"
            )
        if candidates.shape[2] != self.action_dim:
            raise ValueError(
                f"Candidate action dim must match IQL action_dim. "
                f"got A={candidates.shape[2]}, expected {self.action_dim}"
            )

        return {
            "observation": self._build_observation(
                images=images,
                prompt=prompt,
                state=state,
                batch_size=int(candidates.shape[0]),
            ),
            "actions": torch.as_tensor(candidates, dtype=torch.float32),
        }

    @torch.no_grad()
    def score(
        self,
        *,
        images: list[Any] | tuple[Any, ...],
        prompt: str,
        state: np.ndarray,
        candidates: np.ndarray,
    ) -> np.ndarray:
        """Return one Q score per candidate chunk."""
        batch = self.build_batch(images=images, prompt=prompt, state=state, candidates=candidates)
        batch = _move_to_device(batch, self.device)

        if hasattr(self.model, "infer_batch"):
            try:
                out = self.model.infer_batch(batch, device=self.device)
            except TypeError:
                out = self.model.infer_batch(batch)

            if isinstance(out, dict):
                q = out.get("target_q", out.get("q", out.get("adv")))
                if q is None:
                    raise KeyError(f"infer_batch output has no target_q/q/adv. keys={list(out.keys())}")
            else:
                values = tuple(out)
                if len(values) < 2:
                    raise ValueError("infer_batch tuple output must contain at least (adv, q)")
                q = values[1]
            return q.detach().cpu().float().numpy().reshape(-1)

        # Fallback for the current new-method interface. This is still not the
        # old OFT encode/q_values API.
        if not all(hasattr(self.model, name) for name in ("encode_observation", "get_q")):
            raise AttributeError("IQL model must expose infer_batch() or encode_observation()/get_q().")

        obs = batch["observation"]
        actions = batch["actions"]
        state_feat = self.model.encode_observation(obs)
        q_all = self.model.get_q(state_feat, actions, is_target=True).squeeze(-1)
        q = torch.min(q_all, dim=0)[0]
        return q.detach().cpu().float().numpy().reshape(-1)

    def select(
        self,
        *,
        images: list[Any] | tuple[Any, ...],
        prompt: str,
        state: np.ndarray,
        candidates: np.ndarray,
        mode: str = "qselect",
    ) -> tuple[int, np.ndarray]:
        """Return (best_index, scores) for candidates [N, chunk_size, action_dim]."""
        if mode != "qselect":
            raise ValueError(f"QSelector only supports qselect scoring, got {mode!r}")

        scores = self.score(images=images, prompt=prompt, state=state, candidates=candidates)
        if scores.ndim != 1 or scores.shape[0] != np.asarray(candidates).shape[0]:
            raise ValueError(f"Expected scores [N], got {tuple(scores.shape)}")
        best_idx = int(np.argmax(scores))
        return best_idx, scores.astype(np.float32)
