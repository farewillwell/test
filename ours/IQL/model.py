"""
Pi0-input IQL critic.

This file is intended to REPLACE the old OFT-style LightVLValueModel.
It uses the same observation structure as pi0/OpenPI, except that images/text are
encoded by a frozen CLIP encoder.

Expected batch format:
    batch = {
        "observation": {
            "image": {
                "base_0_rgb":        Tensor[B, 3, H, W],
                "left_wrist_0_rgb":  Tensor[B, 3, H, W],
                "right_wrist_0_rgb": Tensor[B, 3, H, W],   # optional
            },
            "image_mask": {
                "base_0_rgb":        BoolTensor[B],
                "left_wrist_0_rgb":  BoolTensor[B],
                "right_wrist_0_rgb": BoolTensor[B],       # optional
            },
            "state": Tensor[B, S],
            "tokenized_prompt": Tensor[B, L],              # CLIP tokenizer ids
            "tokenized_prompt_mask": Tensor[B, L],         # CLIP attention mask
        },
        "next_observation": { ... same keys ... },
        "actions": Tensor[B, H, A],                        # H == replan_steps
        "rewards": Tensor[B, H],
        "is_terminal": BoolTensor[B, H],
        "from_success": BoolTensor[B] or BoolTensor[B, H],  # optional
    }

The critic score is for exactly the action chunk it receives. For your pi0 setup,
set chunk_size == action_horizon == replan_steps, e.g. 5.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


def expectile_loss(diff: torch.Tensor, expectile: float = 0.7) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return weight * diff.pow(2)


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Pi0ObservationEncoder(nn.Module):
    """
    Encode pi0/OpenPI-style observation with frozen CLIP.

    Input observation keys:
        image: dict[view_name, Tensor[B,3,H,W]]
        image_mask: dict[view_name, BoolTensor[B]]
        state: Tensor[B,S]
        tokenized_prompt: Tensor[B,L]           # CLIP token ids
        tokenized_prompt_mask: Tensor[B,L]

    Output:
        state_feat: Tensor[B,d_model]
    """

    def __init__(
        self,
        encoder_name: str,
        robot_state_dim: int,
        d_model: int = 512,
        view_keys: Sequence[str] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        nhead: int = 8,
        dropout: float = 0.1,
        encoder_amp: bool = True,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.robot_state_dim = int(robot_state_dim)
        self.d_model = int(d_model)
        self.view_keys = tuple(view_keys)
        self.encoder_amp = bool(encoder_amp)

        self.vl_encoder = CLIPModel.from_pretrained(encoder_name)
        for p in self.vl_encoder.parameters():
            p.requires_grad = False
        self.vl_encoder.eval()

        clip_projection_dim = self.vl_encoder.config.projection_dim
        vision_hidden_size = self.vl_encoder.config.vision_config.hidden_size

        self.global_proj = nn.Sequential(
            nn.Linear(clip_projection_dim * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.patch_proj = nn.Sequential(
            nn.Linear(vision_hidden_size, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.patch_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.robot_state_proj = nn.Sequential(
            nn.LayerNorm(robot_state_dim),
            nn.Linear(robot_state_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.Sigmoid(),
        )

        self.out_ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def train(self, mode: bool = True):
        super().train(mode)
        self.vl_encoder.eval()
        return self

    def freeze_backbone(self) -> None:
        for p in self.vl_encoder.parameters():
            p.requires_grad = False
        self.vl_encoder.eval()

    def _select_images_and_masks(self, observation: Dict[str, object]) -> Tuple[torch.Tensor, torch.Tensor]:
        if "image" not in observation:
            raise KeyError("pi0 observation must contain observation['image']")
        if "image_mask" not in observation:
            raise KeyError("pi0 observation must contain observation['image_mask']")

        images: Dict[str, torch.Tensor] = observation["image"]  # type: ignore[assignment]
        image_masks: Dict[str, torch.Tensor] = observation["image_mask"]  # type: ignore[assignment]

        available = [k for k in self.view_keys if k in images]
        if not available:
            raise KeyError(
                f"No expected image views found. expected={self.view_keys}, got={tuple(images.keys())}"
            )

        image_list = []
        mask_list = []
        for key in available:
            img = images[key]
            if img.ndim != 4:
                raise ValueError(f"observation['image']['{key}'] must be [B,3,H,W], got {tuple(img.shape)}")
            if img.shape[1] != 3:
                raise ValueError(
                    f"observation['image']['{key}'] must be channels-first [B,3,H,W]. "
                    f"If your loader has [B,H,W,3], convert before model. got={tuple(img.shape)}"
                )
            image_list.append(img.float())

            if key in image_masks:
                mask = image_masks[key].bool()
            else:
                mask = torch.ones(img.shape[0], device=img.device, dtype=torch.bool)
            mask_list.append(mask)

        # [B,V,3,H,W], [B,V]
        stacked_images = torch.stack(image_list, dim=1)
        stacked_masks = torch.stack(mask_list, dim=1)
        return stacked_images, stacked_masks

    def _encode_clip_views(
        self,
        images_bvchw: torch.Tensor,
        image_masks_bv: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return masked pooled CLIP image feature, CLIP text feature and concatenated patch tokens.

        images_bvchw: [B,V,3,H,W]
        image_masks_bv: [B,V]
        input_ids / attention_mask: [B,L]
        """
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError(
                f"tokenized_prompt/tokenized_prompt_mask must be [B,L], "
                f"got {tuple(input_ids.shape)} / {tuple(attention_mask.shape)}"
            )

        B, V, C, H, W = images_bvchw.shape
        flat_images = images_bvchw.reshape(B * V, C, H, W)

        use_amp = bool(self.encoder_amp and flat_images.is_cuda)
        amp_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if use_amp else nullcontext()

        with torch.no_grad():
            with amp_ctx:
                vision_outputs = self.vl_encoder.vision_model(
                    pixel_values=flat_images,
                    output_hidden_states=False,
                    return_dict=True,
                )
                flat_image_feat = self.vl_encoder.visual_projection(vision_outputs.pooler_output)
                flat_patch_tokens = vision_outputs.last_hidden_state[:, 1:]

                text_outputs = self.vl_encoder.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    return_dict=True,
                )
                text_feat = self.vl_encoder.text_projection(text_outputs.pooler_output)

        image_feat = flat_image_feat.float().reshape(B, V, -1)  # [B,V,D_clip]
        patch_tokens = flat_patch_tokens.float().reshape(B, V, flat_patch_tokens.shape[1], -1)

        mask = image_masks_bv.to(image_feat.device).float().unsqueeze(-1)  # [B,V,1]
        denom = mask.sum(dim=1).clamp_min(1.0)  # [B,1]
        pooled_image_feat = (image_feat * mask).sum(dim=1) / denom

        # Keep fixed token length by zeroing invalid views instead of ragged packing.
        patch_mask = image_masks_bv.to(patch_tokens.device).float().view(B, V, 1, 1)
        patch_tokens = patch_tokens * patch_mask
        patch_tokens = patch_tokens.reshape(B, V * patch_tokens.shape[2], patch_tokens.shape[3])

        return pooled_image_feat.float(), text_feat.float(), patch_tokens.float()

    def forward(self, observation: Dict[str, object]) -> torch.Tensor:
        required = ["state", "tokenized_prompt", "tokenized_prompt_mask"]
        for key in required:
            if key not in observation:
                raise KeyError(f"pi0 observation must contain observation['{key}']")

        robot_state = observation["state"].float()  # type: ignore[union-attr]
        if robot_state.ndim != 2:
            raise ValueError(f"observation['state'] must be [B,S], got {tuple(robot_state.shape)}")
        if robot_state.shape[-1] != self.robot_state_dim:
            raise ValueError(
                f"state dim mismatch: got {robot_state.shape[-1]}, expected {self.robot_state_dim}"
            )

        input_ids = observation["tokenized_prompt"].long()  # type: ignore[union-attr]
        attention_mask = observation["tokenized_prompt_mask"].long()  # type: ignore[union-attr]

        images_bvchw, image_masks_bv = self._select_images_and_masks(observation)
        image_feat, text_feat, patch_tokens = self._encode_clip_views(
            images_bvchw=images_bvchw,
            image_masks_bv=image_masks_bv,
            input_ids=input_ids.to(robot_state.device),
            attention_mask=attention_mask.to(robot_state.device),
        )

        image_feat = F.normalize(image_feat.float(), dim=-1)
        text_feat = F.normalize(text_feat.float(), dim=-1)
        global_state = self.global_proj(torch.cat([image_feat, text_feat], dim=-1))

        patch_tokens = self.patch_proj(patch_tokens.float())
        spatial_state, _ = self.patch_attn(
            query=global_state.unsqueeze(1),
            key=patch_tokens,
            value=patch_tokens,
            need_weights=False,
        )
        spatial_state = spatial_state.squeeze(1)

        proprio_state = self.robot_state_proj(robot_state.float())

        gate = self.fusion_gate(torch.cat([global_state, spatial_state, proprio_state], dim=-1))
        state_feat = global_state + gate * spatial_state + proprio_state
        state_feat = state_feat + self.out_ffn(state_feat)
        return self.out_norm(state_feat)


class ActionChunkEncoder(nn.Module):
    def __init__(
        self,
        action_dim: int,
        chunk_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)

        self.action_proj = nn.Sequential(
            nn.Linear(action_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, chunk_size, d_model))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.ndim != 3:
            raise ValueError(f"Expected actions [B,H,A], got {tuple(actions.shape)}")
        if actions.shape[1] != self.chunk_size:
            raise ValueError(
                f"Action chunk length must equal configured chunk_size. "
                f"got H={actions.shape[1]}, expected {self.chunk_size}. "
                f"For pi0 AWBC/IQL, set chunk_size == action_horizon == replan_steps."
            )
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"Action dim mismatch: got {actions.shape[-1]}, expected {self.action_dim}")

        x = self.action_proj(actions.float())
        x = x + self.pos_embed
        x = self.encoder(x)
        return self.out_norm(x)


class TransformerQHead(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        d_model: int = 512,
        nhead: int = 8,
        action_layers: int = 2,
        fusion_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.state_token_proj = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        self.action_encoder = ActionChunkEncoder(
            action_dim=action_dim,
            chunk_size=chunk_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=action_layers,
            dropout=dropout,
        )
        self.action_film = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, 2 * d_model),
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fuser = nn.TransformerEncoder(layer, num_layers=fusion_layers)

        self.out = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1, bias=False),
        )

    def forward(self, state_feat: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        state_token = self.state_token_proj(state_feat).unsqueeze(1)
        action_tokens = self.action_encoder(actions)

        gamma, beta = self.action_film(state_feat).chunk(2, dim=-1)
        action_tokens = action_tokens * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

        tokens = torch.cat([state_token, action_tokens], dim=1)
        fused = self.fuser(tokens)
        fused_state = fused[:, 0]
        fused_action = fused[:, 1:].mean(dim=1)
        return self.out(torch.cat([fused_state, fused_action], dim=-1))


class QEnsemble(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        num_q: int = 2,
        d_model: int = 512,
        nhead: int = 8,
        action_layers: int = 2,
        fusion_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.qs = nn.ModuleList(
            [
                TransformerQHead(
                    state_dim=state_dim,
                    action_dim=action_dim,
                    chunk_size=chunk_size,
                    d_model=d_model,
                    nhead=nhead,
                    action_layers=action_layers,
                    fusion_layers=fusion_layers,
                    dropout=dropout,
                )
                for _ in range(num_q)
            ]
        )

    def forward(self, state_feat: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.stack([q(state_feat, actions) for q in self.qs], dim=0)  # [K,B,1]


class LightVLValueModel(nn.Module):
    """
    Pi0-input chunk-IQL critic.

    This replaces the old OFT/RLDS-style input. The critic only accepts pi0-style
    observations. It scores action chunks with length exactly chunk_size.
    """

    def __init__(
        self,
        encoder_name: str = "openai/clip-vit-base-patch32",
        robot_state_dim: int = 8,
        action_dim: int = 7,
        chunk_size: int = 5,
        num_q: int = 2,
        d_model: int = 512,
        nhead: int = 8,
        action_layers: int = 2,
        fusion_layers: int = 2,
        dropout: float = 0.1,
        view_keys: Sequence[str] = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
        q_l2_coef: float = 1e-4,
        encoder_amp: bool = True,
        rank_coef: float = 0.5,
        rank_margin: float = 0.05,
        rank_noise_std: float = 0.05,
        rank_num_noisy: int = 8,
        rank_action_clip_value: Optional[float] = 1.0,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        self.robot_state_dim = int(robot_state_dim)
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.num_q = int(num_q)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.action_layers = int(action_layers)
        self.fusion_layers = int(fusion_layers)
        self.dropout = float(dropout)
        self.view_keys = tuple(view_keys)
        self.q_l2_coef = float(q_l2_coef)
        self.encoder_amp = bool(encoder_amp)

        # Ranking regularization for successful chunks:
        #   Q(s, a_real) > max_k Q(s, a_noisy_k) + margin
        #
        # This is only used when compute_loss(..., use_q_aug=True).
        self.rank_coef = float(rank_coef)
        self.rank_margin = float(rank_margin)
        self.rank_noise_std = float(rank_noise_std)
        self.rank_num_noisy = int(rank_num_noisy)
        self.rank_action_clip_value = (
            None if rank_action_clip_value is None else float(rank_action_clip_value)
        )

        self.observation_encoder = Pi0ObservationEncoder(
            encoder_name=encoder_name,
            robot_state_dim=robot_state_dim,
            d_model=d_model,
            view_keys=view_keys,
            nhead=nhead,
            dropout=dropout,
            encoder_amp=encoder_amp,
        )
        self.value_head = ValueHead(d_model, dropout=dropout)
        self.q_ensemble = QEnsemble(
            state_dim=d_model,
            action_dim=action_dim,
            chunk_size=chunk_size,
            num_q=num_q,
            d_model=d_model,
            nhead=nhead,
            action_layers=action_layers,
            fusion_layers=fusion_layers,
            dropout=dropout,
        )
        self.target_q_head = copy.deepcopy(self.q_ensemble)
        for p in self.target_q_head.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.observation_encoder.freeze_backbone()
        return self

    def freeze_backbone(self) -> None:
        self.observation_encoder.freeze_backbone()

    def _prepare_actions(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.float()
        if actions.ndim != 3:
            raise ValueError(f"Expected actions [B,H,A], got {tuple(actions.shape)}")
        if actions.shape[1] != self.chunk_size:
            raise ValueError(
                f"Action chunk length mismatch: got H={actions.shape[1]}, expected {self.chunk_size}. "
                f"Set pi0 action_horizon == replan_steps == IQL chunk_size."
            )
        if actions.shape[-1] != self.action_dim:
            raise ValueError(f"Action dim mismatch: got {actions.shape[-1]}, expected {self.action_dim}")
        return actions

    def encode_observation(self, observation: Dict[str, object]) -> torch.Tensor:
        return self.observation_encoder(observation)

    def get_value(self, state_feat: torch.Tensor) -> torch.Tensor:
        return self.value_head(state_feat)

    def get_q(self, state_feat: torch.Tensor, actions: torch.Tensor, is_target: bool = False) -> torch.Tensor:
        actions = self._prepare_actions(actions)
        if is_target:
            return self.target_q_head(state_feat, actions)
        return self.q_ensemble(state_feat, actions)

    def _compute_chunk_return_and_bootstrap(
        self,
        rewards: torch.Tensor,
        is_terminal: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rewards.ndim != 2:
            raise ValueError(f"rewards should be [B,H], got {tuple(rewards.shape)}")
        if is_terminal.ndim != 2:
            raise ValueError(f"is_terminal should be [B,H], got {tuple(is_terminal.shape)}")
        if rewards.shape != is_terminal.shape:
            raise ValueError(f"rewards/is_terminal shape mismatch: {tuple(rewards.shape)} vs {tuple(is_terminal.shape)}")
        if rewards.shape[1] != self.chunk_size:
            raise ValueError(f"reward horizon mismatch: got {rewards.shape[1]}, expected {self.chunk_size}")

        B, H = rewards.shape
        device = rewards.device
        dtype = rewards.dtype
        terminal = is_terminal.bool()
        discounts = gamma ** torch.arange(H, device=device, dtype=dtype)

        alive_mask = torch.ones((B, H), device=device, dtype=dtype)
        if H > 1:
            alive_mask[:, 1:] = torch.cumprod((~terminal[:, :-1]).to(dtype), dim=1)

        chunk_return = (rewards * alive_mask * discounts.unsqueeze(0)).sum(dim=1)

        # Keep the value scale close to single-step reward scale.
        discount_sum = discounts.sum().clamp_min(1e-8)
        chunk_return = chunk_return / discount_sum

        bootstrap_mask = torch.cumprod((~terminal).to(dtype), dim=1)[:, -1]
        return chunk_return, bootstrap_mask

    def _success_mask_from_batch(
        self,
        batch: Dict[str, object],
        device: torch.device | str | int,
        batch_size: int,
    ) -> torch.Tensor:
        """Return [B] bool mask indicating chunks from successful episodes."""
        if "from_success" not in batch:
            raise KeyError(
                "use_q_aug=True requires batch['from_success']. "
                "The IQL dataloader should create it from the strict dataset field `success`."
            )

        from_success = batch["from_success"]
        if not isinstance(from_success, torch.Tensor):
            from_success = torch.as_tensor(from_success)

        from_success = from_success.to(device).bool()
        if from_success.ndim == 2:
            if from_success.shape[0] != batch_size:
                raise ValueError(
                    f"from_success batch mismatch: got {tuple(from_success.shape)}, B={batch_size}"
                )
            return from_success[:, 0]
        if from_success.ndim == 1:
            if from_success.shape[0] != batch_size:
                raise ValueError(
                    f"from_success batch mismatch: got {tuple(from_success.shape)}, B={batch_size}"
                )
            return from_success
        raise ValueError(f"from_success must be [B] or [B,H], got {tuple(from_success.shape)}")

    def _ranking_regularizer(
        self,
        state_feat: torch.Tensor,
        actions: torch.Tensor,
        from_success: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Margin ranking loss on successful trajectory chunks.

        For successful chunks only:
            Q(s, a_real) > max_k Q(s, a_noisy_k) + margin

        The noisy chunks are generated by adding Gaussian noise to the entire
        action chunk. We use the online Q ensemble and pessimistic min-Q for
        both real and noisy actions. The hardest noisy sample is selected per
        state to make the regularizer non-trivial.

        actions:
            [B, H, A], with H == self.chunk_size.
        from_success:
            [B] bool mask.
        """
        if self.rank_coef <= 0.0:
            zero = actions.new_zeros(())
            return zero, {
                "rank_loss": 0.0,
                "rank_good_ratio": float(from_success.float().mean().detach().cpu()),
                "rank_num_good": float(from_success.float().sum().detach().cpu()),
                "rank_q_real_mean": 0.0,
                "rank_q_noisy_mean": 0.0,
                "rank_margin_violation": 0.0,
            }

        actions = self._prepare_actions(actions)
        B, H, A = actions.shape

        good_mask = from_success.bool()
        if good_mask.shape != (B,):
            raise ValueError(f"from_success mask must be [B], got {tuple(good_mask.shape)}")

        if good_mask.sum() == 0:
            zero = actions.new_zeros(())
            return zero, {
                "rank_loss": 0.0,
                "rank_good_ratio": 0.0,
                "rank_num_good": 0.0,
                "rank_q_real_mean": 0.0,
                "rank_q_noisy_mean": 0.0,
                "rank_margin_violation": 0.0,
            }

        num_noisy = max(int(self.rank_num_noisy), 1)

        q_real_all = self.get_q(state_feat, actions, is_target=False).squeeze(-1)  # [Kq, B]
        q_real = torch.min(q_real_all, dim=0)[0]  # [B]

        noise = torch.randn(
            num_noisy,
            B,
            H,
            A,
            device=actions.device,
            dtype=actions.dtype,
        ) * float(self.rank_noise_std)
        noisy_actions = actions.unsqueeze(0) + noise  # [N, B, H, A]

        # LIBERO env actions are effectively clipped in [-1, 1]. Apply the same
        # range to all dimensions, including gripper, because gripper uses a
        # signed convention in the OpenPI/LIBERO path.
        if self.rank_action_clip_value is not None:
            clip = float(self.rank_action_clip_value)
            noisy_actions = torch.clamp(noisy_actions, -clip, clip)

        noisy_actions_flat = noisy_actions.reshape(num_noisy * B, H, A)
        state_noisy = (
            state_feat.unsqueeze(0)
            .expand(num_noisy, B, state_feat.shape[-1])
            .reshape(num_noisy * B, state_feat.shape[-1])
        )

        q_noisy_all = self.get_q(state_noisy, noisy_actions_flat, is_target=False).squeeze(-1)
        q_noisy_all = q_noisy_all.view(q_noisy_all.shape[0], num_noisy, B)  # [Kq,N,B]
        q_noisy = torch.min(q_noisy_all, dim=0)[0]  # [N,B]
        q_hard_noisy = torch.max(q_noisy, dim=0)[0]  # [B]

        violation = float(self.rank_margin) - q_real + q_hard_noisy
        rank_loss_per = F.relu(violation)
        rank_loss = rank_loss_per[good_mask].mean()

        metrics = {
            "rank_loss": float(rank_loss.detach().cpu()),
            "rank_good_ratio": float(good_mask.float().mean().detach().cpu()),
            "rank_num_good": float(good_mask.float().sum().detach().cpu()),
            "rank_q_real_mean": float(q_real[good_mask].mean().detach().cpu()),
            "rank_q_noisy_mean": float(q_hard_noisy[good_mask].mean().detach().cpu()),
            "rank_margin_violation": float((violation[good_mask] > 0).float().mean().detach().cpu()),
        }
        return rank_loss, metrics

    def compute_loss(
        self,
        batch: Dict[str, object],
        gamma: float = 0.99,
        expectile: float = 0.7,
        device: Optional[torch.device | str | int] = None,
        debug: bool = False,
        use_q_aug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        if "observation" not in batch:
            raise KeyError("batch must contain batch['observation'] in pi0 format")
        if "next_observation" not in batch:
            raise KeyError("batch must contain batch['next_observation'] in pi0 format")

        obs = batch["observation"]
        next_obs = batch["next_observation"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        is_terminal = batch["is_terminal"]

        if device is not None:
            obs = move_observation_to_device(obs, device)  # type: ignore[arg-type]
            next_obs = move_observation_to_device(next_obs, device)  # type: ignore[arg-type]
            actions = actions.to(device)  # type: ignore[union-attr]
            rewards = rewards.to(device)  # type: ignore[union-attr]
            is_terminal = is_terminal.to(device)  # type: ignore[union-attr]

        actions = self._prepare_actions(actions.float())  # type: ignore[union-attr]
        rewards = rewards.float()  # type: ignore[union-attr]
        is_terminal = is_terminal.bool()  # type: ignore[union-attr]

        state_feat = self.encode_observation(obs)  # type: ignore[arg-type]
        online_qs = self.get_q(state_feat, actions, is_target=False).squeeze(-1)  # [K,B]

        with torch.no_grad():
            next_state_feat = self.encode_observation(next_obs)  # type: ignore[arg-type]
            next_v = self.get_value(next_state_feat).squeeze(-1)  # [B]
            chunk_return, bootstrap_mask = self._compute_chunk_return_and_bootstrap(
                rewards=rewards,
                is_terminal=is_terminal,
                gamma=gamma,
            )
            q_target = chunk_return + bootstrap_mask * (gamma ** self.chunk_size) * next_v
            q_target = q_target.unsqueeze(0)  # [1,B]

        td_loss = (online_qs - q_target).pow(2).mean()
        q_l2_loss = online_qs.pow(2).mean()
        critic_loss = td_loss + self.q_l2_coef * q_l2_loss

        with torch.no_grad():
            target_qs = self.get_q(state_feat, actions, is_target=True).squeeze(-1)
            target_q = torch.min(target_qs, dim=0)[0]

        v = self.get_value(state_feat).squeeze(-1)
        adv = target_q - v
        value_loss = expectile_loss(adv, expectile).mean()

        rank_loss = actions.new_zeros(())
        rank_metrics: Dict[str, float] = {
            "rank_loss": 0.0,
            "rank_good_ratio": 0.0,
            "rank_num_good": 0.0,
            "rank_q_real_mean": 0.0,
            "rank_q_noisy_mean": 0.0,
            "rank_margin_violation": 0.0,
        }
        if use_q_aug:
            success_mask = self._success_mask_from_batch(
                batch=batch,
                device=actions.device,
                batch_size=actions.shape[0],
            )
            rank_loss, rank_metrics = self._ranking_regularizer(
                state_feat=state_feat,
                actions=actions,
                from_success=success_mask,
            )
            critic_loss = critic_loss + self.rank_coef * rank_loss

        metrics = {
            "critic_td_loss": float(td_loss.detach().cpu()),
            "q_l2_loss": float(q_l2_loss.detach().cpu()),
            "critic_loss": float(critic_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "rank_coef": float(self.rank_coef if use_q_aug else 0.0),
            "q_target_mean": float(q_target.mean().detach().cpu()),
            "online_q_mean": float(online_qs.mean().detach().cpu()),
            "target_q_mean": float(target_q.mean().detach().cpu()),
            "v_mean": float(v.mean().detach().cpu()),
            "adv_mean": float(adv.mean().detach().cpu()),
            "adv_std": float(adv.std(unbiased=False).detach().cpu()),
            "reward_mean": float(rewards.mean().detach().cpu()),
            "terminal_ratio": float(is_terminal.float().mean().detach().cpu()),
        }
        metrics.update(rank_metrics)

        if debug:
            print("\n" + "=" * 80)
            print(">>> PI0-INPUT LIGHT-VL CHUNK-IQL DEBUG <<<")
            print(f"Chunk H      : {self.chunk_size}")
            print(f"Actions      : {tuple(actions.shape)} mean={actions.mean().item():.4f} std={actions.std().item():.4f}")
            print(f"Rewards      : {tuple(rewards.shape)} mean={rewards.mean().item():.4f} min={rewards.min().item():.4f} max={rewards.max().item():.4f}")
            print(f"Terminal     : {is_terminal.float().mean().item():.4f}")
            for k, val in metrics.items():
                print(f"{k:16s}: {val:.6f}")
            print("=" * 80 + "\n")

        return critic_loss, value_loss, metrics

    @torch.no_grad()
    def infer_batch(
        self,
        batch: Dict[str, object],
        device: Optional[torch.device | str | int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if "observation" not in batch:
            raise KeyError("batch must contain batch['observation'] in pi0 format")
        obs = batch["observation"]
        actions = batch["actions"]
        if device is not None:
            obs = move_observation_to_device(obs, device)  # type: ignore[arg-type]
            actions = actions.to(device)  # type: ignore[union-attr]

        actions = self._prepare_actions(actions.float())  # type: ignore[union-attr]
        state_feat = self.encode_observation(obs)  # type: ignore[arg-type]
        target_qs = self.get_q(state_feat, actions, is_target=True).squeeze(-1)
        target_q = torch.min(target_qs, dim=0)[0]
        v = self.get_value(state_feat).squeeze(-1)
        adv = target_q - v
        return adv, target_q, v

    @torch.no_grad()
    def update_target_q(self, tau: float = 0.02) -> None:
        for param, target_param in zip(self.q_ensemble.parameters(), self.target_q_head.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * param.data)

    def save_heads(self, path: str) -> None:
        ckpt = {
            "encoder_name": self.encoder_name,
            "robot_state_dim": self.robot_state_dim,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "num_q": self.num_q,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "action_layers": self.action_layers,
            "fusion_layers": self.fusion_layers,
            "dropout": self.dropout,
            "view_keys": self.view_keys,
            "q_l2_coef": self.q_l2_coef,
            "encoder_amp": self.encoder_amp,
            "rank_coef": self.rank_coef,
            "rank_margin": self.rank_margin,
            "rank_noise_std": self.rank_noise_std,
            "rank_num_noisy": self.rank_num_noisy,
            "rank_action_clip_value": self.rank_action_clip_value,
            "model": self.state_dict(),
        }
        torch.save(ckpt, path)

    @classmethod
    def from_checkpoint(cls, path: str, encoder_name_override: str = "") -> "LightVLValueModel":
        ckpt = torch.load(path, map_location="cpu")
        encoder_name = encoder_name_override or ckpt["encoder_name"]
        model = cls(
            encoder_name=encoder_name,
            robot_state_dim=ckpt["robot_state_dim"],
            action_dim=ckpt["action_dim"],
            chunk_size=ckpt["chunk_size"],
            num_q=ckpt["num_q"],
            d_model=ckpt["d_model"],
            nhead=ckpt["nhead"],
            action_layers=ckpt["action_layers"],
            fusion_layers=ckpt["fusion_layers"],
            dropout=ckpt["dropout"],
            view_keys=tuple(ckpt.get("view_keys", ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"))),
            q_l2_coef=ckpt.get("q_l2_coef", 1e-4),
            encoder_amp=ckpt.get("encoder_amp", True),
            rank_coef=ckpt.get("rank_coef", 0.5),
            rank_margin=ckpt.get("rank_margin", 0.05),
            rank_noise_std=ckpt.get("rank_noise_std", 0.05),
            rank_num_noisy=ckpt.get("rank_num_noisy", 8),
            rank_action_clip_value=ckpt.get("rank_action_clip_value", 1.0),
        )
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state)
        model.freeze_backbone()
        model.eval()
        return model


class LightIQLCritic(LightVLValueModel):
    """Compatibility alias used by some training/selector code.

    Signature:
        LightIQLCritic(encoder_name, state_dim, action_dim, horizon, ...)
    """

    def __init__(
        self,
        encoder_name: str = "openai/clip-vit-base-patch32",
        state_dim: int = 8,
        action_dim: int = 7,
        horizon: int = 5,
        hidden_dim: int = 512,
        num_q: int = 2,
        action_layers: int = 2,
        q_layers: int = 2,
        dropout: float = 0.1,
        nhead: int = 8,
        q_l2_coef: float = 1e-4,
        encoder_amp: bool = True,
        rank_coef: float = 0.5,
        rank_margin: float = 0.05,
        rank_noise_std: float = 0.05,
        rank_num_noisy: int = 8,
        rank_action_clip_value: Optional[float] = 1.0,
    ):
        super().__init__(
            encoder_name=encoder_name,
            robot_state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=horizon,
            num_q=num_q,
            d_model=hidden_dim,
            nhead=nhead,
            action_layers=action_layers,
            fusion_layers=q_layers,
            dropout=dropout,
            q_l2_coef=q_l2_coef,
            encoder_amp=encoder_amp,
            rank_coef=rank_coef,
            rank_margin=rank_margin,
            rank_noise_std=rank_noise_std,
            rank_num_noisy=rank_num_noisy,
            rank_action_clip_value=rank_action_clip_value,
        )


def move_observation_to_device(observation: Dict[str, object], device: torch.device | str | int) -> Dict[str, object]:
    """Move a pi0-style observation dict to a torch device."""
    out: Dict[str, object] = {}
    for key, value in observation.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device)
        elif isinstance(value, dict):
            out[key] = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in value.items()
            }
        else:
            out[key] = value
    return out


def load_light_vl_model_from_head(value_head_path: str, value_encoder_path: str = ""):
    model = LightVLValueModel.from_checkpoint(value_head_path, encoder_name_override=value_encoder_path)
    meta = {
        "encoder_name": model.encoder_name,
        "robot_state_dim": model.robot_state_dim,
        "action_dim": model.action_dim,
        "chunk_size": model.chunk_size,
        "num_q": model.num_q,
        "d_model": model.d_model,
    }
    return model, meta
