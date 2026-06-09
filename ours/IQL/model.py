import copy
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


def expectile_loss(diff, expectile=0.7):
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return weight * diff.pow(2)


class ValueHead(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
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

    def forward(self, x):
        return self.net(x)


class SpatialStateEncoder(nn.Module):
    """
    从 CLIP 的 global image/text feature + vision patch tokens 中构造机器人 critic state feature。

    之前版本:
        state = MLP([image_global, text_global])

    现在版本:
        global_state = MLP([image_global, text_global])
        patch_tokens = MLP(CLIP vision patch tokens)
        spatial_state = AttentionPool(query=global_state, key/value=patch_tokens)
        state = gated_fusion(global_state, spatial_state)

    这样比纯 CLIP global embedding 更适合机器人局部几何判断。
    """

    def __init__(
        self,
        clip_projection_dim,
        vision_hidden_size,
        d_model=512,
        nhead=8,
        dropout=0.1,
    ):
        super().__init__()

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

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
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

    def forward(self, image_feat, text_feat, patch_tokens):
        # image_feat: [B, D_clip]
        # text_feat : [B, D_clip]
        # patch_tokens: [B, N, D_vision]

        image_feat = F.normalize(image_feat.float(), dim=-1)
        text_feat = F.normalize(text_feat.float(), dim=-1)

        global_raw = torch.cat([image_feat, text_feat], dim=-1)
        global_state = self.global_proj(global_raw)  # [B, D]

        patch_tokens = self.patch_proj(patch_tokens.float())  # [B, N, D]

        query = global_state.unsqueeze(1)  # [B, 1, D]
        spatial_state, _ = self.patch_attn(
            query=query,
            key=patch_tokens,
            value=patch_tokens,
            need_weights=False,
        )
        spatial_state = spatial_state.squeeze(1)  # [B, D]

        fusion_gate = self.gate(torch.cat([global_state, spatial_state], dim=-1))
        state = global_state + fusion_gate * spatial_state

        state = state + self.out_ffn(state)
        state = self.out_norm(state)
        return state


class ActionChunkEncoder(nn.Module):
    def __init__(
        self,
        action_dim,
        chunk_size,
        d_model=512,
        nhead=8,
        num_layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.chunk_size = chunk_size

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

    def forward(self, actions):
        # actions: [B, H, A]
        if actions.ndim != 3:
            raise ValueError(f"Expected actions [B, H, A], got {actions.shape}")

        B, H, A = actions.shape
        if H > self.chunk_size:
            raise ValueError(f"Action horizon H={H} exceeds configured chunk_size={self.chunk_size}")

        x = self.action_proj(actions.float())
        x = x + self.pos_embed[:, :H]
        x = self.encoder(x)
        x = self.out_norm(x)
        return x


class TransformerQHead(nn.Module):
    """
    更强的 Q(s, a_chunk):

        action tokens = ActionTransformer(a_chunk)
        action tokens = FiLM(action tokens | state)
        tokens = [state_token, action_tokens]
        fused = Transformer(tokens)
        q = MLP([state_token_after_fusion, pooled_action_tokens_after_fusion])

    比原来的 state/action 简单拼接更适合做 action-conditioned ranking。
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        chunk_size,
        d_model=512,
        nhead=8,
        action_layers=2,
        fusion_layers=2,
        dropout=0.1,
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

        # state-conditioned FiLM: 用 state 调制 action token
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

    def forward(self, state_feat, actions):
        # state_feat: [B, D]
        # actions: [B, H, A]

        state_token = self.state_token_proj(state_feat).unsqueeze(1)  # [B, 1, D]
        action_tokens = self.action_encoder(actions)                  # [B, H, D]

        gamma_beta = self.action_film(state_feat)                     # [B, 2D]
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        action_tokens = action_tokens * (1.0 + gamma) + beta

        tokens = torch.cat([state_token, action_tokens], dim=1)
        fused = self.fuser(tokens)

        fused_state = fused[:, 0]           # [B, D]
        fused_action = fused[:, 1:].mean(1) # [B, D]

        q_input = torch.cat([fused_state, fused_action], dim=-1)
        return self.out(q_input)


class QEnsemble(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        chunk_size,
        num_q=2,
        d_model=512,
        nhead=8,
        action_layers=2,
        fusion_layers=2,
        dropout=0.1,
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

    def forward(self, state_feat, actions):
        qs = [q(state_feat, actions) for q in self.qs]
        return torch.stack(qs, dim=0)  # [K, B, 1]


class LightVLValueModel(nn.Module):
    """
    Ultimate-plus-max 版 lightweight action-conditioned IQL critic。

    输入 batch key:
        input_ids
        attention_mask
        pixel_values
        next_pixel_values
        actions
        rewards
        is_terminal

    输出接口保持不变:
        compute_loss(...) -> critic_loss, value_loss
        infer_batch(...)  -> adv, target_q, v
    """

    def __init__(
        self,
        encoder_name="openai/clip-vit-base-patch32",
        action_dim=7,
        chunk_size=8,
        num_q=2,
        d_model=512,
        nhead=8,
        action_layers=2,
        fusion_layers=2,
        dropout=0.1,
        rank_coef=0.5,
        rank_margin=0.05,
        rank_noise_std=0.05,
        q_l2_coef=1e-4,
        encoder_amp=True,
        action_clip_value=1.0,
    ):
        super().__init__()

        self.encoder_name = encoder_name
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_q = num_q
        self.d_model = d_model
        self.nhead = nhead
        self.action_layers = action_layers
        self.fusion_layers = fusion_layers
        self.dropout = dropout

        self.rank_coef = rank_coef
        self.rank_margin = rank_margin
        self.rank_noise_std = rank_noise_std
        self.q_l2_coef = q_l2_coef
        self.encoder_amp = encoder_amp
        self.action_clip_value = action_clip_value

        self.vl_encoder = CLIPModel.from_pretrained(encoder_name)

        for p in self.vl_encoder.parameters():
            p.requires_grad = False
        self.vl_encoder.eval()

        clip_projection_dim = self.vl_encoder.config.projection_dim
        vision_hidden_size = self.vl_encoder.config.vision_config.hidden_size

        self.state_encoder = SpatialStateEncoder(
            clip_projection_dim=clip_projection_dim,
            vision_hidden_size=vision_hidden_size,
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
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

    def train(self, mode=True):
        super().train(mode)
        # frozen encoder 永远 eval
        self.vl_encoder.eval()
        return self

    def freeze_backbone(self):
        for p in self.vl_encoder.parameters():
            p.requires_grad = False
        self.vl_encoder.eval()

    def _encode_clip_features(self, input_ids, attention_mask, pixel_values):
        """
        只跑一次 CLIP vision/text tower，同时拿:
            image_feat: projected global image embedding
            text_feat : projected global text embedding
            patch_tokens: CLIP vision patch tokens
        """

        use_amp = bool(self.encoder_amp and pixel_values.is_cuda)
        amp_ctx = torch.cuda.amp.autocast(dtype=torch.bfloat16) if use_amp else nullcontext()

        with torch.no_grad():
            with amp_ctx:
                vision_outputs = self.vl_encoder.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=False,
                    return_dict=True,
                )
                text_outputs = self.vl_encoder.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    return_dict=True,
                )

                image_feat = self.vl_encoder.visual_projection(vision_outputs.pooler_output)
                text_feat = self.vl_encoder.text_projection(text_outputs.pooler_output)

                # last_hidden_state: [B, 1 + num_patches, D]
                # 去掉 CLS token，保留 spatial patch tokens
                patch_tokens = vision_outputs.last_hidden_state[:, 1:]

        return image_feat.float(), text_feat.float(), patch_tokens.float()

    def _get_backbone_features(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        labels=None,
    ):
        """
        替代原 OpenVLA _get_backbone_features。
        返回: state_feat [B, d_model]
        """

        image_feat, text_feat, patch_tokens = self._encode_clip_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

        state_feat = self.state_encoder(
            image_feat=image_feat,
            text_feat=text_feat,
            patch_tokens=patch_tokens,
        )
        return state_feat

    def get_value(self, hidden_features):
        return self.value_head(hidden_features)

    def _prepare_actions(self, actions):
        actions = actions.float()

        if actions.ndim == 2:
            expected = self.chunk_size * self.action_dim
            if actions.shape[-1] != expected:
                raise ValueError(
                    f"Flattened actions dim mismatch: got {actions.shape[-1]}, expected {expected}"
                )
            actions = actions.view(actions.shape[0], self.chunk_size, self.action_dim)

        if actions.ndim != 3:
            raise ValueError(f"Expected actions [B, H, A], got {actions.shape}")

        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Action dim mismatch: got {actions.shape[-1]}, expected {self.action_dim}"
            )

        if actions.shape[1] > self.chunk_size:
            raise ValueError(
                f"Action horizon mismatch: got H={actions.shape[1]}, max chunk_size={self.chunk_size}"
            )

        return actions

    def get_q(self, hidden_features, actions, is_target=False):
        actions = self._prepare_actions(actions)

        if is_target:
            return self.target_q_head(hidden_features, actions)
        return self.q_ensemble(hidden_features, actions)

    def _compute_chunk_return_and_bootstrap(self, rewards, is_terminal, gamma):
        """
        rewards: [B, H]
        is_terminal: [B, H]
        """
        if rewards.ndim != 2:
            raise ValueError(f"rewards should be [B, H], got {rewards.shape}")
        if is_terminal.ndim != 2:
            raise ValueError(f"is_terminal should be [B, H], got {is_terminal.shape}")
        if rewards.shape != is_terminal.shape:
            raise ValueError(
                f"rewards/is_terminal shape mismatch: {rewards.shape} vs {is_terminal.shape}"
            )

        B, H = rewards.shape
        device = rewards.device
        dtype = rewards.dtype

        is_terminal = is_terminal.bool()
        discounts = gamma ** torch.arange(H, device=device, dtype=dtype)

        alive_mask = torch.ones((B, H), device=device, dtype=dtype)
        if H > 1:
            alive_mask[:, 1:] = torch.cumprod((~is_terminal[:, :-1]).to(dtype), dim=1)

        masked_rewards = rewards * alive_mask
        chunk_return = (masked_rewards * discounts.unsqueeze(0)).sum(dim=1)

        # 保持和你之前 chunk-IQL 一致：归一化到接近单步 reward 尺度
        discount_sum = discounts.sum().clamp_min(1e-8)
        chunk_return = chunk_return / discount_sum

        # 只有 chunk 内完全没有 terminal 才 bootstrap
        bootstrap_mask = torch.cumprod((~is_terminal).to(dtype), dim=1)[:, -1]
        return chunk_return, bootstrap_mask

    def _ranking_regularizer(self, hidden_states, actions, from_success):
        """
        Multi-noise ranking regularization for successful trajectory chunks:

            Q(s, a_real) > max_k Q(s, a_noisy_k) + margin

        Compared with the old version:
            old: real vs one noisy action
            new: real vs hardest noisy action among K noisy samples
        """

        if self.rank_coef <= 0:
            return actions.new_zeros(())

        actions = self._prepare_actions(actions)  # [B, H, A]
        B, H, A = actions.shape

        # from_success normally has shape [B, H]
        if from_success.ndim == 2:
            good_mask = from_success[:, 0].bool()
        else:
            good_mask = from_success.bool()

        if good_mask.sum() == 0:
            return actions.new_zeros(())

        # 新增：支持多个 noisy action
        # 如果你没有在 __init__ 里定义 self.rank_num_noisy，就默认退化为 1。
        num_noisy = int(getattr(self, "rank_num_noisy", 8))
        num_noisy = max(num_noisy, 1)

        # ------------------------------------------------------------
        # 1. real action Q
        # ------------------------------------------------------------
        q_real_all = self.get_q(
            hidden_features=hidden_states,
            actions=actions,
            is_target=False,
        ).squeeze(-1)  # [Kq, B]

        q_real = torch.min(q_real_all, dim=0)[0]  # [B]

        # ------------------------------------------------------------
        # 2. sample K noisy actions
        # ------------------------------------------------------------
        noise = torch.randn(
            num_noisy,
            B,
            H,
            A,
            device=actions.device,
            dtype=actions.dtype,
        ) * float(self.rank_noise_std)

        noisy_actions = actions.unsqueeze(0) + noise  # [N, B, H, A]

        if self.action_clip_value is not None:
            clip = float(self.action_clip_value)

            noisy_main = torch.clamp(noisy_actions[..., :-1], -clip, clip)
            noisy_gripper = torch.clamp(noisy_actions[..., -1:], 0.0, 1.0)

            noisy_actions = torch.cat([noisy_main, noisy_gripper], dim=-1)

        # [N, B, H, A] -> [N*B, H, A]
        noisy_actions_flat = noisy_actions.reshape(num_noisy * B, H, A)

        # hidden_states: [B, D] -> [N*B, D]
        hidden_noisy = (
            hidden_states.unsqueeze(0)
            .expand(num_noisy, B, hidden_states.shape[-1])
            .reshape(num_noisy * B, hidden_states.shape[-1])
        )

        # ------------------------------------------------------------
        # 3. noisy action Q
        # ------------------------------------------------------------
        q_noisy_all = self.get_q(
            hidden_features=hidden_noisy,
            actions=noisy_actions_flat,
            is_target=False,
        ).squeeze(-1)  # [Kq, N*B]

        # [Kq, N*B] -> [Kq, N, B]
        q_noisy_all = q_noisy_all.view(q_noisy_all.shape[0], num_noisy, B)

        # ensemble min: [N, B]
        q_noisy = torch.min(q_noisy_all, dim=0)[0]

        # hardest noisy per state: [B]
        q_hard_noisy = torch.max(q_noisy, dim=0)[0]

        # ------------------------------------------------------------
        # 4. margin ranking loss
        # ------------------------------------------------------------
        rank_loss = F.relu(
            float(self.rank_margin) - q_real + q_hard_noisy
        )  # [B]

        return rank_loss[good_mask].mean()

    def compute_loss(
        self,
        batch,
        gamma=0.99,
        expectile=0.7,
        device_id=None,
        debug=False,
        use_q_aug=False,
    ):
        rewards = batch["rewards"].to(device_id).float()          # [B, H]
        is_terminal = batch["is_terminal"].to(device_id).bool()   # [B, H]
        actions = batch["actions"].to(device_id).float()          # [B, H, A]
        actions = self._prepare_actions(actions)
        from_success = batch['from_success'].to(device_id).bool()
        input_ids = batch["input_ids"].to(device_id)
        attention_mask = batch["attention_mask"].to(device_id)
        pixel_values = batch["pixel_values"].to(device_id).float()
        next_pixel_values = batch["next_pixel_values"].to(device_id).float()

        horizon = actions.shape[1]

        hidden_states = self._get_backbone_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

        online_qs = self.get_q(
            hidden_features=hidden_states,
            actions=actions,
            is_target=False,
        ).squeeze(-1)  # [K, B]

        online_q = torch.min(online_qs, dim=0)[0]

        with torch.no_grad():
            next_hidden_states = self._get_backbone_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=next_pixel_values,
            )

            next_v = self.get_value(next_hidden_states).squeeze(-1)

            chunk_return, bootstrap_mask = self._compute_chunk_return_and_bootstrap(
                rewards=rewards,
                is_terminal=is_terminal,
                gamma=gamma,
            )

            q_learning_target = chunk_return + bootstrap_mask * (gamma ** horizon) * next_v
            q_learning_target = q_learning_target.unsqueeze(0)  # [1, B]

        td_error = online_qs - q_learning_target
        critic_td_loss = td_error.pow(2).mean()

        # 控制 Q 绝对尺度，防止 scorer 爆炸。
        q_l2_loss = online_qs.pow(2).mean()

        with torch.no_grad():
            target_qs = self.get_q(
                hidden_features=hidden_states,
                actions=actions,
                is_target=True,
            ).squeeze(-1)
            target_q = torch.min(target_qs, dim=0)[0]

        v = self.get_value(hidden_states).squeeze(-1)
        adv = target_q - v
        value_loss = expectile_loss(adv, expectile).mean()
        critic_loss = (
            critic_td_loss
            + self.q_l2_coef * q_l2_loss
        )
        if use_q_aug :
            rank_loss = self._ranking_regularizer(
                hidden_states=hidden_states,
                actions=actions,
                from_success=from_success,
            )
            critic_loss = critic_loss + self.rank_coef * rank_loss
        if debug:
            with torch.no_grad():
                q_disagree = target_qs.std(dim=0).mean()
                positive_reward_ratio = (rewards > 0).float().mean()
                terminal_ratio = is_terminal.float().mean()

                print("\n" + "=" * 80)
                print(">>> LIGHT-VL SPATIAL CHUNK-IQL DEBUG <<<")
                print(f"Encoder      : {self.encoder_name}")
                print(f"Chunk H      : {horizon}")
                print(f"Rewards      | shape={tuple(rewards.shape)} mean={rewards.mean().item():.4f} "
                      f"min={rewards.min().item():.4f} max={rewards.max().item():.4f}")
                print(f"Actions      | shape={tuple(actions.shape)} mean={actions.mean().item():.4f} "
                      f"std={actions.std().item():.4f}")
                print(f"PosReward    | {positive_reward_ratio.item():.4f}")
                print(f"Terminal     | {terminal_ratio.item():.4f}")
                print(f"ChunkReturn  | mean={chunk_return.mean().item():.4f} "
                      f"min={chunk_return.min().item():.4f} max={chunk_return.max().item():.4f}")
                print(f"Bootstrap    | mean={bootstrap_mask.float().mean().item():.4f}")
                print(f"Next V       | mean={next_v.mean().item():.4f} "
                      f"min={next_v.min().item():.4f} max={next_v.max().item():.4f}")
                print(f"Online Q     | mean={online_q.mean().item():.4f} "
                      f"min={online_q.min().item():.4f} max={online_q.max().item():.4f}")
                print(f"Target Q     | mean={target_q.mean().item():.4f} "
                      f"min={target_q.min().item():.4f} max={target_q.max().item():.4f}")
                print(f"Q Target     | mean={q_learning_target.mean().item():.4f} "
                      f"min={q_learning_target.min().item():.4f} max={q_learning_target.max().item():.4f}")
                print(f"V            | mean={v.mean().item():.4f} "
                      f"min={v.min().item():.4f} max={v.max().item():.4f}")
                print(f"Adv          | mean={adv.mean().item():.4f} "
                      f"std={adv.std().item():.4f} min={adv.min().item():.4f} max={adv.max().item():.4f}")
                print(f"Q Disagree   | {q_disagree.item():.6f}")
                print("-" * 80)
                print(f"Critic TD    | {critic_td_loss.item():.6f}")
                print(f"Q L2         | {q_l2_loss.item():.6f} * {self.q_l2_coef}")
                if use_q_aug:
                    print(f"Rank Loss    | {rank_loss.item():.6f} * {self.rank_coef}")
                print(f"Critic Loss  | {critic_loss.item():.6f}")
                print(f"Value Loss   | {value_loss.item():.6f}")
                print("=" * 80 + "\n")

        return critic_loss, value_loss

    @torch.no_grad()
    def infer_batch(self, batch, device_id=None):
        actions = batch["actions"].to(device_id).float()
        actions = self._prepare_actions(actions)

        input_ids = batch["input_ids"].to(device_id)
        attention_mask = batch["attention_mask"].to(device_id)
        pixel_values = batch["pixel_values"].to(device_id).float()

        hidden_states = self._get_backbone_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
        )

        target_qs = self.get_q(
            hidden_features=hidden_states,
            actions=actions,
            is_target=True,
        ).squeeze(-1)

        target_q = torch.min(target_qs, dim=0)[0]
        v = self.get_value(hidden_states).squeeze(-1)
        adv = target_q - v

        return adv, target_q, v

    @torch.no_grad()
    def update_target_q(self, tau=0.02):
        for param, target_param in zip(
            self.q_ensemble.parameters(),
            self.target_q_head.parameters(),
        ):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * param.data)

    def save_heads(self, path):
        ckpt = {
            "encoder_name": self.encoder_name,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "num_q": self.num_q,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "action_layers": self.action_layers,
            "fusion_layers": self.fusion_layers,
            "dropout": self.dropout,
            "rank_coef": self.rank_coef,
            "rank_margin": self.rank_margin,
            "rank_noise_std": self.rank_noise_std,
            "q_l2_coef": self.q_l2_coef,
            "action_clip_value": self.action_clip_value,
            "state_encoder": self.state_encoder.state_dict(),
            "value_head": self.value_head.state_dict(),
            "q_ensemble": self.q_ensemble.state_dict(),
            "target_q": self.target_q_head.state_dict(),
        }
        torch.save(ckpt, path)

    def load_heads(self, path):
        ckpt = torch.load(path, map_location="cpu")

        assert ckpt["action_dim"] == self.action_dim, (
            f"action_dim mismatch: ckpt={ckpt['action_dim']} current={self.action_dim}"
        )
        assert ckpt["chunk_size"] == self.chunk_size, (
            f"chunk_size mismatch: ckpt={ckpt['chunk_size']} current={self.chunk_size}"
        )
        assert ckpt["num_q"] == self.num_q, (
            f"num_q mismatch: ckpt={ckpt['num_q']} current={self.num_q}"
        )
        assert ckpt["d_model"] == self.d_model, (
            f"d_model mismatch: ckpt={ckpt['d_model']} current={self.d_model}"
        )

        # encoder_name 不强制 assert，因为你可能想做迁移实验；
        # 但强烈建议同一个 head 对应同一个 encoder。
        if ckpt.get("encoder_name", self.encoder_name) != self.encoder_name:
            print(
                f"[Warning] Loading head trained with encoder={ckpt.get('encoder_name')} "
                f"into current encoder={self.encoder_name}. This is usually not recommended."
            )

        self.state_encoder.load_state_dict(ckpt["state_encoder"])
        self.value_head.load_state_dict(ckpt["value_head"])
        self.q_ensemble.load_state_dict(ckpt["q_ensemble"])
        self.target_q_head.load_state_dict(ckpt["target_q"])

def load_light_vl_model_from_head(
    value_head_path: str,
    value_encoder_path: str = "",
):
    """
    从 LightVLValueModel.save_heads() 保存的 ckpt 里读取结构参数，再构造模型。
    value_encoder_path 非空时覆盖 ckpt["encoder_name"]，用于跨机器路径不同的情况。
    """
    if not value_head_path:
        raise ValueError("Light-VL selector requires --value_head_path.")

    ckpt = torch.load(value_head_path, map_location="cpu")

    encoder_name = value_encoder_path or ckpt["encoder_name"]

    required_keys = [
        "action_dim",
        "chunk_size",
        "num_q",
        "d_model",
        "nhead",
        "action_layers",
        "fusion_layers",
        "dropout",
    ]
    for key in required_keys:
        if key not in ckpt:
            raise KeyError(
                f"Light-VL checkpoint missing key `{key}`. "
                f"Please make sure it was saved by the new save_heads()."
            )

    model = LightVLValueModel(
        encoder_name=encoder_name,
        action_dim=ckpt["action_dim"],
        chunk_size=ckpt["chunk_size"],
        num_q=ckpt["num_q"],
        d_model=ckpt["d_model"],
        nhead=ckpt["nhead"],
        action_layers=ckpt["action_layers"],
        fusion_layers=ckpt["fusion_layers"],
        dropout=ckpt["dropout"],

        # 这些只影响训练 loss；在线 select 不用。
        # 但构造函数需要，所以从 ckpt 读。
        rank_coef=ckpt.get("rank_coef", 0.0),
        rank_margin=ckpt.get("rank_margin", 0.05),
        rank_noise_std=ckpt.get("rank_noise_std", 0.05),
        q_l2_coef=ckpt.get("q_l2_coef", 1e-4),
        action_clip_value=ckpt.get("action_clip_value", 1.0),
        encoder_amp=True,
    )

    model.load_heads(value_head_path)
    model.freeze_backbone()
    model.eval()

    meta = {
        "encoder_name": encoder_name,
        "action_dim": int(ckpt["action_dim"]),
        "chunk_size": int(ckpt["chunk_size"]),
        "num_q": int(ckpt["num_q"]),
        "d_model": int(ckpt["d_model"]),
    }

    return model, meta