#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serve a pi0 policy with batched stochastic sampling + IQL Q-select.

This server speaks exactly the same websocket protocol as OpenPI's
scripts/serve_policy.py, so the existing LIBERO eval / collector client can
communicate with it unchanged.

Core logic per request:
    1. Receive one unbatched LIBERO observation dict:
         observation/image, observation/wrist_image, observation/state, prompt
    2. Run OpenPI input transforms once.
    3. Replicate the transformed observation N times along batch dimension.
    4. Generate N different noise tensors.
    5. Call pi0.sample_actions once on the N-sized batch.
    6. Apply OpenPI output transforms per candidate to get [N, H, 7].
    7. Run IQL selector once on [N, H, 7].
    8. Return the selected action chunk as {"actions": [H, 7]}.

In --sample-mode simple, this wrapper just delegates to OpenPI Policy.infer().
For pi0 flow matching there is no separate greedy/random/first mode: a single
normal inference call is the simple baseline.

The collector / test env process should connect to this server just like it
connects to scripts/serve_policy.py.

Example:
    cd /data/huangdi/heliqun/pi0/openpi
    source pi_env/bin/activate

    python /data/huangdi/heliqun/pi0/ours/serve_pi0_qselect.py \
      --policy-config pi0_libero_awbc \
      --policy-dir /data/aoss/heliqun/pi0_runs/iter0/5000 \
      --critic-path /data/aoss/heliqun/pi0_iql/head/iter0/final.pt \
      --num-action-samples 16 \
      --sample-mode qselect \
      --num-steps 10 \
      --port 8000
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import pathlib
import socket
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


# Make `from selector import QSelector` work when this file is placed in Qselect/.
THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from selector import QSelector


@dataclasses.dataclass
class Args:
    # Policy checkpoint, same semantic role as serve_policy.py policy:checkpoint.
    policy_config: str
    policy_dir: str

    # Q selector.
    critic_path: str = ""
    sample_mode: str = "qselect"  # qselect | simple
    selector_device: str = ""

    # Batched stochastic sampling.
    num_action_samples: int = 16
    num_steps: int = 10
    noise_scale: float = 1.0
    seed: int = 0

    # Optional: if >0, selector scores only candidate[:, :score_horizon, :].
    # For your unified setup action_horizon == replan_steps == 5, leave this 0.
    score_horizon: int = 0

    # Serving.
    port: int = 8000
    default_prompt: str = ""
    record: bool = False

    # Debug.
    log_every: int = 20


def parse_args() -> Args:
    p = argparse.ArgumentParser()

    p.add_argument("--policy-config", required=True)
    p.add_argument("--policy-dir", required=True)

    p.add_argument("--critic-path", default="")
    p.add_argument("--sample-mode", default="qselect", choices=("qselect", "simple"))
    p.add_argument("--selector-device", default="")

    p.add_argument("--num-action-samples", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--noise-scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--score-horizon", type=int, default=0)

    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--default-prompt", default="")
    p.add_argument("--record", action="store_true")
    p.add_argument("--log-every", type=int, default=20)

    return Args(**vars(p.parse_args()))


def _tree_batch_repeat(tree: Any, batch_size: int, *, framework: str, device: str | None = None) -> Any:
    """Add a batch dimension and repeat the transformed single sample N times."""
    if framework == "jax":
        def convert(x):
            x = jnp.asarray(x)
            x = jnp.expand_dims(x, axis=0)
            return jnp.repeat(x, batch_size, axis=0)

        return jax.tree.map(convert, tree)

    if framework == "torch":
        def convert(x):
            arr = np.asarray(x)
            t = torch.from_numpy(arr).to(device)
            t = t.unsqueeze(0)
            reps = [batch_size] + [1] * (t.ndim - 1)
            return t.repeat(*reps)

        return jax.tree.map(convert, tree)

    raise ValueError(f"Unknown framework: {framework}")


def _tree_take(tree: Any, index: int) -> Any:
    """Take one item from a batched pytree and return numpy leaves."""
    def take(x):
        if torch.is_tensor(x):
            return x[index].detach().cpu().numpy()
        return np.asarray(x[index])
    return jax.tree.map(take, tree)


def _to_numpy_tree(tree: Any) -> Any:
    def convert(x):
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)
    return jax.tree.map(convert, tree)


def _extract_raw_selector_inputs(obs: dict[str, Any]) -> tuple[list[Any], str, np.ndarray]:
    """Extract images/prompt/state from the original untransformed LIBERO request."""
    if "observation/image" not in obs:
        raise KeyError("Expected obs['observation/image']")
    if "observation/wrist_image" not in obs:
        raise KeyError("Expected obs['observation/wrist_image']")
    if "observation/state" not in obs:
        raise KeyError("Expected obs['observation/state']")

    images = [obs["observation/image"], obs["observation/wrist_image"]]
    prompt = str(obs.get("prompt", ""))
    state = np.asarray(obs["observation/state"], dtype=np.float32)

    if state.ndim != 1:
        state = state.reshape(-1)

    return images, prompt, state


class Pi0QSelectPolicy:
    """
    A websocket-servable policy wrapper.

    It intentionally accesses OpenPI Policy private fields:
        _input_transform, _output_transform, _sample_actions, _model, _rng
    because OpenPI's public Policy.infer() only supports a single noise/action
    sample per request. This wrapper keeps the same external infer(obs) API.
    """

    def __init__(self, args: Args) -> None:
        self.args = args
        self.train_config = _config.get_config(args.policy_config)

        sample_kwargs = {"num_steps": int(args.num_steps)}
        default_prompt = args.default_prompt if args.default_prompt else None

        self.policy = _policy_config.create_trained_policy(
            self.train_config,
            args.policy_dir,
            sample_kwargs=sample_kwargs,
            default_prompt=default_prompt,
        )

        self.metadata = dict(getattr(self.policy, "metadata", {}) or {})
        self.metadata.update(
            {
                "qselect": {
                    "enabled": args.sample_mode == "qselect",
                    "num_action_samples": int(args.num_action_samples),
                    "sample_mode": args.sample_mode,
                    "critic_path": args.critic_path,
                    "score_horizon": int(args.score_horizon),
                }
            }
        )

        self.num_action_samples = int(args.num_action_samples)
        if self.num_action_samples <= 0:
            raise ValueError(f"--num-action-samples must be positive, got {self.num_action_samples}")

        self.is_pytorch = bool(getattr(self.policy, "_is_pytorch_model", False))
        self.pytorch_device = getattr(self.policy, "_pytorch_device", "cpu")

        self.action_horizon = int(getattr(self.train_config.model, "action_horizon"))
        self.action_dim = int(getattr(self.train_config.model, "action_dim"))

        if args.sample_mode == "qselect":
            if not args.critic_path:
                raise ValueError("--critic-path is required for qselect mode.")
            selector_device = args.selector_device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.selector = QSelector(args.critic_path, device=selector_device)
        else:
            self.selector = None

        np.random.seed(args.seed)
        self.torch_generator = torch.Generator(device=self.pytorch_device if str(self.pytorch_device).startswith("cuda") else "cpu")
        self.torch_generator.manual_seed(args.seed)
        if not self.is_pytorch:
            # Override Policy's default rng so this server is deterministic under --seed.
            self.policy._rng = jax.random.key(args.seed)

        self.request_count = 0

        logging.info(
            "Pi0QSelectPolicy initialized: config=%s dir=%s horizon=%s action_dim=%s "
            "num_samples=%s mode=%s pytorch=%s",
            args.policy_config,
            args.policy_dir,
            self.action_horizon,
            self.action_dim,
            self.num_action_samples,
            args.sample_mode,
            self.is_pytorch,
        )

    def _sample_batched_raw_actions(self, transformed_single: dict[str, Any]) -> tuple[Any, Any, float]:
        """
        Return:
            transformed_batch: batched transformed model input
            raw_actions: batched raw model actions before output transforms, [N,H,D]
            model_time: seconds
        """
        n = self.num_action_samples

        if self.is_pytorch:
            transformed_batch = _tree_batch_repeat(
                transformed_single,
                n,
                framework="torch",
                device=self.pytorch_device,
            )
            observation = _model.Observation.from_dict(transformed_batch)

            noise = torch.randn(
                (n, self.action_horizon, self.action_dim),
                generator=self.torch_generator,
                device=self.pytorch_device,
                dtype=torch.float32,
            ) * float(self.args.noise_scale)

            sample_kwargs = dict(getattr(self.policy, "_sample_kwargs", {}) or {})
            sample_kwargs["noise"] = noise

            start = time.monotonic()
            with torch.inference_mode():
                raw_actions = self.policy._sample_actions(
                    self.pytorch_device,
                    observation,
                    **sample_kwargs,
                )
            model_time = time.monotonic() - start
            return transformed_batch, raw_actions, model_time

        transformed_batch = _tree_batch_repeat(transformed_single, n, framework="jax")
        observation = _model.Observation.from_dict(transformed_batch)

        self.policy._rng, sample_rng = jax.random.split(self.policy._rng)
        noise = jax.random.normal(
            sample_rng,
            (n, self.action_horizon, self.action_dim),
            dtype=jnp.float32,
        ) * float(self.args.noise_scale)

        sample_kwargs = dict(getattr(self.policy, "_sample_kwargs", {}) or {})
        sample_kwargs["noise"] = noise

        start = time.monotonic()
        raw_actions = self.policy._sample_actions(sample_rng, observation, **sample_kwargs)
        # Ensure computation is finished for timing and conversion.
        raw_actions.block_until_ready()
        model_time = time.monotonic() - start
        return transformed_batch, raw_actions, model_time

    def _apply_output_transform_per_candidate(self, transformed_batch: Any, raw_actions: Any) -> np.ndarray:
        """
        OpenPI output transforms are written for unbatched infer outputs because
        Policy.infer() removes the batch dimension before applying them. Therefore
        for N batched candidates we apply the output transform candidate-by-candidate.

        Returns:
            candidates: [N, H, 7] env-space LIBERO actions.
        """
        raw_actions_np = _to_numpy_tree(raw_actions)
        transformed_batch_np = _to_numpy_tree(transformed_batch)

        candidates = []
        for i in range(self.num_action_samples):
            single_outputs = {
                "state": _tree_take(transformed_batch_np["state"], i),
                "actions": np.asarray(raw_actions_np[i]),
            }
            out = self.policy._output_transform(single_outputs)
            if "actions" not in out:
                raise KeyError(f"Output transform did not return actions. Keys: {list(out.keys())}")
            action_chunk = np.asarray(out["actions"], dtype=np.float32)
            if action_chunk.ndim != 2:
                raise ValueError(f"Expected transformed candidate [H,A], got {action_chunk.shape}")
            if action_chunk.shape[-1] != 7:
                # Safety for malformed output transforms.
                action_chunk = action_chunk[:, :7]
            candidates.append(action_chunk)

        return np.stack(candidates, axis=0).astype(np.float32)

    def _select_candidate(self, obs: dict[str, Any], candidates: np.ndarray) -> tuple[int, np.ndarray]:
        mode = self.args.sample_mode

        if mode != "qselect":
            raise ValueError(f"Unknown sample mode: {mode}")

        if self.selector is None:
            raise RuntimeError("QSelector is not initialized.")

        score_candidates = candidates
        if self.args.score_horizon > 0:
            h = int(self.args.score_horizon)
            if h > candidates.shape[1]:
                raise ValueError(f"--score-horizon={h} > candidate horizon={candidates.shape[1]}")
            score_candidates = candidates[:, :h, :]

        images, prompt, state = _extract_raw_selector_inputs(obs)
        best_idx, scores = self.selector.select(
            images=images,
            prompt=prompt,
            state=state,
            candidates=score_candidates,
            mode="qselect",
        )
        return int(best_idx), np.asarray(scores, dtype=np.float32)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self.request_count += 1

        if self.args.sample_mode == "simple":
            start = time.monotonic()
            out = self.policy.infer(obs)
            infer_time = time.monotonic() - start
            if self.args.log_every > 0 and self.request_count % self.args.log_every == 0:
                actions = np.asarray(out.get("actions", []), dtype=np.float32)
                logging.info(
                    "request=%s mode=simple action_shape=%s infer_ms=%.1f",
                    self.request_count,
                    tuple(actions.shape),
                    infer_time * 1000.0,
                )
            return out

        # Match OpenPI Policy.infer(): copy because transforms may mutate.
        transform_start = time.monotonic()
        transformed_single = jax.tree.map(lambda x: x, obs)
        transformed_single = self.policy._input_transform(transformed_single)
        transform_time = time.monotonic() - transform_start

        transformed_batch, raw_actions, model_time = self._sample_batched_raw_actions(transformed_single)

        output_start = time.monotonic()
        candidates = self._apply_output_transform_per_candidate(transformed_batch, raw_actions)
        output_time = time.monotonic() - output_start

        select_start = time.monotonic()
        best_idx, scores = self._select_candidate(obs, candidates)
        select_time = time.monotonic() - select_start

        selected = candidates[best_idx]

        if self.args.log_every > 0 and self.request_count % self.args.log_every == 0:
            logging.info(
                "request=%s mode=%s best=%s q_mean=%.4f q_std=%.4f q_min=%.4f q_max=%.4f "
                "model_ms=%.1f select_ms=%.1f",
                self.request_count,
                self.args.sample_mode,
                best_idx,
                float(np.mean(scores)) if scores.size else 0.0,
                float(np.std(scores)) if scores.size else 0.0,
                float(np.min(scores)) if scores.size else 0.0,
                float(np.max(scores)) if scores.size else 0.0,
                model_time * 1000.0,
                select_time * 1000.0,
            )

        return {
            "actions": selected,
            "qselect": {
                "best_index": int(best_idx),
                "scores": scores.tolist(),
                "num_action_samples": int(candidates.shape[0]),
                "candidate_horizon": int(candidates.shape[1]),
                "candidate_action_dim": int(candidates.shape[2]),
                "mode": self.args.sample_mode,
            },
            "policy_timing": {
                "transform_ms": transform_time * 1000.0,
                "batched_model_ms": model_time * 1000.0,
                "output_transform_ms": output_time * 1000.0,
                "selector_ms": select_time * 1000.0,
            },
        }


class PolicyRecorder:
    """Minimal recorder wrapper compatible with websocket_policy_server."""

    def __init__(self, policy: Pi0QSelectPolicy, record_dir: str = "qselect_policy_records") -> None:
        self._policy = policy
        self.metadata = policy.metadata
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._step = 0

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        out = self._policy.infer(obs)
        path = self._record_dir / f"step_{self._step:06d}.npz"
        self._step += 1

        # Store only light metadata; raw images can make records too large.
        np.savez_compressed(
            path,
            actions=np.asarray(out["actions"], dtype=np.float32),
            scores=np.asarray(out.get("qselect", {}).get("scores", []), dtype=np.float32),
            best_index=np.asarray([out.get("qselect", {}).get("best_index", -1)], dtype=np.int64),
        )
        return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = parse_args()

    policy = Pi0QSelectPolicy(args)
    server_policy = PolicyRecorder(policy) if args.record else policy

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating Q-select server (host: %s, ip: %s, port: %s)", hostname, local_ip, args.port)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=server_policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
