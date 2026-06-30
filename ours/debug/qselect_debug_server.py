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

In --sample-mode random, this wrapper uses the same batched noise proposal path
as qselect, but selects one candidate uniformly at random instead of scoring it.

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
import datetime as dt
import json
import logging
import os
import pathlib
import socket
import sys
import time
from typing import Any

import numpy as np


# Heavy runtime dependencies are populated only after argparse has handled
# --help. This also keeps pure diagnostic tests independent of CUDA/OpenPI.
jax: Any = None
jnp: Any = None
torch: Any = None
_model: Any = None
_policy_config: Any = None
websocket_policy_server: Any = None
_config: Any = None


# Keep all edits in ours/debug while importing the production selector read-only.
THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
QSELECT_DIR = THIS_DIR.parent / "Qselect"

DEFAULT_PORT = 8000
DEFAULT_NUM_STEPS = 10
DEFAULT_NOISE_SCALE = 1.0
DEFAULT_SCORE_HORIZON = 0
DEFAULT_LOG_EVERY = 20


def _load_runtime_dependencies() -> None:
    global jax, jnp, torch, _model, _policy_config, websocket_policy_server, _config
    if jax is not None:
        return
    import jax as jax_module
    import jax.numpy as jnp_module
    import torch as torch_module
    from openpi.models import model as model_module
    from openpi.policies import policy_config as policy_config_module
    from openpi.serving import websocket_policy_server as websocket_policy_server_module
    from openpi.training import config as config_module

    jax = jax_module
    jnp = jnp_module
    torch = torch_module
    _model = model_module
    _policy_config = policy_config_module
    websocket_policy_server = websocket_policy_server_module
    _config = config_module


def _require_finite(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")


def _pair_l2_stats(values: np.ndarray) -> dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(values.shape[0], -1)
    if flat.shape[0] < 2:
        distances = np.zeros((1,), dtype=np.float64)
    else:
        rows, cols = np.triu_indices(flat.shape[0], k=1)
        distances = np.linalg.norm(flat[rows] - flat[cols], axis=1)
    denom = float(np.sqrt(flat.shape[1]))
    per_dim = distances / denom
    return {
        "mean": float(np.mean(distances)),
        "min": float(np.min(distances)),
        "max": float(np.max(distances)),
        "std": float(np.std(distances)),
        "per_dim_mean": float(np.mean(per_dim)),
        "per_dim_min": float(np.min(per_dim)),
        "per_dim_max": float(np.max(per_dim)),
        "per_dim_std": float(np.std(per_dim)),
    }


def _mean_candidate_std(values: np.ndarray, start: int, stop: int) -> float:
    stop = min(int(stop), int(values.shape[-1]))
    start = min(int(start), stop)
    if start == stop:
        return 0.0
    return float(np.mean(np.std(values[..., start:stop], axis=0)))


def _boundary_stats(values: np.ndarray, selected: np.ndarray, prefix: str) -> dict[str, float]:
    values_abs = np.abs(values)
    selected_abs = np.abs(selected)
    saturated = values_abs >= 0.99
    would_clip = values_abs > 1.0
    candidate_axes = tuple(range(1, values.ndim))
    return {
        f"{prefix}cand_abs_max": float(np.max(values_abs)),
        f"{prefix}cand_saturation_frac": float(np.mean(saturated)),
        f"{prefix}cand_would_clip_frac": float(np.mean(would_clip)),
        f"{prefix}cand_saturated_candidate_frac": float(np.mean(np.any(saturated, axis=candidate_axes))),
        f"{prefix}cand_would_clip_candidate_frac": float(np.mean(np.any(would_clip, axis=candidate_axes))),
        f"{prefix}selected_abs_max": float(np.max(selected_abs)),
        f"{prefix}selected_saturation_frac": float(np.mean(selected_abs >= 0.99)),
        f"{prefix}selected_would_clip_frac": float(np.mean(selected_abs > 1.0)),
    }


def compute_candidate_stats(
    candidates: Any,
    scores: Any,
    best_idx: int,
    exec_horizon: int | None = None,
) -> dict[str, int | float]:
    """Compute JSON-safe diversity, selection, score, and boundary statistics.

    Pair L2 treats one complete action chunk as one point. ``*_per_dim_*``
    divides that L2 by sqrt(H*A), making full/prefix values comparable.
    Boundary metrics are observational only; this function never clips actions.
    """
    candidates_np = np.asarray(candidates, dtype=np.float64)
    scores_np = np.asarray(scores, dtype=np.float64)
    if candidates_np.ndim != 3:
        raise ValueError(f"candidates must have shape [N,H,A], got {candidates_np.shape}")
    n, horizon, action_dim = map(int, candidates_np.shape)
    if n <= 0 or horizon <= 0 or action_dim <= 0:
        raise ValueError(f"candidates dimensions must be positive, got {candidates_np.shape}")
    if scores_np.shape != (n,):
        raise ValueError(f"scores must have shape [{n}], got {scores_np.shape}")
    _require_finite("candidates", candidates_np)
    _require_finite("scores", scores_np)
    best_idx = int(best_idx)
    if not 0 <= best_idx < n:
        raise ValueError(f"best_idx must be in [0, {n}), got {best_idx}")
    if exec_horizon is None or int(exec_horizon) == 0:
        exec_horizon = horizon
    exec_horizon = int(exec_horizon)
    if not 1 <= exec_horizon <= horizon:
        raise ValueError(f"exec_horizon must be in [1, {horizon}], got {exec_horizon}")

    executed = candidates_np[:, :exec_horizon, :]
    selected = candidates_np[best_idx]
    selected_exec = executed[best_idx]
    full_pair = _pair_l2_stats(candidates_np)
    exec_pair = _pair_l2_stats(executed)
    best_delta = selected - candidates_np[0]
    best_exec_delta = selected_exec - executed[0]
    sorted_scores = np.sort(scores_np)
    top_gap = float(sorted_scores[-1] - sorted_scores[-2]) if n >= 2 else 0.0

    stats: dict[str, int | float] = {
        "num_candidates": n,
        "horizon": horizon,
        "action_dim": action_dim,
        "exec_horizon": exec_horizon,
        "cand_pair_l2_mean": full_pair["mean"],
        "cand_pair_l2_min": full_pair["min"],
        "cand_pair_l2_max": full_pair["max"],
        "cand_pair_l2_std": full_pair["std"],
        "cand_pair_l2_per_dim_mean": full_pair["per_dim_mean"],
        "cand_pair_l2_per_dim_min": full_pair["per_dim_min"],
        "cand_pair_l2_per_dim_max": full_pair["per_dim_max"],
        "cand_pair_l2_per_dim_std": full_pair["per_dim_std"],
        "cand_std_all": float(np.mean(np.std(candidates_np, axis=0))),
        "cand_std_trans": _mean_candidate_std(candidates_np, 0, 3),
        "cand_std_rot": _mean_candidate_std(candidates_np, 3, 6),
        "cand_std_grip": _mean_candidate_std(candidates_np, 6, 7),
        "exec_pair_l2_mean": exec_pair["mean"],
        "exec_pair_l2_min": exec_pair["min"],
        "exec_pair_l2_max": exec_pair["max"],
        "exec_pair_l2_std": exec_pair["std"],
        "exec_pair_l2_per_dim_mean": exec_pair["per_dim_mean"],
        "exec_pair_l2_per_dim_min": exec_pair["per_dim_min"],
        "exec_pair_l2_per_dim_max": exec_pair["per_dim_max"],
        "exec_pair_l2_per_dim_std": exec_pair["per_dim_std"],
        "exec_std_all": float(np.mean(np.std(executed, axis=0))),
        "exec_std_trans": _mean_candidate_std(executed, 0, 3),
        "exec_std_rot": _mean_candidate_std(executed, 3, 6),
        "exec_std_grip": _mean_candidate_std(executed, 6, 7),
        "best_idx": best_idx,
        "best_vs_first_l2": float(np.linalg.norm(best_delta)),
        "best_vs_first_l2_per_dim": float(np.linalg.norm(best_delta) / np.sqrt(best_delta.size)),
        "best_vs_first_exec_l2": float(np.linalg.norm(best_exec_delta)),
        "best_vs_first_exec_l2_per_dim": float(
            np.linalg.norm(best_exec_delta) / np.sqrt(best_exec_delta.size)
        ),
        "q_mean": float(np.mean(scores_np)),
        "q_std": float(np.std(scores_np)),
        "q_min": float(np.min(scores_np)),
        "q_max": float(np.max(scores_np)),
        "q_gap": float(np.max(scores_np) - np.mean(scores_np)),
        "q_range": float(np.max(scores_np) - np.min(scores_np)),
        "q_top1_top2_gap": top_gap,
    }
    stats.update(_boundary_stats(candidates_np, selected, ""))
    stats.update(_boundary_stats(executed, selected_exec, "exec_"))
    return stats


def compute_tensor_diversity(
    values: Any,
    exec_horizon: int | None = None,
) -> dict[str, int | float]:
    """Compute layer diversity for noise or pre-transform model actions."""
    values_np = np.asarray(values, dtype=np.float64)
    if values_np.ndim != 3:
        raise ValueError(f"values must have shape [N,H,A], got {values_np.shape}")
    n, horizon, action_dim = map(int, values_np.shape)
    if n <= 0 or horizon <= 0 or action_dim <= 0:
        raise ValueError(f"values dimensions must be positive, got {values_np.shape}")
    _require_finite("values", values_np)
    if exec_horizon is None or int(exec_horizon) == 0:
        exec_horizon = horizon
    exec_horizon = int(exec_horizon)
    if not 1 <= exec_horizon <= horizon:
        raise ValueError(f"exec_horizon must be in [1, {horizon}], got {exec_horizon}")
    executed = values_np[:, :exec_horizon, :]
    full_pair = _pair_l2_stats(values_np)
    exec_pair = _pair_l2_stats(executed)
    return {
        "num_candidates": n,
        "horizon": horizon,
        "action_dim": action_dim,
        "exec_horizon": exec_horizon,
        "pair_l2_mean": full_pair["mean"],
        "pair_l2_min": full_pair["min"],
        "pair_l2_max": full_pair["max"],
        "pair_l2_std": full_pair["std"],
        "pair_l2_per_dim_mean": full_pair["per_dim_mean"],
        "pair_l2_per_dim_min": full_pair["per_dim_min"],
        "pair_l2_per_dim_max": full_pair["per_dim_max"],
        "pair_l2_per_dim_std": full_pair["per_dim_std"],
        "std_all": float(np.mean(np.std(values_np, axis=0))),
        "std_trans": _mean_candidate_std(values_np, 0, 3),
        "std_rot": _mean_candidate_std(values_np, 3, 6),
        "std_grip": _mean_candidate_std(values_np, 6, 7),
        "exec_pair_l2_mean": exec_pair["mean"],
        "exec_pair_l2_min": exec_pair["min"],
        "exec_pair_l2_max": exec_pair["max"],
        "exec_pair_l2_std": exec_pair["std"],
        "exec_pair_l2_per_dim_mean": exec_pair["per_dim_mean"],
        "exec_pair_l2_per_dim_min": exec_pair["per_dim_min"],
        "exec_pair_l2_per_dim_max": exec_pair["per_dim_max"],
        "exec_pair_l2_per_dim_std": exec_pair["per_dim_std"],
        "exec_std_all": float(np.mean(np.std(executed, axis=0))),
        "exec_std_trans": _mean_candidate_std(executed, 0, 3),
        "exec_std_rot": _mean_candidate_std(executed, 3, 6),
        "exec_std_grip": _mean_candidate_std(executed, 6, 7),
    }


def _json_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    arr = np.asarray(value)
    if arr.ndim == 0:
        return arr.item()
    if arr.size == 1:
        return arr.reshape(()).item()
    return str(value)


def extract_request_metadata(obs: dict[str, Any]) -> dict[str, Any]:
    """Extract optional rollout identifiers without guessing from request order."""
    nested = obs.get("metadata", {})
    if not isinstance(nested, dict):
        nested = {}

    def find(aliases: tuple[str, ...]) -> Any:
        for source in (obs, nested):
            for key in aliases:
                if key in source:
                    return _json_scalar(source[key])
        return None

    return {
        "episode_id": find(("episode_id", "trial_id", "rollout_id", "episode")),
        "env_step": find(("env_step", "step", "step_id", "timestep")),
        "task_id": find(("task_id",)),
    }


def strip_request_metadata(obs: dict[str, Any]) -> dict[str, Any]:
    """Remove debug-only identifiers before applying OpenPI model transforms."""
    metadata_keys = {
        "episode_id",
        "trial_id",
        "rollout_id",
        "episode",
        "env_step",
        "step",
        "step_id",
        "timestep",
        "task_id",
        "metadata",
    }
    return {key: value for key, value in obs.items() if key not in metadata_keys}


def build_diagnostic_record(
    *,
    args: Any,
    request_count: int,
    obs: dict[str, Any],
    candidates: Any,
    scores: Any,
    best_idx: int,
    noise: Any,
    raw_actions: Any,
    model_ms: float,
    select_ms: float,
) -> dict[str, Any]:
    exec_horizon = int(getattr(args, "diag_exec_horizon", 0))
    stats = compute_candidate_stats(candidates, scores, best_idx, exec_horizon)
    metadata = extract_request_metadata(obs)
    return {
        "request": int(request_count),
        "time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sample_mode": str(args.sample_mode),
        "q_metrics_available": str(args.sample_mode) == "qselect",
        "noise_scale": float(args.noise_scale),
        "noise_strategy": str(args.noise_strategy),
        "num_action_samples": int(args.num_action_samples),
        "policy_dir": str(args.policy_dir),
        "critic_path": str(getattr(args, "critic_path", "") or ""),
        "episode_id": metadata["episode_id"],
        "env_step": metadata["env_step"],
        "task_id": metadata["task_id"],
        "prompt": str(obs.get("prompt", "")),
        "model_ms": float(model_ms),
        "select_ms": float(select_ms),
        "stats": stats,
        "noise_stats": compute_tensor_diversity(noise, exec_horizon),
        "raw_action_stats": compute_tensor_diversity(raw_actions, exec_horizon),
    }


def append_diag_jsonl(path: str | pathlib.Path, record: dict[str, Any]) -> pathlib.Path:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
    return path


def maybe_save_candidate_record(
    *,
    request_count: int,
    every: int,
    record_dir: str | pathlib.Path,
    candidates: Any,
    scores: Any,
    best_idx: int,
    selected_action: Any,
    stats: dict[str, Any],
    noise: Any,
    raw_actions: Any,
) -> pathlib.Path | None:
    every = int(every)
    if every <= 0 or int(request_count) % every != 0:
        return None
    record_dir = pathlib.Path(record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)
    path = record_dir / f"request_{int(request_count):06d}.npz"
    stats_json = json.dumps(stats, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    np.savez_compressed(
        path,
        candidates=np.asarray(candidates),
        scores=np.asarray(scores),
        best_idx=np.asarray(int(best_idx), dtype=np.int64),
        selected_action=np.asarray(selected_action),
        stats_json=np.asarray(stats_json),
        noise=np.asarray(noise),
        raw_actions=np.asarray(raw_actions),
    )
    return path


def get_noise(
    sample_rng: Any,
    n: int,
    action_horizon: int,
    action_dim: int,
    args: Args | Any,
) -> jnp.ndarray:
    """Generate JAX initial noises for pi0/pi0.5 flow proposal expansion.

    Strategies:
        base:
            Original behavior. IID Gaussian noises, then multiplied by noise_scale.

        hubu:
            Antithetic / complementary noises. For every eps, also use -eps.
            This gives paired opposite proposals.

        zhengjiao:
            Approximately orthogonal noise directions in flattened chunk space.
            This increases directional coverage across candidates.

        guocaiyang:
            Oversample Gaussian noises, then greedily select farthest candidates.
            This maximizes candidate diversity in noise space.

    Returns:
        noise: jnp.ndarray with shape [n, action_horizon, action_dim].
    """
    strategy = getattr(args, "noise_strategy", "base")
    scale = float(getattr(args, "noise_scale", DEFAULT_NOISE_SCALE))

    h = int(action_horizon)
    d = int(action_dim)
    dim = h * d

    if n <= 0:
        raise ValueError(f"num noise samples must be positive, got {n}")

    # 1. 原始默认：保持你现在的行为。
    if strategy == "base":
        noise = jax.random.normal(
            sample_rng,
            (n, h, d),
            dtype=jnp.float32,
        )
        return noise * scale

    # 2. 互补噪声：eps 和 -eps 成对出现。
    if strategy == "hubu":
        half = (n + 1) // 2
        eps = jax.random.normal(
            sample_rng,
            (half, h, d),
            dtype=jnp.float32,
        )
        noise = jnp.concatenate([eps, -eps], axis=0)[:n]
        return noise * scale

    # 3. 正交噪声：flatten 后做近似正交方向，再采样高斯半径。
    if strategy == "zhengjiao":
        if n > dim:
            raise ValueError(
                f"zhengjiao requires n <= action_horizon * action_dim, "
                f"got n={n}, action_horizon={h}, action_dim={d}, dim={dim}"
            )

        key_dir, key_radius = jax.random.split(sample_rng)

        mat = jax.random.normal(
            key_dir,
            (dim, n),
            dtype=jnp.float32,
        )
        q, _ = jnp.linalg.qr(mat)  # [dim, n], orthonormal columns

        # 标准高斯在 dim 维空间里的半径约服从 sqrt(chi-square(dim))。
        # 这样避免正交方向全是单位长度，导致噪声幅度过小。
        radius = jnp.sqrt(
            jax.random.chisquare(
                key_radius,
                df=dim,
                shape=(n,),
            ).astype(jnp.float32)
        )

        flat = q.T * radius[:, None]  # [n, dim]
        noise = flat.reshape(n, h, d)
        return noise * scale

    # 4. 过采样后选最远：先采 M 个，再选出彼此最分散的 n 个。
    if strategy == "guocaiyang":
        # 这里不额外加 args.noise_oversample，保持你当前参数体系最小化。
        # 默认过采样倍数设成 8，至少 32 个。
        m = max(n * 8, 32)

        pool = jax.random.normal(
            sample_rng,
            (m, h, d),
            dtype=jnp.float32,
        )

        pool_np = np.asarray(pool)
        flat = pool_np.reshape(m, -1)

        # 从 norm 最大的点开始，确定性更强。
        first = int(np.argmax(np.sum(flat * flat, axis=1)))
        selected = [first]

        min_dist = np.sum((flat - flat[first]) ** 2, axis=1)
        for _ in range(1, n):
            idx = int(np.argmax(min_dist))
            selected.append(idx)
            dist = np.sum((flat - flat[idx]) ** 2, axis=1)
            min_dist = np.minimum(min_dist, dist)

        selected_np = pool_np[np.asarray(selected, dtype=np.int64)]
        noise = jnp.asarray(selected_np, dtype=jnp.float32)
        return noise * scale

    raise ValueError(
        f"Unknown noise_strategy={strategy!r}. "
        "Expected one of: base, hubu, zhengjiao, guocaiyang."
    )

@dataclasses.dataclass
class Args:
    # Required / important.
    policy_dir: str
    critic_path: str = ""

    # Real experiment knobs.
    sample_mode: str = "qselect"  # qselect | simple | random
    num_action_samples: int = 16
    seed: int = 0

    # Fixed defaults. Usually do not expose from iter.py.
    policy_config: str = "pi05_libero_awbc"
    port: int = DEFAULT_PORT
    num_steps: int = DEFAULT_NUM_STEPS
    noise_scale: float = DEFAULT_NOISE_SCALE
    noise_strategy: str = "base"
    score_horizon: int = DEFAULT_SCORE_HORIZON
    selector_device: str = ""
    default_prompt: str = ""
    record: bool = False
    log_every: int = DEFAULT_LOG_EVERY
    diag_jsonl: str = ""
    diag_save_candidates_every: int = 0
    diag_record_dir: str = ""
    diag_exec_horizon: int = 0


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Isolated pi05 Q-select diagnostic server")

    p.add_argument("--policy-dir", required=True)
    p.add_argument("--critic-path", default="")

    p.add_argument("--sample-mode", default="qselect", choices=("qselect", "simple", "random"))
    p.add_argument("--num-action-samples", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)

    # Keep these as escape hatches, but iter.py normally does not pass them.
    p.add_argument("--policy-config", default="pi05_libero_awbc")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--num-steps", type=int, default=DEFAULT_NUM_STEPS)
    p.add_argument("--noise-scale", type=float, default=DEFAULT_NOISE_SCALE)
    p.add_argument(
        "--noise-strategy",
        default="base",
        choices=("base", "hubu", "zhengjiao", "guocaiyang"),
    )
    p.add_argument("--score-horizon", type=int, default=DEFAULT_SCORE_HORIZON)
    p.add_argument("--selector-device", default="")
    p.add_argument("--default-prompt", default="")
    p.add_argument("--record", action="store_true")
    p.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    p.add_argument("--diag-jsonl", default="")
    p.add_argument("--diag-save-candidates-every", type=int, default=0)
    p.add_argument("--diag-record-dir", default="")
    p.add_argument(
        "--diag-exec-horizon",
        type=int,
        default=0,
        help="Executed action prefix length; 0 uses the full candidate horizon.",
    )

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
        if args.policy_config != "pi05_libero_awbc":
            raise ValueError(
                "This isolated diagnostic server supports only --policy-config=pi05_libero_awbc"
            )

        logging.info("[init] get train config: %s", args.policy_config)
        self.train_config = _config.get_config(args.policy_config)
        logging.info(
            "[init] got train config: action_horizon=%s action_dim=%s model=%s",
            getattr(self.train_config.model, "action_horizon", None),
            getattr(self.train_config.model, "action_dim", None),
            type(self.train_config.model).__name__,
        )

        policy_dir = pathlib.Path(args.policy_dir)
        logging.info("[init] policy_dir=%s exists=%s", policy_dir, policy_dir.exists())
        logging.info("[init] policy params exists=%s", (policy_dir / "params").exists())
        logging.info("[init] policy assets exists=%s", (policy_dir / "assets").exists())

        sample_kwargs = {"num_steps": int(args.num_steps)}
        default_prompt = args.default_prompt if args.default_prompt else None

        logging.info("[init] create_trained_policy begin")
        t0 = time.monotonic()
        self.policy = _policy_config.create_trained_policy(
            self.train_config,
            args.policy_dir,
            sample_kwargs=sample_kwargs,
            default_prompt=default_prompt,
        )
        logging.info("[init] create_trained_policy done in %.1fs", time.monotonic() - t0)

        self.metadata = dict(getattr(self.policy, "metadata", {}) or {})
        self.metadata.update(
            {
                "qselect": {
                    "enabled": args.sample_mode == "qselect",
                    "num_action_samples": int(args.num_action_samples),
                    "sample_mode": args.sample_mode,
                    "critic_path": args.critic_path,
                    "score_horizon": int(args.score_horizon),
                    "noise_strategy": args.noise_strategy,
                    "noise_scale": float(args.noise_scale),
                    "random_select": args.sample_mode == "random",
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
        if not 0 <= int(args.diag_exec_horizon) <= self.action_horizon:
            raise ValueError(
                f"--diag-exec-horizon must be in [0, {self.action_horizon}], "
                f"got {args.diag_exec_horizon}"
            )
        if int(args.diag_save_candidates_every) < 0:
            raise ValueError("--diag-save-candidates-every must be non-negative")
        if int(args.diag_save_candidates_every) > 0 and not args.diag_record_dir:
            raise ValueError(
                "--diag-record-dir is required when --diag-save-candidates-every is positive"
            )

        if args.sample_mode == "qselect":
            if not args.critic_path:
                raise ValueError("--critic-path is required for qselect mode.")
            if str(QSELECT_DIR) not in sys.path:
                sys.path.insert(0, str(QSELECT_DIR))

            logging.info("[init] import QSelector from %s", QSELECT_DIR)
            from selector import QSelector

            critic_path = pathlib.Path(args.critic_path)
            logging.info(
                "[init] load QSelector begin: path=%s exists=%s size_mb=%.2f",
                critic_path,
                critic_path.exists(),
                (critic_path.stat().st_size / 1024 / 1024) if critic_path.exists() else -1.0,
            )
            t0 = time.monotonic()
            self.selector = QSelector(args.critic_path)
            logging.info("[init] load QSelector done in %.1fs", time.monotonic() - t0)
        else:
            logging.info("[init] selector skipped for sample_mode=%s", args.sample_mode)
            self.selector = None

        np.random.seed(args.seed)
        self.random_selector = np.random.default_rng(args.seed)
        self.torch_generator = torch.Generator(device=self.pytorch_device if str(self.pytorch_device).startswith("cuda") else "cpu")
        self.torch_generator.manual_seed(args.seed)
        if not self.is_pytorch:
            # Override Policy's default rng so this server is deterministic under --seed.
            self.policy._rng = jax.random.key(args.seed)

        self.request_count = 0

        logging.info(
            "Pi0QSelectPolicy initialized: config=%s dir=%s horizon=%s action_dim=%s "
            "num_samples=%s mode=%s pytorch=%s noise_strategy=%s noise_scale=%.3f",
            args.policy_config,
            args.policy_dir,
            self.action_horizon,
            self.action_dim,
            self.num_action_samples,
            args.sample_mode,
            self.is_pytorch,
            args.noise_strategy,
            float(args.noise_scale),
        )

    def _sample_batched_raw_actions(
        self, transformed_single: dict[str, Any]
    ) -> tuple[Any, Any, Any, float]:
        """
        Return:
            transformed_batch: batched transformed model input
            raw_actions: batched raw model actions before output transforms, [N,H,D]
            noise: actual model input noise, [N,H,D]
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
            if self.args.noise_strategy != "base":
                raise NotImplementedError("torch not support noise strategy")
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
            return transformed_batch, raw_actions, noise, model_time

        transformed_batch = _tree_batch_repeat(transformed_single, n, framework="jax")
        observation = _model.Observation.from_dict(transformed_batch)

        self.policy._rng, sample_rng = jax.random.split(self.policy._rng)
        noise = get_noise(
            sample_rng=sample_rng,
            n=n,
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
            args=self.args,
        )

        sample_kwargs = dict(getattr(self.policy, "_sample_kwargs", {}) or {})
        sample_kwargs["noise"] = noise

        start = time.monotonic()
        raw_actions = self.policy._sample_actions(sample_rng, observation, **sample_kwargs)
        # Ensure computation is finished for timing and conversion.
        raw_actions.block_until_ready()
        model_time = time.monotonic() - start
        return transformed_batch, raw_actions, noise, model_time

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

        if mode == "random":
            best_idx = int(self.random_selector.integers(0, int(candidates.shape[0])))
            scores = np.zeros((int(candidates.shape[0]),), dtype=np.float32)
            return best_idx, scores

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
                    "[diag] request=%s mode=simple action_shape=%s infer_ms=%.1f",
                    self.request_count,
                    tuple(actions.shape),
                    infer_time * 1000.0,
                )
            return out

        # Match OpenPI Policy.infer(): copy because transforms may mutate.
        transform_start = time.monotonic()
        model_obs = strip_request_metadata(obs)
        transformed_single = jax.tree.map(lambda x: x, model_obs)
        transformed_single = self.policy._input_transform(transformed_single)
        transform_time = time.monotonic() - transform_start

        transformed_batch, raw_actions, noise, model_time = self._sample_batched_raw_actions(
            transformed_single
        )

        output_start = time.monotonic()
        candidates = self._apply_output_transform_per_candidate(transformed_batch, raw_actions)
        output_time = time.monotonic() - output_start

        select_start = time.monotonic()
        best_idx, scores = self._select_candidate(obs, candidates)
        select_time = time.monotonic() - select_start

        selected = candidates[best_idx]

        noise_np = np.asarray(_to_numpy_tree(noise), dtype=np.float32)
        raw_actions_np = np.asarray(_to_numpy_tree(raw_actions), dtype=np.float32)
        record = build_diagnostic_record(
            args=self.args,
            request_count=self.request_count,
            obs=obs,
            candidates=candidates,
            scores=scores,
            best_idx=best_idx,
            noise=noise_np,
            raw_actions=raw_actions_np,
            model_ms=model_time * 1000.0,
            select_ms=select_time * 1000.0,
        )
        if self.args.diag_jsonl:
            append_diag_jsonl(self.args.diag_jsonl, record)
        maybe_save_candidate_record(
            request_count=self.request_count,
            every=self.args.diag_save_candidates_every,
            record_dir=self.args.diag_record_dir,
            candidates=candidates,
            scores=scores,
            best_idx=best_idx,
            selected_action=selected,
            stats=record["stats"],
            noise=noise_np,
            raw_actions=raw_actions_np,
        )

        if self.args.log_every > 0 and self.request_count % self.args.log_every == 0:
            stats = record["stats"]
            base_log = (
                "[diag] request=%s mode=%s scale=%.4g strategy=%s best=%s "
                "cand_l2=%.6f cand_l2_per_dim=%.6f exec_l2=%.6f exec_l2_per_dim=%.6f "
                "best_vs_first=%.6f best_vs_first_exec=%.6f "
                "trans_std=%.6f rot_std=%.6f grip_std=%.6f "
                "saturation=%.6f would_clip=%.6f model_ms=%.1f select_ms=%.1f"
            )
            base_args = (
                self.request_count,
                self.args.sample_mode,
                float(self.args.noise_scale),
                self.args.noise_strategy,
                best_idx,
                stats["cand_pair_l2_mean"],
                stats["cand_pair_l2_per_dim_mean"],
                stats["exec_pair_l2_mean"],
                stats["exec_pair_l2_per_dim_mean"],
                stats["best_vs_first_l2"],
                stats["best_vs_first_exec_l2"],
                stats["cand_std_trans"],
                stats["cand_std_rot"],
                stats["cand_std_grip"],
                stats["cand_saturation_frac"],
                stats["cand_would_clip_frac"],
                model_time * 1000.0,
                select_time * 1000.0,
            )
            if self.args.sample_mode == "qselect":
                logging.info(
                    base_log
                    + " q_available=1 q_mean=%.6f q_std=%.6f q_gap=%.6f q_topgap=%.6f",
                    *base_args,
                    stats["q_mean"],
                    stats["q_std"],
                    stats["q_gap"],
                    stats["q_top1_top2_gap"],
                )
            else:
                logging.info(base_log + " q_available=0", *base_args)

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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        force=True,
    )

    # If initialization hangs, dump a Python stack every 120s.
    # This makes checkpoint restore / JAX compile / selector load hangs visible in policy_server.log.
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)

    logging.info("[boot] parse args")
    args = parse_args()
    logging.info("[boot] args=%s", dataclasses.asdict(args))

    logging.info("[boot] env CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES"))
    logging.info("[boot] env JAX_COMPILATION_CACHE_DIR=%s", os.environ.get("JAX_COMPILATION_CACHE_DIR"))
    logging.info("[boot] env CUDA_CACHE_PATH=%s", os.environ.get("CUDA_CACHE_PATH"))
    logging.info("[boot] env XLA_PYTHON_CLIENT_PREALLOCATE=%s", os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"))
    logging.info("[boot] env XLA_PYTHON_CLIENT_MEM_FRACTION=%s", os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"))

    logging.info("[boot] load runtime dependencies begin")
    t0 = time.monotonic()
    _load_runtime_dependencies()
    logging.info("[boot] load runtime dependencies done in %.1fs", time.monotonic() - t0)

    logging.info(
        "[boot] construct Pi0QSelectPolicy begin: config=%s policy_dir=%s mode=%s critic=%s",
        args.policy_config,
        args.policy_dir,
        args.sample_mode,
        args.critic_path,
    )
    t0 = time.monotonic()
    policy = Pi0QSelectPolicy(args)
    logging.info("[boot] construct Pi0QSelectPolicy done in %.1fs", time.monotonic() - t0)

    server_policy = PolicyRecorder(policy) if args.record else policy

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("[boot] creating websocket server host=%s ip=%s port=%s", hostname, local_ip, args.port)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=server_policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    logging.info("[boot] websocket server created; serve_forever")
    server.serve_forever()


if __name__ == "__main__":
    main()
