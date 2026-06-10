#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train pi0 AWBC through the normal OpenPI training entrypoint.

AWBC is not a separate trainer here.  This wrapper enforces the success-only
LeRobot schema with chunk-level ``adv``, recomputes norm stats for that labeled
repo, then launches:

    scripts/train.py pi0_libero_awbc --data.repo-id <AWBC_REPO>

The repo passed to this script must be the labeled AWBC repo, not the raw
collector repo.
"""

from __future__ import annotations
import torch
import argparse
import dataclasses
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from typing import Any

import numpy as np


REQUIRED_KEYS = {"image", "wrist_image", "state", "actions", "adv", "task"}
FORBIDDEN_KEYS = {"reward", "done", "success", "from_success"}
DEFAULT_ASSET_ID = "physical-intelligence/libero"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate and train pi0 AWBC on a labeled LeRobot repo.")
    p.add_argument("--awbc-repo-id", required=True, help="Success-only LeRobot repo with image/wrist_image/state/actions/adv/task.")
    p.add_argument("--hf-lerobot-home", default="", help="Parent directory for local LeRobot repos.")
    p.add_argument("--model-dir", required=True, help="Output model directory. final/ and steps/ are created below it.")
    p.add_argument("--base-policy-dir", default="", help="Existing pi0 checkpoint dir whose params/ initialize AWBC.")

    p.add_argument("--pi0-root", default="", help="Defaults to this script's pi0 repo root.")
    p.add_argument("--openpi-root", default="", help="Defaults to <pi0-root>/openpi.")
    p.add_argument("--python-bin", default="", help="Defaults to <openpi-root>/pi_env/bin/python.")

    p.add_argument("--config-name", default="pi0_libero_awbc")
    p.add_argument("--assets-base-dir", default="", help="Defaults to <repo_path>/openpi_assets.")
    p.add_argument("--asset-id", default=DEFAULT_ASSET_ID)
    p.add_argument("--project-name", default="openpi")
    p.add_argument("--exp-name", default="awbc")

    p.add_argument("--num-train-steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--save-interval", type=int, default=1000)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--keep-period", default="5000")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fsdp-devices", type=int, default=2)
    p.add_argument("--norm-max-frames", type=int, default=0)

    p.add_argument("--cache-root", default="", help="Defaults to <pi0-root>/cache.")
    p.add_argument("--log-file", default="")

    p.add_argument("--compute-norm-stats", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--check-config", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--wandb-enabled", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def append_log(log_file: Path | None, message: str) -> None:
    line = f"[{now()}] {message}"
    print(line, flush=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def run_cmd(cmd: list[str], *, cwd: Path, env: dict[str, str], log_file: Path | None, dry_run: bool) -> None:
    append_log(log_file, f"[cmd] {quote_cmd(cmd)}")
    append_log(log_file, f"[cwd] {cwd}")
    if dry_run:
        return

    with subprocess.Popen(
        [str(x) for x in cmd],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            if log_file is not None:
                with log_file.open("a", encoding="utf-8") as f:
                    f.write(line)
        rc = proc.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def repo_path_from_args(args: argparse.Namespace) -> Path | None:
    if not args.hf_lerobot_home:
        return None
    return Path(args.hf_lerobot_home) / args.awbc_repo_id


def setup_paths(args: argparse.Namespace) -> dict[str, Path]:
    script_path = Path(__file__).resolve()
    pi0_root = Path(args.pi0_root).resolve() if args.pi0_root else script_path.parents[2]
    openpi_root = Path(args.openpi_root).resolve() if args.openpi_root else pi0_root / "openpi"
    python_bin = Path(args.python_bin).resolve() if args.python_bin else openpi_root / "pi_env" / "bin" / "python"
    model_dir = Path(args.model_dir).resolve()

    repo_path = repo_path_from_args(args)
    if args.assets_base_dir:
        assets_base_dir = Path(args.assets_base_dir).resolve()
    elif repo_path is not None:
        assets_base_dir = repo_path / "openpi_assets"
    else:
        assets_base_dir = model_dir / "openpi_assets"

    cache_root = Path(args.cache_root).resolve() if args.cache_root else pi0_root / "cache"
    return {
        "pi0_root": pi0_root,
        "openpi_root": openpi_root,
        "python_bin": python_bin,
        "model_dir": model_dir,
        "assets_base_dir": assets_base_dir,
        "step_dir": model_dir / "steps",
        "final_dir": model_dir / "final",
        "cache_root": cache_root,
    }


def build_env(args: argparse.Namespace, paths: dict[str, Path]) -> dict[str, str]:
    env = os.environ.copy()
    if args.hf_lerobot_home:
        env["HF_LEROBOT_HOME"] = str(Path(args.hf_lerobot_home).resolve())
    if "OPENPI_DATA_HOME" not in env or not env["OPENPI_DATA_HOME"]:
        raise RuntimeError("OPENPI_DATA_HOME must be set in the shell environment.")
    env["PYTHONUNBUFFERED"] = "1"
    env["JAX_COMPILATION_CACHE_DIR"] = env.get("JAX_COMPILATION_CACHE_DIR", str(paths["cache_root"] / "jax"))
    env["CUDA_CACHE_PATH"] = env.get("CUDA_CACHE_PATH", str(paths["cache_root"] / "cuda"))
    env["CUDA_CACHE_MAXSIZE"] = env.get("CUDA_CACHE_MAXSIZE", "2147483648")
    env["JAX_ENABLE_COMPILATION_CACHE"] = env.get("JAX_ENABLE_COMPILATION_CACHE", "true")
    env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = env.get("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    return env


def to_scalar(x: Any) -> Any:
    if torch.is_tensor(x):
        if x.numel() == 1:
            return x.detach().cpu().item()
        return x.detach().cpu().numpy()
    if isinstance(x, (int, float, bool, str, bytes)):
        return x
    arr = np.asarray(x)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return arr


def validate_awbc_repo(args: argparse.Namespace, paths: dict[str, Path], env: dict[str, str]) -> None:
    if args.hf_lerobot_home:
        os.environ["HF_LEROBOT_HOME"] = env["HF_LEROBOT_HOME"]
    sys.path.insert(0, str(paths["openpi_root"] / "src"))

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.awbc_repo_id)
    if len(ds) == 0:
        raise RuntimeError(f"AWBC repo is empty: {args.awbc_repo_id}")
    sample = ds[0]
    keys = set(sample.keys())

    missing = sorted(REQUIRED_KEYS - keys)
    if missing:
        raise KeyError(f"AWBC repo missing required keys: {missing}. keys={sorted(keys)}")

    bad = sorted(FORBIDDEN_KEYS & keys)
    if bad:
        raise KeyError(f"AWBC repo must not contain collector/IQL-only keys: {bad}")

    adv = np.asarray(sample["adv"])
    if adv.size != 1:
        raise ValueError(f"Expected scalar adv per frame, got shape={adv.shape}")

    state = np.asarray(sample["state"])
    actions = np.asarray(sample["actions"])
    if state.reshape(-1).shape[0] != 8:
        raise ValueError(f"Expected LIBERO state dim 8, got shape={state.shape}")
    if actions.reshape(-1).shape[0] != 7:
        raise ValueError(f"Expected LIBERO action dim 7 per frame, got shape={actions.shape}")

    print("AWBC repo schema is valid.")


def contains_adv_repack(data_config: Any) -> bool:
    for transform in getattr(data_config.repack_transforms, "inputs", ()):
        structure = getattr(transform, "structure", None)
        if structure is not None and "adv" in json.dumps(structure):
            return True
    return False


def validate_awbc_config(args: argparse.Namespace, paths: dict[str, Path]) -> None:
    sys.path.insert(0, str(paths["openpi_root"] / "src"))
    import openpi.training.config as _config

    cfg = _config.get_config(args.config_name)
    if not getattr(cfg, "use_awbc", False):
        raise RuntimeError(f"{args.config_name} must have use_awbc=True")
    if int(getattr(cfg.model, "action_horizon")) != 5:
        raise RuntimeError(f"{args.config_name} must use action_horizon=5, got {cfg.model.action_horizon}")

    cfg = dataclasses.replace(
        cfg,
        data=dataclasses.replace(cfg.data, repo_id=args.awbc_repo_id),
        assets_base_dir=str(paths["assets_base_dir"]),
    )
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    if not contains_adv_repack(data_cfg):
        raise RuntimeError(f"{args.config_name} data config does not repack adv")

    print(f"AWBC config is valid: {args.config_name}, action_horizon={cfg.model.action_horizon}")


def norm_stats_cmd(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    cmd = [
        str(paths["python_bin"]),
        "-u",
        str(paths["pi0_root"] / "train" / "compute_norm_stats_custom.py"),
        args.awbc_repo_id,
        "--asset-id",
        args.asset_id,
        "--config-name",
        args.config_name,
        "--assets-base-dir",
        str(paths["assets_base_dir"]),
        "--checkpoint-base-dir",
        str(paths["model_dir"]),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.norm_max_frames > 0:
        cmd.extend(["--max-frames", str(args.norm_max_frames)])
    return cmd


def train_cmd(args: argparse.Namespace, paths: dict[str, Path]) -> list[str]:
    cmd = [
        str(paths["python_bin"]),
        "-u",
        "scripts/train.py",
        args.config_name,
        f"--exp-name={args.exp_name}",
        f"--project-name={args.project_name}",
        f"--assets-base-dir={paths['assets_base_dir']}",
        f"--checkpoint-base-dir={paths['model_dir']}",
        f"--checkpoint-dir-override={paths['step_dir']}",
        f"--published-checkpoint-dir={paths['final_dir']}",
        f"--batch-size={args.batch_size}",
        f"--num-workers={args.num_workers}",
        f"--num-train-steps={args.num_train_steps}",
        f"--save-interval={args.save_interval}",
        f"--log-interval={args.log_interval}",
        f"--seed={args.seed}",
        f"--fsdp-devices={args.fsdp_devices}",
        f"--data.repo-id={args.awbc_repo_id}",
        f"--data.assets.asset-id={args.asset_id}",
    ]
    if args.base_policy_dir:
        params_path = Path(args.base_policy_dir).resolve() / "params"
        cmd.append(f"--weight-loader.params-path={params_path}")
    if args.keep_period in {"none", "None"}:
        cmd.append("--keep-period=None")
    else:
        cmd.append(f"--keep-period={args.keep_period}")
    if not args.wandb_enabled:
        cmd.append("--no-wandb-enabled")
    if args.overwrite:
        cmd.append("--overwrite")
    if args.resume:
        cmd.append("--resume")
    return cmd


def main() -> None:
    args = parse_args()
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume cannot both be true.")

    paths = setup_paths(args)
    env = build_env(args, paths)
    log_file = Path(args.log_file).resolve() if args.log_file else paths["model_dir"] / "train_awbc.log"

    (paths["cache_root"] / "jax").mkdir(parents=True, exist_ok=True)
    (paths["cache_root"] / "cuda").mkdir(parents=True, exist_ok=True)
    paths["model_dir"].mkdir(parents=True, exist_ok=True)

    append_log(log_file, "========== pi0 AWBC ==========")
    append_log(log_file, f"repo_id={args.awbc_repo_id}")
    append_log(log_file, f"HF_LEROBOT_HOME={env.get('HF_LEROBOT_HOME', '')}")
    append_log(log_file, f"model_dir={paths['model_dir']}")
    append_log(log_file, f"assets_base_dir={paths['assets_base_dir']}")

    validate_awbc_repo(args, paths, env)
    if args.check_config:
        validate_awbc_config(args, paths)

    if args.validate_only:
        append_log(log_file, "validate-only complete.")
        return

    if args.compute_norm_stats:
        run_cmd(norm_stats_cmd(args, paths), cwd=paths["openpi_root"], env=env, log_file=log_file, dry_run=args.dry_run)
    else:
        append_log(log_file, "Skipping norm stats computation.")

    run_cmd(train_cmd(args, paths), cwd=paths["openpi_root"], env=env, log_file=log_file, dry_run=args.dry_run)
    append_log(log_file, f"AWBC final checkpoint: {paths['final_dir']}")


if __name__ == "__main__":
    main()
