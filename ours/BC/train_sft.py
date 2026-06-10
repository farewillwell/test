#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal pi0/OpenPI SFT launcher for the iterative SFT baseline.

Required:
  --data-dir         local ordinary SFT LeRobot repo, e.g. iter0/data/sft
  --model-dir        output checkpoint dir, e.g. iter0/sft_model
  --base-policy-dir  checkpoint dir containing params/, e.g. pi0_base or previous final/

Fixed defaults:
  config-name: pi0_libero_low_mem_finetune
  asset-id: physical-intelligence/libero
  project-name: openpi
  exp-name: sft
  HF_LEROBOT_HOME: dirname(data-dir)
  assets-base-dir: data-dir/openpi_assets
  overwrite: true
  wandb: disabled
  save-interval: steps, so only the final training boundary is saved/published

This wrapper intentionally does not require or read any AWBC/IQL-only fields such
as adv, reward, terminal, success, task_id, trial_id, or step_index. The input
repo is expected to contain the ordinary OpenPI SFT fields:
  image, wrist_image, state, actions, task
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any


CONFIG_NAME = "pi0_libero"
ASSET_ID = "physical-intelligence/libero"
PROJECT_NAME = "openpi"
EXP_NAME = "sft"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal pi0/OpenPI SFT training launcher.")

    p.add_argument("--data-dir", required=True, help="Local ordinary SFT LeRobot repo directory.")
    p.add_argument("--model-dir", required=True, help="Output checkpoint directory.")
    p.add_argument("--base-policy-dir", required=True, help="Checkpoint directory containing params/.")

    p.add_argument("--steps", "--num-train-steps", dest="steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gpus", "--fsdp-devices", dest="gpus", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-file", default="")

    p.add_argument("--pi0-root", default="")
    p.add_argument("--openpi-root", default="")
    p.add_argument("--python-bin", default="")

    args, unknown = p.parse_known_args()
    if unknown:
        print(f"[warn] ignoring unknown args: {' '.join(unknown)}", flush=True)
    return args


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def quote_cmd(cmd: list[Any]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def log(log_file: Path | None, msg: str) -> None:
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def check_env(env: dict[str, str], log_file: Path | None) -> None:
    bad_none = [k for k, v in env.items() if v is None]
    if bad_none:
        for k in sorted(bad_none):
            log(log_file, f"[env-error] {k}=None")
        raise RuntimeError("env contains None values: " + ", ".join(sorted(bad_none)))

    bad_type = [k for k, v in env.items() if not isinstance(v, (str, bytes, os.PathLike))]
    if bad_type:
        for k in sorted(bad_type):
            log(log_file, f"[env-error] {k}: type={type(env[k]).__name__} value={env[k]!r}")
        raise RuntimeError("env contains non-string values: " + ", ".join(sorted(bad_type)))


def run(cmd: list[Any], *, cwd: Path, env: dict[str, str], log_file: Path | None) -> None:
    log(log_file, f"[cmd] {quote_cmd(cmd)}")
    log(log_file, f"[cwd] {cwd}")
    check_env(env, log_file)

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


def infer_roots(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    script_path = Path(__file__)
    pi0_root = Path(args.pi0_root) if args.pi0_root else script_path.parents[2]
    openpi_root = Path(args.openpi_root) if args.openpi_root else pi0_root / "openpi"
    python_bin = Path(args.python_bin) if args.python_bin else openpi_root / "pi_env" / "bin" / "python"
    return pi0_root, openpi_root, python_bin


def require_lerobot_repo(data_dir: Path) -> None:
    info_path = data_dir / "meta" / "info.json"
    if not data_dir.exists():
        raise FileNotFoundError(f"--data-dir does not exist: {data_dir}")
    if not info_path.exists():
        raise FileNotFoundError(f"Invalid local LeRobot repo: {data_dir}\nExpected: {info_path}")


def build_env(data_dir) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["JAX_COMPILATION_CACHE_DIR"] = env.get("JAX_COMPILATION_CACHE_DIR")
    env["HF_LEROBOT_HOME"] = str(data_dir.parent)
    env["CUDA_CACHE_PATH"] = env.get("CUDA_CACHE_PATH")
    env["CUDA_CACHE_MAXSIZE"] = env.get("CUDA_CACHE_MAXSIZE", "2147483648")
    env["JAX_ENABLE_COMPILATION_CACHE"] = env.get("JAX_ENABLE_COMPILATION_CACHE", "true")
    env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = env.get("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0")
    return env


def main() -> None:
    args = parse_args()

    pi0_root, openpi_root, python_bin = infer_roots(args)
    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    base_policy_dir = Path(args.base_policy_dir)

    require_lerobot_repo(data_dir)

    params_path = base_policy_dir / "params"
    if not params_path.exists():
        raise FileNotFoundError(f"Base policy params not found: {params_path}")

    repo_id = data_dir.name
    assets_base_dir = data_dir / "openpi_assets"
    final_dir = model_dir / "final"
    step_dir = model_dir / "steps"
    log_file = Path(args.log_file) if args.log_file else model_dir / "train_sft.log"

    model_dir.mkdir(parents=True, exist_ok=True)
    env = build_env(data_dir)

    log(log_file, "========== pi0 SFT baseline ==========")
    log(log_file, f"data_dir={data_dir}")
    log(log_file, f"repo_id={repo_id}")
    log(log_file, f"HF_LEROBOT_HOME={env['HF_LEROBOT_HOME']}")
    log(log_file, f"model_dir={model_dir}")
    log(log_file, f"base_policy_dir={base_policy_dir}")
    log(log_file, f"steps={args.steps} save_interval={args.steps}")
    log(log_file, f"config={CONFIG_NAME} asset_id={ASSET_ID}")

    norm_cmd = [
        str(python_bin),
        "-u",
        str(pi0_root / "train" / "compute_norm_stats_custom.py"),
        repo_id,
        "--asset-id",
        ASSET_ID,
        "--config-name",
        CONFIG_NAME,
        "--assets-base-dir",
        str(assets_base_dir),
        "--checkpoint-base-dir",
        str(model_dir),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
    ]
    run(norm_cmd, cwd=openpi_root, env=env, log_file=log_file)

    train_cmd = [
        str(python_bin),
        "-u",
        "scripts/train.py",
        CONFIG_NAME,
        "--exp-name=sft",
        f"--project-name={PROJECT_NAME}",
        f"--assets-base-dir={assets_base_dir}",
        f"--checkpoint-base-dir={model_dir}",
        f"--checkpoint-dir-override={step_dir}",
        f"--published-checkpoint-dir={final_dir}",
        f"--batch-size={args.batch_size}",
        f"--num-workers={args.num_workers}",
        f"--num-train-steps={args.steps}",
        f"--save-interval={args.steps}",
        "--log-interval=100",
        f"--seed={args.seed}",
        f"--fsdp-devices={args.gpus}",
        f"--data.repo-id={repo_id}",
        f"--data.assets.asset-id={ASSET_ID}",
        f"--weight-loader.params-path={params_path}",
        "--keep-period=None",
        "--no-wandb-enabled",
        "--overwrite",
    ]
    run(train_cmd, cwd=openpi_root, env=env, log_file=log_file)

    if not (final_dir / "params").exists():
        raise RuntimeError(f"Expected final params not found: {final_dir / 'params'}")

    log(log_file, f"SFT final checkpoint: {final_dir}")


if __name__ == "__main__":
    main()
