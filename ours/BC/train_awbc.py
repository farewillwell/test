#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal AWBC launcher.

Required:
  --data-dir         local labeled LeRobot repo, e.g. iter0/data/labeled
  --model-dir        output checkpoint dir, e.g. iter0/awbc_model
  --base-policy-dir  checkpoint dir containing params/, e.g. pi0_base or previous final/

Optional:
  --steps, --batch-size, --num-workers, --gpus, --seed

Fixed defaults:
  config-name: pi0_libero_awbc
  asset-id: physical-intelligence/libero
  repo-id: basename(data-dir), normally labeled
  assets-base-dir: data-dir/openpi_assets
  overwrite: true
  wandb: false
  keep-period: None
  save-interval: steps + 1, so no intermediate step checkpoints are saved
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any


CONFIG_NAME = "pi0_libero_awbc"
ASSET_ID = "physical-intelligence/libero"
PROJECT_NAME = "openpi"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal pi0 AWBC training launcher.")

    # Minimal new interface.
    p.add_argument("--data-dir", default="", help="Local labeled LeRobot repo directory, e.g. iter0/data/labeled.")
    p.add_argument("--model-dir", required=True, help="Output checkpoint dir.")
    p.add_argument("--base-policy-dir", required=True, help="Checkpoint dir containing params/.")

    p.add_argument("--steps", "--num-train-steps", dest="steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gpus", "--fsdp-devices", dest="gpus", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-file", default="")

    # Optional root overrides. Usually not needed.
    p.add_argument("--pi0-root", default="")
    p.add_argument("--openpi-root", default="")
    p.add_argument("--python-bin", default="")

    # Legacy compatibility with the previous iter.py call.
    # These are accepted so old calls do not immediately break, but the simple
    # path is --data-dir.
    p.add_argument("--awbc-repo-id", default="")
    p.add_argument("--hf-lerobot-home", default="")
    p.add_argument("--config-name", default=CONFIG_NAME)
    p.add_argument("--asset-id", default=ASSET_ID)
    p.add_argument("--project-name", default=PROJECT_NAME)
    p.add_argument("--exp-name", default="awbc")
    p.add_argument("--save-interval", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--keep-period", default="None")
    p.add_argument("--norm-max-frames", type=int, default=0)

    # Accepted and ignored / mapped to fixed behavior.
    p.add_argument("--wandb-enabled", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--compute-norm-stats", action=argparse.BooleanOptionalAction, default=True)

    args, unknown = p.parse_known_args()
    if unknown:
        raise RuntimeError(f"[warn] ignoring unknown args: {' '.join(unknown)}")

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
    bad = [k for k, v in env.items() if v is None]
    if bad:
        for k in sorted(bad):
            log(log_file, f"[env-error] {k}=None")
        raise RuntimeError("env contains None values: " + ", ".join(sorted(bad)))


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
    script_path = Path(__file__).absolute()
    pi0_root = Path(args.pi0_root) if args.pi0_root else script_path.parents[2]
    openpi_root = Path(args.openpi_root) if args.openpi_root else pi0_root / "openpi"
    python_bin = Path(args.python_bin) if args.python_bin else openpi_root / "pi_env" / "bin" / "python"
    return pi0_root, openpi_root, python_bin


def infer_data_dir(args: argparse.Namespace) -> Path:
    if args.data_dir:
        return Path(args.data_dir)

    if args.hf_lerobot_home and args.awbc_repo_id:
        return Path(args.hf_lerobot_home) / args.awbc_repo_id

    raise ValueError(
        "Missing --data-dir. Minimal usage: "
        "--data-dir /path/to/iter0/data/labeled --model-dir ... --base-policy-dir ..."
    )


def require_repo(data_dir: Path) -> None:
    info = data_dir / "meta" / "info.json"
    if not info.exists():
        raise FileNotFoundError(f"Invalid local LeRobot repo: {data_dir}\nExpected: {info}")


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
    data_dir = infer_data_dir(args)
    model_dir = Path(args.model_dir)
    base_policy_dir = Path(args.base_policy_dir)

    require_repo(data_dir)

    repo_id = data_dir.name
    assets_base_dir = data_dir / "openpi_assets"
    params_path = base_policy_dir / "params"
    if not params_path.exists():
        raise FileNotFoundError(f"Base policy params not found: {params_path}")

    model_dir.mkdir(parents=True, exist_ok=True)
    final_dir = model_dir / "final"
    step_dir = model_dir / "steps"
    log_file = Path(args.log_file) if args.log_file else model_dir / "train_awbc.log"

    env = build_env(data_dir)

    save_interval = args.save_interval if args.save_interval > 0 else args.steps+1

    log(log_file, "========== pi0 AWBC simple ==========")
    log(log_file, f"data_dir={data_dir}")
    log(log_file, f"repo_id={repo_id}")
    log(log_file, f"model_dir={model_dir}")
    log(log_file, f"HF_LEROBOT_HOME={env['HF_LEROBOT_HOME']}")
    log(log_file, f"base_policy_dir={base_policy_dir}")
    log(log_file, f"steps={args.steps} save_interval={save_interval}")
    log(log_file, f"config={args.config_name} asset_id={args.asset_id}")

    if args.compute_norm_stats:
        norm_cmd = [
            str(python_bin),
            "-u",
            str(pi0_root / "train" / "compute_norm_stats_custom.py"),
            "--repo-id",
            repo_id,
            "--asset-id",
            args.asset_id,
            "--config-name",
            args.config_name,
            "--assets-base-dir",
            str(assets_base_dir),
            "--checkpoint-base-dir",
            str(model_dir),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
        ]
        if args.norm_max_frames > 0:
            norm_cmd.extend(["--max-frames", str(args.norm_max_frames)])
        run(norm_cmd, cwd=openpi_root, env=env, log_file=log_file)

    train_cmd = [
        str(python_bin),
        "-u",
        "scripts/train.py",
        args.config_name,
        f"--exp-name={args.exp_name}",
        f"--project-name={args.project_name}",
        f"--assets-base-dir={assets_base_dir}",
        f"--checkpoint-base-dir={model_dir}",
        f"--checkpoint-dir-override={step_dir}",
        f"--published-checkpoint-dir={final_dir}",
        f"--batch-size={args.batch_size}",
        f"--num-workers={args.num_workers}",
        f"--num-train-steps={args.steps}",
        f"--save-interval={save_interval}",
        f"--log-interval={args.log_interval}",
        f"--seed={args.seed}",
        f"--fsdp-devices={args.gpus}",
        f"--data.repo-id={repo_id}",
        f"--data.assets.asset-id={args.asset_id}",
        f"--weight-loader.params-path={params_path}",
        "--keep-period=None",
        "--no-wandb-enabled",
        "--overwrite",
        "--save-train-state=false",
    ]

    # Keep resume opt-in only. Default is overwrite.
    if args.resume:
        train_cmd = [x for x in train_cmd if x != "--overwrite"]
        train_cmd.append("--resume")

    run(train_cmd, cwd=openpi_root, env=env, log_file=log_file)

    if not (final_dir / "params").exists():
        raise RuntimeError(f"Expected final params not found: {final_dir / 'params'}")

    log(log_file, f"AWBC final checkpoint: {final_dir}")


if __name__ == "__main__":
    main()
