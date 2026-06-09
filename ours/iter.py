from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path


STAGES = ("train_iql", "label", "make_awbc_data", "train_awbc", "collect")
FINISHED = "finished"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--seed-data-dir", required=True, help="Initial LeRobot dataset.")
    parser.add_argument("--init-model-dir", required=True, help="Initial policy checkpoint, usually a final/ dir.")
    parser.add_argument("--iters", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--iql-steps", type=int, default=4000)
    parser.add_argument("--config-name", default="pi0_libero_low_mem_finetune")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--keep-period", default="5000")
    parser.add_argument("--fsdp-devices", type=int, default=2)
    parser.add_argument("--project-name", default="openpi")
    parser.add_argument("--asset-id", default="physical-intelligence/libero")
    parser.add_argument("--norm-max-frames", type=int, default=0)
    parser.add_argument("--views", default="image,wrist_image")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--iql-batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-q", type=int, default=2)
    parser.add_argument("--num-action-samples", type=int, default=8)
    parser.add_argument("--sample-mode", choices=("qselect", "best", "random", "first"), default="qselect")
    parser.add_argument("--task-suite-name", default="libero_goal")
    parser.add_argument("--task-id", type=int, default=6)
    parser.add_argument("--num-trials-per-task", type=int, default=50)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def quote_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def run(
    cmd: list[str],
    log_path: Path,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
    cwd: Path | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = quote_cmd(cmd)
    print(line)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {line}\n")
        if env:
            for key in sorted(env):
                f.write(f"env {key}={env[key]}\n")
        if cwd:
            f.write(f"cwd {cwd}\n")
    if dry_run:
        return

    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=merged_env, cwd=cwd) as proc:
        assert proc.stdout is not None
        with log_path.open("a", encoding="utf-8") as f:
            for out_line in proc.stdout:
                print(out_line, end="")
                f.write(out_line)
        return_code = proc.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def build_initial_state(args: argparse.Namespace) -> dict:
    return {
        "version": 1,
        "workspace": str(Path(args.workspace).resolve()),
        "iters_total": int(args.iters),
        "iter": 0,
        "next_stage": STAGES[0],
        "data_dir": str(Path(args.seed_data_dir).resolve()),
        "policy_dir": str(Path(args.init_model_dir).resolve()),
        "history": [],
    }


def paths_for_iter(workspace: Path, iter_index: int) -> dict[str, Path]:
    root = workspace / f"iter{iter_index}"
    return {
        "root": root,
        "logs": root / "logs",
        "q_dir": root / "q",
        "label_dir": root / "labels",
        "awbc_data": root / "awbc_lerobot",
        "model_dir": root / "model",
        "collect_data": root / "collected_lerobot",
    }


def stage_train_iql(args: argparse.Namespace, pi0_root: Path, state: dict, p: dict[str, Path]) -> None:
    run(
        [
            "python",
            str(pi0_root / "ours" / "light_iql.py"),
            "--data-dir",
            state["data_dir"],
            "--save-dir",
            str(p["q_dir"]),
            "--max-steps",
            str(args.iql_steps),
            "--batch-size",
            str(args.iql_batch_size),
            "--views",
            args.views,
            "--horizon",
            str(args.horizon),
            "--hidden-dim",
            str(args.hidden_dim),
            "--num-q",
            str(args.num_q),
            "--default-success",
            "--seed",
            str(args.seed),
        ],
        p["logs"] / "train_iql.log",
        dry_run=args.dry_run,
    )


def stage_label(args: argparse.Namespace, pi0_root: Path, state: dict, p: dict[str, Path]) -> None:
    run(
        [
            "python",
            str(pi0_root / "ours" / "label_q.py"),
            "--data-dir",
            state["data_dir"],
            "--critic-path",
            str(p["q_dir"] / "final.pt"),
            "--output-dir",
            str(p["label_dir"]),
        ],
        p["logs"] / "label.log",
        dry_run=args.dry_run,
    )


def stage_make_awbc_data(args: argparse.Namespace, pi0_root: Path, state: dict, p: dict[str, Path]) -> None:
    run(
        [
            "python",
            str(pi0_root / "ours" / "make_awbc_data.py"),
            "--src-data-dir",
            state["data_dir"],
            "--label-dir",
            str(p["label_dir"]),
            "--output-dir",
            str(p["awbc_data"]),
        ],
        p["logs"] / "make_awbc_data.log",
        dry_run=args.dry_run,
    )


def stage_train_awbc(args: argparse.Namespace, pi0_root: Path, state: dict, p: dict[str, Path]) -> None:
    openpi_root = pi0_root / "openpi"
    python = openpi_root / "pi_env" / "bin" / "python"
    assets_base_dir = p["awbc_data"] / "openpi_assets"
    repo_id = p["awbc_data"].name
    hf_lerobot_home = str(p["awbc_data"].parent)
    base_env = {
        "HF_LEROBOT_HOME": hf_lerobot_home,
        "OPENPI_DATA_HOME": os.environ.get("OPENPI_DATA_HOME", "/data/aoss/heliqun/model/pi"),
        "JAX_COMPILATION_CACHE_DIR": os.environ.get("JAX_COMPILATION_CACHE_DIR", str(pi0_root / "cache" / "jax")),
        "CUDA_CACHE_PATH": os.environ.get("CUDA_CACHE_PATH", str(pi0_root / "cache" / "cuda")),
        "CUDA_CACHE_MAXSIZE": os.environ.get("CUDA_CACHE_MAXSIZE", "2147483648"),
        "JAX_ENABLE_COMPILATION_CACHE": os.environ.get("JAX_ENABLE_COMPILATION_CACHE", "true"),
        "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": os.environ.get(
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "0"
        ),
        "PYTHONUNBUFFERED": "1",
    }
    stats_cmd = [
        str(python),
        "-u",
        str(pi0_root / "train" / "compute_norm_stats_custom.py"),
        repo_id,
        "--asset-id",
        args.asset_id,
        "--config-name",
        args.config_name,
        "--assets-base-dir",
        str(assets_base_dir),
        "--checkpoint-base-dir",
        str(p["model_dir"]),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
    ]
    if args.norm_max_frames > 0:
        stats_cmd.extend(["--max-frames", str(args.norm_max_frames)])
    run(
        stats_cmd,
        p["logs"] / "compute_norm_stats.log",
        env=base_env,
        dry_run=args.dry_run,
        cwd=openpi_root,
    )

    params_path = Path(state["policy_dir"]) / "params"
    train_args = [
        str(python),
        "-u",
        "scripts/train.py",
        args.config_name,
        f"--exp-name=awbc_iter{state['iter']}",
        f"--project-name={args.project_name}",
        f"--assets-base-dir={assets_base_dir}",
        f"--checkpoint-base-dir={p['model_dir']}",
        f"--checkpoint-dir-override={p['model_dir'] / 'steps'}",
        f"--published-checkpoint-dir={p['model_dir'] / 'final'}",
        f"--batch-size={args.batch_size}",
        f"--num-workers={args.num_workers}",
        f"--num-train-steps={args.train_steps}",
        f"--save-interval={args.save_interval}",
        f"--log-interval={args.log_interval}",
        f"--seed={args.seed}",
        f"--fsdp-devices={args.fsdp_devices}",
        f"--data.repo-id={repo_id}",
        f"--data.assets.asset-id={args.asset_id}",
        f"--weight-loader.params-path={params_path}",
        "--no-wandb-enabled",
        "--overwrite",
    ]
    if args.keep_period in {"none", "None"}:
        train_args.append("--keep-period=None")
    else:
        train_args.append(f"--keep-period={args.keep_period}")
    run(
        train_args,
        p["logs"] / "train_awbc.log",
        env=base_env,
        dry_run=args.dry_run,
        cwd=openpi_root,
    )


def stage_collect(args: argparse.Namespace, pi0_root: Path, state: dict, p: dict[str, Path]) -> None:
    run(
        ["bash", str(pi0_root / "ours" / "run_collect_with_server.sh")],
        p["logs"] / "collect.log",
        env={
            "MODEL_PATH": str(p["model_dir"] / "final"),
            "OUTPUT_DIR": str(p["collect_data"]),
            "CRITIC_PATH": str(p["q_dir"] / "final.pt"),
            "CONFIG_NAME": args.config_name,
            "HOST": args.host,
            "PORT": str(args.port),
            "NUM_ACTION_SAMPLES": str(args.num_action_samples),
            "SAMPLE_MODE": args.sample_mode,
            "TASK_SUITE_NAME": args.task_suite_name,
            "TASK_ID": str(args.task_id),
            "NUM_TRIALS_PER_TASK": str(args.num_trials_per_task),
            "REPLAN_STEPS": str(args.replan_steps),
            "SEED": str(args.seed),
            "OVERWRITE": "true",
        },
        dry_run=args.dry_run,
    )


STAGE_EXECUTORS = {
    "train_iql": stage_train_iql,
    "label": stage_label,
    "make_awbc_data": stage_make_awbc_data,
    "train_awbc": stage_train_awbc,
    "collect": stage_collect,
}


def advance_after_success(state: dict, stage: str, p: dict[str, Path]) -> dict:
    iter_index = int(state["iter"])
    stage_index = STAGES.index(stage)
    state.setdefault("history", []).append(
        {
            "iter": iter_index,
            "stage": stage,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_dir": state["data_dir"],
            "policy_dir": state["policy_dir"],
        }
    )
    if stage_index + 1 < len(STAGES):
        state["next_stage"] = STAGES[stage_index + 1]
        return state

    state["policy_dir"] = str(p["model_dir"] / "final")
    state["data_dir"] = str(p["collect_data"])
    state["iter"] = iter_index + 1
    if state["iter"] >= int(state["iters_total"]):
        state["next_stage"] = FINISHED
    else:
        state["next_stage"] = STAGES[0]
    return state


def main() -> None:
    args = parse_args()
    pi0_root = Path(__file__).resolve().parents[1]
    workspace = Path(args.workspace).resolve()
    save_path = workspace / "save.json"

    state = load_json(save_path) or build_initial_state(args)
    state["iters_total"] = int(args.iters)
    save_json(save_path, state)

    while state["next_stage"] != FINISHED:
        iter_index = int(state["iter"])
        if iter_index >= int(state["iters_total"]):
            state["next_stage"] = FINISHED
            save_json(save_path, state)
            break
        stage = str(state["next_stage"])
        if stage not in STAGE_EXECUTORS:
            raise ValueError(f"Unknown stage in {save_path}: {stage}")
        p = paths_for_iter(workspace, iter_index)
        p["root"].mkdir(parents=True, exist_ok=True)

        state["running_stage"] = stage
        state["running_iter"] = iter_index
        save_json(save_path, state)
        STAGE_EXECUTORS[stage](args, pi0_root, state, p)

        state.pop("running_stage", None)
        state.pop("running_iter", None)
        state = advance_after_success(state, stage, p)
        save_json(save_path, state)

    print(f"Finished. save file: {save_path}")


if __name__ == "__main__":
    main()
