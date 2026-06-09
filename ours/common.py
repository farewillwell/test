from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def episode_parquet_files(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No LeRobot episode parquet files found under {root}/data")
    return files


def load_tasks(data_dir: str | Path) -> dict[int, str]:
    rows = read_jsonl(Path(data_dir) / "meta" / "tasks.jsonl")
    tasks: dict[int, str] = {}
    for row in rows:
        idx = row.get("task_index", row.get("index"))
        task = row.get("task", row.get("name"))
        if idx is not None and task is not None:
            tasks[int(idx)] = str(task)
    return tasks


def load_episode_metadata(data_dir: str | Path) -> dict[int, dict]:
    rows = read_jsonl(Path(data_dir) / "meta" / "episodes.jsonl")
    out: dict[int, dict] = {}
    for row in rows:
        if "episode_index" in row:
            out[int(row["episode_index"])] = row
    rollout_rows = read_jsonl(Path(data_dir) / "meta" / "ours_rollouts.jsonl")
    for row in rollout_rows:
        if "episode_index" not in row:
            continue
        episode_index = int(row["episode_index"])
        out.setdefault(episode_index, {}).update(row)
    return out


def image_cell_to_pil(cell) -> Image.Image:
    if isinstance(cell, dict) and "bytes" in cell:
        return Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(cell)).convert("RGB")
    arr = np.asarray(cell)
    if arr.ndim == 3:
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")
    raise TypeError(f"Unsupported image cell type: {type(cell)}")


def build_action_chunks(actions: np.ndarray, horizon: int) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected actions [T,A], got {actions.shape}")
    t = actions.shape[0]
    idx = np.arange(t)[:, None] + np.arange(horizon)[None, :]
    idx = np.clip(idx, 0, t - 1)
    return actions[idx]


def episode_reward(meta: dict, default_success: bool = True) -> float:
    for key in ("success", "is_success", "from_success"):
        if key in meta:
            return 1.0 if bool(meta[key]) else 0.0
    if "reward" in meta:
        return float(meta["reward"])
    return 1.0 if default_success else 0.0


def load_episode_df(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)
