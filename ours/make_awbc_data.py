from __future__ import annotations

import argparse
import math
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from common import image_cell_to_pil, load_episode_df, load_tasks, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-data-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max-copies", type=int, default=5)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    src = Path(args.src_data_dir)
    out = Path(args.output_dir)
    if out.exists():
        shutil.rmtree(out)
    labels = read_jsonl(Path(args.label_dir) / "labels.jsonl")
    tasks = load_tasks(src)
    by_ep: dict[int, list[float]] = defaultdict(list)
    for row in labels:
        by_ep[int(row["episode_index"])].append(float(row["adv"]))
    ep_weight = {}
    all_adv = np.asarray([float(r["adv"]) for r in labels], dtype=np.float32)
    mean = float(all_adv.mean()) if all_adv.size else 0.0
    std = float(all_adv.std() + 1e-6) if all_adv.size else 1.0
    for ep, vals in by_ep.items():
        adv = (float(np.mean(vals)) - mean) / std
        ep_weight[ep] = max(1, min(args.max_copies, int(round(math.exp(adv / max(args.beta, 1e-6))))))

    dataset = LeRobotDataset.create(
        repo_id=out.name,
        root=out,
        robot_type="libero",
        fps=args.fps,
        features={
            "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        use_videos=False,
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for p in sorted((src / "data").glob("chunk-*/episode_*.parquet")):
        ep = int(p.stem.split("_")[-1])
        copies = ep_weight.get(ep, 1)
        df = load_episode_df(p)
        for _ in range(copies):
            for _, row in df.iterrows():
                task = tasks.get(int(row["task_index"]), "")
                dataset.add_frame(
                    {
                        "image": image_cell_to_pil(row["image"]),
                        "wrist_image": image_cell_to_pil(row["wrist_image"]),
                        "state": row["state"],
                        "actions": row["actions"],
                        "task": task,
                    }
                )
            dataset.save_episode()
    if hasattr(dataset, "stop_image_writer"):
        dataset.stop_image_writer()
    if (src / "openpi_assets").exists():
        shutil.copytree(src / "openpi_assets", out / "openpi_assets", dirs_exist_ok=True)
    print(f"Saved AWBC resampled LeRobot data to {out}")


if __name__ == "__main__":
    main()
