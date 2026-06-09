from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPProcessor

from common import write_jsonl
from light_iql import LeRobotIQLDataset, LightIQLCritic, collate


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--critic-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    ckpt = torch.load(args.critic_path, map_location="cpu")
    cfg = ckpt["args"]
    processor = CLIPProcessor.from_pretrained(cfg["encoder_name"])
    dataset = LeRobotIQLDataset(
        Path(args.data_dir),
        processor,
        int(cfg["horizon"]),
        int(cfg["image_size"]),
        str(cfg["views"]).split(","),
        bool(cfg.get("default_success", False)),
    )
    sample = dataset[0]
    model = LightIQLCritic(
        cfg["encoder_name"],
        len(sample["state"]),
        sample["actions"].shape[-1],
        int(cfg["horizon"]),
        hidden_dim=int(cfg.get("hidden_dim", 512)),
        num_q=int(cfg.get("num_q", 2)),
        action_layers=int(cfg.get("action_layers", 2)),
        q_layers=int(cfg.get("q_layers", 2)),
    )
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=lambda b: collate(b, processor))
    rows = []
    cursor = 0
    for batch in tqdm(loader, desc="label"):
        size = batch["state"].shape[0]
        tbatch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        s = model.encode(tbatch["pixel_values"], tbatch["input_ids"], tbatch["attention_mask"], tbatch["view_counts"], tbatch["state"])
        q = model.q_values(s, tbatch["actions"]).min(dim=0).values.squeeze(-1)
        v = model.v(s).squeeze(-1)
        adv = (q - v).detach().cpu().float().numpy()
        q_np = q.detach().cpu().float().numpy()
        v_np = v.detach().cpu().float().numpy()
        for i in range(size):
            path, frame_idx, _ = dataset.items[cursor + i]
            episode = int(path.stem.split("_")[-1])
            rows.append(
                {
                    "episode_index": episode,
                    "frame_index": int(frame_idx),
                    "q": float(q_np[i]),
                    "v": float(v_np[i]),
                    "adv": float(adv[i]),
                }
            )
        cursor += size

    out = Path(args.output_dir)
    write_jsonl(out / "labels.jsonl", rows)
    arr = np.asarray([r["adv"] for r in rows], dtype=np.float32)
    summary = {
        "count": int(arr.size),
        "adv_mean": float(arr.mean()) if arr.size else 0.0,
        "adv_std": float(arr.std()) if arr.size else 0.0,
        "adv_min": float(arr.min()) if arr.size else 0.0,
        "adv_max": float(arr.max()) if arr.size else 0.0,
    }
    write_jsonl(out / "summary.jsonl", [summary])
    print(summary)


if __name__ == "__main__":
    main()
