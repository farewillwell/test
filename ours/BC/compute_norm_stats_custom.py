#!/usr/bin/env python3
"""Compute OpenPI norm stats with external dataset/config overrides."""

from __future__ import annotations

import dataclasses

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max(1, max_frames // batch_size)
        shuffle = True
    else:
        num_batches = max(1, len(dataset) // batch_size)
        shuffle = False

    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def build_config(
    config_name: str,
    repo_id: str,
    asset_id: str | None,
    assets_base_dir: str,
    checkpoint_base_dir: str,
    batch_size: int | None,
    num_workers: int | None,
) -> _config.TrainConfig:
    config = _config.get_config(config_name)
    assets = config.data.assets
    if asset_id:
        assets = dataclasses.replace(assets, asset_id=asset_id)
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(config.data, repo_id=repo_id, assets=assets),
        assets_base_dir=assets_base_dir,
        checkpoint_base_dir=checkpoint_base_dir,
    )
    if batch_size is not None:
        config = dataclasses.replace(config, batch_size=batch_size)
    if num_workers is not None:
        config = dataclasses.replace(config, num_workers=num_workers)
    return config


def main(
    repo_id: str,
    asset_id: str | None = None,
    config_name: str = "pi0_libero_low_mem_finetune",
    assets_base_dir: str = "./assets",
    checkpoint_base_dir: str = "./checkpoints",
    batch_size: int | None = None,
    num_workers: int | None = None,
    max_frames: int | None = None,
) -> None:
    config = build_config(config_name, repo_id, asset_id, assets_base_dir, checkpoint_base_dir, batch_size, num_workers)
    data_config = config.data.create(config.assets_dirs, config.model)

    data_loader, num_batches = create_torch_dataloader(
        data_config,
        config.model.action_horizon,
        config.batch_size,
        config.model,
        config.num_workers,
        max_frames,
    )

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in stats:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: running_stats.get_statistics() for key, running_stats in stats.items()}
    output_path = config.assets_dirs / (data_config.asset_id or data_config.repo_id)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
