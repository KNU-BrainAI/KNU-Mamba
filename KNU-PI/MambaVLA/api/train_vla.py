"""train_vla API implementation."""

from __future__ import annotations

from pathlib import Path

import torch
import wandb

from configs.factory import create_model, create_trainer
from configs.io import load_config_from_yaml

from .common import set_seed, sync_device


def train_vla(
    config_file: str,
    dataset_path: str,
    output_path: str,
) -> str:
    """
    Use YAML configuration and dataset path to train the model and save the weights.
    """
    cfg = load_config_from_yaml(config_file)
    cfg.dataset.dataset_path = str(Path(dataset_path).expanduser().resolve())

    sync_device(cfg)
    set_seed(cfg.seed)

    wandb_config = {
        "project": cfg.wandb.project,
        "entity": cfg.wandb.entity,
        "group": cfg.group,
        "seed": cfg.seed,
        "benchmark_type": cfg.dataset.benchmark_type,
        "demos_per_task": cfg.dataset.demos_per_task,
        "chunck_size": cfg.chunck_size,
        "perception_seq_len": cfg.perception_seq_len,
        "action_seq_len": cfg.action_seq_len,
        "train_batch_size": cfg.train_batch_size,
        "epoch": cfg.epoch,
        "device": cfg.device,
        "len_embd": cfg.len_embd,
        "latent_dim": cfg.latent_dim,
        "action_dim": cfg.action_dim,
        "state_dim": cfg.state_dim,
    }

    wandb_run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.group,
        config=wandb_config,
    )

    try:
        model = create_model(cfg)
        trainer = create_trainer(cfg)

        workdir = Path(output_path).expanduser().resolve().parent
        workdir.mkdir(parents=True, exist_ok=True)
        model.working_dir = str(workdir)
        trainer.working_dir = str(workdir)

        trainer.main(model)

        torch.save(model.state_dict(), Path(output_path))
    finally:
        wandb.finish()

    return str(Path(output_path))

