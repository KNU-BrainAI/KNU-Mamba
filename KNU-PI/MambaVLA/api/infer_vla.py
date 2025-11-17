"""infer_vla API implementation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from configs.config import MainConfig
from configs.factory import create_model
from configs.io import load_config_from_yaml

from .common import prepare_observation_batch, sync_device


def infer_vla(
    pth: str,
    language: str,
    config_file: Optional[str] = None,
    observations: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Use saved weights and language prompt to output actions.
    """
    cfg = load_config_from_yaml(config_file) if config_file else MainConfig()
    sync_device(cfg)

    device = torch.device(cfg.device)
    model = create_model(cfg)

    state_dict = torch.load(pth, map_location=device)
    model.load_state_dict(state_dict)

    obs_batch = prepare_observation_batch(cfg, language, observations, device)
    model.eval()
    with torch.no_grad():
        actions = model.predict(obs_batch)
    return actions.squeeze(0).cpu()

