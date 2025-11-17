"""API 모듈에서 공유하는 도우미 함수."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from configs.config import MainConfig


def sync_device(cfg: MainConfig) -> None:
    """CUDA 사용 가능 여부에 따라 디바이스를 동기화."""
    wants_cuda = cfg.device == "cuda"
    if wants_cuda and not torch.cuda.is_available():
        target_device = "cpu"
    else:
        target_device = cfg.device

    cfg.device = target_device
    cfg.model_cfg.device = target_device
    cfg.model_cfg.model.device = target_device
    cfg.model_cfg.model.backbones.device = target_device
    cfg.trainer.device = target_device
    cfg.simulation.device = target_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_observation_batch(
    cfg: MainConfig,
    language: str,
    observations: Optional[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, Any]:
    """이미지/언어 입력을 배치 형태로 준비."""
    obs_dict: Dict[str, Any] = {}
    seq_len = cfg.perception_seq_len

    shape_meta = cfg.shape_meta.obs

    if observations:
        for key, value in observations.items():
            if isinstance(value, torch.Tensor):
                obs_dict[key] = value.to(device)
            elif isinstance(value, np.ndarray):
                obs_dict[key] = torch.from_numpy(value).to(device)
            else:
                obs_dict[key] = value

        for cam in cfg.camera_names:
            cam_key = f"{cam}_image"
            if cam_key not in obs_dict:
                raise ValueError(f"관측 딕셔너리에 '{cam_key}'가 없습니다.")
    else:
        for cam in cfg.camera_names:
            key = f"{cam}_image"
            shape = shape_meta[key]["shape"]
            obs_dict[key] = torch.zeros((1, seq_len, *shape), device=device)

    if observations and "lang" in observations:
        obs_dict["lang"] = observations["lang"]
    else:
        obs_dict["lang"] = [language]

    return obs_dict

