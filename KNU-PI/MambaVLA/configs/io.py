"""
Input/output helper: convert dataclass-based configuration to YAML or load from YAML.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, MutableMapping

import yaml

from .config import MainConfig


def load_config_from_yaml(path: str | Path | None) -> MainConfig:
    """Read YAML file and return MainConfig instance."""
    config = MainConfig()
    if path is None:
        return config

    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    _apply_updates(config, data)
    return config


def save_config_to_yaml(config: MainConfig, path: str | Path) -> None:
    """Save MainConfig instance to YAML file."""
    data = _dataclass_to_dict(config)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


def _apply_updates(target: Any, updates: MutableMapping[str, Any]) -> None:
    """Apply recursive dict updates to dataclass instance."""
    for key, value in updates.items():
        if not hasattr(target, key):
            raise KeyError(f"{target.__class__.__name__} does not have '{key}' field.")

        current = getattr(target, key)
        if is_dataclass(current) and isinstance(value, MutableMapping):
            _apply_updates(current, value)
        else:
            setattr(target, key, value)


def _dataclass_to_dict(obj: Any) -> Any:
    """Convert dataclass and nested structures to YAML-friendly dict."""
    if is_dataclass(obj):
        result = {}
        for field in fields(obj):
            result[field.name] = _dataclass_to_dict(getattr(obj, field.name))
        return result
    if isinstance(obj, dict):
        return {key: _dataclass_to_dict(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(val) for val in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


__all__ = [
    "load_config_from_yaml",
    "save_config_to_yaml",
]

