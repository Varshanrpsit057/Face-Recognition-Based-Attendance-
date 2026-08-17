"""
Configuration loader — load, save, validate, and export configuration.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

import yaml

from config import AppConfig, cfg


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml_config(config: Dict[str, Any], path: Path) -> None:
    """Save a dict as YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def _update_dataclass(obj: Any, overrides: Dict[str, Any]) -> None:
    """Recursively update dataclass fields from a dict."""
    for key, value in overrides.items():
        if hasattr(obj, key):
            attr = getattr(obj, key)
            if dataclasses.is_dataclass(attr) and isinstance(value, dict):
                _update_dataclass(attr, value)
            else:
                setattr(obj, key, value)


def merge_config(base: AppConfig, overrides: Dict[str, Any]) -> AppConfig:
    """Merge YAML overrides into an AppConfig instance."""
    _update_dataclass(base, overrides)
    return base


def validate_config(config: AppConfig) -> List[str]:
    """Return a list of validation warnings for the given config."""
    warnings: List[str] = []
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    level_value = config.logging.level.value if isinstance(config.logging.level, Enum) else str(config.logging.level)
    if level_value not in valid_levels:
        warnings.append(f"Invalid log level: {level_value}")
    if config.recognition.similarity_threshold < 0.0 or config.recognition.similarity_threshold > 1.0:
        warnings.append(f"Similarity threshold out of range: {config.recognition.similarity_threshold}")
    if config.quality.min_face_width < 1:
        warnings.append(f"Minimum face width too small: {config.quality.min_face_width}")
    return warnings


def _serialize(obj: Any) -> Any:
    """Convert non-serializable objects (Path, Enum) for export."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _serialize(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


def export_config(config: AppConfig) -> Dict[str, Any]:
    """Serialize config to a JSON-safe dict."""
    raw = dataclasses.asdict(config)
    return _serialize(raw)


def get_runtime_config() -> Dict[str, Any]:
    """Return current runtime config including hardware info."""
    config_dict = export_config(cfg)
    try:
        from src.device_manager import DeviceManager
        device_info = DeviceManager().get_device_info()
        config_dict["hardware"] = dataclasses.asdict(device_info)
    except Exception:
        config_dict["hardware"] = {}
    return config_dict


def update_config_field(config: AppConfig, field_path: str, value: Any) -> AppConfig:
    """Update a nested config field. E.g. field_path='recognition.similarity_threshold'."""
    parts = field_path.split(".")
    target: Any = config
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)
    return config
