"""Configuration loader and validator for the application."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from pydantic import ValidationError

from app.config.schema import RootConfig


_cached_config: Optional[RootConfig] = None


def find_workspace_root() -> Path:
    """Finds the workspace root directory via centralized PathManager single source of truth."""
    from app.infrastructure.paths import paths
    return paths.root


def load_config(config_path: Optional[str] = None) -> RootConfig:
    """
    Loads configuration following precedence order:
    1. Built-in defaults (RootConfig)
    2. YAML config file if provided or found
    3. Environment variable overrides (e.g. SMART_PARKING_ENV)
    """
    global _cached_config
    from app.infrastructure.paths import paths

    data: Dict[str, Any] = {}

    target_path: Optional[Path] = None
    if config_path:
        target_path = paths.resolve(config_path)
    else:
        # Check environment or default windows config
        env_profile = os.getenv("SMART_PARKING_CONFIG", "configs/windows.yaml")
        candidate = paths.resolve(env_profile)
        if candidate.exists():
            target_path = candidate

    if target_path and target_path.exists():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
                if isinstance(content, dict):
                    data = content
        except Exception as e:
            raise RuntimeError(f"Failed to parse YAML configuration at '{target_path}': {e}") from e
    elif config_path:
        raise FileNotFoundError(f"Specified configuration file not found at '{target_path}'")

    # Environment variable overrides
    if "SMART_PARKING_PORT" in os.environ:
        data.setdefault("app", {})["port"] = int(os.environ["SMART_PARKING_PORT"])
    if "SMART_PARKING_HOST" in os.environ:
        data.setdefault("app", {})["host"] = os.environ["SMART_PARKING_HOST"]
    if "SMART_PARKING_ENV" in os.environ:
        data.setdefault("app", {})["env"] = os.environ["SMART_PARKING_ENV"]

    try:
        config = RootConfig.model_validate(data)
    except ValidationError as ve:
        raise ValueError(f"Configuration schema validation failed: {ve}") from ve

    _cached_config = config
    return config


def get_config() -> RootConfig:
    """Returns currently loaded configuration or loads default."""
    global _cached_config
    if _cached_config is None:
        _cached_config = load_config()
    return _cached_config


def save_config(config: RootConfig, target_path: str) -> None:
    """Saves effective configuration to target YAML path."""
    from app.infrastructure.paths import paths
    dest = paths.resolve(target_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(), f, sort_keys=False)

