"""Pytest configuration and shared test fixtures."""

import pytest
from pathlib import Path
from app.config.schema import RootConfig
from app.config.loader import load_config
from app.infrastructure.paths import PathManager


@pytest.fixture
def test_paths(tmp_path: Path) -> PathManager:
    """Fixture providing temporary isolated paths manager."""
    pm = PathManager(root=tmp_path)
    pm.ensure_directories()
    return pm


@pytest.fixture
def sample_config() -> RootConfig:
    """Fixture returning validated default root config."""
    return load_config("configs/windows.yaml")
