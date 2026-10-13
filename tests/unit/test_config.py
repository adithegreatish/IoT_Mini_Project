"""Unit tests for configuration loading and validation."""

import pytest
from pathlib import Path
from pydantic import ValidationError
from app.config.schema import RootConfig, CameraConfig, ParkingConfig
from app.config.loader import load_config, save_config


def test_default_config_instantiation():
    config = RootConfig()
    assert config.app.name == "Smart Parking & HSRP System"
    assert config.camera.requested_fps == 30
    assert config.parking.occupancy_threshold == 0.35
    assert config.detection.enabled is True
    assert config.database.retention_days == 30


def test_windows_yaml_load():
    config = load_config("configs/windows.yaml")
    assert config.app.env == "windows"
    assert config.camera.requested_width == 1280
    assert config.camera.requested_height == 720
    assert "car" in config.detection.classes


def test_raspberry_pi_yaml_load():
    config = load_config("configs/raspberry_pi.yaml")
    assert config.app.env == "raspberry_pi"
    assert config.camera.requested_fps == 30
    assert config.detection.input_width == 480
    assert config.database.retention_days == 14
    assert config.performance.max_memory_limit_mb == 7168
    assert config.performance.sqlite_cache_size_mb == 1024


def test_config_invalid_threshold_raises():
    with pytest.raises(ValidationError):
        ParkingConfig(occupancy_threshold=1.5)  # threshold must be <= 1.0


def test_config_save_and_reload(tmp_path: Path):
    temp_file = tmp_path / "custom.yaml"
    cfg = RootConfig()
    cfg.app.port = 9000
    cfg.camera.requested_fps = 60
    save_config(cfg, str(temp_file))

    assert temp_file.exists()
    reloaded = load_config(str(temp_file))
    assert reloaded.app.port == 9000
    assert reloaded.camera.requested_fps == 60
