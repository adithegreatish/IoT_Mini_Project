"""Configuration package for Smart Parking system."""

from app.config.schema import (
    AppConfig,
    CameraConfig,
    ParkingConfig,
    DetectionConfig,
    PlatesConfig,
    OCRConfig,
    DatabaseConfig,
    DiagnosticsConfig,
    PrivacyConfig,
    LoggingConfig,
    RootConfig,
)
from app.config.loader import load_config, get_config, save_config, find_workspace_root

__all__ = [
    "AppConfig",
    "CameraConfig",
    "ParkingConfig",
    "DetectionConfig",
    "PlatesConfig",
    "OCRConfig",
    "DatabaseConfig",
    "DiagnosticsConfig",
    "PrivacyConfig",
    "LoggingConfig",
    "RootConfig",
    "load_config",
    "get_config",
    "save_config",
    "find_workspace_root",
]
