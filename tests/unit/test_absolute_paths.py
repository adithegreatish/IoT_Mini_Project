"""Unit tests for absolute path migration, SMART_PARKING_ROOT, and CWD independence."""

import os
from pathlib import Path
import pytest

from app.config.loader import load_config
from app.database.connection import DatabaseManager
from app.infrastructure.paths import PathManager, discover_project_root, paths


def test_default_project_root_is_absolute_and_deterministic():
    root = discover_project_root()
    assert root.is_absolute()
    assert root.exists()
    assert (root / "app").is_dir()
    assert (root / "configs").is_dir()


def test_smart_parking_root_env_override(tmp_path: Path, monkeypatch):
    custom_root = tmp_path / "custom_project_root"
    monkeypatch.setenv("SMART_PARKING_ROOT", str(custom_root))
    discovered = discover_project_root()
    assert discovered == custom_root.resolve()
    assert discovered.is_dir()


def test_path_manager_standard_properties_are_absolute():
    pm = PathManager()
    assert pm.root.is_absolute()
    assert pm.data_dir.is_absolute()
    assert pm.raw_data_dir.is_absolute()
    assert pm.processed_data_dir.is_absolute()
    assert pm.annotations_dir.is_absolute()
    assert pm.splits_dir.is_absolute()
    assert pm.models_dir.is_absolute()
    assert pm.configs_dir.is_absolute()
    assert pm.logs_dir.is_absolute()
    assert pm.calibration_file.is_absolute()
    assert pm.database_file.is_absolute()
    assert pm.models_manifest_file.is_absolute()
    assert pm.vehicle_model_onnx.is_absolute()
    assert pm.plate_model_onnx.is_absolute()

    # All must be children of root
    assert str(pm.data_dir).startswith(str(pm.root))
    assert str(pm.models_dir).startswith(str(pm.root))


def test_cwd_independence_for_path_resolution(tmp_path: Path):
    orig_cwd = Path.cwd()
    try:
        # Switch working directory to a completely foreign temp path
        os.chdir(str(tmp_path))
        assert Path.cwd() == tmp_path.resolve()

        # Relative path resolution MUST resolve against project root, not current working directory!
        resolved = paths.resolve("data/calibration.json")
        assert resolved.is_absolute()
        assert resolved == (paths.root / "data" / "calibration.json").resolve()
        assert not str(resolved).startswith(str(tmp_path))

        # Absolute paths must remain absolute
        abs_target = (tmp_path / "other" / "file.txt").resolve()
        assert paths.resolve(abs_target) == abs_target
    finally:
        os.chdir(str(orig_cwd))


def test_config_loading_from_foreign_cwd(tmp_path: Path):
    orig_cwd = Path.cwd()
    try:
        os.chdir(str(tmp_path))
        cfg = load_config("configs/windows.yaml")
        assert cfg.app.env == "windows"
        assert cfg.camera.requested_width == 1280
    finally:
        os.chdir(str(orig_cwd))


def test_database_access_from_foreign_cwd(tmp_path: Path):
    orig_cwd = Path.cwd()
    try:
        os.chdir(str(tmp_path))
        db_mgr = DatabaseManager("data/parking_events.sqlite")
        assert db_mgr.db_path.is_absolute()
        assert db_mgr.db_path == (paths.root / "data" / "parking_events.sqlite").resolve()
        assert not str(db_mgr.db_path).startswith(str(tmp_path))

        with db_mgr.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            result = cursor.fetchone()[0]
            assert result == 1
    finally:
        os.chdir(str(orig_cwd))
