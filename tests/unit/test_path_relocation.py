"""
Tests for repository root discovery, working-directory independence,
project-relative path resolution, and absence of hardcoded machine paths.
"""

import os
from pathlib import Path
import tempfile
import pytest

from app.config.loader import load_config
from app.config.schema import DetectionConfig, PlatesConfig
from app.database.connection import DatabaseManager
from app.detection.vehicle_detector import VehicleDetector
from app.infrastructure.paths import PathManager, discover_project_root, paths
from app.plates.plate_detector import PlateDetector


def test_discover_root_independent_of_cwd():
    """Verifies that discover_project_root never relies on CWD."""
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as foreign_dir:
        try:
            os.chdir(foreign_dir)
            detected = discover_project_root()
            # Root must be a directory containing pyproject.toml, not foreign_dir
            assert (detected / "pyproject.toml").exists()
            assert detected != Path(foreign_dir)
        finally:
            os.chdir(original_cwd)


def test_env_var_override():
    """Verifies that SMART_PARKING_ROOT environment variable overrides discovery."""
    with tempfile.TemporaryDirectory() as custom_dir:
        os.environ["SMART_PARKING_ROOT"] = custom_dir
        try:
            mgr = PathManager()
            assert mgr.root == Path(custom_dir).resolve()
            assert mgr.resolve("data/test.txt") == (Path(custom_dir).resolve() / "data" / "test.txt")
        finally:
            os.environ.pop("SMART_PARKING_ROOT", None)


def test_relative_path_conversions():
    """Verifies that PathManager properly converts paths to project-relative."""
    p_mgr = PathManager()
    abs_data = p_mgr.data_dir
    rel = p_mgr.to_relative(abs_data)
    assert rel == Path("data")
    assert p_mgr.to_relative_str(abs_data) == "data"

    # Relative path should remain relative
    assert p_mgr.to_relative("models/test.onnx") == Path("models/test.onnx")


def test_dataset_yaml_validity_from_foreign_cwd():
    """Verifies that dataset YAML configurations and paths succeed from another working directory."""
    import yaml

    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as foreign_dir:
        try:
            os.chdir(foreign_dir)
            # Check vehicles.yaml using resolved path
            v_yaml = paths.resolve("data/vehicles.yaml")
            assert v_yaml.exists(), f"vehicles.yaml not found at {v_yaml}"
            with open(v_yaml, "r", encoding="utf-8") as f:
                v_data = yaml.safe_load(f)
            assert "names" in v_data

            # Check plates.yaml using resolved path
            p_yaml = paths.resolve("data/plates.yaml")
            assert p_yaml.exists(), f"plates.yaml not found at {p_yaml}"
            with open(p_yaml, "r", encoding="utf-8") as f:
                p_data = yaml.safe_load(f)
            assert "names" in p_data

            # If ultralytics is installed (training/dev env), also run ultralytics dataset validation
            try:
                from ultralytics.data.utils import check_det_dataset
                res_v = check_det_dataset(str(v_yaml))
                assert Path(res_v["train"]).exists()
                assert Path(res_v["val"]).exists()

                res_p = check_det_dataset(str(p_yaml))
                assert Path(res_p["train"]).exists()
                assert Path(res_p["val"]).exists()
            except (ImportError, ModuleNotFoundError):
                pass  # Ultralytics is omitted from edge/Pi deployments
        finally:
            os.chdir(original_cwd)


def test_model_loading_with_relative_paths_from_foreign_cwd():
    """Verifies models load using relative paths even when CWD is changed."""
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as foreign_dir:
        try:
            os.chdir(foreign_dir)
            v_det = VehicleDetector(DetectionConfig(runtime="onnx", model_path="models/vehicle_detector.onnx"))
            assert v_det.status == "READY"

            p_det = PlateDetector(PlatesConfig(runtime="onnx", model_path="models/plate_detector.onnx"))
            assert p_det.status == "READY"
        finally:
            os.chdir(original_cwd)


def test_database_resolution_from_foreign_cwd():
    """Verifies SQLite connection resolves relative path correctly regardless of CWD."""
    original_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as foreign_dir:
        try:
            os.chdir(foreign_dir)
            db = DatabaseManager("data/test_events.sqlite")
            assert str(paths.root) in str(db.db_path)
            conn = db.get_connection()
            assert conn is not None
            conn.close()
            # Clean up test database
            if db.db_path.exists():
                db.db_path.unlink()
        finally:
            os.chdir(original_cwd)


def test_no_hardcoded_machine_specific_paths_in_configs_and_source():
    """Verifies that no committed source code or configs contain hardcoded machine paths."""
    forbidden = ["C:\\Users\\adity\\Downloads\\IoT", "c:/users/adity/downloads/iot"]
    root = paths.root

    # Check app/ and configs/
    for check_dir in [root / "app", root / "configs"]:
        for file in check_dir.rglob("*"):
            if file.is_file() and file.suffix in [".py", ".yaml", ".json"]:
                content = file.read_text(encoding="utf-8", errors="ignore").lower()
                for bad in forbidden:
                    assert bad.lower() not in content, f"Found hardcoded path in {file}"
