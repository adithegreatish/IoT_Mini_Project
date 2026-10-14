"""Centralized filesystem path management and single source of truth for project paths.

Enforces absolute, canonical path resolution across the entire repository.
Supports SMART_PARKING_ROOT environment variable override or deterministic
repository-structure root discovery that NEVER relies on Path.cwd().
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger("smart_parking.paths")


def discover_project_root() -> Path:
    """
    Deterministically discovers the canonical project root.
    Priority:
    1. SMART_PARKING_ROOT environment variable (if set, resolved and validated).
    2. Structural discovery from current file location (climbing upwards to find markers).
    NEVER falls back to Path.cwd().
    """
    env_root = os.environ.get("SMART_PARKING_ROOT", "").strip()
    if env_root:
        candidate = Path(env_root).resolve()
        try:
            candidate.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(
                f"SMART_PARKING_ROOT configured as '{env_root}' cannot be resolved or created: {e}"
            ) from e
        return candidate

    # Structural discovery starting from this file: app/infrastructure/paths.py
    current = Path(__file__).resolve().parent
    markers = ["pyproject.toml", "configs", "requirements-windows.txt"]
    for _ in range(10):
        if any((current / marker).exists() for marker in markers):
            return current
        if current.parent == current:
            break
        current = current.parent

    # Fallback to 2 levels up from app/infrastructure: <repo>
    return Path(__file__).resolve().parent.parent.parent


class PathManager:
    """
    Centralized path management authority.
    Guarantees every resolved path is absolute, canonical, and normalized.
    """

    def __init__(self, root: Optional[Union[str, Path]] = None):
        self._root: Path = Path(root).resolve() if root else discover_project_root()

    @property
    def root(self) -> Path:
        """Absolute canonical project root."""
        return self._root

    def set_root(self, new_root: Union[str, Path]) -> None:
        """Sets project root dynamically (useful for test isolation)."""
        self._root = Path(new_root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative_or_absolute: Union[str, Path]) -> Path:
        """
        Converts any path (relative or absolute) into a canonical absolute Path.
        Relative paths are strictly resolved against self.root, never against CWD.
        """
        p = Path(relative_or_absolute)
        if p.is_absolute():
            return p.resolve()
        return (self._root / p).resolve()

    def resolve_str(self, relative_or_absolute: Union[str, Path]) -> str:
        """Returns string representation of resolved absolute path."""
        return str(self.resolve(relative_or_absolute))

    def to_relative(self, path: Union[str, Path]) -> Path:
        """
        Converts any path into a Path relative to project root.
        If the path is already relative, returns it normalized.
        If the path is outside project root, returns Path(path).
        """
        p = Path(path)
        if not p.is_absolute():
            return p
        try:
            return p.resolve().relative_to(self._root.resolve())
        except ValueError:
            return p

    def to_relative_str(self, path: Union[str, Path], posix: bool = True) -> str:
        """Returns portable string representation of project-relative path."""
        rel = self.to_relative(path)
        return rel.as_posix() if posix else str(rel)

    # Standard Directory Properties (all returning absolute Paths)
    @property
    def data_dir(self) -> Path:
        return self.resolve("data")

    @property
    def raw_data_dir(self) -> Path:
        return self.resolve("data/raw")

    @property
    def processed_data_dir(self) -> Path:
        return self.resolve("data/processed")

    @property
    def annotations_dir(self) -> Path:
        return self.resolve("data/annotations")

    @property
    def splits_dir(self) -> Path:
        return self.resolve("data/splits")

    @property
    def demo_dir(self) -> Path:
        return self.resolve("data/demo")

    @property
    def test_reports_dir(self) -> Path:
        return self.resolve("data/test_reports")

    @property
    def evaluations_dir(self) -> Path:
        return self.resolve("data/test_reports/evaluations")

    @property
    def models_dir(self) -> Path:
        return self.resolve("models")

    @property
    def training_runs_dir(self) -> Path:
        return self.resolve("models/training_runs")

    @property
    def configs_dir(self) -> Path:
        return self.resolve("configs")

    @property
    def logs_dir(self) -> Path:
        return self.resolve("logs")

    @property
    def docs_dir(self) -> Path:
        return self.resolve("docs")

    @property
    def cache_dir(self) -> Path:
        return self.resolve("data/.cache")

    # Standard File Properties (all returning absolute Paths)
    @property
    def calibration_file(self) -> Path:
        return self.resolve("data/calibration.json")

    @property
    def database_file(self) -> Path:
        return self.resolve("data/parking_events.sqlite")

    @property
    def models_manifest_file(self) -> Path:
        return self.resolve("models/manifest.json")

    @property
    def vehicle_model_onnx(self) -> Path:
        return self.resolve("models/vehicle_detector.onnx")

    @property
    def plate_model_onnx(self) -> Path:
        return self.resolve("models/plate_detector.onnx")

    def ensure_directories(self) -> None:
        """Ensures all standard project directories exist on disk."""
        dirs = [
            self.root,
            self.logs_dir,
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.processed_data_dir / "parking_slots",
            self.processed_data_dir / "plates",
            self.annotations_dir,
            self.splits_dir,
            self.demo_dir,
            self.test_reports_dir,
            self.evaluations_dir,
            self.models_dir,
            self.training_runs_dir,
            self.training_runs_dir / "vehicle",
            self.training_runs_dir / "plate",
            self.configs_dir,
            self.cache_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def log_paths(self) -> None:
        """Logs canonical project root and primary directory paths."""
        logger.info(f"SMART_PARKING_ROOT: {self.root}")
        logger.info(f"Data Directory:     {self.data_dir}")
        logger.info(f"Models Directory:   {self.models_dir}")
        logger.info(f"Configs Directory:  {self.configs_dir}")
        logger.info(f"Database File:      {self.database_file}")

    def get_summary_dict(self) -> Dict[str, str]:
        """Returns dictionary of resolved paths for health & diagnostics reporting."""
        return {
            "root": str(self.root),
            "data": str(self.data_dir),
            "raw": str(self.raw_data_dir),
            "processed": str(self.processed_data_dir),
            "annotations": str(self.annotations_dir),
            "splits": str(self.splits_dir),
            "models": str(self.models_dir),
            "configs": str(self.configs_dir),
            "logs": str(self.logs_dir),
            "database": str(self.database_file),
            "calibration": str(self.calibration_file),
        }


# Global authoritative instance
paths = PathManager()

