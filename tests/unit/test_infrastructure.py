"""Unit tests for paths, platform info, and logging setup."""

import logging
from pathlib import Path
from app.infrastructure.paths import PathManager
from app.infrastructure.platform_info import get_platform_info
from app.infrastructure.logging_setup import setup_logging, PrivacyRedactionFilter
from app.config.schema import LoggingConfig, PrivacyConfig


def test_path_manager_ensure_dirs(tmp_path: Path):
    pm = PathManager(root=tmp_path)
    pm.ensure_directories()
    assert (tmp_path / "logs").is_dir()
    assert (tmp_path / "data" / "test_reports").is_dir()
    assert (tmp_path / "models").is_dir()


def test_platform_info():
    info = get_platform_info()
    assert "os" in info
    assert "architecture" in info
    assert "python_version" in info
    assert info["cpu_count"] >= 1


def test_logging_redaction_filter():
    redactor = PrivacyRedactionFilter(enabled=True)
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="Detected vehicle with plate MH12AB1234 in Bay 1",
        args=(),
        exc_info=None,
    )
    redactor.filter(record)
    assert "MH12AB1234" not in record.msg
    assert "[PLATE-REDACTED]" in record.msg


def test_logging_setup(tmp_path: Path):
    log_file = tmp_path / "logs" / "test.log"
    l_cfg = LoggingConfig(file_path=str(log_file), level="DEBUG")
    p_cfg = PrivacyConfig(redact_log_plates=True)
    logger = setup_logging(l_cfg, p_cfg)

    logger.info("Application initialized with plate DL01CA9999")
    # Flush handlers
    for handler in logger.handlers:
        handler.flush()

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "DL01CA9999" not in content
    assert "[PLATE-REDACTED]" in content
