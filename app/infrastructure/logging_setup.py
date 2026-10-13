"""Structured logging with file rotation and privacy redaction."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import re
from typing import Optional

from app.config.schema import LoggingConfig, PrivacyConfig
from app.infrastructure.paths import paths


class PrivacyRedactionFilter(logging.Filter):
    """Redacts potential vehicle license plate strings from logs."""

    PLATE_PATTERN = re.compile(r"\b([A-Z]{2}\s*[\d]{1,2}\s*[A-Z]{0,3}\s*[\d]{4})\b", re.IGNORECASE)

    def __init__(self, enabled: bool = True):
        super().__init__()
        self.enabled = enabled

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.enabled:
            return True
        if isinstance(record.msg, str):
            record.msg = self.PLATE_PATTERN.sub(r"[PLATE-REDACTED]", record.msg)
        return True


def setup_logging(
    logging_config: Optional[LoggingConfig] = None,
    privacy_config: Optional[PrivacyConfig] = None,
) -> logging.Logger:
    """Configures root application logger with console and rotating file output."""
    if logging_config is None:
        logging_config = LoggingConfig()
    if privacy_config is None:
        privacy_config = PrivacyConfig()

    log_level = getattr(logging, logging_config.level.upper(), logging.INFO)
    logger = logging.getLogger("smart_parking")
    logger.setLevel(log_level)
    logger.propagate = False

    # Clear existing handlers to avoid duplicate log lines
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s.%(module)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    redaction_filter = PrivacyRedactionFilter(enabled=privacy_config.redact_log_plates)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(redaction_filter)
    logger.addHandler(console_handler)

    # File Handler
    log_file = paths.resolve(logging_config.file_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=logging_config.max_bytes,
            backupCount=logging_config.backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redaction_filter)
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Failed to initialize rotating file logger at '{log_file}': {e}")

    return logger
