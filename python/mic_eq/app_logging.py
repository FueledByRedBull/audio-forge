"""Application logging setup for desktop/runtime code."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config_parts import shared


LOG_DIR_NAME = shared.APPDATA_DIR_NAME
LOG_FILE_MAX_BYTES = 1_000_000
LOG_FILE_BACKUP_COUNT = 3


def get_log_file() -> Path:
    """Return the desktop app log file path."""
    return shared._config_base_dir() / LOG_DIR_NAME / "logs" / "app.log"


def configure_app_logging() -> Path:
    """Configure rotating file logging for the desktop app."""
    log_file = get_log_file()
    try:
        # Resolving the config folder runs the legacy migration, so it must come
        # before anything creates or opens files inside the folder being swapped.
        shared.get_config_file()
        if shared.migration_pending():
            logging.getLogger(__name__).warning("File logging is off until the config migration completes")
            return log_file
        log_file.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        # A read-only or locked profile must not stop the app from starting.
        logging.getLogger(__name__).warning("File logging is off: %s", log_file.parent, exc_info=True)
        return log_file

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, RotatingFileHandler) and Path(handler.baseFilename) == log_file:
            return log_file

    handler = RotatingFileHandler(
        log_file,
        maxBytes=LOG_FILE_MAX_BYTES,
        backupCount=LOG_FILE_BACKUP_COUNT,
        encoding="utf-8",
        # A second launch that exits early never opens app.log, so it cannot block the first one's rollover.
        delay=True,
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    root_logger.addHandler(handler)
    if root_logger.level == logging.NOTSET or root_logger.level > logging.INFO:
        root_logger.setLevel(logging.INFO)

    return log_file


__all__ = ["configure_app_logging", "get_log_file"]
