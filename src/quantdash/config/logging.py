"""Structured logging configuration."""

import logging
import sys
from typing import Optional

from .settings import get_settings


def setup_logging(level: Optional[str] = None) -> logging.Logger:
    """Configure structured logging for the application."""
    settings = get_settings()
    log_level = level or settings.LOG_LEVEL

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    # Create package logger
    logger = logging.getLogger("quantdash")
    logger.setLevel(log_level)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a named logger under the quantdash namespace."""
    return logging.getLogger(f"quantdash.{name}")
