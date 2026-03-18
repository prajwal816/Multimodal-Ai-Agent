"""
src/utils/logger.py
───────────────────
Rich-formatted, rotating-file logger for the Multimodal AI Agent.
Level and file path are read from configs/config.yaml at first import.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional

# Rich is optional — degrade gracefully if not installed
try:
    from rich.logging import RichHandler
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


def get_logger(
    name: str = "multimodal_agent",
    level: str = "INFO",
    log_file: Optional[str] = "logs/agent.log",
    rich_console: bool = True,
) -> logging.Logger:
    """
    Return a named logger configured with:
      - RichHandler (colour console) when rich is installed
      - RotatingFileHandler (10 MB / 5 backups)

    Calling this function multiple times with the same *name* returns the
    same logger (standard Python behaviour).
    """
    logger = logging.getLogger(name)

    # Only configure once
    if logger.handlers:
        return logger

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    # ── Console handler ───────────────────────────────────────────────────────
    if rich_console and _RICH_AVAILABLE:
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
            markup=True,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s", datefmt=datefmt))
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    console_handler.setLevel(numeric_level)
    logger.addHandler(console_handler)

    # ── File handler ──────────────────────────────────────────────────────────
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        file_handler.setLevel(numeric_level)
        logger.addHandler(file_handler)

    return logger


def get_logger_from_config(cfg: dict) -> logging.Logger:
    """Convenience wrapper that reads logging settings from config dict."""
    log_cfg = cfg.get("logging", {})
    return get_logger(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("log_file", "logs/agent.log"),
        rich_console=log_cfg.get("rich_console", True),
    )
