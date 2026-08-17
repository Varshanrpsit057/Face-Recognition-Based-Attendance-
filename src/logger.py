"""
Structured logging system with coloured console output and rotating file handlers.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import platform
import sys
from pathlib import Path
from typing import Dict

from config import cfg

# Enable ANSI escape sequences on Windows
if platform.system() == "Windows":
    os.system("")


class ColoredFormatter(logging.Formatter):
    """Console formatter with ANSI colour codes per level."""

    COLORS = {
        "DEBUG": "\033[94m",      # Blue
        "INFO": "\033[92m",       # Green
        "WARNING": "\033[93m",    # Yellow
        "ERROR": "\033[91m",      # Red
        "CRITICAL": "\033[1;91m", # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


_loggers: Dict[str, logging.Logger] = {}


def get_logger(name: str, category: str = "system") -> logging.Logger:
    """
    Create or return a logger with coloured console + rotating file output.

    Args:
        name: Logger name (usually ``__name__``).
        category: Log category — determines the log filename.
    """
    logger_key = f"{name}_{category}"
    if logger_key in _loggers:
        return _loggers[logger_key]

    logger = logging.getLogger(logger_key)

    # Resolve level from enum
    level_str = cfg.logging.level.value if hasattr(cfg.logging.level, "value") else str(cfg.logging.level)
    logger.setLevel(getattr(logging, level_str, logging.INFO))

    if not logger.handlers:
        logger.propagate = False
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # Console handler with colour
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(ColoredFormatter(fmt))
        logger.addHandler(ch)

        # File handler
        try:
            log_dir = Path(cfg.paths.logs_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            max_bytes = int(cfg.logging.max_file_size_mb * 1024 * 1024)
            fh = logging.handlers.RotatingFileHandler(
                log_dir / f"{category}.log",
                maxBytes=max_bytes,
                backupCount=cfg.logging.backup_count,
                encoding="utf-8",
            )
            fh.setFormatter(logging.Formatter(fmt))
            logger.addHandler(fh)
        except Exception:
            pass  # gracefully skip file logging if directory creation fails

    _loggers[logger_key] = logger
    return logger


def log_performance(
    logger: logging.Logger, operation: str, duration: float, **kwargs
) -> None:
    """Helper for consistent performance log messages."""
    extra = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    msg = f"PERF: {operation} took {duration:.4f}s"
    if extra:
        msg += f" ({extra})"
    logger.info(msg)


def setup_logging() -> None:
    """Ensure log directories exist."""
    Path(cfg.paths.logs_dir).mkdir(parents=True, exist_ok=True)


setup_logging()
