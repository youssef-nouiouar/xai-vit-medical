"""Logging utilities — wraps loguru with sensible defaults.

Use this throughout the project — never use plain ``print``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger


DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def setup_logger(
    log_file: Path | str | None = None,
    level: str = "INFO",
    format_str: str | None = None,
) -> None:
    """Configure global ``loguru`` logger.

    Args:
        log_file: Path to log file. If None, logs to stderr only.
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR).
        format_str: Custom format string. If None, uses Hydra-style
            colored format.
    """
    logger.remove()  # remove default handler

    fmt = format_str or DEFAULT_FORMAT

    # stderr handler (colored)
    logger.add(
        sys.stderr,
        format=fmt,
        level=level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File handler (no colors, with rotation)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            format=fmt,
            level=level,
            colorize=False,
            rotation="100 MB",
            retention="30 days",
            backtrace=True,
            diagnose=True,
        )

    logger.info(f"Logger initialized — level={level}, file={log_file}")


def log_config(cfg: Any) -> None:
    """Pretty-print the config at experiment start.

    Args:
        cfg: Hydra ``DictConfig`` object.
    """
    from omegaconf import OmegaConf

    cfg_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    logger.info("Experiment configuration:\n" + cfg_yaml)


# Re-export logger for convenience
__all__ = ["logger", "setup_logger", "log_config"]
