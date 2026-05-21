"""Logging configuration for the ConvViTs project.

Provides a single `setup_logger` entry-point that configures the root
logger (or any named logger) with consistent formatting, optional file
output, and rank-awareness for future distributed training support.

Typical usage (called once in `main.py` before anything else):

    from utils.logger import setup_logger
    logger = setup_logger(name="convvits", log_file="logs/train.log", level=logging.INFO)

All other modules then simply do:

    import logging
    logger = logging.getLogger(__name__)
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------
# Constants
# ---------------------------------------------

# _DEFAULT_FMT = "[%(asctime)s] [%(levelname)-8s] [%(name)s:%(lineno)-4d] %(message)s"
_DEFAULT_FMT = "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# Maps string names (from argparse) -> logging levels.
LEVEL_MAP: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


# ---------------------------------------------
# Internal helpers
# ---------------------------------------------

def _make_formatter(fmt: str = _DEFAULT_FMT, date_fmt: str = _DATE_FMT) -> logging.Formatter:
    """
    Build a `logging.Formatter` with the project-standard layout.

    Args:
        fmt: Log record format string.
        date_fmt: Date/time format string.

    Returns:
        Configured `Formatter` instance.
    """
    return logging.Formatter(fmt=fmt, datefmt=date_fmt)


def _make_stream_handler(stream=sys.stdout, level: int = logging.DEBUG,
                         fmt: str = _DEFAULT_FMT) -> logging.StreamHandler:
    """
    Create a stream handler pointed at *stream*.

    Args:
        stream: Output stream (default: `sys.stdout`).
        level: Minimum level this handler will emit.
        fmt: Log record format string.

    Returns:
        Configured `StreamHandler`.
    """
    handler = logging.StreamHandler(stream)
    handler.setLevel(level)
    handler.setFormatter(_make_formatter(fmt))
    return handler


def _make_file_handler(log_file: Path, level: int = logging.DEBUG, fmt: str = _DEFAULT_FMT) -> logging.FileHandler:
    """
    Create a file handler that appends to *log_file*.

    The parent directory is created automatically if it does not exist.

    Args:
        log_file: Destination path for the log file.
        level: Minimum level this handler will emit.
        fmt: Log record format string.

    Returns:
        Configured `FileHandler`.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(_make_formatter(fmt))
    return handler


def _level_from_str(level_str: str) -> int:
    """
    Convert a case-insensitive level string to a `logging` int constant.

    Args:
        level_str: One of "debug", "info", "warning", "error", "critical" (case-insensitive).

    Returns:
        Corresponding `logging` level integer.

    Raises:
        ValueError: If *level_str* is not a recognized level name.
    """
    key = level_str.strip().lower()
    if key not in LEVEL_MAP:
        raise ValueError(
            f"Unknown log level '{level_str}'. "
            f"Choose from: {list(LEVEL_MAP.keys())}"
        )
    return LEVEL_MAP[key]


# ---------------------------------------------
# Public Methods
# ---------------------------------------------

def setup_logger(name: str = "convvits", level: int = logging.INFO, log_file: Optional[str | Path] = None,
                 fmt: str = _DEFAULT_FMT, propagate: bool = False, rank: int = 0) -> logging.Logger:
    """
    Configure and return a named logger for the project.

    Attaches a `StreamHandler` (stdout) and, optionally, a `FileHandler`. Safe to call multiple times - existing
    handlers are cleared before new ones are attached so the logger is idempotent.

    In distributed settings, pass `rank` so that only the main process (rank 0) emits logs; worker processes will
    have their level set to `WARNING` to suppress noise.

    Args:
        name: Logger name, typically "cvt" or `__name__`.
        level: Root logging level. Use `LEVEL_MAP` to resolve from a string (e.g. from argparse).
        log_file: Optional path to a `.log` file. The directory is created if missing.
        fmt: Log record format string.
        propagate: Whether to propagate messages to the root logger. Usually `False` to avoid duplicate output.
        rank: Process rank in distributed training. Non-zero ranks have their effective level raised to `WARNING`.

    Returns:
        Configured `logging.Logger` instance.

    Example:
        >>> logger = setup_logger(
        ...     name="convvits",
        ...     level=LEVEL_MAP["debug"],
        ...     log_file="logs/exp_01/train.log",
        ... )
        >>> logger.info("Logger ready.")
    """
    log = logging.getLogger(name)

    # Clear any handlers attached by a previous call (idempotency).
    log.handlers.clear()

    # Non-primary ranks: suppress to WARNING so logs aren't duplicated.
    effective_level = level if rank == 0 else logging.WARNING
    log.setLevel(effective_level)
    log.propagate = propagate

    # Always attach a stream handler.
    log.addHandler(_make_stream_handler(level=effective_level, fmt=fmt))

    # Optionally attach a file handler (rank-0 only to avoid file races).
    if log_file is not None and rank == 0:
        log.addHandler(_make_file_handler(Path(log_file), level=logging.DEBUG, fmt=fmt))
        log.debug("File logging enabled -> %s", Path(log_file).resolve())

    log.info(
        "Logger '%s' initialised | level=%s | file=%s",
        name,
        logging.getLevelName(effective_level),
        log_file or "none",
    )
    return log


def get_logger(name: str = "convvits") -> logging.Logger:
    """
    Return an already-configured logger by name.

    Convenience wrapper around `logging.getLogger` - use this in submodules that should not reconfigure the logger,
    only retrieve it.

    Args:
        name: Logger name passed to `logging.getLogger`.

    Returns:
        The named `logging.Logger` instance.

    Example:
        >>> logger = get_logger("convvits")
        >>> logger.info("Using existing logger.")
    """
    return logging.getLogger(name)
