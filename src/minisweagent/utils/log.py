import logging
from pathlib import Path

from rich.logging import RichHandler

MINI_LOGGERS = {}
_MINI_HANDLERS = []


def get_logger(name: str) -> logging.Logger:
    if name in MINI_LOGGERS:
        return MINI_LOGGERS[name]
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = RichHandler(
        show_path=False,
        show_time=False,
        show_level=False,
        markup=False,
    )
    formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    for handler in _MINI_HANDLERS:
        logger.addHandler(handler)
    MINI_LOGGERS[name] = logger
    return logger


def add_handler(handler: logging.Handler) -> None:
    """Add a handler to all existing and future mini loggers."""
    _MINI_HANDLERS.append(handler)
    for logger in MINI_LOGGERS.values():
        logger.addHandler(handler)


def add_file_handler(path: Path | str, level: int = logging.DEBUG) -> None:
    """Add a file handler to all existing and future mini loggers."""
    handler = logging.FileHandler(path)
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    add_handler(handler)


logger = get_logger("minisweagent")
