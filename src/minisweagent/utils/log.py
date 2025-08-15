import logging
from pathlib import Path

MINI_LOGGERS = {}
_EXTRA_HANDLERS = []


def get_logger(name: str) -> logging.Logger:
    if name in MINI_LOGGERS:
        return MINI_LOGGERS[name]
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(name)s: %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    for handler in _EXTRA_HANDLERS:
        logger.addHandler(handler)
    MINI_LOGGERS[name] = logger
    return logger


def add_file_handlers(path: Path):
    handler = logging.FileHandler(path)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    _EXTRA_HANDLERS.append(handler)
    for logger in MINI_LOGGERS.values():
        logger.addHandler(handler)


logger = get_logger("minisweagent")
