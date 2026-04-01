"""
Centralized logging setup.
Call setup_logging() once at the entry point of every script.
"""

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
) -> None:
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=handlers,
        force=True,   # override any existing config
    )
    # Quiet noisy third-party loggers
    for noisy in ("urllib3", "filelock", "fsspec", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
