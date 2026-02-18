import logging
import sys
import os
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if not logger.handlers:
        # Use plain StreamHandler on Windows to avoid Rich unicode crashes
        # Rich uses box-drawing characters internally that cp1252 can't encode
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.INFO)
        stream_fmt = logging.Formatter(
            "%(asctime)s | %(name)-14s | %(levelname)-7s | %(message)s",
            datefmt="%H:%M:%S",
        )
        stream_handler.setFormatter(stream_fmt)
        logger.addHandler(stream_handler)

        # File handler (always works, UTF-8)
        log_file = log_dir / f"bot_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    return logger
