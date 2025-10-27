""" Logging Configuration """

import logging
import sys
from pathlib import Path
from src.config import settings


def setup_logger(name: str) -> logging.Logger:
    """ Setup logger with file and console handlers """

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level.upper()))

    # Ensure the root logger also captures debug messages for full tracebacks
    # This might be necessary if the error is not caught by a specific logger
    logging.getLogger().setLevel(logging.DEBUG)

    if logger.handlers:
        return logger
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        # Add exc_info to the format to ensure tracebacks are printed to console
        # This will be handled by the logger.error(exc_info=True) call
    )
    console_handler.setFormatter(console_format)

    file_handler = logging.FileHandler(log_dir / "app.log")
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
