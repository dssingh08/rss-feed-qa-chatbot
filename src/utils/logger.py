""" Logging Configuration """

import logging
import sys
from pathlib import Path
from src.config import settings


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[94m',      # Blue
        'INFO': '\033[92m',       # Green
        'WARNING': '\033[93m',    # Yellow
        'ERROR': '\033[91m',      # Red
        'CRITICAL': '\033[1;91m', # Bold Red
    }
    RESET = '\033[0m'
    TIME_COLOR = '\033[96m'       # Cyan
    NAME_COLOR = '\033[95m'       # Magenta

    def format(self, record):
        level_color = self.COLORS.get(record.levelname, self.RESET)
        time_str = self.formatTime(record, self.datefmt)
        msg = record.getMessage()
        
        formatted_message = (
            f"{self.TIME_COLOR}{time_str}{self.RESET} - "
            f"{self.NAME_COLOR}{record.name}{self.RESET} - "
            f"{level_color}{record.levelname}{self.RESET} - "
            f"{level_color}{msg}{self.RESET}"
        )
        
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            formatted_message += f"\n{level_color}{record.exc_text}{self.RESET}"
            
        return formatted_message


def setup_logger(name: str) -> logging.Logger:
    """ Setup logger with file and console handlers """

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level.upper()))

    # Ensure the root logger also captures debug messages for full tracebacks
    logging.getLogger().setLevel(logging.DEBUG)

    if logger.handlers:
        return logger
    
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(ColoredFormatter())

    file_handler = logging.FileHandler(log_dir / "app.log")
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
