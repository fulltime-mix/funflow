"""Logging utilities for FunFlow framework."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union

CONSOLE = 25
logging.addLevelName(CONSOLE, "CONSOLE")

_loggers = {}


class FunFlowLogger:
    """FunFlow logger that writes all logs to file and important logs to console.

    Log levels: DEBUG/INFO (file only), CONSOLE/WARNING/ERROR/CRITICAL (file + console)
    """

    def __init__(
        self,
        work_dir: Union[str, Path],
        name: str = "FunFlow",
        log_file: str = "train.log",
        file_mode: str = "a",
        console_level: int = CONSOLE,
    ):
        """
        Args:
            work_dir: Work directory for log files
            name: Logger name
            log_file: Log file name
            file_mode: File mode ('w' overwrite, 'a' append)
            console_level: Minimum level for console output
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(logging.DEBUG)

        self._logger.handlers.clear()

        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_formatter = logging.Formatter(
            "%(asctime)s - %(message)s", datefmt="%H:%M:%S"
        )

        log_path = self.work_dir / log_file
        fh = logging.FileHandler(str(log_path), mode=file_mode, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(file_formatter)
        self._logger.addHandler(fh)

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(console_level)
        ch.setFormatter(console_formatter)
        self._logger.addHandler(ch)

        self._logger.propagate = False

        _loggers[name] = self

    @property
    def logger(self) -> logging.Logger:
        """Get underlying logging.Logger instance."""
        return self._logger

    def debug(self, msg: str, *args, **kwargs):
        """Log DEBUG level (file only)."""
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        """Log INFO level (file only)."""
        self._logger.info(msg, *args, **kwargs)

    def console(self, msg: str, *args, **kwargs):
        """Log CONSOLE level (file + console)."""
        self._logger.log(CONSOLE, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        """Log WARNING level (file + console)."""
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        """Log ERROR level (file + console)."""
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        """Log CRITICAL level (file + console)."""
        self._logger.critical(msg, *args, **kwargs)

    def log(self, level: int, msg: str, *args, **kwargs):
        """Log at specified level."""
        self._logger.log(level, msg, *args, **kwargs)

    def separator(self, char: str = "=", length: int = 60):
        """Print separator line to console."""
        self.console(char * length)

    def section(self, title: str, char: str = "=", length: int = 60):
        """Print section with title to console."""
        self.console(char * length)
        self.console(title)
        self.console(char * length)


def setup_logger(
    name: str = "FunFlow",
    log_file: Optional[str] = None,
    work_dir: Optional[Union[str, Path]] = None,
    file_mode: str = "a",
    console_level: int = CONSOLE,
) -> FunFlowLogger:
    """Setup logger (first-time initialization).

    Args:
        name: Logger name
        log_file: Full path to log file
        work_dir: Work directory
        file_mode: File mode ('w' or 'a')
        console_level: Minimum level for console output

    Returns:
        Configured logger
    """
    if name in _loggers:
        return _loggers[name]

    if work_dir is not None:
        work_dir = Path(work_dir)
    elif log_file is not None:
        work_dir = Path(log_file).parent
    else:
        work_dir = Path(".")

    if log_file is not None:
        log_file_name = Path(log_file).name
    else:
        log_file_name = "train.log"

    return FunFlowLogger(
        work_dir=work_dir,
        name=name,
        log_file=log_file_name,
        file_mode=file_mode,
        console_level=console_level,
    )


def get_logger(name: str = "FunFlow") -> Union[FunFlowLogger, logging.Logger]:
    """Get logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    return logging.getLogger(name)


def reset_logger(name: str = "FunFlow"):
    """Reset logger (clear cache).

    Args:
        name: Logger name
    """
    if name in _loggers:
        logger = _loggers[name]
        if isinstance(logger, FunFlowLogger):
            logger._logger.handlers.clear()
        del _loggers[name]
