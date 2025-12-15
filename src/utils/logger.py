"""
Centralized logging configuration.
Memory efficient with rotation and compression.
"""
import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_dir: str = "./logs",
    log_level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
    compression: str = "gz"
) -> None:
    """
    Configure application logger with rotation and compression.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        rotation: When to rotate log files
        retention: How long to keep old logs
        compression: Compression format for rotated logs
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Add console handler (minimal format for performance)
    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        colorize=True,
        backtrace=False,
        diagnose=False
    )
    
    # Add file handler with rotation
    logger.add(
        f"{log_dir}/app_{{time:YYYY-MM-DD}}.log",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=rotation,
        retention=retention,
        compression=compression,
        backtrace=True,
        diagnose=True,
        enqueue=True  # Thread-safe async logging
    )
    
    # Add error-only file handler
    logger.add(
        f"{log_dir}/errors_{{time:YYYY-MM-DD}}.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=rotation,
        retention=retention,
        compression=compression,
        backtrace=True,
        diagnose=True,
        enqueue=True
    )


def get_logger(name: str):
    """Get a logger instance with the given name."""
    return logger.bind(name=name)

