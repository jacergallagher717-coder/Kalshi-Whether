"""
Logging configuration for the Weather Kalshi Paper Trading System.

Log format:
2025-01-15 14:30:00 | INFO | [module_name] | Message here

Log files:
- logs/system.log - All logs
- logs/trades.log - Trade-specific logs only
- logs/errors.log - Warnings and errors only
"""

import logging
import os
from datetime import datetime
from config.settings import LOG_LEVEL, LOG_DIR


class CustomFormatter(logging.Formatter):
    """Custom formatter with module name in brackets."""

    def format(self, record):
        # Format: timestamp | level | [module] | message
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        return f"{timestamp} | {record.levelname:8} | [{record.name}] | {record.getMessage()}"


def setup_logger(name: str = "weather_trader") -> logging.Logger:
    """
    Set up and configure the main logger.

    Args:
        name: Logger name

    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    os.makedirs(LOG_DIR, exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = CustomFormatter()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # System log handler (all logs)
    system_handler = logging.FileHandler(os.path.join(LOG_DIR, "system.log"))
    system_handler.setLevel(logging.DEBUG)
    system_handler.setFormatter(formatter)
    logger.addHandler(system_handler)

    # Error log handler (warnings and errors only)
    error_handler = logging.FileHandler(os.path.join(LOG_DIR, "errors.log"))
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)

    return logger


def get_logger(module_name: str) -> logging.Logger:
    """
    Get a child logger for a specific module.

    Args:
        module_name: Name of the module requesting the logger

    Returns:
        Logger instance for the module
    """
    return logging.getLogger(f"weather_trader.{module_name}")


def setup_trade_logger() -> logging.Logger:
    """
    Set up dedicated trade logger.

    Returns:
        Logger instance for trade-specific logs
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger("weather_trader.trades")

    if not any(isinstance(h, logging.FileHandler) and "trades.log" in h.baseFilename
               for h in logger.handlers):
        formatter = CustomFormatter()
        trade_handler = logging.FileHandler(os.path.join(LOG_DIR, "trades.log"))
        trade_handler.setLevel(logging.INFO)
        trade_handler.setFormatter(formatter)
        logger.addHandler(trade_handler)

    return logger
