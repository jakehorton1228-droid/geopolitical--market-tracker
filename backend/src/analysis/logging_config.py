"""
Logging Configuration for Analysis Modules.

This module provides consistent logging setup for all analysis components.
It configures both file and console logging with appropriate formatting.

WHY PROPER LOGGING?
-------------------
1. Debugging: See what's happening during long analysis runs
2. Auditing: Track what analyses were run and when
3. Performance: Identify slow operations
4. Errors: Capture and diagnose failures

LOG LEVELS:
-----------
- DEBUG: Detailed information for diagnosing problems
- INFO: General operational messages (default)
- WARNING: Something unexpected but not an error
- ERROR: A failure that prevented an operation
- CRITICAL: A serious failure

USAGE:
------
    # At the start of your script or notebook:
    from src.analysis.logging_config import setup_logging

    # Basic setup (INFO level to console)
    setup_logging()

    # Verbose mode (DEBUG level)
    setup_logging(level="DEBUG")

    # Log to file as well
    setup_logging(log_file="analysis.log")

    # Then in your code, logging just works:
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Starting analysis...")
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# Default format: timestamp - module - level - message
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DETAILED_FORMAT = "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"
SIMPLE_FORMAT = "%(levelname)s: %(message)s"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_style: str = "default",
    modules: list[str] = None,
) -> None:
    """
    Configure logging for analysis modules.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
        format_style: "default", "detailed", or "simple"
        modules: List of module names to configure. Default is analysis modules.

    Example:
        # Basic usage
        setup_logging()

        # Debug mode with file logging
        setup_logging(level="DEBUG", log_file="analysis.log")

        # Simple output for notebooks
        setup_logging(format_style="simple")
    """
    # Select format
    if format_style == "detailed":
        log_format = DETAILED_FORMAT
    elif format_style == "simple":
        log_format = SIMPLE_FORMAT
    else:
        log_format = DEFAULT_FORMAT

    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Default modules to configure
    if modules is None:
        modules = [
            "src.analysis",
            "src.analysis.correlation",
            "src.analysis.feature_engineering",
            "src.analysis.historical_patterns",
            "src.analysis.production_regression",
            "src.analysis.production_event_study",
            "src.analysis.production_anomaly",
            "src.agent",
            "src.ingestion",
            "src.ingestion.gdelt",
            "src.ingestion.market_data",
            "src.db",
        ]

    # Create formatter
    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)

    # File handler (optional)
    file_handler = None
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)

    # Configure each module's logger
    for module_name in modules:
        logger = logging.getLogger(module_name)
        logger.setLevel(numeric_level)

        # Remove existing handlers to avoid duplicates
        logger.handlers = []

        # Add handlers
        logger.addHandler(console_handler)
        if file_handler:
            logger.addHandler(file_handler)

        # Don't propagate to root logger
        logger.propagate = False


def get_analysis_logger(name: str) -> logging.Logger:
    """
    Get a logger for an analysis module.

    This is a convenience function that ensures the logger is properly named.

    Args:
        name: Module name (usually __name__)

    Returns:
        Configured logger

    Example:
        from src.analysis.logging_config import get_analysis_logger
        logger = get_analysis_logger(__name__)
        logger.info("Starting analysis...")
    """
    return logging.getLogger(name)


def log_analysis_start(
    logger: logging.Logger,
    analysis_type: str,
    symbol: str = None,
    start_date=None,
    end_date=None,
    **kwargs,
) -> None:
    """
    Log the start of an analysis with consistent formatting.

    Args:
        logger: The logger to use
        analysis_type: Type of analysis (e.g., "event_study", "regression")
        symbol: Optional symbol being analyzed
        start_date: Optional start date
        end_date: Optional end date
        **kwargs: Additional parameters to log
    """
    parts = [f"Starting {analysis_type}"]

    if symbol:
        parts.append(f"symbol={symbol}")
    if start_date:
        parts.append(f"start={start_date}")
    if end_date:
        parts.append(f"end={end_date}")

    for key, value in kwargs.items():
        parts.append(f"{key}={value}")

    logger.info(" | ".join(parts))


def log_analysis_complete(
    logger: logging.Logger,
    analysis_type: str,
    duration_seconds: float = None,
    result_count: int = None,
    **kwargs,
) -> None:
    """
    Log the completion of an analysis with consistent formatting.

    Args:
        logger: The logger to use
        analysis_type: Type of analysis
        duration_seconds: How long the analysis took
        result_count: Number of results produced
        **kwargs: Additional metrics to log
    """
    parts = [f"Completed {analysis_type}"]

    if duration_seconds is not None:
        parts.append(f"duration={duration_seconds:.2f}s")
    if result_count is not None:
        parts.append(f"results={result_count}")

    for key, value in kwargs.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")

    logger.info(" | ".join(parts))


class AnalysisTimer:
    """
    Context manager for timing analysis operations.

    Usage:
        from src.analysis.logging_config import AnalysisTimer, get_analysis_logger

        logger = get_analysis_logger(__name__)

        with AnalysisTimer(logger, "regression analysis"):
            # ... do analysis ...
            pass

        # Logs: "regression analysis completed in 2.34 seconds"
    """

    def __init__(self, logger: logging.Logger, operation_name: str):
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()

        if exc_type is not None:
            self.logger.error(
                f"{self.operation_name} failed after {duration:.2f}s: {exc_val}"
            )
        else:
            self.logger.info(
                f"{self.operation_name} completed in {duration:.2f}s"
            )

        return False  # Don't suppress exceptions


# Quick setup for common use cases
def setup_debug_logging():
    """Quick setup for debug-level logging."""
    setup_logging(level="DEBUG", format_style="detailed")


def setup_quiet_logging():
    """Quick setup for minimal logging (warnings and errors only)."""
    setup_logging(level="WARNING", format_style="simple")


def setup_file_logging(filename: str = "analysis.log"):
    """Quick setup for file logging."""
    setup_logging(level="INFO", log_file=filename)
