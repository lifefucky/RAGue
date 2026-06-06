"""Logging setup for ingestion CLI runs."""

from __future__ import annotations

import logging


INGESTION_LOGGER_NAME = "rague.ingestion"


def configure_ingestion_logging(level: str = "INFO") -> logging.Logger:
    """Configure the ingestion logger for terminal progress output."""
    logger = logging.getLogger(INGESTION_LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s  %(message)s"))
    handler.setLevel(logger.level)
    logger.addHandler(handler)
    return logger


def get_ingestion_logger() -> logging.Logger:
    """Return the ingestion logger, configuring INFO if not yet set up."""
    logger = logging.getLogger(INGESTION_LOGGER_NAME)
    if not logger.handlers:
        return configure_ingestion_logging("INFO")
    return logger
