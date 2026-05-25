"""Structured stdlib logging configuration."""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional


_CONFIGURED = False


def configure_logging(level: Optional[str] = None) -> None:
    """Initialize root logging once.

    Parameters
    ----------
    level
        Optional level name (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``).
        Falls back to the ``LLM_MEMORY_EVAL_LOG_LEVEL`` environment variable
        and then to ``"INFO"``.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    chosen = (level or os.environ.get("LLM_MEMORY_EVAL_LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, chosen, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stderr,
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for *name*."""
    configure_logging()
    return logging.getLogger(name)
