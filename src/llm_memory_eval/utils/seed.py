"""Deterministic seeding across NumPy, the Python ``random`` module, and any
optional torch backend that is installed."""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed every randomness source the package may touch.

    Parameters
    ----------
    seed
        Non-negative integer seed. Persisted to ``PYTHONHASHSEED`` so that
        any subprocess inherits the same hash-randomisation state.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    _seed_torch_if_available(seed)


def _seed_torch_if_available(seed: int) -> None:
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)  # type: ignore[attr-defined]
    except Exception:
        # torch is an optional dependency; absence is not an error
        return
