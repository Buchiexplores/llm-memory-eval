"""Descriptive statistics for the paired outcome variables."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def descriptive_summary(
    df: pd.DataFrame,
    variables: Sequence[tuple[str, str, str]],
) -> list[dict[str, float | str]]:
    """Return descriptive statistics for paired Summarization vs RAG variables.

    Parameters
    ----------
    df
        Experiment results.
    variables
        Iterable of ``(summ_column, rag_column, display_name)`` tuples.
    """
    rows: list[dict[str, float | str]] = []
    for sv, rv, name in variables:
        s = df[sv].astype(float)
        r = df[rv].astype(float)
        rows.append(
            {
                "Variable": name,
                "Summ_M": _round(np.mean(s)),
                "Summ_SD": _round(np.std(s, ddof=1)) if len(s) > 1 else 0.0,
                "Summ_Mdn": _round(np.median(s)),
                "Summ_Min": _round(np.min(s)),
                "Summ_Max": _round(np.max(s)),
                "RAG_M": _round(np.mean(r)),
                "RAG_SD": _round(np.std(r, ddof=1)) if len(r) > 1 else 0.0,
                "RAG_Mdn": _round(np.median(r)),
                "RAG_Min": _round(np.min(r)),
                "RAG_Max": _round(np.max(r)),
            }
        )
    return rows


def _round(value: float) -> float:
    return float(round(float(value), 4))
