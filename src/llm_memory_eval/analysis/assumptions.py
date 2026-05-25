"""Statistical-assumption diagnostics."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import levene, shapiro, ttest_rel, wilcoxon


def _fmt_p(p: float) -> str:
    return "< .001" if p < 0.001 else f"= {p:.3f}"


def shapiro_normality(
    df: pd.DataFrame,
    variables: Sequence[Tuple[str, str, str]],
) -> List[Dict[str, object]]:
    """Shapiro-Wilk test on within-instance difference scores."""
    rows: List[Dict[str, object]] = []
    for sv, rv, name in variables:
        diff = df[rv].astype(float).to_numpy() - df[sv].astype(float).to_numpy()
        try:
            w_stat, p_val = shapiro(diff[: min(len(diff), 5000)])
        except Exception:  # noqa: BLE001
            w_stat, p_val = 0.0, 0.0
        skew = float(pd.Series(diff).skew())
        kurt = float(pd.Series(diff).kurtosis())
        rows.append(
            {
                "Variable": name,
                "W_stat": round(float(w_stat), 4),
                "p": float(p_val),
                "p_fmt": _fmt_p(float(p_val)),
                "Skewness": round(skew, 3),
                "Kurtosis": round(kurt, 3),
                "Normal": "Yes" if p_val >= 0.05 else "No",
            }
        )
    return rows


def levene_test(
    df: pd.DataFrame,
    variables: Iterable[Tuple[str, str, str]],
) -> List[Dict[str, object]]:
    """Levene's test of homogeneity of variance across strategy conditions."""
    rows: List[Dict[str, object]] = []
    for sv, rv, name in variables:
        lev_stat, lev_p = levene(df[sv].astype(float), df[rv].astype(float))
        rows.append(
            {
                "Variable": name,
                "F": round(float(lev_stat), 3),
                "p": float(lev_p),
                "p_fmt": _fmt_p(float(lev_p)),
                "Equal_Variance": "Yes" if lev_p >= 0.05 else "No",
            }
        )
    return rows


def log_transform_check(
    summ_latency: np.ndarray,
    rag_latency: np.ndarray,
) -> Dict[str, object]:
    """Examine whether a natural-log transformation normalises latency differences."""
    summ = np.asarray(summ_latency, dtype=float)
    rag = np.asarray(rag_latency, dtype=float)
    offset = 1e-6
    log_summ = np.log(summ + offset)
    log_rag = np.log(rag + offset)
    log_diff = log_rag - log_summ

    try:
        sw_stat, sw_p = shapiro(log_diff[: min(len(log_diff), 5000)])
    except Exception:  # noqa: BLE001
        sw_stat, sw_p = 0.0, 0.0

    still_normal = bool(sw_p >= 0.05)
    if still_normal:
        t_stat, p_val = ttest_rel(log_summ, log_rag)
        test_used = "Paired t-test (log-transformed)"
    else:
        nz = log_diff[log_diff != 0]
        if len(nz) > 0:
            t_stat, p_val = wilcoxon(nz)
        else:
            t_stat, p_val = 0.0, 1.0
        test_used = "Wilcoxon (log transform did not normalise)"

    return {
        "log_shapiro_W": round(float(sw_stat), 4),
        "log_shapiro_p": float(sw_p),
        "log_shapiro_p_fmt": _fmt_p(float(sw_p)),
        "log_normalized": still_normal,
        "log_test_used": test_used,
        "log_statistic": round(float(t_stat), 3),
        "log_p": float(p_val),
        "log_p_fmt": _fmt_p(float(p_val)),
    }
