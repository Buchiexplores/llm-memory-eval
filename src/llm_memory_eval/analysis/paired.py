"""Paired-samples t-tests, Wilcoxon confirmation, and Holm-Bonferroni."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon


def _fmt_p(p: float) -> str:
    if p < 0.001:
        return "< .001"
    return f"= {p:.3f}"


def paired_test(
    summ_values: np.ndarray,
    rag_values: np.ndarray,
    name: str,
) -> Dict[str, object]:
    """Run the paired t-test and Wilcoxon signed-rank confirmatory test.

    Returns a result dict with means, SDs, t-statistic, exact p-values,
    Cohen's d for paired designs, the 95 percent CI of the mean
    difference, and the Wilcoxon confirmatory result.
    """
    summ = np.asarray(summ_values, dtype=float)
    rag = np.asarray(rag_values, dtype=float)
    if summ.shape != rag.shape:
        raise ValueError("summ and rag arrays must have equal length")

    diff = summ - rag
    n = int(len(diff))

    mean_s = float(np.mean(summ))
    sd_s = float(np.std(summ, ddof=1)) if n > 1 else 0.0
    mean_r = float(np.mean(rag))
    sd_r = float(np.std(rag, ddof=1)) if n > 1 else 0.0
    mean_d = float(np.mean(diff))
    sd_d = float(np.std(diff, ddof=1)) if n > 1 else 0.0

    p_norm = 1.0
    if n >= 8:
        _, p_norm = shapiro(diff[: min(n, 5000)])

    t_stat, p_val = ttest_rel(summ, rag)
    cohen_d = mean_d / sd_d if sd_d > 0 else 0.0
    se = sd_d / np.sqrt(n) if n > 0 else 0.0
    ci_lo = mean_d - 1.96 * se
    ci_hi = mean_d + 1.96 * se

    diff_nonzero = diff[diff != 0]
    if len(diff_nonzero) > 0:
        w_stat, p_w = wilcoxon(diff_nonzero)
    else:
        w_stat, p_w = 0.0, 1.0

    return {
        "Variable": name,
        "Test": "Paired t-test",
        "N": n,
        "M_Summ": round(mean_s, 4),
        "SD_Summ": round(sd_s, 4),
        "M_RAG": round(mean_r, 4),
        "SD_RAG": round(sd_r, 4),
        "Mean_Diff": round(mean_d, 4),
        "Statistic": round(float(t_stat), 3),
        "p": float(p_val),
        "p_fmt": _fmt_p(float(p_val)),
        "Cohen_d": round(cohen_d, 3),
        "CI_95": f"[{ci_lo:.4f}, {ci_hi:.4f}]",
        "Sig": "Yes" if p_val < 0.05 else "No",
        "Normality_p": round(float(p_norm), 4),
        "Null_Decision": "Rejected" if p_val < 0.05 else "Failed to Reject",
        "Wilcoxon_W": round(float(w_stat), 1),
        "Wilcoxon_p": float(p_w),
        "Wilcoxon_p_fmt": _fmt_p(float(p_w)),
        "Wilcoxon_Decision": "Rejected" if p_w < 0.05 else "Failed to Reject",
    }


def holm_correction(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Apply Holm-Bonferroni correction to a family of paired tests.

    Adds ``p_adj``, ``p_adj_fmt``, ``Sig_adj``, and ``Null_Decision_adj``
    keys to each result dict in place and also returns the list.
    """
    n = len(results)
    if n == 0:
        return results

    indexed = sorted(enumerate(results), key=lambda iv: float(iv[1]["p"]))
    last_adj = 0.0
    for rank, (orig_idx, item) in enumerate(indexed):
        raw_p = float(item["p"])
        adj = min(raw_p * (n - rank), 1.0)
        adj = max(adj, last_adj)  # monotone
        results[orig_idx]["p_adj"] = round(adj, 6)
        results[orig_idx]["p_adj_fmt"] = _fmt_p(adj)
        results[orig_idx]["Sig_adj"] = "Yes" if adj < 0.05 else "No"
        results[orig_idx]["Null_Decision_adj"] = (
            "Rejected" if adj < 0.05 else "Failed to Reject"
        )
        last_adj = adj
    return results
