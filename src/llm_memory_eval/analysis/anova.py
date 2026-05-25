"""2 x 3 ANOVA on Memory Strategy x Conversation Length."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from scipy.stats import ttest_rel


_LEVELS = ["short", "medium", "long"]
_STRATEGIES = ["Summarization", "RAG"]


def _fmt_p(p: float) -> str:
    return "< .001" if p < 0.001 else f"= {p:.3f}"


def two_way_anova(df: pd.DataFrame, sv: str, rv: str, name: str) -> Dict[str, object]:
    """Run a 2 x 3 ANOVA (Strategy x Length) on a single outcome variable.

    The data are reshaped into a long-format dataframe so the SS partition
    handles unbalanced cells correctly.
    """
    rows = []
    for _, r in df.iterrows():
        rows.append({"strategy": "Summarization", "length": r["length_category"], "value": float(r[sv])})
        rows.append({"strategy": "RAG", "length": r["length_category"], "value": float(r[rv])})
    long = pd.DataFrame(rows)

    grand = float(long["value"].mean())
    n_total = len(long)

    def ss_factor(col: str, levels) -> float:
        total = 0.0
        for lev in levels:
            grp = long[long[col] == lev]["value"]
            if len(grp) == 0:
                continue
            total += len(grp) * (float(grp.mean()) - grand) ** 2
        return total

    ss_s = ss_factor("strategy", _STRATEGIES)
    ss_l = ss_factor("length", _LEVELS)

    ss_int = 0.0
    for s in _STRATEGIES:
        s_mean = float(long[long["strategy"] == s]["value"].mean())
        for c in _LEVELS:
            cell = long[(long["strategy"] == s) & (long["length"] == c)]["value"]
            if len(cell) == 0:
                continue
            c_mean = float(long[long["length"] == c]["value"].mean())
            ss_int += len(cell) * (float(cell.mean()) - s_mean - c_mean + grand) ** 2

    ss_tot = float(((long["value"] - grand) ** 2).sum())
    ss_w = ss_tot - ss_s - ss_l - ss_int

    df_s, df_l, df_i = 1, 2, 2
    df_w = max(1, n_total - 6)

    ms_s = ss_s / df_s
    ms_l = ss_l / df_l
    ms_i = ss_int / df_i
    ms_w = ss_w / df_w if ss_w > 0 else 1e-12

    f_s = ms_s / ms_w
    f_l = ms_l / ms_w
    f_i = ms_i / ms_w

    p_s = 1 - f_dist.cdf(f_s, df_s, df_w)
    p_l = 1 - f_dist.cdf(f_l, df_l, df_w)
    p_i = 1 - f_dist.cdf(f_i, df_i, df_w)

    eta_s = ss_s / (ss_s + ss_w) if (ss_s + ss_w) > 0 else 0.0
    eta_l = ss_l / (ss_l + ss_w) if (ss_l + ss_w) > 0 else 0.0
    eta_i = ss_int / (ss_int + ss_w) if (ss_int + ss_w) > 0 else 0.0

    cell_stats: Dict[str, object] = {}
    for s in _STRATEGIES:
        for c in _LEVELS:
            cell = long[(long["strategy"] == s) & (long["length"] == c)]["value"]
            key = f"{s[:4]}_{c}"
            cell_stats[f"{key}_M"] = round(float(cell.mean()), 4) if len(cell) > 0 else 0.0
            cell_stats[f"{key}_SD"] = round(float(cell.std(ddof=1)), 4) if len(cell) > 1 else 0.0
            cell_stats[f"{key}_n"] = int(len(cell))

    return {
        "Variable": name,
        "N": n_total,
        "F_Strategy": round(float(f_s), 3),
        "df_s": f"{df_s}, {df_w}",
        "p_Strategy": float(p_s),
        "p_Strategy_fmt": _fmt_p(float(p_s)),
        "eta_Strategy": round(float(eta_s), 4),
        "F_Length": round(float(f_l), 3),
        "df_l": f"{df_l}, {df_w}",
        "p_Length": float(p_l),
        "p_Length_fmt": _fmt_p(float(p_l)),
        "eta_Length": round(float(eta_l), 4),
        "F_Interaction": round(float(f_i), 3),
        "df_i": f"{df_i}, {df_w}",
        "p_Interaction": float(p_i),
        "p_Interaction_fmt": _fmt_p(float(p_i)),
        "eta_Interaction": round(float(eta_i), 4),
        "Interaction_Sig": "Yes" if p_i < 0.05 else "No",
        **cell_stats,
    }


def simple_effects_tests(df: pd.DataFrame, sv: str, rv: str, name: str) -> List[Dict[str, object]]:
    """Bonferroni-corrected simple effects within each length category."""
    results: List[Dict[str, object]] = []
    for cat in _LEVELS:
        mask = df["length_category"] == cat
        summ_vals = df.loc[mask, sv].astype(float).to_numpy()
        rag_vals = df.loc[mask, rv].astype(float).to_numpy()
        n = int(len(summ_vals))
        if n < 3:
            continue

        diff = summ_vals - rag_vals
        mean_d = float(np.mean(diff))
        sd_d = float(np.std(diff, ddof=1)) if n > 1 else 0.0
        d = mean_d / sd_d if sd_d > 0 else 0.0

        t_stat, p_val = ttest_rel(summ_vals, rag_vals)
        p_bonf = min(float(p_val) * 3, 1.0)

        results.append(
            {
                "Variable": name,
                "Length": cat.capitalize(),
                "n": n,
                "M_Summ": round(float(np.mean(summ_vals)), 4),
                "M_RAG": round(float(np.mean(rag_vals)), 4),
                "Mean_Diff": round(mean_d, 4),
                "t": round(float(t_stat), 3),
                "df": n - 1,
                "p_raw": float(p_val),
                "p_raw_fmt": _fmt_p(float(p_val)),
                "p_bonf": float(p_bonf),
                "p_bonf_fmt": _fmt_p(float(p_bonf)),
                "d": round(d, 3),
                "Sig_bonf": "Yes" if p_bonf < 0.05 else "No",
            }
        )
    return results
