"""Publication-quality matplotlib figures.

The figure set covers all three research questions plus diagnostic
quantile-quantile plots used to inspect the normality assumption. Style
defaults follow common publication conventions (serif typeface, gridlines
for legibility, no top/right spines, 300 dpi).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_COLOR_SUMM = "#3A86FF"
_COLOR_RAG = "#FF6B6B"
_COLOR_SUMM_DARK = "#2563EB"
_COLOR_RAG_DARK = "#DC2626"


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


def _stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "†"
    return "ns"


def generate_figures(results_dir: Path) -> None:
    """Render the seven publication figures into ``results_dir/figures/``."""
    results_dir = Path(results_dir)
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    with (results_dir / "statistical_analyses.json").open("r", encoding="utf-8") as fh:
        analyses = json.load(fh)
    df = pd.read_csv(results_dir / "experiment_results.csv")
    df["summ_total_latency"] = df["summ_prep_time"] + df["summ_latency"]
    df["rag_total_latency"] = df["rag_prep_time"] + df["rag_latency"]
    if "summ_prompt_tokens" in df.columns:
        df["summ_total_tokens"] = df["summ_prompt_tokens"] + df["summ_output_tokens"]
        df["rag_total_tokens"] = df["rag_prompt_tokens"] + df["rag_output_tokens"]

    _figure_rq1(analyses, fig_dir)
    _figure_rq2(df, analyses, fig_dir)
    _figure_rq3_interaction(
        analyses, fig_dir, index=0, key="f1_interaction", ylabel="Mean Recall Accuracy (F1)"
    )
    _figure_rq3_interaction(
        analyses, fig_dir, index=2, key="latency_interaction", ylabel="Response Latency (s)"
    )
    _figure_rq3_storage(analyses, fig_dir)
    _figure_benchmark(analyses, fig_dir)
    _figure_qq(df, fig_dir)


def _figure_rq1(analyses, fig_dir: Path) -> None:
    rq1 = analyses["rq1"]
    labels = ["Recall Accuracy\n(F1)", "Exact Match", "Consistency\nRate", "Contradiction\nRate"]
    summ = [r["M_Summ"] for r in rq1]
    rag = [r["M_RAG"] for r in rq1]
    summ_sd = [r["SD_Summ"] for r in rq1]
    rag_sd = [r["SD_RAG"] for r in rq1]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    w = 0.32
    ax.bar(
        x - w / 2,
        summ,
        w,
        yerr=summ_sd,
        capsize=5,
        color=_COLOR_SUMM,
        edgecolor=_COLOR_SUMM_DARK,
        linewidth=1.2,
        label="Summarization",
        alpha=0.85,
    )
    ax.bar(
        x + w / 2,
        rag,
        w,
        yerr=rag_sd,
        capsize=5,
        color=_COLOR_RAG,
        edgecolor=_COLOR_RAG_DARK,
        linewidth=1.2,
        label="RAG",
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Score", fontweight="bold")
    top = max(
        max(s + e for s, e in zip(summ, summ_sd, strict=False)),
        max(r + e for r, e in zip(rag, rag_sd, strict=False)),
    )
    ax.set_ylim(0, max(0.05, top) * 1.2)
    for i, r in enumerate(rq1):
        ax.text(
            i,
            top * 1.07,
            _stars(r.get("p_adj", r["p"])),
            ha="center",
            fontsize=11,
            fontweight="bold",
        )
    ax.legend(loc="upper right")
    ax.text(
        0.02,
        0.98,
        "* p < .05    ** p < .01    *** p < .001    † p < .10    "
        "ns = not significant (Holm-Bonferroni adjusted)",
        transform=ax.transAxes,
        fontsize=8.5,
        va="top",
        fontstyle="italic",
        color="#666666",
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "figure1_rq1_recall_consistency.png")
    plt.close()


def _figure_rq2(df, analyses, fig_dir: Path) -> None:
    rq2_lookup = {r["Variable"]: r for r in analyses["rq2"]}
    lat_p = rq2_lookup.get("Response Latency (s)", {}).get("p_adj", 1.0)
    tok_p = rq2_lookup.get("Token Usage", {}).get("p_adj", 1.0)
    sto_p = rq2_lookup.get("Storage Overhead (bytes)", {}).get("p_adj", 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    lat_cap = max(df["summ_total_latency"].quantile(0.95), df["rag_total_latency"].quantile(0.95))
    _boxplot(
        axes[0],
        [
            df["summ_total_latency"].clip(upper=lat_cap),
            df["rag_total_latency"].clip(upper=lat_cap),
        ],
        title="(a) Response Latency",
        ylabel="Response Latency (s)",
        marker=_stars(lat_p),
    )
    _boxplot(
        axes[1],
        [
            df["summ_total_tokens"][df["summ_total_tokens"] > 0],
            df["rag_total_tokens"][df["rag_total_tokens"] > 0],
        ],
        title="(b) Token Usage",
        ylabel="Token Usage",
        marker=_stars(tok_p),
    )
    _boxplot(
        axes[2],
        [
            df["summ_storage"][df["summ_storage"] > 0] / 1000,
            df["rag_storage"][df["rag_storage"] > 0] / 1000,
        ],
        title="(c) Storage Overhead",
        ylabel="Storage Overhead (KB)",
        marker=_stars(sto_p),
    )

    fig.text(
        0.5,
        -0.03,
        "* p < .05    ** p < .01    *** p < .001    † p < .10    " "ns = not significant",
        ha="center",
        fontsize=10,
        fontstyle="italic",
        color="#666666",
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "figure2_rq2_efficiency.png")
    plt.close()


def _boxplot(ax, data, *, title, ylabel, marker) -> None:
    bp = ax.boxplot(
        data,
        tick_labels=["Summarization", "RAG"],
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "white", "linewidth": 2},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#888888", "alpha": 0.4},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )
    bp["boxes"][0].set(facecolor=_COLOR_SUMM, alpha=0.8)
    bp["boxes"][1].set(facecolor=_COLOR_RAG, alpha=0.8)
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title)
    ax.text(
        1.5,
        ax.get_ylim()[1] * 0.92,
        marker,
        ha="center",
        fontsize=14,
        fontweight="bold",
        color=_COLOR_RAG_DARK,
    )


def _figure_rq3_interaction(analyses, fig_dir: Path, *, index: int, key: str, ylabel: str) -> None:
    rq = analyses["rq3"][index]
    cats = ["short", "medium", "long"]
    summ = [rq[f"Summ_{c}_M"] for c in cats]
    rag = [rq[f"RAG_{c}_M"] for c in cats]
    summ_sd = [rq[f"Summ_{c}_SD"] for c in cats]
    rag_sd = [rq[f"RAG_{c}_SD"] for c in cats]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(3)
    ax.errorbar(
        x,
        summ,
        yerr=summ_sd,
        marker="s",
        color=_COLOR_SUMM_DARK,
        linewidth=2,
        capsize=6,
        label="Summarization",
        linestyle="--",
        markersize=10,
        markerfacecolor=_COLOR_SUMM,
    )
    ax.errorbar(
        x,
        rag,
        yerr=rag_sd,
        marker="^",
        color=_COLOR_RAG_DARK,
        linewidth=2,
        capsize=6,
        label="RAG",
        linestyle="-",
        markersize=10,
        markerfacecolor=_COLOR_RAG,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["Short", "Medium", "Long"], fontweight="bold")
    ax.set_xlabel("Conversation Length Category", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.legend(loc="upper right")
    top = max(
        max(m + s for m, s in zip(summ, summ_sd, strict=False)),
        max(m + s for m, s in zip(rag, rag_sd, strict=False)),
    )
    ax.set_ylim(0, max(0.05, top) * 1.15)
    plt.tight_layout()
    plt.savefig(fig_dir / f"figure3_rq3_{key}.png")
    plt.close()


def _figure_rq3_storage(analyses, fig_dir: Path) -> None:
    rq = analyses["rq3"][4]  # storage
    cats = ["short", "medium", "long"]
    summ = [rq[f"Summ_{c}_M"] / 1000 for c in cats]
    rag = [rq[f"RAG_{c}_M"] / 1000 for c in cats]
    summ_sd = [rq[f"Summ_{c}_SD"] / 1000 for c in cats]
    rag_sd = [rq[f"RAG_{c}_SD"] / 1000 for c in cats]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(3)
    ax.errorbar(
        x,
        summ,
        yerr=summ_sd,
        marker="s",
        color=_COLOR_SUMM_DARK,
        linewidth=2,
        capsize=6,
        label="Summarization",
        linestyle="--",
        markersize=10,
        markerfacecolor=_COLOR_SUMM,
    )
    ax.errorbar(
        x,
        rag,
        yerr=rag_sd,
        marker="^",
        color=_COLOR_RAG_DARK,
        linewidth=2,
        capsize=6,
        label="RAG",
        linestyle="-",
        markersize=10,
        markerfacecolor=_COLOR_RAG,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["Short", "Medium", "Long"], fontweight="bold")
    ax.set_xlabel("Conversation Length Category", fontweight="bold")
    ax.set_ylabel("Memory Storage Overhead (KB)", fontweight="bold")
    ax.legend(loc="upper left")

    eta = rq.get("eta_Interaction", 0.0)
    p_int = rq.get("p_Interaction", 1.0)
    p_txt = "< .001" if p_int < 0.001 else f"= {p_int:.3f}"
    top = max(rag) + max(rag_sd) + 20
    ax.annotate(
        f"Strategy x Length Interaction\n(ηp² = {eta:.3f}, p {p_txt})",
        xy=(2, rag[2]),
        xytext=(1.2, top),
        fontsize=9,
        fontstyle="italic",
        color=_COLOR_RAG_DARK,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": _COLOR_RAG_DARK, "lw": 1.5},
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "figure4_rq3_storage_interaction.png")
    plt.close()


def _figure_benchmark(analyses, fig_dir: Path) -> None:
    bench = analyses["by_benchmark"]
    names = [b["Benchmark"] for b in bench]
    summ = [b["Summ_F1_M"] for b in bench]
    rag = [b["RAG_F1_M"] for b in bench]
    summ_sd = [b["Summ_F1_SD"] for b in bench]
    rag_sd = [b["RAG_F1_SD"] for b in bench]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(names))
    w = 0.32
    bars1 = ax.bar(
        x - w / 2,
        summ,
        w,
        yerr=summ_sd,
        capsize=5,
        color=_COLOR_SUMM,
        edgecolor=_COLOR_SUMM_DARK,
        linewidth=1.2,
        label="Summarization",
        alpha=0.85,
    )
    bars2 = ax.bar(
        x + w / 2,
        rag,
        w,
        yerr=rag_sd,
        capsize=5,
        color=_COLOR_RAG,
        edgecolor=_COLOR_RAG_DARK,
        linewidth=1.2,
        label="RAG",
        alpha=0.85,
    )
    for group in (bars1, bars2):
        for bar in group:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#444444",
            )
    ax.set_ylabel("Mean Recall Accuracy (F1)", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontweight="bold")
    top = max(
        max(s + e for s, e in zip(summ, summ_sd, strict=False)),
        max(r + e for r, e in zip(rag, rag_sd, strict=False)),
    )
    ax.set_ylim(0, max(0.05, top) * 1.15)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure5_benchmark_comparison.png")
    plt.close()


def _figure_qq(df, fig_dir: Path) -> None:
    from scipy import stats as sp_stats

    if "summ_consistency" not in df.columns:
        df["summ_consistency"] = (df["summ_f1"] >= 0.30).astype(float)
        df["rag_consistency"] = (df["rag_f1"] >= 0.30).astype(float)
        df["summ_contradiction"] = (df["summ_f1"] < 0.05).astype(float)
        df["rag_contradiction"] = (df["rag_f1"] < 0.05).astype(float)

    qq_vars = [
        ("summ_f1", "rag_f1", "F1 Score"),
        ("summ_em", "rag_em", "Exact Match"),
        ("summ_consistency", "rag_consistency", "Consistency Rate"),
        ("summ_contradiction", "rag_contradiction", "Contradiction Rate"),
        ("summ_total_latency", "rag_total_latency", "Latency (s)"),
        ("summ_total_tokens", "rag_total_tokens", "Token Usage"),
        ("summ_storage", "rag_storage", "Storage (bytes)"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()
    for idx, (sv, rv, label) in enumerate(qq_vars):
        ax = axes[idx]
        diff = df[rv].to_numpy(dtype=float) - df[sv].to_numpy(dtype=float)
        (osm, osr), (slope, intercept, _r) = sp_stats.probplot(diff, dist="norm")
        ax.scatter(
            osm, osr, s=12, alpha=0.6, color=_COLOR_SUMM, edgecolor=_COLOR_SUMM_DARK, linewidth=0.5
        )
        fit = slope * np.array(osm) + intercept
        ax.plot(osm, fit, color=_COLOR_RAG_DARK, linewidth=1.5, linestyle="--")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Theoretical Quantiles", fontsize=8)
        ax.set_ylabel("Sample Quantiles", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[7].set_visible(False)
    fig.suptitle(
        "Q-Q Plots of Paired Difference Scores (RAG − Summarization)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(fig_dir / "figure7_qq_plots.png")
    plt.close()
