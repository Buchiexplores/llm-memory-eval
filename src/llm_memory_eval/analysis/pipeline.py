"""End-to-end analysis pipeline producing the study's statistical tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from llm_memory_eval.analysis.anova import simple_effects_tests, two_way_anova
from llm_memory_eval.analysis.assumptions import (
    levene_test,
    log_transform_check,
    shapiro_normality,
)
from llm_memory_eval.analysis.descriptive import descriptive_summary
from llm_memory_eval.analysis.paired import holm_correction, paired_test
from llm_memory_eval.utils.logging import get_logger


log = get_logger(__name__)


_RQ1_VARS = [
    ("summ_f1", "rag_f1", "Recall Accuracy (F1)"),
    ("summ_em", "rag_em", "Exact Match"),
    ("summ_consistency", "rag_consistency", "Consistency Rate"),
    ("summ_contradiction", "rag_contradiction", "Contradiction Rate"),
]

_RQ2_VARS = [
    ("summ_total_latency", "rag_total_latency", "Response Latency (s)"),
    ("summ_total_tokens", "rag_total_tokens", "Token Usage"),
    ("summ_storage", "rag_storage", "Storage Overhead (bytes)"),
]

_RQ3_VARS = [
    ("summ_f1", "rag_f1", "Recall Accuracy (F1)"),
    ("summ_consistency", "rag_consistency", "Consistency Rate"),
    ("summ_total_latency", "rag_total_latency", "Response Latency (s)"),
    ("summ_total_tokens", "rag_total_tokens", "Token Usage"),
    ("summ_storage", "rag_storage", "Storage Overhead (bytes)"),
]


def _load_results(results_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(results_dir / "experiment_results.csv")
    df["summ_total_latency"] = df["summ_prep_time"] + df["summ_latency"]
    df["rag_total_latency"] = df["rag_prep_time"] + df["rag_latency"]
    if "summ_prompt_tokens" in df.columns:
        df["summ_total_tokens"] = df["summ_prompt_tokens"] + df["summ_output_tokens"]
        df["rag_total_tokens"] = df["rag_prompt_tokens"] + df["rag_output_tokens"]
    if "summ_consistency" not in df.columns:
        df["summ_consistency"] = (df["summ_f1"] >= 0.30).astype(float)
        df["rag_consistency"] = (df["rag_f1"] >= 0.30).astype(float)
        df["summ_contradiction"] = (df["summ_f1"] < 0.05).astype(float)
        df["rag_contradiction"] = (df["rag_f1"] < 0.05).astype(float)
    return df


def run_full_analysis(results_dir: Path) -> Dict[str, Any]:
    """Run the full statistical analysis and write outputs.

    Produces:
      - ``results_dir/statistical_analyses.json`` (all analyses combined)
      - ``results_dir/tables/table_descriptive.csv``
      - ``results_dir/tables/table_rq1.csv``
      - ``results_dir/tables/table_rq2.csv``
      - ``results_dir/tables/table_rq3.csv``
      - ``results_dir/tables/table_simple_effects.csv``
      - ``results_dir/tables/table_by_benchmark.csv``
    """
    results_dir = Path(results_dir)
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    df = _load_results(results_dir)
    log.info("Loaded %d paired instances", len(df))

    analyses: Dict[str, Any] = {}

    analyses["descriptive"] = descriptive_summary(df, _RQ1_VARS + _RQ2_VARS)

    rq1 = [paired_test(df[sv].to_numpy(), df[rv].to_numpy(), name) for sv, rv, name in _RQ1_VARS]
    analyses["rq1"] = holm_correction(rq1)

    rq2 = [paired_test(df[sv].to_numpy(), df[rv].to_numpy(), name) for sv, rv, name in _RQ2_VARS]
    analyses["rq2"] = holm_correction(rq2)

    analyses["log_transform_latency"] = log_transform_check(
        df["summ_total_latency"].to_numpy(),
        df["rag_total_latency"].to_numpy(),
    )

    rq3 = [two_way_anova(df, sv, rv, name) for sv, rv, name in _RQ3_VARS]
    analyses["rq3"] = rq3

    simple: List[Dict[str, Any]] = []
    for sv, rv, name in _RQ3_VARS:
        matching = next((r for r in rq3 if r["Variable"] == name), None)
        if matching and matching["Interaction_Sig"] == "Yes":
            simple.extend(simple_effects_tests(df, sv, rv, name))
    analyses["simple_effects"] = simple

    analyses["levene"] = levene_test(df, _RQ3_VARS)

    analyses["normality"] = shapiro_normality(df, _RQ1_VARS + _RQ2_VARS)

    analyses["wilcoxon_rq1"] = [
        {
            "Variable": r["Variable"],
            "W": r["Wilcoxon_W"],
            "p": r["Wilcoxon_p"],
            "p_fmt": r["Wilcoxon_p_fmt"],
            "Decision": r["Wilcoxon_Decision"],
        }
        for r in rq1
    ]
    analyses["wilcoxon_rq2"] = [
        {
            "Variable": r["Variable"],
            "W": r["Wilcoxon_W"],
            "p": r["Wilcoxon_p"],
            "p_fmt": r["Wilcoxon_p_fmt"],
            "Decision": r["Wilcoxon_Decision"],
        }
        for r in rq2
    ]

    bench_rows = []
    for bench in sorted(df["benchmark"].unique()):
        sub = df[df["benchmark"] == bench]
        bench_rows.append(
            {
                "Benchmark": bench,
                "N": int(len(sub)),
                "Summ_F1_M": round(float(sub["summ_f1"].mean()), 4),
                "Summ_F1_SD": round(float(sub["summ_f1"].std(ddof=1)), 4),
                "RAG_F1_M": round(float(sub["rag_f1"].mean()), 4),
                "RAG_F1_SD": round(float(sub["rag_f1"].std(ddof=1)), 4),
                "Summ_Lat_M": round(float(sub["summ_total_latency"].mean()), 4),
                "RAG_Lat_M": round(float(sub["rag_total_latency"].mean()), 4),
            }
        )
    analyses["by_benchmark"] = bench_rows

    (results_dir / "statistical_analyses.json").write_text(
        json.dumps(analyses, indent=2, default=str), encoding="utf-8"
    )

    pd.DataFrame(analyses["descriptive"]).to_csv(tables_dir / "table_descriptive.csv", index=False)
    pd.DataFrame(rq1).to_csv(tables_dir / "table_rq1.csv", index=False)
    pd.DataFrame(rq2).to_csv(tables_dir / "table_rq2.csv", index=False)
    pd.DataFrame(rq3).to_csv(tables_dir / "table_rq3.csv", index=False)
    if simple:
        pd.DataFrame(simple).to_csv(tables_dir / "table_simple_effects.csv", index=False)
    pd.DataFrame(bench_rows).to_csv(tables_dir / "table_by_benchmark.csv", index=False)

    log.info("Analyses written to %s", results_dir)
    return analyses
