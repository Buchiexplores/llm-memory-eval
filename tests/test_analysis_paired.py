"""Unit tests for paired-test wrapper and Holm-Bonferroni correction."""

from __future__ import annotations

import numpy as np

from llm_memory_eval.analysis.paired import holm_correction, paired_test


class TestPairedTest:
    def test_returns_expected_keys(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.normal(0.4, 0.1, 50)
        b = rng.normal(0.5, 0.1, 50)
        result = paired_test(a, b, "Demo")
        expected = {
            "Variable",
            "Test",
            "N",
            "M_Summ",
            "SD_Summ",
            "M_RAG",
            "SD_RAG",
            "Mean_Diff",
            "Statistic",
            "p",
            "Cohen_d",
            "CI_95",
            "Wilcoxon_W",
            "Wilcoxon_p",
        }
        assert expected.issubset(result.keys())

    def test_rejects_mismatched_lengths(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            paired_test(np.array([1, 2]), np.array([1, 2, 3]), "Mismatch")

    def test_significant_when_means_differ(self) -> None:
        rng = np.random.default_rng(1)
        a = rng.normal(0.2, 0.05, 100)
        b = rng.normal(0.4, 0.05, 100)
        result = paired_test(a, b, "Difference")
        assert result["p"] < 0.001
        assert result["Null_Decision"] == "Rejected"


class TestHolmCorrection:
    def test_monotonic(self) -> None:
        results = [
            {"Variable": "A", "p": 0.001},
            {"Variable": "B", "p": 0.02},
            {"Variable": "C", "p": 0.5},
            {"Variable": "D", "p": 0.9},
        ]
        adjusted = holm_correction(results)
        adj_vals = [r["p_adj"] for r in adjusted]
        # Sorted by raw p, adjusted p should be non-decreasing
        sorted_pairs = sorted(zip([r["p"] for r in results], adj_vals, strict=False))
        sorted_adj = [a for _, a in sorted_pairs]
        for i in range(1, len(sorted_adj)):
            assert sorted_adj[i] >= sorted_adj[i - 1]

    def test_decisions_added(self) -> None:
        results = [{"Variable": "A", "p": 0.0001}, {"Variable": "B", "p": 0.40}]
        adjusted = holm_correction(results)
        assert adjusted[0]["Sig_adj"] in {"Yes", "No"}
        assert adjusted[0]["Null_Decision_adj"] in {"Rejected", "Failed to Reject"}
